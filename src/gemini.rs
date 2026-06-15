//! Native Google Gemini provider.
//!
//! Unlike Groq/DeepSeek/etc., Gemini does not speak the OpenAI
//! `/chat/completions` format, so it gets a dedicated client (like Anthropic).
//! Notable wire differences handled here:
//! - roles are `user` / `model` (no `system`/`tool`/`assistant`);
//! - the system prompt is a top-level `system_instruction`;
//! - tool calls/results use `functionCall` / `functionResponse` parts keyed by
//!   function *name*, not an id — so we synthesise stable ids and correlate
//!   tool results back to names from the preceding assistant turn;
//! - streaming is SSE with no `[DONE]` sentinel (the stream just ends).

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tracing::{info, warn};
use web_time::Instant;

use crate::error::{ProviderError, RetryPolicy, execute_with_retry};
use crate::http::{HttpRequest, HttpResponse, SharedHttpClient, collect_stream_to_string};
use crate::provider::{AgentProviderKind, ModelTiers, ProviderInfo, TextProvider};
use crate::streaming::should_flush_delta;
use crate::{AssistantTurn, ChatMessage, EventSink, MessageRole, RuntimeEvent, ToolCall, ToolDefinition};

const DEFAULT_GEMINI_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone)]
pub struct GeminiClientConfig {
    pub base_url: String,
    pub verbose: bool,
    pub max_tokens: u32,
    pub model_tiers: ModelTiers,
    pub retry: RetryPolicy,
}

impl Default for GeminiClientConfig {
    fn default() -> Self {
        Self {
            base_url: AgentProviderKind::Gemini
                .default_base_url()
                .expect("gemini has a default base url")
                .to_string(),
            verbose: false,
            max_tokens: DEFAULT_GEMINI_MAX_TOKENS,
            model_tiers: AgentProviderKind::Gemini.default_model_tiers(),
            retry: RetryPolicy::default(),
        }
    }
}

impl GeminiClientConfig {
    fn endpoint(&self, model: &str, method: &str) -> String {
        format!(
            "{}/models/{}:{}",
            self.base_url.trim_end_matches('/'),
            model,
            method
        )
    }
}

#[derive(Clone)]
pub struct GeminiClient {
    http_client: SharedHttpClient,
    api_key: String,
    config: GeminiClientConfig,
}

impl GeminiClient {
    pub fn new(http_client: SharedHttpClient, api_key: impl Into<String>) -> Self {
        Self::with_config(http_client, api_key, GeminiClientConfig::default())
    }

    pub fn with_verbose(
        http_client: SharedHttpClient,
        api_key: impl Into<String>,
        verbose: bool,
    ) -> Self {
        Self {
            http_client,
            api_key: api_key.into(),
            config: GeminiClientConfig {
                verbose,
                ..GeminiClientConfig::default()
            },
        }
    }

    pub fn with_config(
        http_client: SharedHttpClient,
        api_key: impl Into<String>,
        config: GeminiClientConfig,
    ) -> Self {
        Self {
            http_client,
            api_key: api_key.into(),
            config,
        }
    }

    pub fn verbose(&self) -> bool {
        self.config.verbose
    }

    fn provider_name(&self) -> &str {
        AgentProviderKind::Gemini.as_str()
    }

    fn prepare(&self, request: HttpRequest) -> HttpRequest {
        request.header("x-goog-api-key", &self.api_key)
    }

    async fn send_classified(&self, request: HttpRequest) -> Result<HttpResponse, ProviderError> {
        let name = self.provider_name();
        execute_with_retry(&self.config.retry, || {
            let request = request.clone();
            async move {
                let response = self
                    .http_client
                    .send(request)
                    .await
                    .map_err(|err| ProviderError::transport(name, err))?;
                if response.is_success() {
                    Ok(response)
                } else {
                    let body = response
                        .text()
                        .unwrap_or_else(|_| "<failed to read response body>".to_string());
                    Err(ProviderError::from_status(name, response.status, body))
                }
            }
        })
        .await
    }

    fn build_payload(
        &self,
        system_prompt: &str,
        messages: &[ChatMessage],
        tool_definitions: &[ToolDefinition],
    ) -> Result<Value> {
        let contents = messages_to_gemini_contents(messages)?;
        let mut payload = Map::new();
        payload.insert(
            "system_instruction".to_string(),
            json!({ "parts": [{ "text": system_prompt }] }),
        );
        payload.insert("contents".to_string(), Value::Array(contents));
        payload.insert(
            "generationConfig".to_string(),
            json!({ "maxOutputTokens": self.config.max_tokens }),
        );
        if !tool_definitions.is_empty() {
            payload.insert(
                "tools".to_string(),
                json!([{ "function_declarations": gemini_function_declarations(tool_definitions) }]),
            );
            payload.insert(
                "tool_config".to_string(),
                json!({ "function_calling_config": { "mode": "AUTO" } }),
            );
        }
        Ok(Value::Object(payload))
    }

    async fn request_assistant_turn_impl(
        &self,
        model: &str,
        system_prompt: &str,
        history: &[ChatMessage],
        tool_definitions: &[ToolDefinition],
    ) -> Result<AssistantTurn> {
        let request_started_at = Instant::now();

        if self.verbose() {
            info!(
                model = %model,
                history_count = history.len(),
                tool_count = tool_definitions.len(),
                system_prompt_chars = system_prompt.chars().count(),
                "requesting assistant turn"
            );
        }

        let payload = self.build_payload(system_prompt, history, tool_definitions)?;
        let request = self
            .prepare(HttpRequest::post(
                self.config.endpoint(model, "generateContent"),
            ))
            .json_body(&payload)?;
        let response = self
            .send_classified(request)
            .await
            .context("failed to call Gemini generateContent")?;

        let body: GenerateContentResponse = response
            .json()
            .context("failed to decode Gemini response")?;
        let assistant_turn = assistant_turn_from_gemini(&body)?;

        if self.verbose() {
            info!(
                model = %model,
                duration_ms = request_started_at.elapsed().as_millis() as u64,
                tool_calls = assistant_turn.tool_calls.len(),
                content_chars = assistant_turn
                    .content
                    .as_deref()
                    .unwrap_or_default()
                    .chars()
                    .count(),
                "assistant turn completed"
            );
        }

        Ok(assistant_turn)
    }

    async fn stream_message_impl(
        &self,
        model: &str,
        system_prompt: &str,
        messages: &[ChatMessage],
        sink: &mut dyn EventSink,
    ) -> Result<String> {
        let request_started_at = Instant::now();
        sink.emit(RuntimeEvent::AssistantStarted {
            model: model.to_string(),
        })
        .await?;

        if self.verbose() {
            info!(
                model = %model,
                message_count = messages.len(),
                system_prompt_chars = system_prompt.chars().count(),
                "requesting streamed assistant message"
            );
        }

        let payload = self.build_payload(system_prompt, messages, &[])?;
        let url = format!(
            "{}?alt=sse",
            self.config.endpoint(model, "streamGenerateContent")
        );
        let request = self.prepare(HttpRequest::post(url)).json_body(&payload)?;
        let response = self
            .http_client
            .send_streaming(request)
            .await
            .context("failed to call Gemini streamGenerateContent")?;

        if !(200..300).contains(&response.status) {
            let status = response.status;
            let body = collect_stream_to_string(response.body).await;
            if self.verbose() {
                warn!(
                    model = %model,
                    duration_ms = request_started_at.elapsed().as_millis() as u64,
                    status,
                    body = %body,
                    "streamed assistant message request failed"
                );
            }
            return Err(ProviderError::from_status(self.provider_name(), status, body).into());
        }

        let mut response_stream = response.body;
        let mut raw_event_buffer = String::new();
        let mut full_message = String::new();
        let mut pending_delta = String::new();
        let mut chunk_count = 0usize;
        let mut parsed_event_count = 0usize;
        let mut flush_count = 0usize;

        while let Some(chunk) = response_stream.next().await {
            let chunk = chunk.context("failed to read Gemini stream response chunk")?;
            let chunk_text = std::str::from_utf8(&chunk)
                .context("failed to decode Gemini stream chunk as UTF-8")?;
            chunk_count += 1;
            raw_event_buffer.push_str(&chunk_text.replace('\r', ""));

            while let Some(event_end_index) = raw_event_buffer.find("\n\n") {
                let raw_event = raw_event_buffer[..event_end_index].to_string();
                raw_event_buffer = raw_event_buffer[event_end_index + 2..].to_string();
                parsed_event_count += 1;

                match parse_gemini_stream_event(&raw_event) {
                    Ok(deltas) => {
                        for delta in deltas {
                            full_message.push_str(&delta);
                            pending_delta.push_str(&delta);
                            if should_flush_delta(&pending_delta) {
                                let flush_value = pending_delta.clone();
                                pending_delta.clear();
                                flush_count += 1;
                                sink.emit(RuntimeEvent::AssistantDelta { delta: flush_value })
                                    .await?;
                            }
                        }
                    }
                    Err(error) => {
                        warn!(
                            raw_event = %raw_event,
                            error = %format!("{error:#}"),
                            "failed to parse Gemini stream event"
                        );
                    }
                }
            }
        }

        if !pending_delta.is_empty() {
            let flush_value = pending_delta.clone();
            pending_delta.clear();
            flush_count += 1;
            sink.emit(RuntimeEvent::AssistantDelta { delta: flush_value })
                .await?;
        }

        let message = full_message.trim().to_string();
        if message.is_empty() {
            return Err(anyhow!("Gemini streamed message returned an empty message"));
        }

        if self.verbose() {
            info!(
                model = %model,
                duration_ms = request_started_at.elapsed().as_millis() as u64,
                chunk_count,
                parsed_event_count,
                flush_count,
                message_chars = message.chars().count(),
                "streamed assistant message completed"
            );
        }

        Ok(message)
    }
}

impl ProviderInfo for GeminiClient {
    fn kind(&self) -> AgentProviderKind {
        AgentProviderKind::Gemini
    }

    fn verbose(&self) -> bool {
        self.config.verbose
    }

    fn model_tiers(&self) -> &ModelTiers {
        &self.config.model_tiers
    }
}

#[async_trait]
impl TextProvider for GeminiClient {
    async fn request_assistant_turn(
        &self,
        model: &str,
        system_prompt: &str,
        history: &[ChatMessage],
        tool_definitions: &[ToolDefinition],
    ) -> Result<AssistantTurn> {
        self.request_assistant_turn_impl(model, system_prompt, history, tool_definitions)
            .await
    }

    async fn stream_message(
        &self,
        model: &str,
        system_prompt: &str,
        messages: &[ChatMessage],
        sink: &mut dyn EventSink,
    ) -> Result<String> {
        self.stream_message_impl(model, system_prompt, messages, sink)
            .await
    }
}

/// Deterministic synthetic id for a Gemini function call, since the API keys
/// tool calls/results by name rather than id. Stable within a turn so tool
/// results round-trip back to the right function name.
fn synth_tool_call_id(name: &str, index: usize) -> String {
    format!("call_{name}_{index}")
}

pub fn parse_gemini_stream_event(raw_event: &str) -> Result<Vec<String>> {
    let data_lines = raw_event
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .collect::<Vec<_>>();

    if data_lines.is_empty() {
        return Ok(Vec::new());
    }

    let data = data_lines.join("\n");
    let payload: GenerateContentResponse = serde_json::from_str(&data)
        .with_context(|| format!("failed to decode Gemini stream payload: {data}"))?;

    let mut deltas = Vec::new();
    for candidate in &payload.candidates {
        for part in &candidate.content.parts {
            if let Some(text) = &part.text
                && !text.is_empty()
            {
                deltas.push(text.clone());
            }
        }
    }
    Ok(deltas)
}

fn gemini_function_declarations(definitions: &[ToolDefinition]) -> Vec<Value> {
    definitions
        .iter()
        .map(|definition| {
            json!({
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.input_schema,
            })
        })
        .collect()
}

fn messages_to_gemini_contents(messages: &[ChatMessage]) -> Result<Vec<Value>> {
    // Correlate tool_call_id -> function name from assistant turns, so tool
    // results (which only carry the id) can be expressed as Gemini
    // functionResponse parts keyed by name.
    let mut id_to_name: HashMap<String, String> = HashMap::new();
    let mut contents = Vec::new();
    let mut index = 0usize;

    while index < messages.len() {
        let message = &messages[index];
        match message.role {
            MessageRole::System => {
                index += 1;
            }
            MessageRole::User => {
                contents.push(json!({
                    "role": "user",
                    "parts": gemini_user_parts(message),
                }));
                index += 1;
            }
            MessageRole::Assistant => {
                let mut parts = Vec::new();
                if let Some(text) = message.content.clone()
                    && !text.is_empty()
                {
                    parts.push(json!({ "text": text }));
                }
                for call in &message.tool_calls {
                    id_to_name.insert(call.id.clone(), call.name.clone());
                    parts.push(json!({
                        "functionCall": {
                            "name": call.name,
                            "args": call.arguments,
                        }
                    }));
                }
                contents.push(json!({ "role": "model", "parts": parts }));
                index += 1;
            }
            MessageRole::Tool => {
                let mut parts = Vec::new();
                while index < messages.len() && messages[index].role == MessageRole::Tool {
                    let tool_message = &messages[index];
                    let tool_call_id = tool_message
                        .tool_call_id
                        .clone()
                        .ok_or_else(|| anyhow!("tool message missing tool_call_id"))?;
                    let name = id_to_name
                        .get(&tool_call_id)
                        .cloned()
                        .unwrap_or_else(|| tool_call_id.clone());
                    let response_value: Value =
                        serde_json::from_str(&tool_message.content.clone().unwrap_or_default())
                            .unwrap_or_else(|_| {
                                json!({ "result": tool_message.content.clone().unwrap_or_default() })
                            });
                    parts.push(json!({
                        "functionResponse": {
                            "name": name,
                            "response": { "result": response_value },
                        }
                    }));
                    index += 1;
                }
                if !parts.is_empty() {
                    contents.push(json!({ "role": "user", "parts": parts }));
                }
            }
        }
    }

    Ok(contents)
}

/// Build Gemini `parts` (text + inline_data/file_data) for a user message,
/// including any attachments.
fn gemini_user_parts(message: &ChatMessage) -> Vec<Value> {
    use crate::message::AttachmentSource;

    let mut parts = Vec::new();
    parts.push(json!({ "text": message.content.clone().unwrap_or_default() }));

    for attachment in &message.attachments {
        let part = match &attachment.source {
            AttachmentSource::Base64(data) => json!({
                "inline_data": {
                    "mime_type": attachment.media_type,
                    "data": data,
                }
            }),
            AttachmentSource::Url(url) => json!({
                "file_data": {
                    "mime_type": attachment.media_type,
                    "file_uri": url,
                }
            }),
        };
        parts.push(part);
    }

    parts
}

fn assistant_turn_from_gemini(response: &GenerateContentResponse) -> Result<AssistantTurn> {
    let mut text = String::new();
    let mut tool_calls = Vec::new();

    if let Some(candidate) = response.candidates.first() {
        for (part_index, part) in candidate.content.parts.iter().enumerate() {
            if let Some(value) = &part.text {
                text.push_str(value);
            }
            if let Some(call) = &part.function_call {
                tool_calls.push(ToolCall {
                    id: synth_tool_call_id(&call.name, part_index),
                    name: call.name.clone(),
                    arguments: call.args.clone().unwrap_or_else(|| Value::Object(Map::new())),
                });
            }
        }
    }

    let content = (!text.trim().is_empty()).then_some(text);

    Ok(AssistantTurn {
        content,
        tool_calls,
    })
}

#[derive(Debug, Deserialize)]
struct GenerateContentResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    #[serde(default)]
    content: GeminiContent,
}

#[derive(Debug, Default, Deserialize)]
struct GeminiContent {
    #[serde(default)]
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
struct GeminiPart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default, rename = "functionCall")]
    function_call: Option<GeminiFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    #[serde(default)]
    args: Option<Value>,
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde_json::json;

    use super::{messages_to_gemini_contents, parse_gemini_stream_event};
    use crate::{ChatMessage, ToolCall};

    #[test]
    fn maps_tool_result_to_function_response_by_name() -> Result<()> {
        let messages = vec![
            ChatMessage::assistant_with_tools(
                Some("Checking.".to_string()),
                vec![ToolCall {
                    id: "call_search_products_0".to_string(),
                    name: "search_products".to_string(),
                    arguments: json!({ "query": "lagavulin" }),
                }],
            ),
            ChatMessage::tool("call_search_products_0", "{\"ok\":true}"),
        ];

        let contents = messages_to_gemini_contents(&messages)?;
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0]["role"], json!("model"));
        assert_eq!(
            contents[0]["parts"][1]["functionCall"]["name"],
            json!("search_products")
        );
        assert_eq!(contents[1]["role"], json!("user"));
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["name"],
            json!("search_products")
        );
        Ok(())
    }

    #[test]
    fn parses_streamed_text_part() -> Result<()> {
        let raw_event =
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]}}]}\n\n";
        assert_eq!(parse_gemini_stream_event(raw_event)?, vec!["Hello".to_string()]);
        Ok(())
    }
}
