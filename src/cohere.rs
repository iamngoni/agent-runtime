//! Native Cohere v2 chat provider.
//!
//! Cohere's `/v2/chat` API is close to OpenAI's shape (role-tagged messages,
//! function tools, JSON-string tool arguments) but differs enough to warrant a
//! dedicated client: assistant `content` comes back as an array of typed
//! blocks, tool calls live in a sibling `tool_calls` field, and the SSE stream
//! uses typed events (`content-delta`, `message-end`) with no `[DONE]`
//! sentinel.

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tracing::{info, warn};
use web_time::Instant;

use crate::error::{ProviderError, RetryPolicy, execute_with_retry};
use crate::http::{HttpRequest, HttpResponse, SharedHttpClient, collect_stream_to_string};
use crate::message::{AttachmentKind, AttachmentSource};
use crate::provider::{AgentProviderKind, ModelTiers, ProviderInfo, TextProvider};
use crate::streaming::should_flush_delta;
use crate::{AssistantTurn, ChatMessage, EventSink, MessageRole, RuntimeEvent, ToolCall, ToolDefinition};

const DEFAULT_COHERE_BASE_URL: &str = "https://api.cohere.com/v2";

#[derive(Debug, Clone)]
pub struct CohereClientConfig {
    pub base_url: String,
    pub verbose: bool,
    pub model_tiers: ModelTiers,
    pub retry: RetryPolicy,
}

impl Default for CohereClientConfig {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_COHERE_BASE_URL.to_string(),
            verbose: false,
            model_tiers: ModelTiers::new("command-r-plus", "command-r", "command-r-plus"),
            retry: RetryPolicy::default(),
        }
    }
}

impl CohereClientConfig {
    fn chat_url(&self) -> String {
        format!("{}/chat", self.base_url.trim_end_matches('/'))
    }
}

#[derive(Clone)]
pub struct CohereClient {
    http_client: SharedHttpClient,
    api_key: String,
    config: CohereClientConfig,
}

impl CohereClient {
    pub fn new(http_client: SharedHttpClient, api_key: impl Into<String>) -> Self {
        Self::with_config(http_client, api_key, CohereClientConfig::default())
    }

    pub fn with_verbose(
        http_client: SharedHttpClient,
        api_key: impl Into<String>,
        verbose: bool,
    ) -> Self {
        Self {
            http_client,
            api_key: api_key.into(),
            config: CohereClientConfig {
                verbose,
                ..CohereClientConfig::default()
            },
        }
    }

    pub fn with_config(
        http_client: SharedHttpClient,
        api_key: impl Into<String>,
        config: CohereClientConfig,
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
        // Cohere is reached via the Custom kind; surface a stable name.
        "cohere"
    }

    fn prepare(&self, request: HttpRequest) -> HttpRequest {
        request.bearer_auth(&self.api_key)
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
        model: &str,
        system_prompt: &str,
        messages: &[ChatMessage],
        tool_definitions: &[ToolDefinition],
        stream: bool,
    ) -> Result<Value> {
        let mut wire_messages = Vec::with_capacity(messages.len() + 1);
        wire_messages.push(json!({ "role": "system", "content": system_prompt }));
        for message in messages {
            wire_messages.push(message_to_cohere_json(message)?);
        }

        let mut payload = Map::new();
        payload.insert("model".to_string(), Value::String(model.to_string()));
        payload.insert("messages".to_string(), Value::Array(wire_messages));
        if stream {
            payload.insert("stream".to_string(), Value::Bool(true));
        }
        if !tool_definitions.is_empty() {
            payload.insert(
                "tools".to_string(),
                Value::Array(cohere_tools(tool_definitions)),
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
                "requesting assistant turn"
            );
        }

        let payload = self.build_payload(model, system_prompt, history, tool_definitions, false)?;
        let request = self
            .prepare(HttpRequest::post(self.config.chat_url()))
            .json_body(&payload)?;
        let response = self
            .send_classified(request)
            .await
            .context("failed to call Cohere chat")?;

        let body: CohereChatResponse =
            response.json().context("failed to decode Cohere response")?;
        let assistant_turn = assistant_turn_from_cohere(body.message)?;

        if self.verbose() {
            info!(
                model = %model,
                duration_ms = request_started_at.elapsed().as_millis() as u64,
                tool_calls = assistant_turn.tool_calls.len(),
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

        let payload = self.build_payload(model, system_prompt, messages, &[], true)?;
        let request = self
            .prepare(HttpRequest::post(self.config.chat_url()))
            .json_body(&payload)?;
        let response = self
            .http_client
            .send_streaming(request)
            .await
            .context("failed to call Cohere streaming chat")?;

        if !(200..300).contains(&response.status) {
            let status = response.status;
            let body = collect_stream_to_string(response.body).await;
            if self.verbose() {
                warn!(model = %model, status, body = %body, "streamed message failed");
            }
            return Err(ProviderError::from_status(self.provider_name(), status, body).into());
        }

        let mut response_stream = response.body;
        let mut raw_event_buffer = String::new();
        let mut full_message = String::new();
        let mut pending_delta = String::new();
        let mut chunk_count = 0usize;
        let mut flush_count = 0usize;

        while let Some(chunk) = response_stream.next().await {
            let chunk = chunk.context("failed to read Cohere stream chunk")?;
            let chunk_text = std::str::from_utf8(&chunk)
                .context("failed to decode Cohere stream chunk as UTF-8")?;
            chunk_count += 1;
            raw_event_buffer.push_str(&chunk_text.replace('\r', ""));

            while let Some(event_end_index) = raw_event_buffer.find("\n\n") {
                let raw_event = raw_event_buffer[..event_end_index].to_string();
                raw_event_buffer = raw_event_buffer[event_end_index + 2..].to_string();

                match parse_cohere_stream_event(&raw_event) {
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
                        warn!(raw_event = %raw_event, error = %format!("{error:#}"), "failed to parse Cohere stream event");
                    }
                }
            }
        }

        if !pending_delta.is_empty() {
            flush_count += 1;
            sink.emit(RuntimeEvent::AssistantDelta { delta: pending_delta.clone() })
                .await?;
        }

        let message = full_message.trim().to_string();
        if message.is_empty() {
            return Err(anyhow!("Cohere streamed message returned an empty message"));
        }

        if self.verbose() {
            info!(
                model = %model,
                duration_ms = request_started_at.elapsed().as_millis() as u64,
                chunk_count,
                flush_count,
                message_chars = message.chars().count(),
                "streamed assistant message completed"
            );
        }

        Ok(message)
    }
}

impl ProviderInfo for CohereClient {
    fn kind(&self) -> AgentProviderKind {
        AgentProviderKind::Custom("cohere".to_string())
    }

    fn verbose(&self) -> bool {
        self.config.verbose
    }

    fn model_tiers(&self) -> &ModelTiers {
        &self.config.model_tiers
    }
}

#[async_trait]
impl TextProvider for CohereClient {
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

pub fn parse_cohere_stream_event(raw_event: &str) -> Result<Vec<String>> {
    let data_lines = raw_event
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .collect::<Vec<_>>();

    if data_lines.is_empty() {
        return Ok(Vec::new());
    }

    let data = data_lines.join("\n");
    if data == "[DONE]" {
        return Ok(Vec::new());
    }

    let event: CohereStreamEvent = serde_json::from_str(&data)
        .with_context(|| format!("failed to decode Cohere stream payload: {data}"))?;

    if event.kind.as_deref() == Some("content-delta")
        && let Some(text) = event
            .delta
            .and_then(|d| d.message)
            .and_then(|m| m.content)
            .and_then(|c| c.text)
            .filter(|t| !t.is_empty())
    {
        return Ok(vec![text]);
    }
    Ok(Vec::new())
}

fn cohere_tools(definitions: &[ToolDefinition]) -> Vec<Value> {
    definitions
        .iter()
        .map(|definition| {
            json!({
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.input_schema,
                }
            })
        })
        .collect()
}

fn message_to_cohere_json(message: &ChatMessage) -> Result<Value> {
    Ok(match message.role {
        MessageRole::System => json!({
            "role": "system",
            "content": message.content.clone().unwrap_or_default(),
        }),
        MessageRole::User if message.has_attachments() => json!({
            "role": "user",
            "content": cohere_user_content_parts(message),
        }),
        MessageRole::User => json!({
            "role": "user",
            "content": message.content.clone().unwrap_or_default(),
        }),
        MessageRole::Tool => json!({
            "role": "tool",
            "tool_call_id": message
                .tool_call_id
                .clone()
                .ok_or_else(|| anyhow!("tool message missing tool_call_id"))?,
            "content": message.content.clone().unwrap_or_default(),
        }),
        MessageRole::Assistant => {
            let mut object = Map::new();
            object.insert("role".to_string(), Value::String("assistant".to_string()));
            if let Some(content) = message.content.clone() {
                object.insert("content".to_string(), Value::String(content));
            }
            if !message.tool_calls.is_empty() {
                object.insert(
                    "tool_calls".to_string(),
                    Value::Array(message.tool_calls.iter().map(tool_call_to_cohere_json).collect()),
                );
            }
            Value::Object(object)
        }
    })
}

fn cohere_user_content_parts(message: &ChatMessage) -> Vec<Value> {
    let mut parts = Vec::new();
    if let Some(text) = message.content.as_deref()
        && !text.is_empty()
    {
        parts.push(json!({ "type": "text", "text": text }));
    }
    for attachment in &message.attachments {
        // Cohere v2 multimodal supports images via image_url parts; documents
        // use a separate RAG channel, so non-image attachments are skipped.
        if attachment.kind == AttachmentKind::Image {
            let url = match &attachment.source {
                AttachmentSource::Url(url) => url.clone(),
                AttachmentSource::Base64(_) => attachment.data_uri().unwrap_or_default(),
            };
            parts.push(json!({ "type": "image_url", "image_url": { "url": url } }));
        }
    }
    parts
}

fn tool_call_to_cohere_json(call: &ToolCall) -> Value {
    json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string()),
        }
    })
}

fn assistant_turn_from_cohere(message: CohereMessage) -> Result<AssistantTurn> {
    let mut text = String::new();
    for block in message.content.unwrap_or_default() {
        if block.kind.as_deref() == Some("text")
            && let Some(value) = block.text
        {
            text.push_str(&value);
        }
    }

    let mut tool_calls = Vec::new();
    for wire in message.tool_calls.unwrap_or_default() {
        let arguments = match wire.function.arguments {
            Some(raw) if !raw.trim().is_empty() => serde_json::from_str(&raw)
                .with_context(|| format!("failed to decode Cohere tool arguments: {raw}"))?,
            _ => Value::Object(Map::new()),
        };
        tool_calls.push(ToolCall {
            id: wire.id.unwrap_or_default(),
            name: wire.function.name,
            arguments,
        });
    }

    let content = (!text.trim().is_empty()).then_some(text);
    Ok(AssistantTurn { content, tool_calls })
}

#[derive(Debug, Deserialize)]
struct CohereChatResponse {
    message: CohereMessage,
}

#[derive(Debug, Deserialize)]
struct CohereMessage {
    #[serde(default)]
    content: Option<Vec<CohereContentBlock>>,
    #[serde(default)]
    tool_calls: Option<Vec<CohereToolCallWire>>,
}

#[derive(Debug, Deserialize)]
struct CohereContentBlock {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CohereToolCallWire {
    #[serde(default)]
    id: Option<String>,
    function: CohereToolFunctionWire,
}

#[derive(Debug, Deserialize)]
struct CohereToolFunctionWire {
    name: String,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CohereStreamEvent {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    delta: Option<CohereStreamDelta>,
}

#[derive(Debug, Deserialize)]
struct CohereStreamDelta {
    #[serde(default)]
    message: Option<CohereStreamMessage>,
}

#[derive(Debug, Deserialize)]
struct CohereStreamMessage {
    #[serde(default)]
    content: Option<CohereStreamContent>,
}

#[derive(Debug, Deserialize)]
struct CohereStreamContent {
    #[serde(default)]
    text: Option<String>,
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde_json::json;

    use super::{assistant_turn_from_cohere, parse_cohere_stream_event};

    #[test]
    fn parses_content_delta_event() -> Result<()> {
        let raw = "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"Hi\"}}}}\n\n";
        assert_eq!(parse_cohere_stream_event(raw)?, vec!["Hi".to_string()]);
        Ok(())
    }

    #[test]
    fn ignores_non_text_events() -> Result<()> {
        let raw = "data: {\"type\":\"message-start\"}\n\n";
        assert!(parse_cohere_stream_event(raw)?.is_empty());
        Ok(())
    }

    #[test]
    fn decodes_text_and_tool_calls() -> Result<()> {
        let message = serde_json::from_value(json!({
            "role": "assistant",
            "content": [{ "type": "text", "text": "Let me check." }],
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": { "name": "lookup", "arguments": "{\"q\":\"x\"}" }
            }]
        }))?;
        let turn = assistant_turn_from_cohere(message)?;
        assert_eq!(turn.content.as_deref(), Some("Let me check."));
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "lookup");
        assert_eq!(turn.tool_calls[0].arguments, json!({ "q": "x" }));
        Ok(())
    }
}
