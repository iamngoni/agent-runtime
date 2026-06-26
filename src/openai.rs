use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tracing::{info, warn};
use web_time::Instant;

use crate::error::{ProviderError, RetryPolicy, execute_with_retry};
use crate::http::{HttpRequest, HttpResponse, SharedHttpClient, collect_stream_to_string};
use crate::provider::{
    AgentProviderKind, EmbeddingProvider, ModelTiers, ProviderInfo, TextProvider,
};
use crate::streaming::should_flush_delta;
use crate::{
    AssistantTurn, ChatMessage, EventSink, MessageRole, ResponseFormat, RuntimeEvent, ToolCall,
    ToolDefinition,
};

/// Configuration for an OpenAI-compatible provider. The same client drives
/// OpenAI, Groq, DeepSeek, xAI, Mistral, Ollama, OpenRouter and any other
/// endpoint that speaks the `/chat/completions` wire format — they differ only
/// by `kind`, `base_url`, headers and model tiers.
#[derive(Debug, Clone)]
pub struct OpenAiClientConfig {
    pub kind: AgentProviderKind,
    /// API base URL, e.g. `https://api.groq.com/openai/v1` (no trailing slash
    /// required; one is trimmed). `/chat/completions` and `/embeddings` are
    /// appended to it.
    pub base_url: String,
    /// Extra headers sent on every request (e.g. OpenRouter's `HTTP-Referer`).
    pub extra_headers: Vec<(String, String)>,
    pub model_tiers: ModelTiers,
    pub retry: RetryPolicy,
    pub verbose: bool,
}

impl Default for OpenAiClientConfig {
    fn default() -> Self {
        Self::for_kind(AgentProviderKind::OpenAi)
    }
}

impl OpenAiClientConfig {
    /// Build a config seeded with a vendor's default base URL and model tiers.
    /// Panics for [`AgentProviderKind::Custom`] (no default base URL); use
    /// [`OpenAiClientConfig::new`] and supply one explicitly.
    pub fn for_kind(kind: AgentProviderKind) -> Self {
        let base_url = kind
            .default_base_url()
            .unwrap_or_else(|| {
                panic!(
                    "provider '{}' has no default base URL; construct OpenAiClientConfig::new with one",
                    kind.as_str()
                )
            })
            .to_string();
        let model_tiers = kind.default_model_tiers();
        Self {
            kind,
            base_url,
            extra_headers: Vec::new(),
            model_tiers,
            retry: RetryPolicy::default(),
            verbose: false,
        }
    }

    /// Build a config with an explicit base URL — required for custom vendors.
    pub fn new(kind: AgentProviderKind, base_url: impl Into<String>) -> Self {
        let model_tiers = kind.default_model_tiers();
        Self {
            kind,
            base_url: base_url.into(),
            extra_headers: Vec::new(),
            model_tiers,
            retry: RetryPolicy::default(),
            verbose: false,
        }
    }

    fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    fn embeddings_url(&self) -> String {
        format!("{}/embeddings", self.base_url.trim_end_matches('/'))
    }
}

#[derive(Clone)]
pub struct OpenAiClient {
    http_client: SharedHttpClient,
    api_key: String,
    config: OpenAiClientConfig,
}

impl OpenAiClient {
    /// A plain OpenAI client against the public API.
    pub fn new(http_client: SharedHttpClient, api_key: impl Into<String>) -> Self {
        Self::with_config(http_client, api_key, OpenAiClientConfig::default())
    }

    pub fn with_verbose(
        http_client: SharedHttpClient,
        api_key: impl Into<String>,
        verbose: bool,
    ) -> Self {
        let config = OpenAiClientConfig {
            verbose,
            ..OpenAiClientConfig::default()
        };
        Self::with_config(http_client, api_key, config)
    }

    /// Build a client for any OpenAI-compatible provider.
    pub fn with_config(
        http_client: SharedHttpClient,
        api_key: impl Into<String>,
        config: OpenAiClientConfig,
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
        self.config.kind.as_str()
    }

    /// Apply auth + configured extra headers to a request.
    fn prepare(&self, request: HttpRequest) -> HttpRequest {
        let mut request = request.bearer_auth(&self.api_key);
        for (name, value) in &self.config.extra_headers {
            request = request.header(name, value);
        }
        request
    }

    /// Send a buffered request with retry + typed error classification.
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

    async fn request_assistant_turn_impl(
        &self,
        model: &str,
        system_prompt: &str,
        history: &[ChatMessage],
        tool_definitions: &[ToolDefinition],
    ) -> Result<AssistantTurn> {
        let request_started_at = Instant::now();
        let mut messages = Vec::with_capacity(history.len() + 1);
        messages.push(json!({
            "role": "system",
            "content": system_prompt,
        }));
        messages.extend(history.iter().map(message_to_openai_json));

        if self.verbose() {
            info!(
                provider = self.provider_name(),
                model = %model,
                history_count = history.len(),
                tool_count = tool_definitions.len(),
                system_prompt_chars = system_prompt.chars().count(),
                "requesting assistant turn"
            );
        }

        let mut request_payload = Map::new();
        request_payload.insert("model".to_string(), Value::String(model.to_string()));
        request_payload.insert("messages".to_string(), Value::Array(messages));
        if !tool_definitions.is_empty() {
            request_payload.insert(
                "tools".to_string(),
                Value::Array(openai_function_tools(tool_definitions)),
            );
            request_payload.insert("tool_choice".to_string(), Value::String("auto".to_string()));
        }

        let request = self
            .prepare(HttpRequest::post(self.config.chat_completions_url()))
            .json_body(&Value::Object(request_payload))?;
        let response = self
            .send_classified(request)
            .await
            .context("failed to call OpenAI-compatible assistant with tools")?;

        let body: ToolChatCompletionResponse = response
            .json()
            .context("failed to decode assistant tool response")?;

        let assistant_turn: AssistantTurn = body
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.try_into())
            .transpose()?
            .ok_or_else(|| anyhow!("assistant response did not contain a choice"))?;

        if self.verbose() {
            info!(
                provider = self.provider_name(),
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

    async fn request_structured_impl(
        &self,
        model: &str,
        system_prompt: &str,
        history: &[ChatMessage],
        format: &ResponseFormat,
    ) -> Result<Value> {
        let request_started_at = Instant::now();
        let mut messages = Vec::with_capacity(history.len() + 1);
        messages.push(json!({
            "role": "system",
            "content": system_prompt,
        }));
        messages.extend(history.iter().map(message_to_openai_json));

        if self.verbose() {
            info!(
                provider = self.provider_name(),
                model = %model,
                history_count = history.len(),
                schema = %format.name,
                "requesting structured output"
            );
        }

        // Force a single synthetic function whose parameters are the requested
        // schema; `tool_choice: function` requires the model to call it.
        let response_tool = ToolDefinition {
            name: format.name.clone(),
            description: format.description.clone(),
            input_schema: format.schema.clone(),
        };

        let mut request_payload = Map::new();
        request_payload.insert("model".to_string(), Value::String(model.to_string()));
        request_payload.insert("messages".to_string(), Value::Array(messages));
        request_payload.insert(
            "tools".to_string(),
            Value::Array(openai_function_tools(&[response_tool])),
        );
        request_payload.insert(
            "tool_choice".to_string(),
            json!({ "type": "function", "function": { "name": format.name } }),
        );

        let request = self
            .prepare(HttpRequest::post(self.config.chat_completions_url()))
            .json_body(&Value::Object(request_payload))?;
        let response = self
            .send_classified(request)
            .await
            .context("failed to call OpenAI-compatible provider for structured output")?;

        let body: ToolChatCompletionResponse = response
            .json()
            .context("failed to decode structured tool response")?;
        let assistant_turn: AssistantTurn = body
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.try_into())
            .transpose()?
            .ok_or_else(|| anyhow!("structured response did not contain a choice"))?;

        if self.verbose() {
            info!(
                provider = self.provider_name(),
                model = %model,
                duration_ms = request_started_at.elapsed().as_millis() as u64,
                "structured output completed"
            );
        }

        assistant_turn
            .tool_calls
            .into_iter()
            .next()
            .map(|call| call.arguments)
            .ok_or_else(|| anyhow!("structured response did not contain a tool call"))
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

        let mut all_messages = Vec::with_capacity(messages.len() + 1);
        all_messages.push(json!({
            "role": "system",
            "content": system_prompt,
        }));
        all_messages.extend(messages.iter().map(message_to_openai_json));

        if self.verbose() {
            info!(
                provider = self.provider_name(),
                model = %model,
                message_count = messages.len(),
                system_prompt_chars = system_prompt.chars().count(),
                "requesting streamed assistant message"
            );
        }

        let request_payload = json!({
            "model": model,
            "stream": true,
            "messages": all_messages,
        });

        let request = self
            .prepare(HttpRequest::post(self.config.chat_completions_url()))
            .json_body(&request_payload)?;
        let response = self
            .http_client
            .send_streaming(request)
            .await
            .context("failed to call chat completions for streamed message")?;

        if !(200..300).contains(&response.status) {
            let status = response.status;
            let body = collect_stream_to_string(response.body).await;
            if self.verbose() {
                warn!(
                    provider = self.provider_name(),
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
        let mut saw_done = false;
        let mut chunk_count = 0usize;
        let mut parsed_event_count = 0usize;
        let mut flush_count = 0usize;

        while let Some(chunk) = response_stream.next().await {
            let chunk = chunk.context("failed to read streamed response chunk")?;
            let chunk_text = std::str::from_utf8(&chunk)
                .context("failed to decode streamed response chunk as UTF-8")?;
            chunk_count += 1;
            raw_event_buffer.push_str(&chunk_text.replace('\r', ""));

            while let Some(event_end_index) = raw_event_buffer.find("\n\n") {
                let raw_event = raw_event_buffer[..event_end_index].to_string();
                raw_event_buffer = raw_event_buffer[event_end_index + 2..].to_string();
                parsed_event_count += 1;

                match parse_openai_stream_event(&raw_event) {
                    Ok(ParsedStreamEvent::Done) => {
                        saw_done = true;
                        break;
                    }
                    Ok(ParsedStreamEvent::Empty) => {}
                    Ok(ParsedStreamEvent::Deltas(deltas)) => {
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
                            "failed to parse OpenAI stream event"
                        );
                    }
                }
            }

            if saw_done {
                break;
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
            return Err(anyhow!("streamed assistant message returned an empty message"));
        }

        if self.verbose() {
            info!(
                provider = self.provider_name(),
                model = %model,
                duration_ms = request_started_at.elapsed().as_millis() as u64,
                chunk_count,
                parsed_event_count,
                flush_count,
                saw_done,
                message_chars = message.chars().count(),
                "streamed assistant message completed"
            );
        }

        Ok(message)
    }

    async fn embed_impl(&self, model: &str, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let request_payload = json!({
            "model": model,
            "input": inputs,
        });

        let request = self
            .prepare(HttpRequest::post(self.config.embeddings_url()))
            .json_body(&request_payload)?;
        let response = self
            .send_classified(request)
            .await
            .context("failed to call embeddings endpoint")?;

        let body: EmbeddingResponse = response
            .json()
            .context("failed to decode embeddings response")?;

        let mut data = body.data;
        data.sort_by_key(|item| item.index);
        Ok(data.into_iter().map(|item| item.embedding).collect())
    }
}

impl ProviderInfo for OpenAiClient {
    fn kind(&self) -> AgentProviderKind {
        self.config.kind.clone()
    }

    fn verbose(&self) -> bool {
        self.config.verbose
    }

    fn model_tiers(&self) -> &ModelTiers {
        &self.config.model_tiers
    }
}

#[async_trait]
impl TextProvider for OpenAiClient {
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

    async fn request_structured(
        &self,
        model: &str,
        system_prompt: &str,
        messages: &[ChatMessage],
        format: &ResponseFormat,
    ) -> Result<Value> {
        self.request_structured_impl(model, system_prompt, messages, format)
            .await
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiClient {
    async fn embed(&self, model: &str, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_impl(model, inputs).await
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedStreamEvent {
    Deltas(Vec<String>),
    Done,
    Empty,
}

pub fn parse_openai_stream_event(raw_event: &str) -> Result<ParsedStreamEvent> {
    let data_lines = raw_event
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .collect::<Vec<_>>();

    if data_lines.is_empty() {
        return Ok(ParsedStreamEvent::Empty);
    }

    let data = data_lines.join("\n");
    if data == "[DONE]" {
        return Ok(ParsedStreamEvent::Done);
    }

    let chunk: ChatCompletionStreamChunk = serde_json::from_str(&data)
        .with_context(|| format!("failed to decode OpenAI stream payload: {data}"))?;

    let deltas = chunk
        .choices
        .into_iter()
        .filter_map(|choice| choice.delta.content)
        .filter(|delta| !delta.is_empty())
        .collect::<Vec<_>>();

    if deltas.is_empty() {
        Ok(ParsedStreamEvent::Empty)
    } else {
        Ok(ParsedStreamEvent::Deltas(deltas))
    }
}

fn openai_function_tools(definitions: &[ToolDefinition]) -> Vec<Value> {
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

fn message_to_openai_json(message: &ChatMessage) -> Value {
    match message.role {
        MessageRole::User if message.has_attachments() => json!({
            "role": "user",
            "content": openai_user_content_parts(message),
        }),
        MessageRole::System | MessageRole::User => json!({
            "role": role_name(message.role),
            "content": message.content,
        }),
        MessageRole::Tool => json!({
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }),
        MessageRole::Assistant => {
            let mut object = Map::new();
            object.insert(
                "role".to_string(),
                Value::String(role_name(message.role).to_string()),
            );
            if let Some(content) = message.content.clone() {
                object.insert("content".to_string(), Value::String(content));
            }
            if !message.tool_calls.is_empty() {
                object.insert(
                    "tool_calls".to_string(),
                    Value::Array(
                        message
                            .tool_calls
                            .iter()
                            .map(tool_call_to_openai_json)
                            .collect(),
                    ),
                );
            }
            Value::Object(object)
        }
    }
}

/// Build OpenAI's multimodal `content` array (text + image/file parts) for a
/// user message that carries attachments.
fn openai_user_content_parts(message: &ChatMessage) -> Vec<Value> {
    use crate::message::{AttachmentKind, AttachmentSource};

    let mut parts = Vec::new();
    if let Some(text) = message.content.as_deref()
        && !text.is_empty()
    {
        parts.push(json!({ "type": "text", "text": text }));
    }

    for attachment in &message.attachments {
        let part = match attachment.kind {
            AttachmentKind::Image => {
                let url = match &attachment.source {
                    AttachmentSource::Url(url) => url.clone(),
                    AttachmentSource::Base64(_) => {
                        attachment.data_uri().unwrap_or_default()
                    }
                };
                json!({ "type": "image_url", "image_url": { "url": url } })
            }
            AttachmentKind::Document => match &attachment.source {
                AttachmentSource::Base64(_) => json!({
                    "type": "file",
                    "file": { "file_data": attachment.data_uri().unwrap_or_default() },
                }),
                AttachmentSource::Url(url) => json!({
                    "type": "file",
                    "file": { "file_url": url },
                }),
            },
        };
        parts.push(part);
    }

    parts
}

fn tool_call_to_openai_json(call: &ToolCall) -> Value {
    json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string()),
        }
    })
}

fn role_name(role: MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

fn tool_call_from_openai_wire(wire: OpenAiToolCallWire) -> Result<ToolCall> {
    let arguments = match wire.function.arguments {
        OpenAiToolArgumentsWire::JsonString(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                Value::Object(Default::default())
            } else {
                serde_json::from_str(trimmed)
                    .with_context(|| format!("failed to decode OpenAI tool arguments: {trimmed}"))?
            }
        }
        OpenAiToolArgumentsWire::JsonValue(value) => value,
    };

    Ok(ToolCall {
        id: wire.id,
        name: wire.function.name,
        arguments,
    })
}

impl TryFrom<OpenAiAssistantMessage> for AssistantTurn {
    type Error = anyhow::Error;

    fn try_from(message: OpenAiAssistantMessage) -> Result<Self> {
        let tool_calls = message
            .tool_calls
            .into_iter()
            .map(tool_call_from_openai_wire)
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            content: message.content,
            tool_calls,
        })
    }
}

#[derive(Debug, Deserialize)]
struct ChatCompletionStreamChunk {
    choices: Vec<ChatCompletionStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionStreamChoice {
    delta: ChatCompletionDelta,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ToolChatCompletionResponse {
    choices: Vec<ToolChatCompletionChoice>,
}

#[derive(Debug, Deserialize)]
struct ToolChatCompletionChoice {
    message: OpenAiAssistantMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiAssistantMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAiToolCallWire>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCallWire {
    id: String,
    function: OpenAiToolFunctionWire,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolFunctionWire {
    name: String,
    arguments: OpenAiToolArgumentsWire,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAiToolArgumentsWire {
    JsonString(String),
    JsonValue(Value),
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    #[serde(default)]
    index: usize,
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde_json::json;

    use super::{
        OpenAiAssistantMessage, OpenAiClientConfig, parse_openai_stream_event,
        tool_call_to_openai_json,
    };
    use crate::provider::AgentProviderKind;
    use crate::{AssistantTurn, ParsedStreamEvent, ToolCall};

    #[test]
    fn parses_stream_done_event() -> Result<()> {
        assert_eq!(parse_openai_stream_event("data: [DONE]")?, ParsedStreamEvent::Done);
        Ok(())
    }

    #[test]
    fn maps_provider_neutral_tool_call_to_openai_shape() {
        let value = tool_call_to_openai_json(&ToolCall {
            id: "call_123".to_string(),
            name: "search_products".to_string(),
            arguments: json!({
                "query": "lagavulin",
            }),
        });

        assert_eq!(value["type"], json!("function"));
        assert_eq!(value["function"]["name"], json!("search_products"));
    }

    #[test]
    fn decodes_openai_assistant_tool_calls() -> Result<()> {
        let message: OpenAiAssistantMessage = serde_json::from_value(json!({
            "content": "Let me look that up.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "search_products",
                        "arguments": "{\"query\":\"lagavulin\"}"
                    }
                }
            ]
        }))?;

        let turn: AssistantTurn = message.try_into()?;
        assert_eq!(turn.content.as_deref(), Some("Let me look that up."));
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "search_products");
        assert_eq!(turn.tool_calls[0].arguments, json!({ "query": "lagavulin" }));
        Ok(())
    }

    #[test]
    fn derives_endpoints_from_base_url() {
        let config = OpenAiClientConfig::for_kind(AgentProviderKind::Groq);
        assert_eq!(
            config.chat_completions_url(),
            "https://api.groq.com/openai/v1/chat/completions"
        );
        assert_eq!(
            config.embeddings_url(),
            "https://api.groq.com/openai/v1/embeddings"
        );
    }

    #[test]
    fn custom_base_url_trims_trailing_slash() {
        let config = OpenAiClientConfig::new(
            AgentProviderKind::Custom("local".to_string()),
            "http://localhost:8000/v1/",
        );
        assert_eq!(
            config.chat_completions_url(),
            "http://localhost:8000/v1/chat/completions"
        );
    }
}
