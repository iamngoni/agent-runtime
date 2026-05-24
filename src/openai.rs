use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tracing::{info, warn};

use crate::{
    AgentProvider, AgentProviderKind, AssistantTurn, ChatMessage, EventSink, MessageRole,
    RuntimeEvent, ToolCall, ToolDefinition,
};
use crate::streaming::should_flush_delta;

const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

#[derive(Debug, Clone, Default)]
pub struct OpenAiClientConfig {
    pub verbose: bool,
}

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    http_client: Client,
    api_key: String,
    config: OpenAiClientConfig,
}

impl OpenAiClient {
    pub fn new(http_client: Client, api_key: impl Into<String>) -> Self {
        Self::with_config(http_client, api_key, OpenAiClientConfig::default())
    }

    pub fn with_verbose(http_client: Client, api_key: impl Into<String>, verbose: bool) -> Self {
        Self {
            http_client,
            api_key: api_key.into(),
            config: OpenAiClientConfig { verbose },
        }
    }

    pub fn with_config(
        http_client: Client,
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

        let response = self
            .http_client
            .post(OPENAI_CHAT_COMPLETIONS_URL)
            .bearer_auth(&self.api_key)
            .json(&Value::Object(request_payload))
            .send()
            .await
            .context("failed to call OpenAI assistant with tools")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read response body>".to_string());
            if self.verbose() {
                warn!(
                    model = %model,
                    duration_ms = request_started_at.elapsed().as_millis() as u64,
                    status = %status,
                    body = %body,
                    "assistant turn request failed"
                );
            }
            return Err(anyhow!("OpenAI assistant call returned {status}: {body}"));
        }

        let body: ToolChatCompletionResponse = response
            .json()
            .await
            .context("failed to decode OpenAI assistant tool response")?;

        let assistant_turn: AssistantTurn = body
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.try_into())
            .transpose()?
            .ok_or_else(|| anyhow!("OpenAI assistant response did not contain a choice"))?;

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

        let mut all_messages = Vec::with_capacity(messages.len() + 1);
        all_messages.push(json!({
            "role": "system",
            "content": system_prompt,
        }));
        all_messages.extend(messages.iter().map(message_to_openai_json));

        if self.verbose() {
            info!(
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

        let response = self
            .http_client
            .post(OPENAI_CHAT_COMPLETIONS_URL)
            .bearer_auth(&self.api_key)
            .json(&request_payload)
            .send()
            .await
            .context("failed to call OpenAI chat completions for tool follow-up")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read response body>".to_string());
            if self.verbose() {
                warn!(
                    model = %model,
                    duration_ms = request_started_at.elapsed().as_millis() as u64,
                    status = %status,
                    body = %body,
                    "streamed assistant message request failed"
                );
            }
            return Err(anyhow!("OpenAI tool follow-up returned {status}: {body}"));
        }

        let mut response_stream = response.bytes_stream();
        let mut raw_event_buffer = String::new();
        let mut full_message = String::new();
        let mut pending_delta = String::new();
        let mut saw_done = false;
        let mut chunk_count = 0usize;
        let mut parsed_event_count = 0usize;
        let mut flush_count = 0usize;

        while let Some(chunk) = response_stream.next().await {
            let chunk = chunk.context("failed to read OpenAI tool follow-up response chunk")?;
            let chunk_text = std::str::from_utf8(&chunk)
                .context("failed to decode OpenAI tool follow-up response chunk as UTF-8")?;
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
            return Err(anyhow!("OpenAI tool follow-up returned an empty message"));
        }

        if self.verbose() {
            info!(
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
}

#[async_trait]
impl AgentProvider for OpenAiClient {
    fn kind(&self) -> AgentProviderKind {
        AgentProviderKind::OpenAi
    }

    fn verbose(&self) -> bool {
        self.verbose()
    }

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

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde_json::json;

    use super::{OpenAiAssistantMessage, parse_openai_stream_event, tool_call_to_openai_json};
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
}
