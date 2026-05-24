use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tracing::{info, warn};
use web_time::Instant;

use crate::http::{HttpRequest, SharedHttpClient, collect_stream_to_string};
use crate::streaming::should_flush_delta;
use crate::{
    AgentProvider, AgentProviderKind, AssistantTurn, ChatMessage, EventSink, MessageRole,
    RuntimeEvent, ToolCall, ToolDefinition,
};

const ANTHROPIC_MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone)]
pub struct AnthropicClientConfig {
    pub verbose: bool,
    pub max_tokens: u32,
}

impl Default for AnthropicClientConfig {
    fn default() -> Self {
        Self {
            verbose: false,
            max_tokens: DEFAULT_ANTHROPIC_MAX_TOKENS,
        }
    }
}

#[derive(Clone)]
pub struct AnthropicClient {
    http_client: SharedHttpClient,
    api_key: String,
    config: AnthropicClientConfig,
}

impl AnthropicClient {
    pub fn new(http_client: SharedHttpClient, api_key: impl Into<String>) -> Self {
        Self::with_config(http_client, api_key, AnthropicClientConfig::default())
    }

    pub fn with_verbose(
        http_client: SharedHttpClient,
        api_key: impl Into<String>,
        verbose: bool,
    ) -> Self {
        Self {
            http_client,
            api_key: api_key.into(),
            config: AnthropicClientConfig {
                verbose,
                ..AnthropicClientConfig::default()
            },
        }
    }

    pub fn with_config(
        http_client: SharedHttpClient,
        api_key: impl Into<String>,
        config: AnthropicClientConfig,
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
        let messages = messages_to_anthropic_json(history)?;

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
        request_payload.insert("max_tokens".to_string(), json!(self.config.max_tokens));
        request_payload.insert("system".to_string(), Value::String(system_prompt.to_string()));
        request_payload.insert("messages".to_string(), Value::Array(messages));
        if !tool_definitions.is_empty() {
            request_payload.insert(
                "tools".to_string(),
                Value::Array(anthropic_tools(tool_definitions)),
            );
            request_payload.insert("tool_choice".to_string(), json!({ "type": "auto" }));
        }

        let request = HttpRequest::post(ANTHROPIC_MESSAGES_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json_body(&Value::Object(request_payload))?;
        let response = self
            .http_client
            .send(request)
            .await
            .context("failed to call Anthropic assistant with tools")?;

        if !response.is_success() {
            let status = response.status;
            let body = response
                .text()
                .unwrap_or_else(|_| "<failed to read response body>".to_string());
            if self.verbose() {
                warn!(
                    model = %model,
                    duration_ms = request_started_at.elapsed().as_millis() as u64,
                    status,
                    body = %body,
                    "assistant turn request failed"
                );
            }
            return Err(anyhow!("Anthropic assistant call returned {status}: {body}"));
        }

        let body: AnthropicMessageResponse = response
            .json()
            .context("failed to decode Anthropic assistant response")?;
        let assistant_turn = assistant_turn_from_anthropic_content(&body.content)?;

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
                stop_reason = ?body.stop_reason,
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

        let anthropic_messages = messages_to_anthropic_json(messages)?;

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
            "max_tokens": self.config.max_tokens,
            "stream": true,
            "system": system_prompt,
            "messages": anthropic_messages,
        });

        let request = HttpRequest::post(ANTHROPIC_MESSAGES_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json_body(&request_payload)?;
        let response = self
            .http_client
            .send_streaming(request)
            .await
            .context("failed to call Anthropic streaming messages API")?;

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
            return Err(anyhow!("Anthropic streamed message returned {status}: {body}"));
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
            let chunk = chunk.context("failed to read Anthropic stream response chunk")?;
            let chunk_text = std::str::from_utf8(&chunk)
                .context("failed to decode Anthropic stream chunk as UTF-8")?;
            chunk_count += 1;
            raw_event_buffer.push_str(&chunk_text.replace('\r', ""));

            while let Some(event_end_index) = raw_event_buffer.find("\n\n") {
                let raw_event = raw_event_buffer[..event_end_index].to_string();
                raw_event_buffer = raw_event_buffer[event_end_index + 2..].to_string();
                parsed_event_count += 1;

                match parse_anthropic_stream_event(&raw_event) {
                    Ok(AnthropicParsedStreamEvent::Done) => {
                        saw_done = true;
                        break;
                    }
                    Ok(AnthropicParsedStreamEvent::Empty) => {}
                    Ok(AnthropicParsedStreamEvent::Deltas(deltas)) => {
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
                            "failed to parse Anthropic stream event"
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
            return Err(anyhow!("Anthropic streamed message returned an empty message"));
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
impl AgentProvider for AnthropicClient {
    fn kind(&self) -> AgentProviderKind {
        AgentProviderKind::Anthropic
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
pub enum AnthropicParsedStreamEvent {
    Deltas(Vec<String>),
    Done,
    Empty,
}

pub fn parse_anthropic_stream_event(raw_event: &str) -> Result<AnthropicParsedStreamEvent> {
    let event_name = raw_event
        .lines()
        .find_map(|line| line.strip_prefix("event:"))
        .map(str::trim);
    let data_lines = raw_event
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .collect::<Vec<_>>();

    if data_lines.is_empty() {
        return Ok(AnthropicParsedStreamEvent::Empty);
    }

    let data = data_lines.join("\n");
    let payload: AnthropicStreamEvent = serde_json::from_str(&data)
        .with_context(|| format!("failed to decode Anthropic stream payload: {data}"))?;

    match event_name.unwrap_or(payload.kind.as_str()) {
        "message_stop" => Ok(AnthropicParsedStreamEvent::Done),
        "content_block_start" => {
            let delta = payload
                .content_block
                .as_ref()
                .and_then(|block| (block.kind == "text").then_some(block.text.as_deref()))
                .flatten()
                .filter(|text| !text.is_empty())
                .map(|text| vec![text.to_string()]);
            Ok(match delta {
                Some(deltas) => AnthropicParsedStreamEvent::Deltas(deltas),
                None => AnthropicParsedStreamEvent::Empty,
            })
        }
        "content_block_delta" => {
            let delta = payload
                .delta
                .as_ref()
                .and_then(|delta| {
                    (delta.kind.as_deref() == Some("text_delta")).then_some(delta.text.as_deref())
                })
                .flatten()
                .filter(|text| !text.is_empty())
                .map(|text| vec![text.to_string()]);
            Ok(match delta {
                Some(deltas) => AnthropicParsedStreamEvent::Deltas(deltas),
                None => AnthropicParsedStreamEvent::Empty,
            })
        }
        _ => Ok(AnthropicParsedStreamEvent::Empty),
    }
}

fn anthropic_tools(definitions: &[ToolDefinition]) -> Vec<Value> {
    definitions
        .iter()
        .map(|definition| {
            json!({
                "name": definition.name,
                "description": definition.description,
                "input_schema": definition.input_schema,
            })
        })
        .collect()
}

fn messages_to_anthropic_json(messages: &[ChatMessage]) -> Result<Vec<Value>> {
    let mut converted = Vec::new();
    let mut index = 0usize;

    while index < messages.len() {
        let message = &messages[index];
        match message.role {
            MessageRole::System => {
                index += 1;
            }
            MessageRole::User => {
                converted.push(json!({
                    "role": "user",
                    "content": message.content.clone().unwrap_or_default(),
                }));
                index += 1;
            }
            MessageRole::Assistant => {
                converted.push(assistant_message_to_anthropic_json(message));
                index += 1;
            }
            MessageRole::Tool => {
                let mut tool_results = Vec::new();
                while index < messages.len() && messages[index].role == MessageRole::Tool {
                    let tool_message = &messages[index];
                    let tool_use_id = tool_message
                        .tool_call_id
                        .clone()
                        .ok_or_else(|| anyhow!("tool message missing tool_call_id"))?;
                    tool_results.push(json!({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": tool_message.content.clone().unwrap_or_default(),
                    }));
                    index += 1;
                }

                if !tool_results.is_empty() {
                    converted.push(json!({
                        "role": "user",
                        "content": tool_results,
                    }));
                }
            }
        }
    }

    Ok(converted)
}

fn assistant_message_to_anthropic_json(message: &ChatMessage) -> Value {
    let mut content = Vec::new();

    if let Some(text) = message.content.clone()
        && !text.is_empty()
    {
        content.push(json!({
            "type": "text",
            "text": text,
        }));
    }

    for tool_call in &message.tool_calls {
        content.push(json!({
            "type": "tool_use",
            "id": tool_call.id,
            "name": tool_call.name,
            "input": tool_call.arguments,
        }));
    }

    json!({
        "role": "assistant",
        "content": content,
    })
}

fn assistant_turn_from_anthropic_content(content: &[AnthropicContentBlock]) -> Result<AssistantTurn> {
    let mut text = String::new();
    let mut tool_calls = Vec::new();

    for block in content {
        match block.kind.as_str() {
            "text" => {
                if let Some(value) = block.text.as_deref() {
                    text.push_str(value);
                }
            }
            "tool_use" => {
                let id = block
                    .id
                    .clone()
                    .ok_or_else(|| anyhow!("Anthropic tool_use block missing id"))?;
                let name = block
                    .name
                    .clone()
                    .ok_or_else(|| anyhow!("Anthropic tool_use block missing name"))?;
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments: block.input.clone().unwrap_or_else(|| Value::Object(Map::new())),
                });
            }
            _ => {}
        }
    }

    let content = (!text.trim().is_empty()).then_some(text);

    Ok(AssistantTurn {
        content,
        tool_calls,
    })
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageResponse {
    content: Vec<AnthropicContentBlock>,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    content_block: Option<AnthropicStreamContentBlock>,
    #[serde(default)]
    delta: Option<AnthropicStreamDelta>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamDelta {
    #[serde(rename = "type")]
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde_json::json;

    use super::{AnthropicParsedStreamEvent, messages_to_anthropic_json, parse_anthropic_stream_event};
    use crate::{ChatMessage, ToolCall};

    #[test]
    fn groups_tool_results_into_user_message() -> Result<()> {
        let messages = vec![
            ChatMessage::assistant_with_tools(
                Some("Checking.".to_string()),
                vec![ToolCall {
                    id: "toolu_01".to_string(),
                    name: "search_products".to_string(),
                    arguments: json!({ "query": "lagavulin" }),
                }],
            ),
            ChatMessage::tool("toolu_01", "{\"ok\":true}"),
        ];

        let converted = messages_to_anthropic_json(&messages)?;
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0]["role"], json!("assistant"));
        assert_eq!(converted[1]["role"], json!("user"));
        assert_eq!(converted[1]["content"][0]["type"], json!("tool_result"));
        Ok(())
    }

    #[test]
    fn parses_text_delta_stream_event() -> Result<()> {
        let raw_event = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n"
        );

        assert_eq!(
            parse_anthropic_stream_event(raw_event)?,
            AnthropicParsedStreamEvent::Deltas(vec!["Hello".to_string()])
        );
        Ok(())
    }

    #[test]
    fn ignores_message_delta_stream_event() -> Result<()> {
        let raw_event = concat!(
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":31}}\n\n"
        );

        assert_eq!(
            parse_anthropic_stream_event(raw_event)?,
            AnthropicParsedStreamEvent::Empty
        );
        Ok(())
    }
}
