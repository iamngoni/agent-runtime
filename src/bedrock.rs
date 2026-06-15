//! AWS Bedrock provider via the Converse API, using **bearer-token** auth.
//!
//! This first cut deliberately uses Bedrock's API-key / bearer-token mode
//! (`Authorization: Bearer <token>`) instead of SigV4 request signing, so it
//! stays api-key-shaped like every other provider here and avoids pulling in
//! the AWS SDK (which would jeopardise the wasm/Workers build). SigV4 / IAM
//! auth can be layered on later behind the same client.
//!
//! Streaming note: Bedrock's `converse-stream` uses AWS's binary
//! `vnd.amazon.eventstream` framing, which doesn't map onto our SSE-oriented
//! byte stream. To honour the [`TextProvider`] contract without a binary frame
//! parser, [`stream_message`](BedrockClient) buffers a single `converse` call
//! and emits the result as one delta. Real token streaming can be added later.

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tracing::info;
use web_time::Instant;

use crate::error::{ProviderError, RetryPolicy, execute_with_retry};
use crate::http::{HttpRequest, HttpResponse, SharedHttpClient};
use crate::message::{AttachmentKind, AttachmentSource};
use crate::provider::{AgentProviderKind, ModelTiers, ProviderInfo, TextProvider};
use crate::{AssistantTurn, ChatMessage, EventSink, MessageRole, RuntimeEvent, ToolCall, ToolDefinition};

const DEFAULT_BEDROCK_REGION: &str = "us-east-1";
const DEFAULT_BEDROCK_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone)]
pub struct BedrockClientConfig {
    pub region: String,
    /// Full base URL override. When `None`, it is derived from `region`:
    /// `https://bedrock-runtime.{region}.amazonaws.com`.
    pub base_url: Option<String>,
    pub verbose: bool,
    pub max_tokens: u32,
    pub model_tiers: ModelTiers,
    pub retry: RetryPolicy,
}

impl Default for BedrockClientConfig {
    fn default() -> Self {
        Self {
            region: DEFAULT_BEDROCK_REGION.to_string(),
            base_url: None,
            verbose: false,
            max_tokens: DEFAULT_BEDROCK_MAX_TOKENS,
            model_tiers: AgentProviderKind::Bedrock.default_model_tiers(),
            retry: RetryPolicy::default(),
        }
    }
}

impl BedrockClientConfig {
    fn base_url(&self) -> String {
        self.base_url.clone().unwrap_or_else(|| {
            format!("https://bedrock-runtime.{}.amazonaws.com", self.region)
        })
    }

    fn converse_url(&self, model: &str) -> String {
        format!("{}/model/{}/converse", self.base_url().trim_end_matches('/'), model)
    }
}

#[derive(Clone)]
pub struct BedrockClient {
    http_client: SharedHttpClient,
    api_key: String,
    config: BedrockClientConfig,
}

impl BedrockClient {
    pub fn new(http_client: SharedHttpClient, api_key: impl Into<String>) -> Self {
        Self::with_config(http_client, api_key, BedrockClientConfig::default())
    }

    pub fn with_verbose(
        http_client: SharedHttpClient,
        api_key: impl Into<String>,
        verbose: bool,
    ) -> Self {
        Self {
            http_client,
            api_key: api_key.into(),
            config: BedrockClientConfig {
                verbose,
                ..BedrockClientConfig::default()
            },
        }
    }

    pub fn with_config(
        http_client: SharedHttpClient,
        api_key: impl Into<String>,
        config: BedrockClientConfig,
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
        AgentProviderKind::Bedrock.as_str()
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
        system_prompt: &str,
        messages: &[ChatMessage],
        tool_definitions: &[ToolDefinition],
    ) -> Result<Value> {
        let mut payload = Map::new();
        payload.insert(
            "messages".to_string(),
            Value::Array(messages_to_bedrock(messages)?),
        );
        payload.insert("system".to_string(), json!([{ "text": system_prompt }]));
        payload.insert(
            "inferenceConfig".to_string(),
            json!({ "maxTokens": self.config.max_tokens }),
        );
        if !tool_definitions.is_empty() {
            payload.insert(
                "toolConfig".to_string(),
                json!({ "tools": bedrock_tools(tool_definitions) }),
            );
        }
        Ok(Value::Object(payload))
    }

    async fn converse(
        &self,
        model: &str,
        system_prompt: &str,
        messages: &[ChatMessage],
        tool_definitions: &[ToolDefinition],
    ) -> Result<AssistantTurn> {
        let request_started_at = Instant::now();
        let payload = self.build_payload(system_prompt, messages, tool_definitions)?;
        let request = self
            .prepare(HttpRequest::post(self.config.converse_url(model)))
            .json_body(&payload)?;
        let response = self
            .send_classified(request)
            .await
            .context("failed to call Bedrock Converse")?;

        let body: ConverseResponse =
            response.json().context("failed to decode Bedrock response")?;
        let assistant_turn = assistant_turn_from_bedrock(body.output.message)?;

        if self.verbose() {
            info!(
                model = %model,
                duration_ms = request_started_at.elapsed().as_millis() as u64,
                tool_calls = assistant_turn.tool_calls.len(),
                "Bedrock converse completed"
            );
        }

        Ok(assistant_turn)
    }
}

impl ProviderInfo for BedrockClient {
    fn kind(&self) -> AgentProviderKind {
        AgentProviderKind::Bedrock
    }

    fn verbose(&self) -> bool {
        self.config.verbose
    }

    fn model_tiers(&self) -> &ModelTiers {
        &self.config.model_tiers
    }
}

#[async_trait]
impl TextProvider for BedrockClient {
    async fn request_assistant_turn(
        &self,
        model: &str,
        system_prompt: &str,
        history: &[ChatMessage],
        tool_definitions: &[ToolDefinition],
    ) -> Result<AssistantTurn> {
        self.converse(model, system_prompt, history, tool_definitions)
            .await
    }

    async fn stream_message(
        &self,
        model: &str,
        system_prompt: &str,
        messages: &[ChatMessage],
        sink: &mut dyn EventSink,
    ) -> Result<String> {
        // First-cut: buffer a single Converse call and emit it as one delta.
        // See the module-level note on Bedrock's binary event-stream protocol.
        sink.emit(RuntimeEvent::AssistantStarted {
            model: model.to_string(),
        })
        .await?;

        let turn = self.converse(model, system_prompt, messages, &[]).await?;
        let message = turn.content.unwrap_or_default().trim().to_string();
        if message.is_empty() {
            return Err(anyhow!("Bedrock returned an empty message"));
        }
        sink.emit(RuntimeEvent::AssistantDelta {
            delta: message.clone(),
        })
        .await?;
        Ok(message)
    }
}

fn bedrock_tools(definitions: &[ToolDefinition]) -> Vec<Value> {
    definitions
        .iter()
        .map(|definition| {
            json!({
                "toolSpec": {
                    "name": definition.name,
                    "description": definition.description,
                    "inputSchema": { "json": definition.input_schema },
                }
            })
        })
        .collect()
}

fn messages_to_bedrock(messages: &[ChatMessage]) -> Result<Vec<Value>> {
    let mut converted = Vec::new();
    let mut index = 0usize;

    while index < messages.len() {
        let message = &messages[index];
        match message.role {
            MessageRole::System => index += 1,
            MessageRole::User => {
                converted.push(json!({
                    "role": "user",
                    "content": bedrock_user_content(message),
                }));
                index += 1;
            }
            MessageRole::Assistant => {
                converted.push(json!({
                    "role": "assistant",
                    "content": bedrock_assistant_content(message),
                }));
                index += 1;
            }
            MessageRole::Tool => {
                let mut content = Vec::new();
                while index < messages.len() && messages[index].role == MessageRole::Tool {
                    let tool_message = &messages[index];
                    let tool_use_id = tool_message
                        .tool_call_id
                        .clone()
                        .ok_or_else(|| anyhow!("tool message missing tool_call_id"))?;
                    let raw = tool_message.content.clone().unwrap_or_default();
                    let result_content = match serde_json::from_str::<Value>(&raw) {
                        Ok(value) => json!([{ "json": value }]),
                        Err(_) => json!([{ "text": raw }]),
                    };
                    content.push(json!({
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": result_content,
                        }
                    }));
                    index += 1;
                }
                if !content.is_empty() {
                    converted.push(json!({ "role": "user", "content": content }));
                }
            }
        }
    }

    Ok(converted)
}

fn bedrock_user_content(message: &ChatMessage) -> Vec<Value> {
    let mut content = Vec::new();
    if let Some(text) = message.content.as_deref()
        && !text.is_empty()
    {
        content.push(json!({ "text": text }));
    }
    for attachment in &message.attachments {
        // Converse takes image bytes (base64) with an explicit format.
        if attachment.kind == AttachmentKind::Image
            && let AttachmentSource::Base64(data) = &attachment.source
        {
            content.push(json!({
                "image": {
                    "format": bedrock_image_format(&attachment.media_type),
                    "source": { "bytes": data },
                }
            }));
        }
    }
    content
}

fn bedrock_assistant_content(message: &ChatMessage) -> Vec<Value> {
    let mut content = Vec::new();
    if let Some(text) = message.content.as_deref()
        && !text.is_empty()
    {
        content.push(json!({ "text": text }));
    }
    for call in &message.tool_calls {
        content.push(json!({
            "toolUse": {
                "toolUseId": call.id,
                "name": call.name,
                "input": call.arguments,
            }
        }));
    }
    content
}

fn bedrock_image_format(media_type: &str) -> &str {
    match media_type {
        "image/jpeg" | "image/jpg" => "jpeg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        _ => "png",
    }
}

fn assistant_turn_from_bedrock(message: ConverseMessage) -> Result<AssistantTurn> {
    let mut text = String::new();
    let mut tool_calls = Vec::new();

    for block in message.content {
        if let Some(value) = block.text {
            text.push_str(&value);
        }
        if let Some(tool_use) = block.tool_use {
            tool_calls.push(ToolCall {
                id: tool_use.tool_use_id,
                name: tool_use.name,
                arguments: tool_use.input.unwrap_or_else(|| Value::Object(Map::new())),
            });
        }
    }

    let content = (!text.trim().is_empty()).then_some(text);
    Ok(AssistantTurn { content, tool_calls })
}

#[derive(Debug, Deserialize)]
struct ConverseResponse {
    output: ConverseOutput,
}

#[derive(Debug, Deserialize)]
struct ConverseOutput {
    message: ConverseMessage,
}

#[derive(Debug, Deserialize)]
struct ConverseMessage {
    #[serde(default)]
    content: Vec<ConverseContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ConverseContentBlock {
    #[serde(default)]
    text: Option<String>,
    #[serde(default, rename = "toolUse")]
    tool_use: Option<ConverseToolUse>,
}

#[derive(Debug, Deserialize)]
struct ConverseToolUse {
    #[serde(rename = "toolUseId")]
    tool_use_id: String,
    name: String,
    #[serde(default)]
    input: Option<Value>,
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde_json::json;

    use super::{
        BedrockClientConfig, assistant_turn_from_bedrock, messages_to_bedrock,
    };
    use crate::{ChatMessage, ToolCall};

    #[test]
    fn derives_regional_converse_url() {
        let config = BedrockClientConfig::default();
        assert_eq!(
            config.converse_url("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet-20241022-v2:0/converse"
        );
    }

    #[test]
    fn groups_tool_results_into_user_message() -> Result<()> {
        let messages = vec![
            ChatMessage::assistant_with_tools(
                Some("Checking.".to_string()),
                vec![ToolCall {
                    id: "tool_1".to_string(),
                    name: "lookup".to_string(),
                    arguments: json!({ "q": "x" }),
                }],
            ),
            ChatMessage::tool("tool_1", "{\"ok\":true}"),
        ];
        let converted = messages_to_bedrock(&messages)?;
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0]["role"], json!("assistant"));
        assert_eq!(converted[0]["content"][1]["toolUse"]["name"], json!("lookup"));
        assert_eq!(converted[1]["role"], json!("user"));
        assert_eq!(
            converted[1]["content"][0]["toolResult"]["toolUseId"],
            json!("tool_1")
        );
        Ok(())
    }

    #[test]
    fn decodes_text_and_tool_use() -> Result<()> {
        let message = serde_json::from_value(json!({
            "role": "assistant",
            "content": [
                { "text": "One sec." },
                { "toolUse": { "toolUseId": "t1", "name": "lookup", "input": { "q": "x" } } }
            ]
        }))?;
        let turn = assistant_turn_from_bedrock(message)?;
        assert_eq!(turn.content.as_deref(), Some("One sec."));
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].arguments, json!({ "q": "x" }));
        Ok(())
    }
}
