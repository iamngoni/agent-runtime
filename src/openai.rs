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

/// How many times a structured answer is requested before giving up.
///
/// Covers the one failure this retry exists for: a model that returns a
/// well-formed response whose generated answer text does not parse. Transport
/// faults are already retried by [`execute_with_retry`], so this counts only
/// attempts lost to unreadable output.
const STRUCTURED_ANSWER_ATTEMPTS: u32 = 2;

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
    /// Sent as `max_completion_tokens` on every request when set. Reasoning
    /// ("thinking") models such as Kimi's share one token budget between
    /// `reasoning_content` and the final `content` - the Kimi Code CLI's own
    /// docs warn a too-small budget "can cause a 200 response with no
    /// `content`" and its agent loop always sets an explicit cap (32,000 as
    /// the fallback for models, like `kimi-for-coding`, with no known
    /// context-window entry) specifically to stop reasoning from starving
    /// out the answer. `None` sends no cap (provider/model default).
    pub max_completion_tokens: Option<u64>,
    /// Reasoning ("thinking") effort/level for models that support it, e.g.
    /// Kimi's `k3` (`"low"`/`"high"`/`"max"`). Wire shape depends on `kind` -
    /// see [`OpenAiClient::apply_reasoning_effort`]. `None` sends nothing
    /// (provider/model default).
    pub reasoning_effort: Option<String>,
    /// How the structured path asks the model for a schema-shaped answer.
    /// Defaults to [`StructuredStrategy::ForcedTool`], the behaviour every
    /// existing caller already depends on.
    pub structured_strategy: StructuredStrategy,
}

/// How [`OpenAiClient`] obtains a schema-conformant answer.
///
/// Both variants declare one synthetic function whose parameters are the
/// requested schema; they differ only in whether the call is compelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StructuredStrategy {
    /// `tool_choice: {"type":"function",...}` — the provider guarantees the
    /// call, so an answer is always schema-shaped.
    ///
    /// Reasoning ("thinking") models reject a compelled choice: DeepSeek's V4
    /// family answers `400 Thinking mode does not support this tool_choice`,
    /// which fails every request rather than degrading.
    #[default]
    ForcedTool,
    /// `tool_choice: "auto"` — the model decides, which thinking models accept.
    ///
    /// With a single tool that is the only answer channel there is nothing
    /// else for the model to do, and it calls it in practice. The guarantee is
    /// nonetheless advisory, so the response path also reads a JSON `content`
    /// body when no call is returned.
    AutoTool,
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
            max_completion_tokens: None,
            reasoning_effort: None,
            structured_strategy: StructuredStrategy::default(),
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
            max_completion_tokens: None,
            reasoning_effort: None,
            structured_strategy: StructuredStrategy::default(),
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

    /// Insert `max_completion_tokens` into a request body when configured.
    /// See [`OpenAiClientConfig::max_completion_tokens`].
    fn apply_max_completion_tokens(&self, payload: &mut Map<String, Value>) {
        if let Some(cap) = self.config.max_completion_tokens {
            payload.insert("max_completion_tokens".to_string(), json!(cap));
        }
    }

    /// Insert the configured reasoning effort into a request body, in
    /// whatever wire shape `kind` expects. See
    /// [`OpenAiClientConfig::reasoning_effort`].
    fn apply_reasoning_effort(&self, payload: &mut Map<String, Value>) {
        let Some(effort) = self.config.reasoning_effort.as_deref() else {
            return;
        };
        match &self.config.kind {
            // Moonshot's Kimi Code endpoint controls reasoning depth via a
            // proprietary `extra_body.thinking` block, not a top-level field.
            AgentProviderKind::Custom(name) if name == "kimi_code" => {
                payload.insert(
                    "thinking".to_string(),
                    json!({ "type": "enabled", "effort": effort }),
                );
            }
            // Standard o-series convention, for whenever a kind adopts it.
            _ => {
                payload.insert("reasoning_effort".to_string(), json!(effort));
            }
        }
    }

    /// The `tool_choice` value for one strategy and schema name.
    fn structured_tool_choice(strategy: StructuredStrategy, schema_name: &str) -> Value {
        match strategy {
            StructuredStrategy::ForcedTool => {
                json!({ "type": "function", "function": { "name": schema_name } })
            }
            StructuredStrategy::AutoTool => Value::String("auto".to_string()),
        }
    }

    /// The schema-shaped answer carried by one assistant turn.
    ///
    /// A tool call is the expected channel. Falling back to a JSON body covers
    /// the model answering directly under `auto`, and any provider that drops
    /// the call but still returns the text.
    /// Reads the first choice into a turn, naming `finish_reason` when it fails.
    ///
    /// A malformed tool-argument string and a truncated one produce the same
    /// parse error, so the error alone cannot tell them apart. `finish_reason`
    /// can: `length` means the provider stopped mid-write and the answer was
    /// never complete, anything else means it considered the answer finished
    /// and sent it that way.
    fn turn_from_choices(choices: Vec<ToolChatCompletionChoice>) -> Result<AssistantTurn> {
        let choice = choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("response did not contain a choice"))?;
        let finish_reason = choice
            .finish_reason
            .unwrap_or_else(|| "unreported".to_string());
        AssistantTurn::try_from(choice.message)
            .with_context(|| format!("finish_reason: {finish_reason}"))
    }

    fn answer_from_turn(turn: AssistantTurn) -> Result<Value> {
        if let Some(call) = turn.tool_calls.into_iter().next() {
            return Ok(call.arguments);
        }
        if let Some(content) = turn.content.as_deref() {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                return serde_json::from_str(trimmed).with_context(|| {
                    format!(
                        "structured response contained neither a tool call nor JSON content \
                         (first 120 chars: {})",
                        trimmed.chars().take(120).collect::<String>()
                    )
                });
            }
        }
        Err(anyhow!("structured response did not contain a tool call"))
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
        self.apply_max_completion_tokens(&mut request_payload);
        self.apply_reasoning_effort(&mut request_payload);

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

        let assistant_turn: AssistantTurn =
            Self::turn_from_choices(body.choices).context("assistant turn could not be read")?;

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

        // One synthetic function whose parameters are the requested schema. It
        // is not a capability the model might want but the channel the answer
        // travels through, so the strategy only decides whether taking it is
        // compelled or merely the obvious thing to do.
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
        self.apply_max_completion_tokens(&mut request_payload);
        self.apply_reasoning_effort(&mut request_payload);
        request_payload.insert(
            "tool_choice".to_string(),
            Self::structured_tool_choice(self.config.structured_strategy, &format.name),
        );

        // A model that writes an unparseable answer usually writes a clean one
        // when asked again: the fault is in the generated text, not the request.
        // Retrying here keeps a transient malformation from costing the caller a
        // decision, while a fixed cap keeps a genuinely broken model from
        // turning every call into several.
        let mut last_error = None;
        for attempt in 1..=STRUCTURED_ANSWER_ATTEMPTS {
            let request = self
                .prepare(HttpRequest::post(self.config.chat_completions_url()))
                .json_body(&Value::Object(request_payload.clone()))?;
            let response = self
                .send_classified(request)
                .await
                .context("failed to call OpenAI-compatible provider for structured output")?;

            let body: ToolChatCompletionResponse = response
                .json()
                .context("failed to decode structured tool response")?;

            let answer = Self::turn_from_choices(body.choices)
                .and_then(Self::answer_from_turn)
                .context("structured output could not be read");

            match answer {
                Ok(answer) => {
                    if self.verbose() {
                        info!(
                            provider = self.provider_name(),
                            model = %model,
                            duration_ms = request_started_at.elapsed().as_millis() as u64,
                            attempt,
                            "structured output completed"
                        );
                    }
                    return Ok(answer);
                }
                Err(error) => {
                    warn!(
                        provider = self.provider_name(),
                        model = %model,
                        attempt,
                        attempts = STRUCTURED_ANSWER_ATTEMPTS,
                        error = %format!("{error:#}"),
                        "structured answer unreadable; retrying"
                    );
                    last_error = Some(error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("structured output could not be read")))
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

        let mut request_payload = Map::new();
        request_payload.insert("model".to_string(), Value::String(model.to_string()));
        request_payload.insert("stream".to_string(), Value::Bool(true));
        request_payload.insert("messages".to_string(), Value::Array(all_messages));
        self.apply_max_completion_tokens(&mut request_payload);
        self.apply_reasoning_effort(&mut request_payload);

        let request = self
            .prepare(HttpRequest::post(self.config.chat_completions_url()))
            .json_body(&Value::Object(request_payload))?;
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
        let mut reasoning_chars = 0usize;
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
                    Ok(ParsedStreamEvent::ReasoningDeltas(deltas)) => {
                        for delta in deltas {
                            reasoning_chars += delta.chars().count();
                        }
                    }
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
            if reasoning_chars > 0 {
                // A reasoning ("thinking") model spent its entire response
                // budget on chain-of-thought (DeepSeek/Kimi-style
                // `reasoning_content`) and never emitted a final answer.
                // Distinguish this from a truly empty/broken response so
                // callers can tell "the model didn't converge in time" apart
                // from "something is wrong with the connection or provider".
                return Err(anyhow!(
                    "streamed assistant message spent its entire response on internal reasoning ({reasoning_chars} chars) without producing a final answer"
                ));
            }
            return Err(anyhow!(
                "streamed assistant message returned an empty message"
            ));
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
                reasoning_chars,
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
    /// Reasoning ("thinking") model chain-of-thought chunks - DeepSeek/Kimi-
    /// style APIs stream these via a separate `delta.reasoning_content`
    /// field, distinct from the final-answer `delta.content`. Callers that
    /// don't care about reasoning can treat this like [`Self::Empty`]; it
    /// exists so a caller CAN tell "the model is still thinking" apart from
    /// "nothing happened at all" when the final answer never arrives.
    ReasoningDeltas(Vec<String>),
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

    let mut content_deltas = Vec::new();
    let mut reasoning_deltas = Vec::new();
    for choice in chunk.choices {
        if let Some(content) = choice.delta.content
            && !content.is_empty()
        {
            content_deltas.push(content);
        }
        if let Some(reasoning) = choice.delta.reasoning_content
            && !reasoning.is_empty()
        {
            reasoning_deltas.push(reasoning);
        }
    }

    if !content_deltas.is_empty() {
        Ok(ParsedStreamEvent::Deltas(content_deltas))
    } else if !reasoning_deltas.is_empty() {
        Ok(ParsedStreamEvent::ReasoningDeltas(reasoning_deltas))
    } else {
        Ok(ParsedStreamEvent::Empty)
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
                    AttachmentSource::Base64(_) => attachment.data_uri().unwrap_or_default(),
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
    /// Reasoning-model ("thinking") chain-of-thought chunks, sent by
    /// DeepSeek/Kimi-style APIs as a field separate from `content`.
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ToolChatCompletionResponse {
    choices: Vec<ToolChatCompletionChoice>,
}

#[derive(Debug, Deserialize)]
struct ToolChatCompletionChoice {
    message: OpenAiAssistantMessage,
    /// Why generation stopped. `length` means the answer was cut off, which is
    /// the difference between "the model wrote bad JSON" and "the model wrote
    /// good JSON we only received half of" — indistinguishable from the parse
    /// error alone, and the first question worth asking when one arrives.
    #[serde(default)]
    finish_reason: Option<String>,
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
    use serde_json::{Map, json};

    use super::{
        OpenAiAssistantMessage, OpenAiClient, OpenAiClientConfig, StructuredStrategy,
        parse_openai_stream_event, tool_call_to_openai_json,
    };
    use crate::http::{HttpClient, HttpRequest, HttpResponse, HttpStreamResponse};
    use crate::provider::AgentProviderKind;
    use crate::{AssistantTurn, ParsedStreamEvent, ToolCall};

    struct NoopHttpClient;

    #[async_trait::async_trait]
    impl HttpClient for NoopHttpClient {
        async fn send(&self, _request: HttpRequest) -> anyhow::Result<HttpResponse> {
            unreachable!("not exercised by this test")
        }

        async fn send_streaming(
            &self,
            _request: HttpRequest,
        ) -> anyhow::Result<HttpStreamResponse> {
            unreachable!("not exercised by this test")
        }
    }

    #[test]
    fn parses_stream_done_event() -> Result<()> {
        assert_eq!(
            parse_openai_stream_event("data: [DONE]")?,
            ParsedStreamEvent::Done
        );
        Ok(())
    }

    #[test]
    fn parses_content_delta_event() -> Result<()> {
        let raw = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
        assert_eq!(
            parse_openai_stream_event(raw)?,
            ParsedStreamEvent::Deltas(vec!["hi".to_string()])
        );
        Ok(())
    }

    #[test]
    fn parses_reasoning_delta_event_separately_from_content() -> Result<()> {
        // DeepSeek/Kimi-style reasoning models stream `reasoning_content`
        // chunks distinct from `content` - reasoning-only chunks must not be
        // mistaken for a real answer.
        let raw = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking...\"}}]}";
        assert_eq!(
            parse_openai_stream_event(raw)?,
            ParsedStreamEvent::ReasoningDeltas(vec!["thinking...".to_string()])
        );
        Ok(())
    }

    #[test]
    fn prefers_content_delta_when_both_present_in_one_chunk() -> Result<()> {
        let raw = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\",\"reasoning_content\":\"thinking\"}}]}";
        assert_eq!(
            parse_openai_stream_event(raw)?,
            ParsedStreamEvent::Deltas(vec!["hi".to_string()])
        );
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
        assert_eq!(
            turn.tool_calls[0].arguments,
            json!({ "query": "lagavulin" })
        );
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
    fn derives_kimi_endpoints_from_base_url() {
        let config = OpenAiClientConfig::for_kind(AgentProviderKind::Kimi);
        assert_eq!(
            config.chat_completions_url(),
            "https://api.moonshot.cn/v1/chat/completions"
        );
        assert_eq!(
            config.embeddings_url(),
            "https://api.moonshot.cn/v1/embeddings"
        );
    }

    #[test]
    fn forced_strategy_compels_the_answer_channel_and_auto_does_not() {
        assert_eq!(
            OpenAiClient::structured_tool_choice(StructuredStrategy::ForcedTool, "veyra_step"),
            json!({ "type": "function", "function": { "name": "veyra_step" } })
        );
        assert_eq!(
            OpenAiClient::structured_tool_choice(StructuredStrategy::AutoTool, "veyra_step"),
            json!("auto")
        );
        // Callers that never opt in keep the compelled call they rely on.
        assert_eq!(
            OpenAiClientConfig::for_kind(AgentProviderKind::OpenAi).structured_strategy,
            StructuredStrategy::ForcedTool
        );
    }

    #[test]
    fn a_tool_call_is_the_answer_and_a_json_body_is_the_fallback() {
        let call = AssistantTurn {
            content: Some("ignored when a call is present".to_string()),
            tool_calls: vec![ToolCall {
                id: "1".to_string(),
                name: "veyra_step".to_string(),
                arguments: json!({ "action": "none" }),
            }],
        };
        assert_eq!(
            OpenAiClient::answer_from_turn(call).unwrap(),
            json!({ "action": "none" })
        );

        // Under `auto` the model may answer directly; that is still an answer.
        let body = AssistantTurn {
            content: Some("  {\"action\":\"open\"}  ".to_string()),
            tool_calls: Vec::new(),
        };
        assert_eq!(
            OpenAiClient::answer_from_turn(body).unwrap(),
            json!({ "action": "open" })
        );
    }

    #[test]
    fn prose_without_a_call_fails_loudly_and_quotes_what_arrived() {
        let prose = AssistantTurn {
            content: Some("I think we should probably wait and see.".to_string()),
            tool_calls: Vec::new(),
        };
        let error = OpenAiClient::answer_from_turn(prose)
            .unwrap_err()
            .to_string();
        // The text is quoted so the failure is diagnosable from the log alone.
        assert!(error.contains("I think we should probably wait"), "{error}");

        let empty = AssistantTurn {
            content: None,
            tool_calls: Vec::new(),
        };
        assert!(OpenAiClient::answer_from_turn(empty).is_err());
    }

    #[test]
    fn applies_configured_max_completion_tokens_to_request_body() {
        let mut with_cap = OpenAiClientConfig::for_kind(AgentProviderKind::Kimi);
        with_cap.max_completion_tokens = Some(32_000);
        let client =
            OpenAiClient::with_config(std::sync::Arc::new(NoopHttpClient), "key", with_cap);
        let mut payload = Map::new();
        client.apply_max_completion_tokens(&mut payload);
        assert_eq!(payload.get("max_completion_tokens"), Some(&json!(32_000)));

        let without_cap = OpenAiClientConfig::for_kind(AgentProviderKind::OpenAi);
        let client =
            OpenAiClient::with_config(std::sync::Arc::new(NoopHttpClient), "key", without_cap);
        let mut payload = Map::new();
        client.apply_max_completion_tokens(&mut payload);
        assert!(!payload.contains_key("max_completion_tokens"));
    }

    #[test]
    fn applies_kimi_code_reasoning_effort_as_thinking_block() {
        let mut config = OpenAiClientConfig::new(
            AgentProviderKind::Custom("kimi_code".to_string()),
            "https://api.kimi.com/coding/v1",
        );
        config.reasoning_effort = Some("max".to_string());
        let client = OpenAiClient::with_config(std::sync::Arc::new(NoopHttpClient), "key", config);
        let mut payload = Map::new();
        client.apply_reasoning_effort(&mut payload);
        assert_eq!(
            payload.get("thinking"),
            Some(&json!({ "type": "enabled", "effort": "max" }))
        );
        assert!(!payload.contains_key("reasoning_effort"));
    }

    #[test]
    fn applies_other_kinds_reasoning_effort_as_standard_field() {
        let mut config = OpenAiClientConfig::for_kind(AgentProviderKind::OpenAi);
        config.reasoning_effort = Some("high".to_string());
        let client = OpenAiClient::with_config(std::sync::Arc::new(NoopHttpClient), "key", config);
        let mut payload = Map::new();
        client.apply_reasoning_effort(&mut payload);
        assert_eq!(payload.get("reasoning_effort"), Some(&json!("high")));
        assert!(!payload.contains_key("thinking"));
    }

    #[test]
    fn skips_reasoning_effort_when_not_configured() {
        let config = OpenAiClientConfig::new(
            AgentProviderKind::Custom("kimi_code".to_string()),
            "https://api.kimi.com/coding/v1",
        );
        let client = OpenAiClient::with_config(std::sync::Arc::new(NoopHttpClient), "key", config);
        let mut payload = Map::new();
        client.apply_reasoning_effort(&mut payload);
        assert!(payload.is_empty());
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
