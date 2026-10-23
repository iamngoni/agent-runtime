//! End-to-end flow tests driving the real `Agent` against a mock `HttpClient`.
//!
//! These exercise the request → stream → tool-session pipeline that the
//! in-crate unit tests (which only cover pure wire-format mapping) can't reach:
//! `stream_message`, `execute_tool_session`, retry/backoff, and event emission.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_runtime::ResponseFormat;
use agent_runtime::{
    Agent, AgentProviderKind, AgentTool, ChatMessage, EventSink, HttpByteStream, HttpClient,
    HttpRequest, HttpResponse, HttpStreamResponse, JsonTool, Llm, RetryPolicy, RuntimeEvent,
    Timeouts, ToolRegistry, ToolSessionOutcome, ToolSessionRequest,
};
use anyhow::Result;
use async_trait::async_trait;
use futures_util::stream;
use serde::Deserialize;
use serde_json::json;

/// A mock HTTP backend that replays queued responses. `send` pops from the
/// buffered queue; `send_streaming` pops from the stream queue. Cloneable so a
/// test can configure it, hand a clone to the agent, and still inspect captured
/// requests afterwards.
#[derive(Clone, Default)]
struct MockHttpClient {
    buffered: Arc<Mutex<VecDeque<(u16, String)>>>,
    streamed: Arc<Mutex<VecDeque<(u16, String)>>>,
    requests: Arc<Mutex<Vec<HttpRequest>>>,
}

impl MockHttpClient {
    fn new() -> Self {
        Self::default()
    }

    fn push_buffered(&self, status: u16, body: impl Into<String>) -> &Self {
        self.buffered
            .lock()
            .unwrap()
            .push_back((status, body.into()));
        self
    }

    fn push_stream(&self, status: u16, body: impl Into<String>) -> &Self {
        self.streamed
            .lock()
            .unwrap()
            .push_back((status, body.into()));
        self
    }

    fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }

    fn last_request_body(&self) -> Vec<u8> {
        self.requests
            .lock()
            .unwrap()
            .last()
            .map(|r| r.body.clone())
            .unwrap_or_default()
    }
}

#[async_trait]
impl HttpClient for MockHttpClient {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse> {
        self.requests.lock().unwrap().push(request);
        let (status, body) = self
            .buffered
            .lock()
            .unwrap()
            .pop_front()
            .expect("no buffered mock response queued");
        Ok(HttpResponse {
            status,
            body: body.into_bytes(),
        })
    }

    async fn send_streaming(&self, request: HttpRequest) -> Result<HttpStreamResponse> {
        self.requests.lock().unwrap().push(request);
        let (status, body) = self
            .streamed
            .lock()
            .unwrap()
            .pop_front()
            .expect("no stream mock response queued");
        let chunk: Result<Vec<u8>> = Ok(body.into_bytes());
        let body: HttpByteStream = Box::pin(stream::once(async move { chunk }));
        Ok(HttpStreamResponse { status, body })
    }
}

/// Records every event the runtime emits for assertions.
#[derive(Default)]
struct RecordingSink {
    events: Vec<RuntimeEvent>,
}

#[async_trait]
impl EventSink for RecordingSink {
    async fn emit(&mut self, event: RuntimeEvent) -> Result<()> {
        self.events.push(event);
        Ok(())
    }
}

impl RecordingSink {
    fn deltas(&self) -> String {
        self.events
            .iter()
            .filter_map(|e| match e {
                RuntimeEvent::AssistantDelta { delta } => Some(delta.as_str()),
                _ => None,
            })
            .collect()
    }

    fn tool_started_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, RuntimeEvent::ToolStarted { .. }))
            .count()
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct EchoArgs {
    text: String,
}

fn build_llm(mock: MockHttpClient) -> Llm {
    Llm::builder()
        .provider(AgentProviderKind::OpenAi)
        .api_key("test-key")
        .with_http_client(mock)
        .build()
        .expect("llm builds")
}

#[tokio::test]
async fn streams_assistant_deltas_and_emits_events() -> Result<()> {
    let mock = MockHttpClient::new();
    mock.push_stream(
        200,
        concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\", world\"}}]}\n\n",
            "data: [DONE]\n\n",
        ),
    );
    let agent = build_llm(mock);

    let mut sink = RecordingSink::default();
    let message = agent
        .stream_message("gpt-4o", "be brief", &[ChatMessage::user("hi")], &mut sink)
        .await?;

    assert_eq!(message, "Hello, world");
    assert_eq!(sink.deltas(), "Hello, world");
    assert!(matches!(
        sink.events.first(),
        Some(RuntimeEvent::AssistantStarted { .. })
    ));
    Ok(())
}

#[tokio::test]
async fn runs_a_full_tool_session_end_to_end() -> Result<()> {
    let mock = MockHttpClient::new();
    // 1) Buffered decision turn: the model asks to call the `echo` tool.
    mock.push_buffered(
        200,
        json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "echo", "arguments": "{\"text\":\"ping\"}" }
                    }]
                }
            }]
        })
        .to_string(),
    );
    // 2) Streamed follow-up synthesis after the tool result.
    mock.push_stream(
        200,
        "data: {\"choices\":[{\"delta\":{\"content\":\"echoed: ping\"}}]}\n\ndata: [DONE]\n\n",
    );
    let agent = build_llm(mock.clone());

    let mut registry = ToolRegistry::<()>::new();
    registry.register(JsonTool::new(
        "echo",
        "Echo the text back",
        |_ctx: (), args: EchoArgs| async move { Ok(json!({ "echoed": args.text })) },
    ));

    let mut sink = RecordingSink::default();
    let outcome = agent
        .execute_tool_session(
            ToolSessionRequest {
                model: "gpt-4o",
                decision_system_prompt: "decide",
                followup_system_prompt: "answer",
                history: &[ChatMessage::user("ping the echo tool")],
                tool_registry: &registry,
                tool_context: (),
                max_tool_calls: 4,
            },
            &mut sink,
        )
        .await?;

    match outcome {
        ToolSessionOutcome::ToolBacked {
            executed_tool_calls,
            message,
            ..
        } => {
            assert_eq!(executed_tool_calls.len(), 1);
            assert_eq!(executed_tool_calls[0].call.name, "echo");
            assert_eq!(message, "echoed: ping");
        }
        ToolSessionOutcome::Direct { .. } => panic!("expected a tool-backed outcome"),
    }

    assert_eq!(sink.tool_started_count(), 1);
    // Two upstream calls: the decision turn + the streamed synthesis.
    assert_eq!(mock.request_count(), 2);
    Ok(())
}

#[tokio::test]
async fn retries_retryable_errors_then_succeeds() -> Result<()> {
    let mock = MockHttpClient::new();
    // First attempt: overloaded (retryable). Second: success.
    mock.push_buffered(503, "overloaded");
    mock.push_buffered(
        200,
        json!({ "choices": [{ "message": { "content": "ok" } }] }).to_string(),
    );

    let agent = Llm::builder()
        .provider(AgentProviderKind::OpenAi)
        .api_key("test-key")
        .with_http_client(mock.clone())
        .retry(RetryPolicy::with_retries(2))
        .build()?;

    let turn = agent
        .request_assistant_turn("gpt-4o", "sys", &[ChatMessage::user("hi")], &[])
        .await?;

    assert_eq!(turn.content.as_deref(), Some("ok"));
    assert_eq!(mock.request_count(), 2); // one retry
    Ok(())
}

/// A declarative agent with its own tool, exercised through `Llm::run`.
struct CashierAgent;

impl Agent for CashierAgent {
    fn instructions(&self) -> String {
        "You are a cashier. Use tools to ring up items.".into()
    }
    fn model(&self) -> String {
        "gpt-4o".into()
    }
    fn tools(&self) -> ToolRegistry<()> {
        let mut r = ToolRegistry::new();
        r.register(JsonTool::new(
            "echo",
            "Echo text",
            |_ctx: (), args: EchoArgs| async move { Ok(json!({ "echoed": args.text })) },
        ));
        r
    }
}

#[tokio::test]
async fn high_level_run_drives_a_declarative_agent() -> Result<()> {
    let mock = MockHttpClient::new();
    // Decision turn asks for the echo tool…
    mock.push_buffered(
        200,
        json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "echo", "arguments": "{\"text\":\"latte\"}" }
                    }]
                }
            }]
        })
        .to_string(),
    );
    // …then the synthesis stream.
    mock.push_stream(
        200,
        "data: {\"choices\":[{\"delta\":{\"content\":\"Rang up: latte\"}}]}\n\ndata: [DONE]\n\n",
    );

    let llm = build_llm(mock);
    // The whole tool session is hidden behind one call:
    let reply = llm.run(&CashierAgent, "ring up a latte").await?;
    assert_eq!(reply, "Rang up: latte");
    Ok(())
}

/// A tool-less agent (so the single buffered turn carries the attachment).
struct VisionAgent;

impl Agent for VisionAgent {
    fn instructions(&self) -> String {
        "Describe images briefly.".into()
    }
    fn model(&self) -> String {
        "gpt-4o".into()
    }
}

#[tokio::test]
async fn run_message_carries_attachments_through_high_level_api() -> Result<()> {
    use agent_runtime::Attachment;

    let mock = MockHttpClient::new();
    // No tools => one buffered decision turn that resolves directly.
    mock.push_buffered(
        200,
        json!({ "choices": [{ "message": { "content": "A cat." } }] }).to_string(),
    );
    let llm = build_llm(mock.clone());

    let message = ChatMessage::user_with_attachments(
        "What's in this image?",
        vec![Attachment::image_url("https://example.com/cat.png")],
    );
    let reply = llm.run_message(&VisionAgent, &[], message).await?;
    assert_eq!(reply, "A cat.");

    // The attachment must have reached the wire as a multimodal image part.
    let sent = String::from_utf8(mock.last_request_body()).unwrap();
    assert!(sent.contains("image_url"), "request body: {sent}");
    assert!(sent.contains("https://example.com/cat.png"));
    Ok(())
}

/// A declarative sub-agent the parent can delegate to.
struct RefundsAgent;

impl Agent for RefundsAgent {
    fn instructions(&self) -> String {
        "You process customer refunds.".into()
    }
    fn model(&self) -> String {
        "gpt-4o".into()
    }
}

#[tokio::test]
async fn parent_agent_delegates_to_sub_agent() -> Result<()> {
    let mock = MockHttpClient::new();
    // 1) Parent decides to delegate to the refunds sub-agent.
    mock.push_buffered(
        200,
        json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "refunds_agent",
                            "arguments": "{\"task\":\"refund order 42\"}"
                        }
                    }]
                }
            }]
        })
        .to_string(),
    );
    // 2) Sub-agent runs its own session (no tools) → one buffered direct turn.
    mock.push_buffered(
        200,
        json!({ "choices": [{ "message": { "content": "Refund issued for order 42" } }] })
            .to_string(),
    );
    // 3) Parent's follow-up synthesis after the delegation result.
    mock.push_stream(
        200,
        "data: {\"choices\":[{\"delta\":{\"content\":\"Done: refunded\"}}]}\n\ndata: [DONE]\n\n",
    );

    let parent = build_llm(mock.clone());
    let refunds_llm = build_llm(mock.clone());

    let mut registry = ToolRegistry::<()>::new();
    registry.register(AgentTool::new(
        "refunds_agent",
        "Delegates refund handling to a specialist sub-agent",
        refunds_llm,
        RefundsAgent,
    ));

    let mut sink = RecordingSink::default();
    let outcome = parent
        .execute_tool_session(
            ToolSessionRequest {
                model: "gpt-4o",
                decision_system_prompt: "decide",
                followup_system_prompt: "answer",
                history: &[ChatMessage::user("please refund order 42")],
                tool_registry: &registry,
                tool_context: (),
                max_tool_calls: 4,
            },
            &mut sink,
        )
        .await?;

    match outcome {
        ToolSessionOutcome::ToolBacked {
            executed_tool_calls,
            message,
            ..
        } => {
            assert_eq!(executed_tool_calls[0].call.name, "refunds_agent");
            // The sub-agent's reply is captured in the tool output.
            assert_eq!(
                executed_tool_calls[0].output.content["response"],
                json!("Refund issued for order 42")
            );
            assert_eq!(message, "Done: refunded");
        }
        ToolSessionOutcome::Direct { .. } => panic!("expected delegation to run"),
    }
    // 3 upstream calls: parent decision + sub-agent turn + parent synthesis.
    assert_eq!(mock.request_count(), 3);
    Ok(())
}

#[derive(Debug, Deserialize, schemars::JsonSchema, PartialEq)]
struct Summary {
    sentiment: String,
    score: i64,
}

/// A tool-less agent used to exercise single-shot structured output.
struct SummarizerAgent;

impl Agent for SummarizerAgent {
    fn instructions(&self) -> String {
        "Summarize the sentiment of the input.".into()
    }
    fn model(&self) -> String {
        "gpt-4o".into()
    }
}

#[tokio::test]
async fn run_structured_with_format_sends_caller_schema() -> Result<()> {
    let mock = MockHttpClient::new();
    mock.push_buffered(
        200,
        json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_f",
                        "type": "function",
                        "function": {
                            "name": "trade_intent",
                            "arguments": "{\"side\":\"buy\",\"lots\":0.1}"
                        }
                    }]
                }
            }]
        })
        .to_string(),
    );
    let llm = build_llm(mock.clone());

    let schema = json!({
        "type": "object",
        "properties": {
            "side": {"type": "string", "enum": ["buy", "sell"]},
            "lots": {"type": "number"}
        },
        "required": ["side", "lots"]
    });
    let format = ResponseFormat::new("trade_intent", schema);

    let value = llm
        .run_structured_with_format(&SummarizerAgent, "propose a trade", format)
        .await?;
    assert_eq!(value["side"], "buy");
    assert_eq!(value["lots"], 0.1);

    let sent = String::from_utf8(mock.last_request_body()).unwrap();
    assert!(sent.contains("trade_intent"), "request body: {sent}");
    assert!(
        sent.contains("\"enum\":[\"buy\",\"sell\"]"),
        "request body: {sent}"
    );
    assert_eq!(mock.request_count(), 1);
    Ok(())
}

/// A model that writes a stray closing brace costs a retry, not the answer.
///
/// Observed in production against `deepseek/deepseek-v4.1-flash`: a complete,
/// schema-correct answer followed by one extra `}`, which fails the parse as
/// "trailing characters". The answer itself was never in doubt, so asking again
/// is the proportionate response.
#[tokio::test]
async fn run_structured_retries_an_unparseable_answer() -> Result<()> {
    let mock = MockHttpClient::new();
    let response = |arguments: &str| {
        json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_f",
                        "type": "function",
                        "function": { "name": "trade_intent", "arguments": arguments }
                    }]
                }
            }]
        })
        .to_string()
    };
    // Trailing brace: valid answer, invalid JSON.
    mock.push_buffered(200, response("{\"side\":\"buy\",\"lots\":0.1}}"));
    mock.push_buffered(200, response("{\"side\":\"buy\",\"lots\":0.1}"));
    let llm = build_llm(mock.clone());

    let format = ResponseFormat::new(
        "trade_intent",
        json!({
            "type": "object",
            "properties": {
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "lots": {"type": "number"}
            },
            "required": ["side", "lots"]
        }),
    );

    let value = llm
        .run_structured_with_format(&SummarizerAgent, "propose a trade", format)
        .await?;

    assert_eq!(value["side"], "buy");
    assert_eq!(value["lots"], 0.1);
    assert_eq!(
        mock.request_count(),
        2,
        "the bad answer should cost a retry"
    );
    Ok(())
}

/// Two unparseable answers give up rather than asking forever.
#[tokio::test]
async fn run_structured_stops_after_the_attempt_cap() -> Result<()> {
    let mock = MockHttpClient::new();
    let bad = json!({
        "choices": [{
            "finish_reason": "length",
            "message": {
                "content": null,
                "tool_calls": [{
                    "id": "call_f",
                    "type": "function",
                    "function": { "name": "trade_intent", "arguments": "{\"side\":\"bu" }
                }]
            }
        }]
    })
    .to_string();
    mock.push_buffered(200, bad.clone());
    mock.push_buffered(200, bad);
    let llm = build_llm(mock.clone());

    let format = ResponseFormat::new("trade_intent", json!({"type": "object"}));
    let error = llm
        .run_structured_with_format(&SummarizerAgent, "propose a trade", format)
        .await
        .expect_err("an unreadable answer must not be reported as success");

    // The reported cause carries finish_reason, which separates a truncated
    // answer from a malformed one without another round trip.
    let rendered = format!("{error:#}");
    assert!(rendered.contains("length"), "error: {rendered}");
    assert_eq!(mock.request_count(), 2, "attempts are capped");
    Ok(())
}

#[tokio::test]
async fn run_structured_returns_typed_payload() -> Result<()> {
    let mock = MockHttpClient::new();
    // Forced-function structured response — the schema name is derived from the
    // type (`Summary`), and the arguments are the typed payload.
    mock.push_buffered(
        200,
        json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_s",
                        "type": "function",
                        "function": {
                            "name": "Summary",
                            "arguments": "{\"sentiment\":\"positive\",\"score\":4}"
                        }
                    }]
                }
            }]
        })
        .to_string(),
    );
    let llm = build_llm(mock.clone());

    let summary: Summary = llm.run_structured(&SummarizerAgent, "I love this!").await?;
    assert_eq!(
        summary,
        Summary {
            sentiment: "positive".into(),
            score: 4
        }
    );

    // Enforcement: the request forced the schema tool, not free text.
    let sent = String::from_utf8(mock.last_request_body()).unwrap();
    assert!(sent.contains("tool_choice"), "request body: {sent}");
    assert!(
        sent.contains("\"name\":\"Summary\""),
        "request body: {sent}"
    );
    assert_eq!(mock.request_count(), 1);
    Ok(())
}

/// An agent with a tool, to prove structured output runs tools first and only
/// then constrains the *final* answer to the schema.
struct StructuredCashierAgent;

impl Agent for StructuredCashierAgent {
    fn instructions(&self) -> String {
        "Ring up items with the echo tool, then summarize.".into()
    }
    fn model(&self) -> String {
        "gpt-4o".into()
    }
    fn tools(&self) -> ToolRegistry<()> {
        let mut r = ToolRegistry::new();
        r.register(JsonTool::new(
            "echo",
            "Echo text",
            |_ctx: (), args: EchoArgs| async move { Ok(json!({ "echoed": args.text })) },
        ));
        r
    }
}

#[tokio::test]
async fn run_structured_runs_tools_then_constrains_final_answer() -> Result<()> {
    let mock = MockHttpClient::new();
    // 1) Tool-gathering decision turn: call echo.
    mock.push_buffered(
        200,
        json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "echo", "arguments": "{\"text\":\"latte\"}" }
                    }]
                }
            }]
        })
        .to_string(),
    );
    // 2) Forced structured finalizer after the tool result.
    mock.push_buffered(
        200,
        json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_s",
                        "type": "function",
                        "function": {
                            "name": "Summary",
                            "arguments": "{\"sentiment\":\"served\",\"score\":1}"
                        }
                    }]
                }
            }]
        })
        .to_string(),
    );
    let llm = build_llm(mock.clone());

    let summary: Summary = llm
        .run_structured(&StructuredCashierAgent, "one latte")
        .await?;
    assert_eq!(summary.sentiment, "served");
    // Two upstream calls: the tool-gathering decision turn + the structured
    // finalizer (no streamed synthesis on the structured path).
    assert_eq!(mock.request_count(), 2);
    Ok(())
}

#[tokio::test]
async fn surfaces_non_retryable_errors() -> Result<()> {
    let mock = MockHttpClient::new();
    mock.push_buffered(400, "bad request");
    let agent = build_llm(mock.clone());

    let result = agent
        .request_assistant_turn("gpt-4o", "sys", &[ChatMessage::user("hi")], &[])
        .await;

    assert!(result.is_err());
    assert_eq!(mock.request_count(), 1); // no retry on 400
    Ok(())
}

/// A client whose streaming call answers headers and then hangs forever. The
/// guard signals when the body stream is dropped, proving cancellation reaches
/// the transport.
struct HangingStream {
    started: Arc<tokio::sync::Notify>,
    dropped: Arc<std::sync::atomic::AtomicBool>,
}

#[async_trait]
impl HttpClient for HangingStream {
    async fn send(&self, _request: HttpRequest) -> Result<HttpResponse> {
        Err(anyhow::anyhow!("buffered call unused"))
    }

    async fn send_streaming(&self, _request: HttpRequest) -> Result<HttpStreamResponse> {
        struct Guard(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let guard = Guard(self.dropped.clone());
        let started = self.started.clone();
        let body = stream::once(async move {
            started.notify_one();
            let _guard = guard;
            std::future::pending::<Result<Vec<u8>>>().await
        });
        Ok(HttpStreamResponse {
            status: 200,
            body: Box::pin(body),
        })
    }
}

#[tokio::test]
async fn dropping_a_stream_aborts_the_request() -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};

    let started = Arc::new(tokio::sync::Notify::new());
    let dropped = Arc::new(AtomicBool::new(false));
    let client = HangingStream {
        started: started.clone(),
        dropped: dropped.clone(),
    };
    let llm = Llm::builder()
        .provider(AgentProviderKind::OpenAi)
        .api_key("test-key")
        .with_http_client(client)
        .timeouts(agent_runtime::Timeouts::none())
        .build()?;

    let task = tokio::spawn(async move {
        let mut sink = RecordingSink::default();
        let _ = llm
            .stream_message("gpt-4o", "be brief", &[ChatMessage::user("hi")], &mut sink)
            .await;
    });

    started.notified().await;
    task.abort();
    let _ = task.await;
    // Give the aborted task's destructors a moment to run.
    for _ in 0..100 {
        if dropped.load(Ordering::SeqCst) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert!(
        dropped.load(Ordering::SeqCst),
        "dropping the stream must drop the response body"
    );
    Ok(())
}

#[tokio::test]
async fn a_silent_server_is_a_typed_timeout_not_a_hang() {
    struct Silent;
    #[async_trait]
    impl HttpClient for Silent {
        async fn send(&self, _request: HttpRequest) -> Result<HttpResponse> {
            std::future::pending().await
        }
        async fn send_streaming(&self, _request: HttpRequest) -> Result<HttpStreamResponse> {
            std::future::pending().await
        }
    }

    let llm = Llm::builder()
        .provider(AgentProviderKind::OpenAi)
        .api_key("test-key")
        .with_http_client(Silent)
        .timeouts(
            agent_runtime::Timeouts::none().response(Some(std::time::Duration::from_millis(30))),
        )
        .build()
        .expect("llm builds");

    let error = llm
        .request_assistant_turn("gpt-4o", "sys", &[ChatMessage::user("hi")], &[])
        .await
        .expect_err("a silent server must fail");
    let provider_error = error
        .chain()
        .find_map(|error| error.downcast_ref::<agent_runtime::ProviderError>())
        .expect("typed provider error");
    assert!(matches!(
        provider_error,
        agent_runtime::ProviderError::Timeout {
            stage: agent_runtime::TimeoutStage::Response,
            ..
        }
    ));
}

#[tokio::test]
async fn a_stream_without_its_end_marker_is_reported_as_incomplete() -> Result<()> {
    let mock = MockHttpClient::new();
    mock.push_stream(
        200,
        "data: {\"choices\":[{\"delta\":{\"content\":\"half an answer\"}}]}\n\n",
    );
    let llm = build_llm(mock);
    let mut sink = RecordingSink::default();
    let message = llm
        .stream_message("gpt-4o", "be brief", &[ChatMessage::user("hi")], &mut sink)
        .await?;
    assert_eq!(message, "half an answer");
    assert!(matches!(
        sink.events.last(),
        Some(RuntimeEvent::AssistantEnded { complete: false })
    ));
    Ok(())
}

#[tokio::test]
async fn a_complete_stream_is_reported_as_complete() -> Result<()> {
    let mock = MockHttpClient::new();
    mock.push_stream(
        200,
        "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\ndata: [DONE]\n\n",
    );
    let llm = build_llm(mock);
    let mut sink = RecordingSink::default();
    llm.stream_message("gpt-4o", "be brief", &[ChatMessage::user("hi")], &mut sink)
        .await?;
    assert!(matches!(
        sink.events.last(),
        Some(RuntimeEvent::AssistantEnded { complete: true })
    ));
    Ok(())
}

#[tokio::test]
async fn max_completion_tokens_reaches_the_openai_wire() -> Result<()> {
    let mock = MockHttpClient::new();
    mock.push_buffered(
        200,
        json!({ "choices": [{ "message": { "content": "ok" } }] }).to_string(),
    );
    let llm = Llm::builder()
        .provider(AgentProviderKind::OpenAi)
        .api_key("test-key")
        .with_http_client(mock.clone())
        .max_completion_tokens(4096)
        .build()?;
    llm.request_assistant_turn("gpt-4o", "sys", &[ChatMessage::user("hi")], &[])
        .await?;
    let sent = String::from_utf8(mock.last_request_body()).unwrap();
    assert!(
        sent.contains("\"max_completion_tokens\":4096"),
        "request body: {sent}"
    );
    Ok(())
}

#[tokio::test]
async fn a_real_socket_that_accepts_and_stays_silent_ends_in_a_typed_timeout() -> Result<()> {
    use std::time::Duration;

    // A real TCP listener: it accepts the connection, reads nothing and never
    // writes a response, exactly the server that used to hang forever.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.expect("one connection");
        tokio::time::sleep(Duration::from_secs(30)).await;
        drop(socket);
    });

    let llm = Llm::builder()
        .provider(AgentProviderKind::OpenAi)
        .api_key("test-key")
        .base_url(format!("http://{address}/v1"))
        .timeouts(Timeouts::none().response(Some(Duration::from_millis(200))))
        .build()?;

    let started = std::time::Instant::now();
    let error = llm
        .request_assistant_turn("gpt-4o", "sys", &[ChatMessage::user("hi")], &[])
        .await
        .expect_err("a silent server must time out");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the timeout must fire promptly, not hang"
    );
    let provider_error = error
        .chain()
        .find_map(|error| error.downcast_ref::<agent_runtime::ProviderError>())
        .expect("typed provider error");
    assert!(matches!(
        provider_error,
        agent_runtime::ProviderError::Timeout {
            stage: agent_runtime::TimeoutStage::Response,
            ..
        }
    ));
    server.abort();
    Ok(())
}

#[tokio::test]
async fn dropping_the_streaming_future_drops_the_transport_future() -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};

    struct HangingFuture {
        started: Arc<tokio::sync::Notify>,
        dropped: Arc<AtomicBool>,
    }

    #[async_trait]
    impl HttpClient for HangingFuture {
        async fn send(&self, _request: HttpRequest) -> Result<HttpResponse> {
            Err(anyhow::anyhow!("buffered call unused"))
        }

        async fn send_streaming(&self, _request: HttpRequest) -> Result<HttpStreamResponse> {
            struct Guard(Arc<AtomicBool>);
            impl Drop for Guard {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::SeqCst);
                }
            }
            let _guard = Guard(self.dropped.clone());
            self.started.notify_one();
            std::future::pending::<Result<HttpStreamResponse>>().await
        }
    }

    let started = Arc::new(tokio::sync::Notify::new());
    let dropped = Arc::new(AtomicBool::new(false));
    let llm = Llm::builder()
        .provider(AgentProviderKind::OpenAi)
        .api_key("test-key")
        .with_http_client(HangingFuture {
            started: started.clone(),
            dropped: dropped.clone(),
        })
        .timeouts(Timeouts::none())
        .build()?;

    let mut sink = RecordingSink::default();
    let messages = [ChatMessage::user("hi")];
    let mut future = Box::pin(llm.stream_message("gpt-4o", "be brief", &messages, &mut sink));
    tokio::select! {
        _ = &mut future => panic!("the transport must not answer"),
        _ = started.notified() => {}
    }
    drop(future);
    assert!(
        dropped.load(Ordering::SeqCst),
        "dropping the streaming future must drop the transport future"
    );
    Ok(())
}

fn build_llm_for(
    mock: MockHttpClient,
    provider: AgentProviderKind,
    max_completion_tokens: Option<u64>,
) -> Llm {
    let mut builder = Llm::builder()
        .provider(provider)
        .api_key("test-key")
        .with_http_client(mock);
    if let Some(cap) = max_completion_tokens {
        builder = builder.max_completion_tokens(cap);
    }
    builder.build().expect("llm builds")
}

/// Anthropic rejects a follow-up carrying `tool_use`/`tool_result` blocks
/// unless the request also declares `tools`. The streamed synthesis after a
/// tool call must include them (with `tool_choice: none`).
#[tokio::test]
async fn anthropic_followup_after_a_tool_call_declares_the_tools() -> Result<()> {
    let mock = MockHttpClient::new();
    mock.push_buffered(
        200,
        json!({
            "content": [{
                "type": "tool_use",
                "id": "tool_1",
                "name": "echo",
                "input": { "text": "ping" }
            }],
            "stop_reason": "tool_use"
        })
        .to_string(),
    );
    mock.push_stream(
        200,
        concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"echoed: ping\"}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ),
    );

    let agent = build_llm_for(mock.clone(), AgentProviderKind::Anthropic, None);
    let mut registry = ToolRegistry::<()>::new();
    registry.register(JsonTool::new(
        "echo",
        "Echo the text back",
        |_ctx: (), args: EchoArgs| async move { Ok(json!({ "echoed": args.text })) },
    ));

    let mut sink = RecordingSink::default();
    let outcome = agent
        .execute_tool_session(
            ToolSessionRequest {
                model: "claude-3-5-sonnet-latest",
                decision_system_prompt: "decide",
                followup_system_prompt: "answer",
                history: &[ChatMessage::user("ping the echo tool")],
                tool_registry: &registry,
                tool_context: (),
                max_tool_calls: 4,
            },
            &mut sink,
        )
        .await?;

    assert_eq!(outcome.message(), "echoed: ping");
    let sent = String::from_utf8(mock.last_request_body()).unwrap();
    assert!(sent.contains("\"tools\""), "request body: {sent}");
    assert!(sent.contains("echo"), "request body: {sent}");
    assert!(
        sent.contains("\"tool_choice\":{\"type\":\"none\"}"),
        "request body: {sent}"
    );
    Ok(())
}

/// Bedrock Converse rejects `toolUse`/`toolResult` blocks unless the request
/// carries `toolConfig`; the follow-up must keep the definitions.
#[tokio::test]
async fn bedrock_followup_after_a_tool_call_declares_the_tools() -> Result<()> {
    let mock = MockHttpClient::new();
    mock.push_buffered(
        200,
        json!({
            "output": { "message": { "role": "assistant", "content": [{
                "toolUse": { "toolUseId": "tool_1", "name": "echo", "input": { "text": "ping" } }
            }] } }
        })
        .to_string(),
    );
    mock.push_buffered(
        200,
        json!({
            "output": { "message": { "role": "assistant", "content": [{ "text": "echoed: ping" }] } }
        })
        .to_string(),
    );

    let agent = build_llm_for(mock.clone(), AgentProviderKind::Bedrock, None);
    let mut registry = ToolRegistry::<()>::new();
    registry.register(JsonTool::new(
        "echo",
        "Echo the text back",
        |_ctx: (), args: EchoArgs| async move { Ok(json!({ "echoed": args.text })) },
    ));

    let mut sink = RecordingSink::default();
    let outcome = agent
        .execute_tool_session(
            ToolSessionRequest {
                model: "anthropic.claude-3-5-sonnet-20241022-v2:0",
                decision_system_prompt: "decide",
                followup_system_prompt: "answer",
                history: &[ChatMessage::user("ping the echo tool")],
                tool_registry: &registry,
                tool_context: (),
                max_tool_calls: 4,
            },
            &mut sink,
        )
        .await?;

    assert_eq!(outcome.message(), "echoed: ping");
    let sent = String::from_utf8(mock.last_request_body()).unwrap();
    assert!(sent.contains("\"toolConfig\""), "request body: {sent}");
    assert!(sent.contains("echo"), "request body: {sent}");
    Ok(())
}

/// Mistral's chat-completions API documents `max_tokens`, not
/// `max_completion_tokens`; a recorded request must use the documented field.
#[tokio::test]
async fn mistral_takes_max_tokens_on_the_wire() -> Result<()> {
    let mock = MockHttpClient::new();
    mock.push_stream(
        200,
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
    );
    let agent = build_llm_for(mock.clone(), AgentProviderKind::Mistral, Some(1024));

    let mut sink = RecordingSink::default();
    agent
        .stream_message(
            "mistral-small-latest",
            "be brief",
            &[ChatMessage::user("hi")],
            &mut sink,
        )
        .await?;

    let sent = String::from_utf8(mock.last_request_body()).unwrap();
    assert!(sent.contains("\"max_tokens\":1024"), "request body: {sent}");
    assert!(
        !sent.contains("max_completion_tokens"),
        "request body: {sent}"
    );
    Ok(())
}
