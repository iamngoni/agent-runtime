//! End-to-end flow tests driving the real `Agent` against a mock `HttpClient`.
//!
//! These exercise the request → stream → tool-session pipeline that the
//! in-crate unit tests (which only cover pure wire-format mapping) can't reach:
//! `stream_message`, `execute_tool_session`, retry/backoff, and event emission.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_runtime::{
    Agent, AgentProviderKind, AgentTool, ChatMessage, EventSink, HttpByteStream, HttpClient,
    HttpRequest, HttpResponse, HttpStreamResponse, JsonTool, RetryPolicy, RuntimeEvent,
    ToolRegistry, ToolSessionOutcome, ToolSessionRequest,
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
        self.buffered.lock().unwrap().push_back((status, body.into()));
        self
    }

    fn push_stream(&self, status: u16, body: impl Into<String>) -> &Self {
        self.streamed.lock().unwrap().push_back((status, body.into()));
        self
    }

    fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
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

fn build_agent(mock: MockHttpClient) -> Agent {
    Agent::builder()
        .provider(AgentProviderKind::OpenAi)
        .api_key("test-key")
        .with_http_client(mock)
        .build()
        .expect("agent builds")
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
    let agent = build_agent(mock);

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
    let agent = build_agent(mock.clone());

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

    let agent = Agent::builder()
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
    // 2) Sub-agent's streamed answer (AgentTool drives stream_message).
    mock.push_stream(
        200,
        "data: {\"choices\":[{\"delta\":{\"content\":\"Refund issued for order 42\"}}]}\n\ndata: [DONE]\n\n",
    );
    // 3) Parent's follow-up synthesis after the delegation result.
    mock.push_stream(
        200,
        "data: {\"choices\":[{\"delta\":{\"content\":\"Done: refunded\"}}]}\n\ndata: [DONE]\n\n",
    );

    let parent = build_agent(mock.clone());
    let refunds_agent = build_agent(mock.clone());

    let mut registry = ToolRegistry::<()>::new();
    registry.register(AgentTool::new(
        "refunds_agent",
        "Delegates refund handling to a specialist sub-agent",
        refunds_agent,
        "gpt-4o",
        "You process customer refunds.",
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
    // 3 upstream calls: parent decision + sub-agent stream + parent synthesis.
    assert_eq!(mock.request_count(), 3);
    Ok(())
}

#[tokio::test]
async fn surfaces_non_retryable_errors() -> Result<()> {
    let mock = MockHttpClient::new();
    mock.push_buffered(400, "bad request");
    let agent = build_agent(mock.clone());

    let result = agent
        .request_assistant_turn("gpt-4o", "sys", &[ChatMessage::user("hi")], &[])
        .await;

    assert!(result.is_err());
    assert_eq!(mock.request_count(), 1); // no retry on 400
    Ok(())
}
