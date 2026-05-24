# agent-runtime

A small, runtime-agnostic LLM agent core in Rust. Handles the boring bits — provider plumbing for OpenAI and Anthropic, streaming, tool calling, multi-turn sessions — without locking you into one HTTP stack.

The crate runs on:

- **Native** (tokio, axum, reqwest) — default, via the built-in `reqwest` backend.
- **Cloudflare Workers / wasm32-unknown-unknown** — plug in a `worker::Fetch`-backed `HttpClient` impl and the same `Agent` runs unchanged inside a Worker.
- **Anywhere else with an HTTP client** — implement the `HttpClient` trait and you're in.

## Why

LLM crates that hard-wire `reqwest` (let alone `rustls`) can't compile to `wasm32-unknown-unknown`, which makes them unusable inside Cloudflare Workers / edge runtimes. `agent-runtime` decouples the HTTP transport from the provider logic, so the same agent code drives a tokio server *and* a Worker.

## Install

```toml
[dependencies]
agent-runtime = { git = "https://github.com/modestnerd/agent-runtime", default-features = true }
```

For a Worker (or any other wasm target), turn the reqwest backend off and bring your own:

```toml
[dependencies]
agent-runtime = { git = "...", default-features = false }
```

## Quick start (native, default features)

```rust
use agent_runtime::{Agent, AgentProviderKind, ChatMessage};

let agent = Agent::builder()
    .provider(AgentProviderKind::Anthropic)
    .api_key(std::env::var("ANTHROPIC_API_KEY")?)
    .build()?;

let reply = agent
    .request_assistant_turn(
        "claude-haiku-4-5-20251001",
        "You are a helpful assistant.",
        &[ChatMessage::user("Hello")],
        &[],
    )
    .await?;

println!("{}", reply.content.unwrap_or_default());
```

The default `ReqwestHttpClient` is wired automatically when the `reqwest-http` feature is on (it is, by default).

## Quick start (Cloudflare Worker)

Inside a Worker, `reqwest` won't compile — write a thin adapter on top of `worker::Fetch`:

```rust
use std::sync::Arc;
use agent_runtime::{
    Agent, AgentProviderKind, ChatMessage, HttpClient, HttpMethod, HttpRequest,
    HttpResponse, HttpStreamResponse,
};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use send_wrapper::SendWrapper;
use wasm_bindgen::JsValue;
use worker::{Env, Fetch, Headers, Method, Request, RequestInit};

pub struct WorkerHttpClient;

#[async_trait]
impl HttpClient for WorkerHttpClient {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse> {
        SendWrapper::new(async move {
            let mut headers = Headers::new();
            for (k, v) in &req.headers {
                headers.set(k, v).map_err(|e| anyhow!("{e:?}"))?;
            }
            let body = String::from_utf8(req.body)?;
            let mut init = RequestInit::new();
            init.with_method(convert(req.method))
                .with_headers(headers)
                .with_body(Some(JsValue::from_str(&body)));
            let worker_req = Request::new_with_init(&req.url, &init)
                .map_err(|e| anyhow!("{e:?}"))?;
            let mut resp = Fetch::Request(worker_req).send().await
                .map_err(|e| anyhow!("{e:?}"))?;
            Ok(HttpResponse {
                status: resp.status_code(),
                body: resp.bytes().await.map_err(|e| anyhow!("{e:?}"))?,
            })
        })
        .await
    }

    async fn send_streaming(&self, _req: HttpRequest) -> Result<HttpStreamResponse> {
        // implement once you need streaming
        Err(anyhow!("not implemented"))
    }
}

fn convert(m: HttpMethod) -> Method {
    match m {
        HttpMethod::Get => Method::Get,
        HttpMethod::Post => Method::Post,
        HttpMethod::Put => Method::Put,
        HttpMethod::Patch => Method::Patch,
        HttpMethod::Delete => Method::Delete,
    }
}

// Then:
let agent = Agent::builder()
    .provider(AgentProviderKind::Anthropic)
    .api_key(env.secret("ANTHROPIC_API_KEY")?.to_string())
    .http_client(Arc::new(WorkerHttpClient))
    .build()?;
```

`SendWrapper` is needed because wasm-bindgen futures aren't `Send` and the trait's futures are. workerd is single-threaded so it never trips.

## Streaming

Both providers stream via `agent.stream_message(...)`. You pass an `EventSink` and tokens land via `RuntimeEvent::AssistantDelta { delta }` as the model produces them. A built-in `should_flush_delta` helper batches deltas at sentence breaks if you want chunky output.

```rust
use agent_runtime::{EventSink, RuntimeEvent};

struct PrintSink;

#[async_trait::async_trait]
impl EventSink for PrintSink {
    async fn emit(&mut self, event: RuntimeEvent) -> anyhow::Result<()> {
        if let RuntimeEvent::AssistantDelta { delta } = event {
            print!("{delta}");
        }
        Ok(())
    }
}

agent.stream_message(model, system, &messages, &mut PrintSink).await?;
```

## Tool calling

`ToolRegistry<C>` holds tools keyed by name; `Tool` and `JsonTool` are the two impls. `Agent::execute_tool_session(...)` runs a single tool-using turn end to end: ask the LLM, dispatch any tool calls it requests, return the final natural-language answer.

```rust
use agent_runtime::{JsonTool, ToolRegistry, ToolSessionRequest};

let mut registry = ToolRegistry::<MyContext>::new();
registry.register(JsonTool::new(
    "search_products",
    "Search the catalogue",
    schema_for!(SearchInput),
    |ctx, input: SearchInput| async move { /* … */ Ok(json!({ "results": [...] })) },
));

let outcome = agent
    .execute_tool_session(
        ToolSessionRequest {
            model: "claude-sonnet-4-5-20250929",
            decision_system_prompt: "...",
            followup_system_prompt: "...",
            history: &messages,
            tool_registry: &registry,
            tool_context: ctx,
            max_tool_calls: 4,
        },
        &mut PrintSink,
    )
    .await?;
```

## Feature flags

| Flag | Default | Effect |
|---|---|---|
| `reqwest-http` | on | Pulls in `reqwest` + `rustls-tls`; provides `ReqwestHttpClient` and the `AgentBuilder::reqwest_client(Client)` compat helper. Turn off for wasm. |

## Architecture

```
                         AgentBuilder
                              │
                         (Agent)
                              │
                  ┌───────────┴────────────┐
                  │                        │
            AgentProvider          (you pick:
                  │                  OpenAi or Anthropic)
   ┌──────────────┼──────────────┐
   │              │              │
OpenAiClient   AnthropicClient   (more later)
   │              │
   └─── HttpClient (Arc<dyn>) ───┘
              │
   ┌──────────┴──────────┐
   │                     │
ReqwestHttpClient   WorkerHttpClient   (you write this for non-native)
   (default)        (custom impl)
```

- `HttpClient` is the seam. Two methods: `send` (buffered) and `send_streaming` (chunked).
- `AgentProvider` translates the runtime's neutral `ChatMessage` / `ToolCall` into provider-specific JSON.
- `Agent` is a thin facade over a provider, used directly or via `execute_tool_session` for tool loops.
- `EventSink` is the streaming-progress callback. Implement `emit` to ship deltas wherever (stdout, SSE, websocket).

## Status

- ✅ OpenAI chat completions (streaming + tools)
- ✅ Anthropic messages (streaming + tools)
- ✅ Multi-step tool sessions
- ✅ Native + wasm32 (Cloudflare Workers verified)
- ⏳ More providers as needed (Bedrock, Gemini, Workers AI, local Ollama)

## License

UNLICENSED — internal use.
