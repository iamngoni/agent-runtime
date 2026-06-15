# agent-runtime

A small, runtime-agnostic LLM agent core in Rust. Handles the boring bits — provider plumbing across many vendors, streaming, tool calling, sub-agent delegation, image/document attachments, embeddings, conversation memory, multi-turn sessions, typed errors and retries — without locking you into one HTTP stack.

**Providers:** OpenAI, Anthropic, Google Gemini, Cohere, AWS Bedrock, plus every OpenAI-compatible endpoint — Groq, DeepSeek, xAI (Grok), Mistral, Ollama, OpenRouter, and any custom gateway (vLLM, LiteLLM, self-hosted) via a base URL.

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

## Choosing a provider

Named vendors carry their own default base URL and model tiers — just pick one:

```rust
use agent_runtime::{Agent, AgentProviderKind, ModelTier};

// Groq, DeepSeek, xAI, Mistral, Ollama, OpenRouter all work the same way —
// they speak the OpenAI wire format, so they reuse one client.
let agent = Agent::builder()
    .provider(AgentProviderKind::Groq)
    .api_key(std::env::var("GROQ_API_KEY")?)
    .build()?;

// Ask for a capability tier instead of hardcoding a model string:
let model = agent.model_for(ModelTier::Smartest); // -> "llama-3.3-70b-versatile"
```

Any other OpenAI-compatible endpoint (self-hosted vLLM/LiteLLM, a vendor without
a preset) goes through the escape hatch — supply a name and base URL:

```rust
let agent = Agent::builder()
    .openai_compatible("local-vllm", "http://localhost:8000/v1")
    .api_key("…")
    .model_tiers(agent_runtime::ModelTiers::new("my-model", "my-model", "my-model"))
    .header("X-Title", "my-app") // extra headers sent on every request
    .build()?;
```

Google Gemini has its own native client (distinct wire format) but the same builder API:

```rust
let agent = Agent::builder()
    .provider(AgentProviderKind::Gemini)
    .api_key(std::env::var("GEMINI_API_KEY")?)
    .build()?;
```

## Typed errors & retries

Provider failures are classified into [`ProviderError`] variants — `RateLimited`
(429), `InsufficientCredits` (402), `Overloaded` (503/529), `Status`, and
`Transport` — so callers can tell "retry me" from "fatal." Opt into automatic
backoff with a `RetryPolicy` (default is no retries):

```rust
use agent_runtime::RetryPolicy;

let agent = Agent::builder()
    .provider(AgentProviderKind::OpenAi)
    .api_key("…")
    .retry(RetryPolicy::with_retries(3)) // exponential backoff on retryable errors
    .build()?;
```

## Embeddings

OpenAI-compatible providers implement the `EmbeddingProvider` capability:

```rust
use agent_runtime::{EmbeddingProvider, OpenAiClient, ReqwestHttpClient};
use std::sync::Arc;

let client = OpenAiClient::new(Arc::new(ReqwestHttpClient::default()), api_key);
let vectors = client
    .embed("text-embedding-3-small", &["hello".to_string(), "world".to_string()])
    .await?;
```

## Attachments (images & documents)

Ride media alongside a user message; each provider maps it to its own
multimodal format (OpenAI `image_url`/`file`, Anthropic `image`/`document`,
Gemini `inline_data`, Cohere `image_url`, Bedrock image bytes). Non-supported
media types are skipped rather than erroring.

```rust
use agent_runtime::{Attachment, ChatMessage};

let message = ChatMessage::user_with_attachments(
    "What's in this screenshot?",
    vec![Attachment::image_base64("image/png", base64_png)],
);
// or by URL, or a document:
// Attachment::image_url("https://…/chart.png")
// Attachment::document_base64("application/pdf", base64_pdf)
```

## Conversation memory

Multi-turn history is a pluggable seam, `ConversationStore`, mirroring the
`HttpClient` pattern. The bundled `InMemoryConversationStore` is for **tests and
single-process dev only** — it isn't durable or shared across instances/Worker
isolates. For production, implement `ConversationStore` over Cloudflare KV /
Durable Objects, Postgres, Redis, etc.

```rust
use agent_runtime::{ConversationStore, InMemoryConversationStore, ChatMessage};

let store = InMemoryConversationStore::new();
let convo = store.create_conversation(Some("user-1"), "support").await?;

// load history, run the agent, then persist the new turns:
let history = store.latest_messages(&convo, 100).await?;
store.append_message(&convo, &ChatMessage::user("…")).await?;
store.append_message(&convo, &ChatMessage::assistant("…")).await?;
```

## Sub-agents (agents as tools)

Expose a specialised agent to another agent as a callable tool — the parent
delegates a self-contained task and gets the sub-agent's answer back as the tool
result. The sub-agent runs in isolation (no access to the parent conversation).

```rust
use agent_runtime::{AgentTool, ToolRegistry};

let refunds_agent = Agent::builder()
    .provider(AgentProviderKind::OpenAi).api_key(key.clone()).build()?;

let mut registry = ToolRegistry::<MyContext>::new();
registry.register(AgentTool::new(
    "refunds_agent",
    "Delegates refund handling to a specialist sub-agent",
    refunds_agent,
    "gpt-4o",
    "You process customer refunds.",
));
// The parent now calls `refunds_agent` like any other tool in a tool session.
```

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
    // the JSON schema is derived from the `SearchInput: JsonSchema` arg type
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
              Capability traits (a provider implements what it supports)
        ┌────────────────────┬──────────────────────────┐
        │                    │                           │
   ProviderInfo        TextProvider                EmbeddingProvider
  (kind, tiers)   (turns, streaming, tools)        (embed vectors)
        │                    │                           │
   ┌────┴──────────┬─────────┴───────────┐               │
   │               │                     │               │
OpenAiClient   AnthropicClient      GeminiClient          │
(OpenAI +      (native)             (native)              │
 Groq/DeepSeek/                                           │
 Xai/Mistral/         OpenAiClient also implements ───────┘
 Ollama/OpenRouter/   EmbeddingProvider
 Custom — base URL)
   │               │                     │
   └────────── HttpClient (Arc<dyn>) ────┘
                       │
            ┌──────────┴──────────┐
            │                     │
     ReqwestHttpClient     WorkerHttpClient   (you write this for non-native)
        (default)          (custom impl)
```

- **Capability traits** (à la Laravel AI's provider contracts): `ProviderInfo` carries identity + model tiers; `TextProvider` is text/tools/streaming; `EmbeddingProvider` is vectors. A provider implements only what it offers, so misuse (embeddings on a text-only provider) is a compile error. `AgentProvider` remains as a backward-compatible alias for `TextProvider`.
- **One client, many vendors:** `OpenAiClient` is parameterised by base URL + headers, so Groq/DeepSeek/xAI/Mistral/Ollama/OpenRouter/custom all reuse the same wire code. Anthropic and Gemini have native clients.
- `HttpClient` is the transport seam. Two methods: `send` (buffered) and `send_streaming` (chunked).
- `ProviderError` + `RetryPolicy` classify failures and drive optional backoff/failover.
- `Agent` is a thin facade over a provider, used directly or via `execute_tool_session` for tool loops.
- `EventSink` is the streaming-progress callback. Implement `emit` to ship deltas wherever (stdout, SSE, websocket).

## Status

- ✅ OpenAI chat completions (streaming + tools)
- ✅ Anthropic messages (streaming + tools)
- ✅ Google Gemini (streaming + tools, native client)
- ✅ Cohere v2 chat (streaming + tools, native client)
- ✅ AWS Bedrock Converse (bearer-token auth; streaming buffered — see note below)
- ✅ OpenAI-compatible vendors: Groq, DeepSeek, xAI, Mistral, Ollama, OpenRouter, custom
- ✅ Embeddings (OpenAI-compatible providers)
- ✅ Image & document attachments (multimodal input)
- ✅ Conversation memory (`ConversationStore` seam + in-memory impl)
- ✅ Sub-agent delegation (agents as tools)
- ✅ Typed provider errors + retry/backoff
- ✅ Multi-step tool sessions
- ✅ Native + wasm32 (Cloudflare Workers verified)
- ⏳ Bedrock SigV4/IAM auth and native token streaming (binary event-stream)
- ⏳ More providers as needed (Workers AI, Azure OpenAI)

> **Bedrock streaming note:** Bedrock's `converse-stream` uses AWS's binary `vnd.amazon.eventstream` framing, which doesn't fit the SSE-oriented byte stream. The first cut therefore buffers a single `converse` call and emits it as one delta. Real token streaming (and SigV4 auth) are follow-ups.

## License

UNLICENSED — internal use.
