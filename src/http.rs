//! Pluggable HTTP backend for LLM provider calls.
//!
//! Providers (OpenAI, Anthropic, …) talk to their APIs through [`HttpClient`]
//! rather than depending on a specific HTTP client. That lets the runtime
//! work on Cloudflare Workers (where `reqwest` + `rustls` can't compile to
//! `wasm32-unknown-unknown`) by plugging in a `worker::Fetch`-backed impl,
//! while keeping `reqwest` as the simple default for native servers.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::stream::BoxStream;
use serde::de::DeserializeOwned;

pub type HttpHeaders = Vec<(String, String)>;

/// A stream of response-body chunks. `Send + 'static` so it works through
/// `BoxStream` on multi-threaded runtimes; wasm impls can wrap their
/// non-`Send` JS types with `send_wrapper::SendWrapper` (workerd is
/// single-threaded so the wrapper never trips).
pub type HttpByteStream = BoxStream<'static, Result<Vec<u8>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

impl HttpMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            HttpMethod::Get => "GET",
            HttpMethod::Post => "POST",
            HttpMethod::Put => "PUT",
            HttpMethod::Patch => "PATCH",
            HttpMethod::Delete => "DELETE",
        }
    }
}

/// An HTTP request to be sent through an [`HttpClient`].
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: HttpHeaders,
    pub body: Vec<u8>,
}

impl HttpRequest {
    pub fn post(url: impl Into<String>) -> Self {
        Self {
            method: HttpMethod::Post,
            url: url.into(),
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn bearer_auth(self, token: impl AsRef<str>) -> Self {
        self.header("authorization", format!("Bearer {}", token.as_ref()))
    }

    pub fn json_body<T: serde::Serialize>(mut self, body: &T) -> Result<Self> {
        self.body = serde_json::to_vec(body).context("failed to serialise request body as JSON")?;
        self.headers
            .push(("content-type".to_string(), "application/json".to_string()));
        Ok(self)
    }
}

/// A buffered (non-streaming) HTTP response.
#[derive(Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

impl HttpResponse {
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    pub fn text(&self) -> Result<String> {
        String::from_utf8(self.body.clone()).context("response body is not valid UTF-8")
    }

    pub fn json<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_slice(&self.body).context("failed to decode response body as JSON")
    }
}

/// A streaming HTTP response — the body arrives as a sequence of byte chunks.
pub struct HttpStreamResponse {
    pub status: u16,
    pub body: HttpByteStream,
}

/// Type-erased handle to the active HTTP backend.
pub type SharedHttpClient = Arc<dyn HttpClient>;

#[async_trait]
pub trait HttpClient: Send + Sync {
    /// Send a request and return the full response, buffered.
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse>;

    /// Send a request and stream the response body. Used for SSE / chunked
    /// completions where we want to flush tokens to the caller as they arrive.
    async fn send_streaming(&self, request: HttpRequest) -> Result<HttpStreamResponse>;
}

/// Drain a body stream into a UTF-8 string (best-effort; non-UTF-8 bytes are
/// replaced with the standard replacement character). Used for surfacing
/// error response bodies when a streaming call comes back non-2xx.
pub async fn collect_stream_to_string(mut stream: HttpByteStream) -> String {
    use futures_util::StreamExt;
    let mut buf = Vec::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => buf.extend_from_slice(&bytes),
            Err(_) => break,
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

#[cfg(feature = "reqwest-http")]
pub use reqwest_impl::ReqwestHttpClient;

#[cfg(feature = "reqwest-http")]
mod reqwest_impl {
    use std::sync::Arc;

    use anyhow::{Context, Result, anyhow};
    use async_trait::async_trait;
    use futures_util::TryStreamExt;
    use reqwest::{Client, Method};

    use super::{HttpClient, HttpMethod, HttpRequest, HttpResponse, HttpStreamResponse};

    /// HTTP backend that uses `reqwest`. The default on native targets.
    #[derive(Debug, Clone, Default)]
    pub struct ReqwestHttpClient {
        client: Client,
    }

    impl ReqwestHttpClient {
        pub fn new(client: Client) -> Self {
            Self { client }
        }

        /// Reach the underlying `reqwest::Client` for callers that still want
        /// to drive it directly.
        pub fn inner(&self) -> &Client {
            &self.client
        }

        /// Wrap into the type-erased `Arc<dyn HttpClient>` the agent builder
        /// expects.
        pub fn into_shared(self) -> Arc<dyn HttpClient> {
            Arc::new(self)
        }
    }

    fn convert_method(method: HttpMethod) -> Method {
        match method {
            HttpMethod::Get => Method::GET,
            HttpMethod::Post => Method::POST,
            HttpMethod::Put => Method::PUT,
            HttpMethod::Patch => Method::PATCH,
            HttpMethod::Delete => Method::DELETE,
        }
    }

    fn build_request(client: &Client, request: HttpRequest) -> reqwest::RequestBuilder {
        let mut builder = client.request(convert_method(request.method), &request.url);
        for (name, value) in request.headers {
            builder = builder.header(name, value);
        }
        if !request.body.is_empty() {
            builder = builder.body(request.body);
        }
        builder
    }

    #[async_trait]
    impl HttpClient for ReqwestHttpClient {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse> {
            let response = build_request(&self.client, request)
                .send()
                .await
                .context("HTTP request failed")?;
            let status = response.status().as_u16();
            let body = response
                .bytes()
                .await
                .context("failed to read response body")?
                .to_vec();
            Ok(HttpResponse { status, body })
        }

        async fn send_streaming(&self, request: HttpRequest) -> Result<HttpStreamResponse> {
            let response = build_request(&self.client, request)
                .send()
                .await
                .context("HTTP streaming request failed")?;
            let status = response.status().as_u16();
            let stream = response
                .bytes_stream()
                .map_ok(|bytes| bytes.to_vec())
                .map_err(|err| anyhow!("stream read failed: {err}"));
            Ok(HttpStreamResponse {
                status,
                body: Box::pin(stream),
            })
        }
    }
}
