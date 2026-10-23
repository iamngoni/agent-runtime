//! Timeouts for provider calls, enforced for every [`HttpClient`] backend.
//!
//! The transport trait itself has no notion of time: a backend may hang on
//! connect, on the response headers, or between streamed chunks. [`Timeouts`]
//! describes the four limits callers care about, and [`TimeoutHttpClient`]
//! applies them around whatever backend is plugged in — including custom
//! `wasm` ones that cannot set a timeout themselves. The bundled reqwest
//! backend additionally maps its own connect/read timeouts onto the same typed
//! [`HttpTimeout`] error.
//!
//! All timers come from `futures_timer`, which works on native and
//! `wasm32-unknown-unknown`, so the wrapper stays runtime-agnostic.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures_timer::Delay;
use futures_util::StreamExt;
use futures_util::future::{Either, select};
use futures_util::stream;
use web_time::Instant;

use crate::http::{HttpClient, HttpRequest, HttpResponse, HttpStreamResponse, SharedHttpClient};

/// Which limit elapsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeoutStage {
    /// Opening the connection (reported by the reqwest backend).
    Connect,
    /// Waiting for the response status and headers.
    ResponseHeaders,
    /// Waiting for the next chunk of a streamed body.
    Idle,
    /// The optional overall deadline for the whole call.
    Total,
}

impl TimeoutStage {
    pub fn as_str(self) -> &'static str {
        match self {
            TimeoutStage::Connect => "connect",
            TimeoutStage::ResponseHeaders => "response headers",
            TimeoutStage::Idle => "idle",
            TimeoutStage::Total => "total",
        }
    }
}

/// A provider call exceeded one of its configured limits. Transports return
/// this inside `anyhow::Error`; providers convert it into
/// [`ProviderError::Timeout`](crate::ProviderError::Timeout).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpTimeout {
    pub stage: TimeoutStage,
    pub after: Duration,
}

impl fmt::Display for HttpTimeout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} timed out after {}s",
            self.stage.as_str(),
            self.after.as_secs_f64()
        )
    }
}

impl std::error::Error for HttpTimeout {}

/// Limits applied to every provider request.
///
/// Defaults are deliberately finite: a server that accepts a connection and
/// never answers must fail, not hang. `total` is opt-in for callers that want
/// a hard deadline across connect, headers, and all streamed chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeouts {
    pub connect: Option<Duration>,
    pub response_headers: Option<Duration>,
    pub idle: Option<Duration>,
    pub total: Option<Duration>,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            connect: Some(Duration::from_secs(15)),
            response_headers: Some(Duration::from_secs(60)),
            idle: Some(Duration::from_secs(120)),
            total: None,
        }
    }
}

impl Timeouts {
    /// No limits at all. Useful for tests that drive a mock transport.
    pub fn none() -> Self {
        Self {
            connect: None,
            response_headers: None,
            idle: None,
            total: None,
        }
    }

    pub fn connect(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.connect = timeout.into();
        self
    }

    pub fn response_headers(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.response_headers = timeout.into();
        self
    }

    pub fn idle(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.idle = timeout.into();
        self
    }

    pub fn total(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.total = timeout.into();
        self
    }

    /// The limit on receiving the response status + headers: the tighter of
    /// the header limit and the optional overall deadline. The stage is the
    /// one that actually elapsed, so a total deadline stays non-retryable
    /// even when it fires before the headers arrive.
    fn headers_limit(&self) -> Option<(TimeoutStage, Duration)> {
        [
            self.response_headers
                .map(|after| (TimeoutStage::ResponseHeaders, after)),
            self.total.map(|after| (TimeoutStage::Total, after)),
        ]
        .into_iter()
        .flatten()
        .min_by(|a, b| a.1.cmp(&b.1))
    }
}

/// Wraps an [`HttpClient`] and enforces [`Timeouts`] for it.
#[derive(Clone)]
pub struct TimeoutHttpClient {
    inner: SharedHttpClient,
    timeouts: Timeouts,
}

impl TimeoutHttpClient {
    pub fn new(inner: SharedHttpClient, timeouts: Timeouts) -> Self {
        Self { inner, timeouts }
    }

    pub fn into_shared(self) -> SharedHttpClient {
        Arc::new(self)
    }
}

/// Await `future`, failing with a typed [`HttpTimeout`] after `limit`.
async fn with_timeout<T, F>(future: F, limit: Option<(TimeoutStage, Duration)>) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    let Some((stage, after)) = limit else {
        return future.await;
    };
    if after.is_zero() {
        return Err(anyhow!(HttpTimeout { stage, after }));
    }
    let future = Box::pin(future);
    let timer = Box::pin(Delay::new(after));
    match select(future, timer).await {
        Either::Left((result, _)) => result,
        Either::Right(((), _)) => Err(anyhow!(HttpTimeout { stage, after })),
    }
}

/// Ends a body stream when it goes idle for too long or passes its overall
/// deadline. The idle timer resets after every chunk; the total deadline does
/// not.
fn timeout_stream(
    inner: crate::http::HttpByteStream,
    idle: Option<Duration>,
    total: Option<(TimeoutStage, Duration)>,
    started_at: Instant,
) -> crate::http::HttpByteStream {
    struct State {
        inner: crate::http::HttpByteStream,
        idle: Option<Duration>,
        total: Option<(TimeoutStage, Duration)>,
        started_at: Instant,
    }

    Box::pin(stream::unfold(
        State {
            inner,
            idle,
            total,
            started_at,
        },
        |mut state| async move {
            let total_left = match state.total {
                Some((stage, after)) => {
                    let elapsed = state.started_at.elapsed();
                    if elapsed >= after {
                        return Some((Err(anyhow!(HttpTimeout { stage, after })), state));
                    }
                    Some((stage, after - elapsed))
                }
                None => None,
            };
            let idle = state.idle.map(|after| (TimeoutStage::Idle, after));
            let limit = match (idle, total_left) {
                (Some(idle), Some(total)) => Some(if idle.1 <= total.1 { idle } else { total }),
                (Some(idle), None) => Some(idle),
                (None, Some(total)) => Some(total),
                (None, None) => None,
            };

            match limit {
                None => state.inner.next().await.map(|chunk| (chunk, state)),
                Some((stage, after)) => {
                    let next = Box::pin(state.inner.next());
                    let timer = Box::pin(Delay::new(after));
                    match select(next, timer).await {
                        Either::Left((Some(chunk), _)) => Some((chunk, state)),
                        Either::Left((None, _)) => None,
                        Either::Right(((), _)) => {
                            Some((Err(anyhow!(HttpTimeout { stage, after })), state))
                        }
                    }
                }
            }
        },
    ))
}

#[async_trait]
impl HttpClient for TimeoutHttpClient {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse> {
        // A buffered call has no chunks to reset an idle timer on; the header
        // limit bounds the whole request, and the total deadline can only
        // shorten it.
        with_timeout(self.inner.send(request), self.timeouts.headers_limit()).await
    }

    async fn send_streaming(&self, request: HttpRequest) -> Result<HttpStreamResponse> {
        let started_at = Instant::now();
        let response = with_timeout(
            self.inner.send_streaming(request),
            self.timeouts.headers_limit(),
        )
        .await?;
        let total = self
            .timeouts
            .total
            .map(|after| (TimeoutStage::Total, after));
        Ok(HttpStreamResponse {
            status: response.status,
            body: timeout_stream(response.body, self.timeouts.idle, total, started_at),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::{Result, anyhow};
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use futures_util::TryStreamExt;
    use futures_util::stream;

    use super::{HttpTimeout, TimeoutHttpClient, Timeouts};
    use crate::http::{HttpClient, HttpRequest, HttpResponse, HttpStreamResponse};

    /// Never answers.
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

    /// Answers headers, then sends one chunk and hangs.
    struct StallsAfterFirstChunk;

    #[async_trait]
    impl HttpClient for StallsAfterFirstChunk {
        async fn send(&self, _request: HttpRequest) -> Result<HttpResponse> {
            unreachable!("streaming only")
        }

        async fn send_streaming(&self, _request: HttpRequest) -> Result<HttpStreamResponse> {
            let stream = stream::iter(vec![Ok(b"one".to_vec())]).chain(stream::pending());
            Ok(HttpStreamResponse {
                status: 200,
                body: Box::pin(stream),
            })
        }
    }

    fn timeout_of(error: &anyhow::Error) -> &HttpTimeout {
        error
            .chain()
            .find_map(|error| error.downcast_ref::<HttpTimeout>())
            .expect("a typed timeout")
    }

    #[tokio::test]
    async fn a_silent_server_fails_with_a_typed_header_timeout() {
        let client = TimeoutHttpClient::new(
            Arc::new(Silent),
            Timeouts::none().response_headers(Some(Duration::from_millis(30))),
        );
        let error = client
            .send(HttpRequest::post("http://example.test"))
            .await
            .expect_err("a silent server must time out");
        let timeout = timeout_of(&error);
        assert_eq!(timeout.stage, super::TimeoutStage::ResponseHeaders);
    }

    #[tokio::test]
    async fn a_stream_that_goes_quiet_fails_with_a_typed_idle_timeout() {
        let client = TimeoutHttpClient::new(
            Arc::new(StallsAfterFirstChunk),
            Timeouts::none().idle(Some(Duration::from_millis(30))),
        );
        let response = client
            .send_streaming(HttpRequest::post("http://example.test"))
            .await
            .expect("headers arrive");
        let mut body = response.body;
        assert_eq!(body.next().await.unwrap().unwrap(), b"one");
        let error = body
            .next()
            .await
            .expect("a chunk")
            .expect_err("an idle stream must time out");
        assert_eq!(timeout_of(&error).stage, super::TimeoutStage::Idle);
    }

    #[tokio::test]
    async fn a_total_deadline_bounds_the_whole_stream() {
        let client = TimeoutHttpClient::new(
            Arc::new(StallsAfterFirstChunk),
            Timeouts::none().total(Some(Duration::from_millis(30))),
        );
        let response = client
            .send_streaming(HttpRequest::post("http://example.test"))
            .await
            .expect("headers arrive");
        let mut body = response.body;
        assert_eq!(body.next().await.unwrap().unwrap(), b"one");
        let error = body
            .next()
            .await
            .expect("a chunk")
            .expect_err("a total deadline must fire");
        assert_eq!(timeout_of(&error).stage, super::TimeoutStage::Total);
    }

    #[tokio::test]
    async fn a_total_deadline_alone_keeps_its_total_stage() {
        let client = TimeoutHttpClient::new(
            Arc::new(Silent),
            Timeouts::none().total(Some(Duration::from_millis(30))),
        );
        let error = client
            .send(HttpRequest::post("http://example.test"))
            .await
            .expect_err("a total deadline must fire");
        assert_eq!(timeout_of(&error).stage, super::TimeoutStage::Total);
    }

    #[tokio::test]
    async fn no_limits_leave_a_mock_alone() {
        let client = TimeoutHttpClient::new(Arc::new(Silent), Timeouts::none());
        let result = tokio::time::timeout(
            Duration::from_millis(30),
            client.send(HttpRequest::post("http://example.test")),
        )
        .await;
        assert!(
            result.is_err(),
            "no timeout configured, so it stays pending"
        );
    }

    #[tokio::test]
    async fn chunks_keep_resetting_the_idle_timer() {
        struct Steady;
        #[async_trait]
        impl HttpClient for Steady {
            async fn send(&self, _request: HttpRequest) -> Result<HttpResponse> {
                Err(anyhow!("unused"))
            }
            async fn send_streaming(&self, _request: HttpRequest) -> Result<HttpStreamResponse> {
                let chunks = stream::unfold(0u8, |n| async move {
                    if n == 4 {
                        return None;
                    }
                    futures_timer::Delay::new(Duration::from_millis(10)).await;
                    Some((Ok(vec![n]), n + 1))
                });
                Ok(HttpStreamResponse {
                    status: 200,
                    body: Box::pin(chunks),
                })
            }
        }

        let client = TimeoutHttpClient::new(
            Arc::new(Steady),
            Timeouts::none().idle(Some(Duration::from_millis(40))),
        );
        let response = client
            .send_streaming(HttpRequest::post("http://example.test"))
            .await
            .unwrap();
        let body: Vec<Vec<u8>> = response.body.try_collect().await.unwrap();
        assert_eq!(body.len(), 4);
    }
}
