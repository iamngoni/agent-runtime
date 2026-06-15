//! Typed provider errors and retry/backoff.
//!
//! Adapted from Laravel AI's `HandlesFailoverErrors`: instead of collapsing
//! every non-2xx into one opaque string, providers classify HTTP failures into
//! semantic variants so callers can distinguish "retry me" (rate limited /
//! overloaded) from "fatal" (bad request / insufficient credits). That
//! classification is also what makes automatic retry — and, later, multi-provider
//! failover — possible.

use std::fmt;
use std::time::Duration;

use futures_util::future::Future;

/// A classified failure from an LLM provider call.
#[derive(Debug)]
pub enum ProviderError {
    /// HTTP 429 — the provider is rate limiting us. Retryable.
    RateLimited {
        provider: String,
        retry_after: Option<Duration>,
        body: String,
    },
    /// HTTP 402 (or a credit/quota message) — out of credits. Not retryable.
    InsufficientCredits { provider: String, body: String },
    /// HTTP 503/529 — the provider is temporarily overloaded. Retryable.
    Overloaded { provider: String, body: String },
    /// Any other non-2xx status. Not retryable by default.
    Status {
        provider: String,
        status: u16,
        body: String,
    },
    /// The HTTP layer itself failed (connection reset, DNS, TLS…). Retryable.
    Transport {
        provider: String,
        source: anyhow::Error,
    },
    /// The response decoded successfully but carried no usable content.
    EmptyResponse { provider: String },
}

impl ProviderError {
    /// Classify a non-2xx response into a semantic variant, mirroring Laravel's
    /// status → exception mapping (429 → rate limited, 402 → credits,
    /// 503/529 → overloaded).
    pub fn from_status(provider: impl Into<String>, status: u16, body: impl Into<String>) -> Self {
        let provider = provider.into();
        let body = body.into();
        match status {
            429 => ProviderError::RateLimited {
                provider,
                retry_after: parse_retry_after(&body),
                body,
            },
            402 => ProviderError::InsufficientCredits { provider, body },
            503 | 529 => ProviderError::Overloaded { provider, body },
            _ if body_signals_insufficient_credits(&body) => {
                ProviderError::InsufficientCredits { provider, body }
            }
            _ => ProviderError::Status {
                provider,
                status,
                body,
            },
        }
    }

    /// Wrap a transport-layer failure.
    pub fn transport(provider: impl Into<String>, source: anyhow::Error) -> Self {
        ProviderError::Transport {
            provider: provider.into(),
            source,
        }
    }

    /// Whether retrying the same request could plausibly succeed.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ProviderError::RateLimited { .. }
                | ProviderError::Overloaded { .. }
                | ProviderError::Transport { .. }
        )
    }

    /// The provider name this error originated from.
    pub fn provider(&self) -> &str {
        match self {
            ProviderError::RateLimited { provider, .. }
            | ProviderError::InsufficientCredits { provider, .. }
            | ProviderError::Overloaded { provider, .. }
            | ProviderError::Status { provider, .. }
            | ProviderError::Transport { provider, .. }
            | ProviderError::EmptyResponse { provider } => provider,
        }
    }

    /// The server-suggested retry delay, if the provider sent one (429 only).
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            ProviderError::RateLimited { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::RateLimited { provider, body, .. } => {
                write!(f, "{provider} rate limited (429): {body}")
            }
            ProviderError::InsufficientCredits { provider, body } => {
                write!(f, "{provider} insufficient credits (402): {body}")
            }
            ProviderError::Overloaded { provider, body } => {
                write!(f, "{provider} overloaded: {body}")
            }
            ProviderError::Status {
                provider,
                status,
                body,
            } => write!(f, "{provider} call returned {status}: {body}"),
            ProviderError::Transport { provider, source } => {
                write!(f, "{provider} transport error: {source}")
            }
            ProviderError::EmptyResponse { provider } => {
                write!(f, "{provider} returned an empty response")
            }
        }
    }
}

impl std::error::Error for ProviderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProviderError::Transport { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

/// Controls automatic retry with exponential backoff for retryable
/// [`ProviderError`]s. The default performs **no** retries, so opting in is
/// explicit and timer-free builds stay timer-free.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    /// Honour a server-provided `Retry-After` over the computed backoff.
    pub respect_retry_after: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 0,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(20),
            respect_retry_after: true,
        }
    }
}

impl RetryPolicy {
    /// A sensible retrying policy: `max_retries` attempts with 500ms base,
    /// doubling each time up to 20s.
    pub fn with_retries(max_retries: u32) -> Self {
        Self {
            max_retries,
            ..Self::default()
        }
    }

    /// Exponential backoff for a given (zero-based) attempt index, capped at
    /// `max_delay`.
    pub fn backoff_for(&self, attempt: u32) -> Duration {
        let factor = 2u32.saturating_pow(attempt);
        let delay = self.base_delay.saturating_mul(factor);
        delay.min(self.max_delay)
    }
}

/// Run `op`, retrying retryable [`ProviderError`]s per `policy`. The operation
/// is a closure so each attempt builds a fresh request future.
pub async fn execute_with_retry<F, Fut, T>(
    policy: &RetryPolicy,
    mut op: F,
) -> Result<T, ProviderError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, ProviderError>>,
{
    let mut attempt = 0u32;
    loop {
        match op().await {
            Ok(value) => return Ok(value),
            Err(error) if error.is_retryable() && attempt < policy.max_retries => {
                let delay = policy
                    .respect_retry_after
                    .then(|| error.retry_after())
                    .flatten()
                    .map(|d| d.min(policy.max_delay))
                    .unwrap_or_else(|| policy.backoff_for(attempt));
                tracing::warn!(
                    provider = %error.provider(),
                    attempt = attempt + 1,
                    max_retries = policy.max_retries,
                    delay_ms = delay.as_millis() as u64,
                    error = %error,
                    "retrying provider call after retryable error"
                );
                futures_timer::Delay::new(delay).await;
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

fn parse_retry_after(body: &str) -> Option<Duration> {
    // Some providers echo a retry hint in the JSON error body; the HTTP header
    // is handled separately by callers that have header access. Best-effort.
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let secs = value
        .get("error")
        .and_then(|e| e.get("retry_after"))
        .or_else(|| value.get("retry_after"))
        .and_then(|v| v.as_f64())?;
    (secs.is_finite() && secs >= 0.0).then(|| Duration::from_secs_f64(secs))
}

fn body_signals_insufficient_credits(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    ["insufficient", "quota", "exceeded your current", "billing"]
        .iter()
        .any(|needle| lower.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_status_codes() {
        assert!(matches!(
            ProviderError::from_status("groq", 429, "slow down"),
            ProviderError::RateLimited { .. }
        ));
        assert!(matches!(
            ProviderError::from_status("openai", 402, "no funds"),
            ProviderError::InsufficientCredits { .. }
        ));
        assert!(matches!(
            ProviderError::from_status("anthropic", 529, "overloaded"),
            ProviderError::Overloaded { .. }
        ));
        assert!(matches!(
            ProviderError::from_status("openai", 400, "bad request"),
            ProviderError::Status { status: 400, .. }
        ));
    }

    #[test]
    fn detects_credit_messages_on_generic_status() {
        let error = ProviderError::from_status("xai", 403, "You exceeded your current quota");
        assert!(matches!(error, ProviderError::InsufficientCredits { .. }));
    }

    #[test]
    fn retryability_matches_intent() {
        assert!(ProviderError::from_status("g", 429, "").is_retryable());
        assert!(ProviderError::from_status("g", 503, "").is_retryable());
        assert!(!ProviderError::from_status("g", 400, "").is_retryable());
        assert!(!ProviderError::from_status("g", 402, "").is_retryable());
    }

    #[test]
    fn backoff_is_exponential_and_capped() {
        let policy = RetryPolicy {
            max_retries: 5,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(500),
            respect_retry_after: true,
        };
        assert_eq!(policy.backoff_for(0), Duration::from_millis(100));
        assert_eq!(policy.backoff_for(1), Duration::from_millis(200));
        assert_eq!(policy.backoff_for(2), Duration::from_millis(400));
        assert_eq!(policy.backoff_for(3), Duration::from_millis(500)); // capped
    }
}
