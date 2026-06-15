use std::sync::Arc;

use anyhow::{Context, Result};

use crate::anthropic::AnthropicClientConfig;
use crate::bedrock::BedrockClientConfig;
use crate::cohere::CohereClientConfig;
use crate::error::RetryPolicy;
use crate::gemini::GeminiClientConfig;
use crate::http::{HttpClient, SharedHttpClient};
use crate::openai::OpenAiClientConfig;
use crate::provider::{ModelTier, ModelTiers, TextProvider};
use crate::{
    AgentProviderKind, AnthropicClient, AssistantTurn, BedrockClient, ChatMessage, CohereClient,
    EventSink, GeminiClient, OpenAiClient, ToolDefinition, ToolSessionOutcome, ToolSessionRequest,
    execute_tool_session,
};

#[derive(Debug, Clone, Default)]
pub struct AgentConfig {
    pub verbose: bool,
}

#[derive(Default)]
pub struct AgentBuilder {
    http_client: Option<SharedHttpClient>,
    provider: AgentProviderKind,
    api_key: Option<String>,
    base_url: Option<String>,
    region: Option<String>,
    extra_headers: Vec<(String, String)>,
    model_tiers: Option<ModelTiers>,
    max_tokens: Option<u32>,
    retry: RetryPolicy,
    config: AgentConfig,
}

#[derive(Clone)]
pub struct Agent {
    provider: Arc<dyn TextProvider>,
    config: AgentConfig,
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    pub fn provider_kind(&self) -> AgentProviderKind {
        self.provider.kind()
    }

    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    pub fn verbose(&self) -> bool {
        self.provider.verbose()
    }

    /// The provider's configured model tiers (default/cheapest/smartest).
    pub fn model_tiers(&self) -> &ModelTiers {
        self.provider.model_tiers()
    }

    /// Resolve a capability tier to a concrete model name for this provider.
    pub fn model_for(&self, tier: ModelTier) -> &str {
        self.provider.model_for(tier)
    }

    pub async fn request_assistant_turn(
        &self,
        model: &str,
        system_prompt: &str,
        history: &[ChatMessage],
        tool_definitions: &[ToolDefinition],
    ) -> Result<AssistantTurn> {
        self.provider
            .request_assistant_turn(model, system_prompt, history, tool_definitions)
            .await
    }

    pub async fn stream_message<E>(
        &self,
        model: &str,
        system_prompt: &str,
        messages: &[ChatMessage],
        sink: &mut E,
    ) -> Result<String>
    where
        E: EventSink,
    {
        self.provider
            .stream_message(model, system_prompt, messages, sink)
            .await
    }

    pub async fn execute_tool_session<C, E>(
        &self,
        request: ToolSessionRequest<'_, C>,
        sink: &mut E,
    ) -> Result<ToolSessionOutcome>
    where
        C: Clone + Send + Sync + 'static,
        E: EventSink,
    {
        execute_tool_session(self.provider.as_ref(), request, sink).await
    }
}

impl AgentBuilder {
    /// Plug in a custom HTTP backend (Cloudflare Worker, browser, mock…).
    /// The builder defaults to a `reqwest`-backed client when the
    /// `reqwest-http` feature is enabled.
    pub fn http_client(mut self, http_client: SharedHttpClient) -> Self {
        self.http_client = Some(http_client);
        self
    }

    /// Convenience for the common case of building an [`Arc`] from a concrete
    /// HTTP client. Equivalent to `.http_client(Arc::new(client))`.
    pub fn with_http_client<C>(mut self, http_client: C) -> Self
    where
        C: HttpClient + 'static,
    {
        self.http_client = Some(Arc::new(http_client));
        self
    }

    /// Compatibility shim: accept a raw `reqwest::Client` and wrap it in the
    /// default [`ReqwestHttpClient`](crate::ReqwestHttpClient) backend. Only
    /// available when the `reqwest-http` feature is on.
    #[cfg(feature = "reqwest-http")]
    pub fn reqwest_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(crate::ReqwestHttpClient::new(client).into_shared());
        self
    }

    pub fn provider(mut self, provider: AgentProviderKind) -> Self {
        self.provider = provider;
        self
    }

    /// Target any OpenAI-compatible endpoint by name + base URL (self-hosted
    /// gateways, vLLM, LiteLLM, a vendor we don't ship a preset for). Shorthand
    /// for `.provider(AgentProviderKind::Custom(name)).base_url(url)`.
    pub fn openai_compatible(
        mut self,
        name: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        self.provider = AgentProviderKind::Custom(name.into());
        self.base_url = Some(base_url.into());
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Override the API base URL. Required for
    /// [`AgentProviderKind::Custom`]; optional otherwise (defaults to the
    /// vendor's standard endpoint).
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// AWS region for the Bedrock provider (default `us-east-1`). Ignored by
    /// other providers. Overridden by an explicit `.base_url(...)`.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Add a header sent on every request (e.g. OpenRouter attribution
    /// headers). Only applied to OpenAI-compatible providers.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    /// Override the default model tiers for the selected provider.
    pub fn model_tiers(mut self, tiers: ModelTiers) -> Self {
        self.model_tiers = Some(tiers);
        self
    }

    /// Maximum output tokens (Anthropic and Gemini only; OpenAI-compatible
    /// providers ignore this and let the model decide).
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Configure automatic retry with backoff for retryable provider errors
    /// (rate limits, overloads, transport failures). Defaults to no retries.
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    pub fn build(self) -> Result<Agent> {
        let api_key = self.api_key.context(format!(
            "agent builder requires an API key for {}",
            self.provider.as_str()
        ))?;
        let http_client = match self.http_client {
            Some(client) => client,
            None => default_http_client()?,
        };
        let model_tiers = self
            .model_tiers
            .unwrap_or_else(|| self.provider.default_model_tiers());

        let provider: Arc<dyn TextProvider> = match &self.provider {
            AgentProviderKind::Anthropic => {
                let mut config = AnthropicClientConfig {
                    verbose: self.config.verbose,
                    model_tiers,
                    retry: self.retry,
                    ..AnthropicClientConfig::default()
                };
                if let Some(base_url) = self.base_url {
                    config.base_url = base_url;
                }
                if let Some(max_tokens) = self.max_tokens {
                    config.max_tokens = max_tokens;
                }
                Arc::new(AnthropicClient::with_config(http_client, api_key, config))
            }
            AgentProviderKind::Gemini => {
                let mut config = GeminiClientConfig {
                    verbose: self.config.verbose,
                    model_tiers,
                    retry: self.retry,
                    ..GeminiClientConfig::default()
                };
                if let Some(base_url) = self.base_url {
                    config.base_url = base_url;
                }
                if let Some(max_tokens) = self.max_tokens {
                    config.max_tokens = max_tokens;
                }
                Arc::new(GeminiClient::with_config(http_client, api_key, config))
            }
            AgentProviderKind::Cohere => {
                let mut config = CohereClientConfig {
                    verbose: self.config.verbose,
                    model_tiers,
                    retry: self.retry,
                    ..CohereClientConfig::default()
                };
                if let Some(base_url) = self.base_url {
                    config.base_url = base_url;
                }
                Arc::new(CohereClient::with_config(http_client, api_key, config))
            }
            AgentProviderKind::Bedrock => {
                let mut config = BedrockClientConfig {
                    verbose: self.config.verbose,
                    model_tiers,
                    retry: self.retry,
                    ..BedrockClientConfig::default()
                };
                if let Some(region) = self.region {
                    config.region = region;
                }
                // For Bedrock, .base_url(...) overrides the region-derived URL.
                if let Some(base_url) = self.base_url {
                    config.base_url = Some(base_url);
                }
                if let Some(max_tokens) = self.max_tokens {
                    config.max_tokens = max_tokens;
                }
                Arc::new(BedrockClient::with_config(http_client, api_key, config))
            }
            kind => {
                // OpenAI-compatible: OpenAI, Groq, DeepSeek, Xai, Mistral,
                // Ollama, OpenRouter, or a Custom endpoint.
                let base_url = self
                    .base_url
                    .or_else(|| kind.default_base_url().map(str::to_string))
                    .with_context(|| {
                        format!(
                            "provider '{}' requires an explicit base URL; call .base_url(...) \
                             or .openai_compatible(name, url)",
                            kind.as_str()
                        )
                    })?;
                let config = OpenAiClientConfig {
                    kind: kind.clone(),
                    base_url,
                    extra_headers: self.extra_headers,
                    model_tiers,
                    retry: self.retry,
                    verbose: self.config.verbose,
                };
                Arc::new(OpenAiClient::with_config(http_client, api_key, config))
            }
        };

        Ok(Agent {
            provider,
            config: self.config,
        })
    }
}

#[cfg(feature = "reqwest-http")]
fn default_http_client() -> Result<SharedHttpClient> {
    Ok(crate::ReqwestHttpClient::default().into_shared())
}

#[cfg(not(feature = "reqwest-http"))]
fn default_http_client() -> Result<SharedHttpClient> {
    anyhow::bail!(
        "no HTTP backend configured: enable the `reqwest-http` feature or call \
         `AgentBuilder::http_client(...)` with a custom impl"
    )
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::Agent;
    use crate::provider::ModelTier;
    use crate::AgentProviderKind;

    #[cfg(feature = "reqwest-http")]
    #[test]
    fn builder_defaults_to_non_verbose() -> Result<()> {
        let agent = Agent::builder().api_key("test-key").build()?;
        assert!(!agent.verbose());
        assert_eq!(agent.provider_kind(), AgentProviderKind::OpenAi);
        Ok(())
    }

    #[cfg(feature = "reqwest-http")]
    #[test]
    fn builder_can_enable_verbose_logging() -> Result<()> {
        let agent = Agent::builder().api_key("test-key").verbose(true).build()?;
        assert!(agent.verbose());
        Ok(())
    }

    #[cfg(feature = "reqwest-http")]
    #[test]
    fn builder_can_select_anthropic_provider() -> Result<()> {
        let agent = Agent::builder()
            .provider(AgentProviderKind::Anthropic)
            .api_key("test-key")
            .build()?;
        assert_eq!(agent.provider_kind(), AgentProviderKind::Anthropic);
        Ok(())
    }

    #[cfg(feature = "reqwest-http")]
    #[test]
    fn builder_selects_openai_compatible_vendor() -> Result<()> {
        let agent = Agent::builder()
            .provider(AgentProviderKind::Groq)
            .api_key("test-key")
            .build()?;
        assert_eq!(agent.provider_kind(), AgentProviderKind::Groq);
        // Groq's default tier list should be wired through.
        assert_eq!(agent.model_for(ModelTier::Cheapest), "llama-3.1-8b-instant");
        Ok(())
    }

    #[cfg(feature = "reqwest-http")]
    #[test]
    fn builder_supports_custom_openai_compatible_endpoint() -> Result<()> {
        let agent = Agent::builder()
            .openai_compatible("local-vllm", "http://localhost:8000/v1")
            .api_key("test-key")
            .model_tiers(crate::ModelTiers::new("my-model", "my-model", "my-model"))
            .build()?;
        assert_eq!(
            agent.provider_kind(),
            AgentProviderKind::Custom("local-vllm".to_string())
        );
        assert_eq!(agent.model_for(ModelTier::Default), "my-model");
        Ok(())
    }

    #[cfg(feature = "reqwest-http")]
    #[test]
    fn custom_provider_without_base_url_is_rejected() {
        let result = Agent::builder()
            .provider(AgentProviderKind::Custom("mystery".to_string()))
            .api_key("test-key")
            .build();
        assert!(result.is_err());
    }
}
