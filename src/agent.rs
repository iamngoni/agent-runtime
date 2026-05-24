use std::sync::Arc;

use anyhow::{Context, Result};

use crate::http::{HttpClient, SharedHttpClient};
use crate::{
    AgentProvider, AgentProviderKind, AnthropicClient, AssistantTurn, ChatMessage, EventSink,
    OpenAiClient, ToolDefinition, ToolSessionOutcome, ToolSessionRequest, execute_tool_session,
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
    config: AgentConfig,
}

#[derive(Clone)]
pub struct Agent {
    provider: Arc<dyn AgentProvider>,
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

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
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
        let provider: Arc<dyn AgentProvider> = match self.provider {
            AgentProviderKind::OpenAi => Arc::new(OpenAiClient::with_verbose(
                http_client,
                api_key,
                self.config.verbose,
            )),
            AgentProviderKind::Anthropic => Arc::new(AnthropicClient::with_verbose(
                http_client,
                api_key,
                self.config.verbose,
            )),
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
}
