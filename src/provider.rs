use std::str::FromStr;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::{AssistantTurn, ChatMessage, EventSink, ResponseFormat, ToolDefinition};

/// Identifies which upstream API a provider speaks to.
///
/// Named variants cover the vendors we ship defaults for; [`Custom`] is the
/// escape hatch for any other OpenAI-compatible endpoint (self-hosted gateways,
/// vLLM, LiteLLM, a new vendor we haven't enumerated yet). Most of these speak
/// the OpenAI `/chat/completions` wire format and differ only by base URL —
/// see [`AgentProviderKind::default_base_url`].
///
/// [`Custom`]: AgentProviderKind::Custom
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AgentProviderKind {
    #[default]
    OpenAi,
    Anthropic,
    Gemini,
    Cohere,
    Bedrock,
    Groq,
    DeepSeek,
    Xai,
    Mistral,
    Ollama,
    OpenRouter,
    /// Any other OpenAI-compatible endpoint, identified by a free-form name.
    /// Requires an explicit base URL on the builder.
    Custom(String),
}

impl AgentProviderKind {
    /// Short, stable identifier for logging and `kind()`.
    pub fn as_str(&self) -> &str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
            Self::Cohere => "cohere",
            Self::Bedrock => "bedrock",
            Self::Groq => "groq",
            Self::DeepSeek => "deepseek",
            Self::Xai => "xai",
            Self::Mistral => "mistral",
            Self::Ollama => "ollama",
            Self::OpenRouter => "openrouter",
            Self::Custom(name) => name,
        }
    }

    /// Whether this provider is driven by the OpenAI `/chat/completions`
    /// client. Anthropic and Gemini have native clients; everything else
    /// (including [`Custom`](AgentProviderKind::Custom)) is OpenAI-compatible.
    pub fn is_openai_compatible(&self) -> bool {
        !matches!(
            self,
            Self::Anthropic | Self::Gemini | Self::Cohere | Self::Bedrock
        )
    }

    /// The default API base URL for vendors we ship presets for. `Custom`
    /// returns `None` — the caller must supply a base URL.
    pub fn default_base_url(&self) -> Option<&'static str> {
        match self {
            Self::OpenAi => Some("https://api.openai.com/v1"),
            Self::Groq => Some("https://api.groq.com/openai/v1"),
            Self::DeepSeek => Some("https://api.deepseek.com/v1"),
            Self::Xai => Some("https://api.x.ai/v1"),
            Self::Mistral => Some("https://api.mistral.ai/v1"),
            Self::Ollama => Some("http://localhost:11434/v1"),
            Self::OpenRouter => Some("https://openrouter.ai/api/v1"),
            Self::Anthropic => Some("https://api.anthropic.com/v1"),
            Self::Gemini => Some("https://generativelanguage.googleapis.com/v1beta"),
            Self::Cohere => Some("https://api.cohere.com/v2"),
            // Bedrock's endpoint is region-derived; the client builds it from
            // the configured region rather than a fixed default.
            Self::Bedrock => None,
            Self::Custom(_) => None,
        }
    }

    /// Sensible default model tiers per vendor. All are overridable on the
    /// builder; they exist so callers can ask for "the smart one" without
    /// hardcoding model strings (mirrors Laravel's default/cheapest/smartest).
    pub fn default_model_tiers(&self) -> ModelTiers {
        match self {
            Self::OpenAi => ModelTiers::new("gpt-4o", "gpt-4o-mini", "gpt-4o"),
            Self::Anthropic => ModelTiers::new(
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
                "claude-3-5-sonnet-latest",
            ),
            Self::Gemini => ModelTiers::new(
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
                "gemini-1.5-pro",
            ),
            Self::Cohere => ModelTiers::new("command-r-plus", "command-r", "command-r-plus"),
            Self::Bedrock => ModelTiers::new(
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "anthropic.claude-3-5-haiku-20241022-v1:0",
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
            ),
            Self::Groq => ModelTiers::new(
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
            ),
            Self::DeepSeek => {
                ModelTiers::new("deepseek-chat", "deepseek-chat", "deepseek-reasoner")
            }
            Self::Xai => ModelTiers::new("grok-2-latest", "grok-2-latest", "grok-2-latest"),
            Self::Mistral => ModelTiers::new(
                "mistral-large-latest",
                "mistral-small-latest",
                "mistral-large-latest",
            ),
            Self::Ollama => ModelTiers::new("llama3.2", "llama3.2", "llama3.2"),
            Self::OpenRouter => ModelTiers::new(
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
            ),
            Self::Custom(_) => ModelTiers::new("", "", ""),
        }
    }
}

impl FromStr for AgentProviderKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        Ok(match value.trim().to_ascii_lowercase().as_str() {
            "openai" => Self::OpenAi,
            "anthropic" | "claude" => Self::Anthropic,
            "gemini" | "google" => Self::Gemini,
            "groq" => Self::Groq,
            "deepseek" => Self::DeepSeek,
            "xai" | "grok" => Self::Xai,
            "mistral" => Self::Mistral,
            "ollama" => Self::Ollama,
            "openrouter" => Self::OpenRouter,
            other => Self::Custom(other.to_string()),
        })
    }
}

/// Named model tiers for a provider. Lets callers request a capability level
/// ("cheapest", "smartest") without pinning a model string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelTiers {
    pub default: String,
    pub cheapest: String,
    pub smartest: String,
}

impl ModelTiers {
    pub fn new(
        default: impl Into<String>,
        cheapest: impl Into<String>,
        smartest: impl Into<String>,
    ) -> Self {
        Self {
            default: default.into(),
            cheapest: cheapest.into(),
            smartest: smartest.into(),
        }
    }

    /// Resolve a [`ModelTier`] to its configured model name.
    pub fn resolve(&self, tier: ModelTier) -> &str {
        match tier {
            ModelTier::Default => &self.default,
            ModelTier::Cheapest => &self.cheapest,
            ModelTier::Smartest => &self.smartest,
        }
    }
}

/// A capability level rather than a concrete model name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    Default,
    Cheapest,
    Smartest,
}

/// Metadata shared by every provider, regardless of capability. Split out from
/// the capability traits so a provider declares its identity/model tiers once
/// and then implements only the capabilities it actually supports.
pub trait ProviderInfo: Send + Sync {
    fn kind(&self) -> AgentProviderKind;

    fn verbose(&self) -> bool;

    fn model_tiers(&self) -> &ModelTiers;

    /// Resolve a capability tier to a model name for this provider.
    fn model_for(&self, tier: ModelTier) -> &str {
        self.model_tiers().resolve(tier)
    }
}

/// The text-generation capability: request an assistant turn (with optional
/// tool calls) and stream a follow-up message. This is the core capability the
/// agent runtime drives today.
///
/// Previously this was the monolithic `AgentProvider` trait; it is now one of
/// several capability traits (see [`EmbeddingProvider`]). `AgentProvider`
/// remains available as a re-exported alias for backward compatibility.
#[async_trait]
pub trait TextProvider: ProviderInfo {
    async fn request_assistant_turn(
        &self,
        model: &str,
        system_prompt: &str,
        history: &[ChatMessage],
        tool_definitions: &[ToolDefinition],
    ) -> Result<AssistantTurn>;

    async fn stream_message(
        &self,
        model: &str,
        system_prompt: &str,
        messages: &[ChatMessage],
        sink: &mut dyn EventSink,
    ) -> Result<String>;

    /// Request a response constrained to `format`'s JSON Schema, returning the
    /// structured value. Backends that support forced/structured output
    /// (Anthropic, OpenAI-compatible) override this; the default reports the
    /// capability as unavailable rather than silently degrading to free text.
    async fn request_structured(
        &self,
        _model: &str,
        _system_prompt: &str,
        _messages: &[ChatMessage],
        _format: &ResponseFormat,
    ) -> Result<Value> {
        anyhow::bail!(
            "structured output is not supported by the '{}' provider",
            self.kind().as_str()
        )
    }
}

/// The embeddings capability: turn text inputs into vectors. Implemented by the
/// OpenAI-compatible providers (OpenAI, Mistral, Ollama, …) via the
/// `/embeddings` endpoint. Providers without embeddings simply don't implement
/// this trait, so misuse is a compile error rather than a runtime 404.
#[async_trait]
pub trait EmbeddingProvider: ProviderInfo {
    /// Embed a batch of inputs, returning one vector per input in order.
    async fn embed(&self, model: &str, inputs: &[String]) -> Result<Vec<Vec<f32>>>;
}
