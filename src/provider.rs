use std::str::FromStr;

use anyhow::{Result, anyhow};
use async_trait::async_trait;

use crate::{AssistantTurn, ChatMessage, EventSink, ToolDefinition};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentProviderKind {
    OpenAi,
    Anthropic,
}

impl AgentProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

impl Default for AgentProviderKind {
    fn default() -> Self {
        Self::OpenAi
    }
}

impl FromStr for AgentProviderKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "openai" => Ok(Self::OpenAi),
            "anthropic" => Ok(Self::Anthropic),
            other => Err(anyhow!("unsupported agent provider '{other}'")),
        }
    }
}

#[async_trait]
pub trait AgentProvider: Send + Sync {
    fn kind(&self) -> AgentProviderKind;

    fn verbose(&self) -> bool;

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
}
