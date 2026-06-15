mod agent;
mod anthropic;
mod bedrock;
mod cohere;
mod delegate;
mod error;
mod event;
mod gemini;
mod http;
mod message;
mod openai;
mod provider;
mod session;
mod store;
mod streaming;
mod tool;

pub use agent::{Agent, AgentBuilder, AgentConfig};
pub use anthropic::{
    AnthropicClient, AnthropicClientConfig, AnthropicParsedStreamEvent,
    parse_anthropic_stream_event,
};
pub use bedrock::{BedrockClient, BedrockClientConfig};
pub use cohere::{CohereClient, CohereClientConfig, parse_cohere_stream_event};
pub use delegate::AgentTool;
pub use error::{ProviderError, RetryPolicy, execute_with_retry};
pub use event::{EventSink, NullEventSink, RuntimeEvent};
pub use gemini::{GeminiClient, GeminiClientConfig, parse_gemini_stream_event};
pub use http::{
    HttpByteStream, HttpClient, HttpHeaders, HttpMethod, HttpRequest, HttpResponse,
    HttpStreamResponse, SharedHttpClient,
};
#[cfg(feature = "reqwest-http")]
pub use http::ReqwestHttpClient;
pub use message::{
    Attachment, AttachmentKind, AttachmentSource, AssistantTurn, ChatMessage, MessageRole,
};
pub use openai::{OpenAiClient, OpenAiClientConfig, ParsedStreamEvent, parse_openai_stream_event};
pub use provider::{
    AgentProviderKind, EmbeddingProvider, ModelTier, ModelTiers, ProviderInfo, TextProvider,
};
/// Backward-compatible alias for the text-generation capability trait, which
/// was previously the monolithic provider trait. Prefer [`TextProvider`].
pub use provider::TextProvider as AgentProvider;
pub use session::{
    ExecutedToolCall, ToolSessionOutcome, ToolSessionRequest, build_followup_messages,
    execute_tool_calls, execute_tool_session,
};
pub use store::{ConversationStore, InMemoryConversationStore};
pub use streaming::should_flush_delta;
pub use tool::{JsonTool, Tool, ToolCall, ToolDefinition, ToolOutput, ToolRegistry};
