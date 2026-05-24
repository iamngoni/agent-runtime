mod agent;
mod anthropic;
mod event;
mod http;
mod message;
mod openai;
mod provider;
mod session;
mod streaming;
mod tool;

pub use agent::{Agent, AgentBuilder, AgentConfig};
pub use anthropic::{
    AnthropicClient, AnthropicParsedStreamEvent, parse_anthropic_stream_event,
};
pub use event::{EventSink, NullEventSink, RuntimeEvent};
pub use http::{
    HttpByteStream, HttpClient, HttpHeaders, HttpMethod, HttpRequest, HttpResponse,
    HttpStreamResponse, SharedHttpClient,
};
#[cfg(feature = "reqwest-http")]
pub use http::ReqwestHttpClient;
pub use message::{AssistantTurn, ChatMessage, MessageRole};
pub use openai::{OpenAiClient, ParsedStreamEvent, parse_openai_stream_event};
pub use provider::{AgentProvider, AgentProviderKind};
pub use session::{
    ExecutedToolCall, ToolSessionOutcome, ToolSessionRequest, build_followup_messages,
    execute_tool_calls, execute_tool_session,
};
pub use streaming::should_flush_delta;
pub use tool::{JsonTool, Tool, ToolCall, ToolDefinition, ToolOutput, ToolRegistry};
