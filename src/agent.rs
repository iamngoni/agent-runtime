//! Declarative agents.
//!
//! An [`Agent`] is the thing with identity: instructions, a model, and tools —
//! defined *outside* the run logic and bound to an [`Llm`](crate::Llm) at call
//! time via [`Llm::run`](crate::Llm::run). This mirrors Laravel AI's
//! `class CustomerSupportAgent implements Agent, HasTools`: you describe the
//! agent declaratively and let the runtime drive the tool session.
//!
//! ```ignore
//! struct CustomerSupportAgent;
//!
//! impl Agent for CustomerSupportAgent {
//!     fn instructions(&self) -> String {
//!         "You help customers with account, order, and billing questions.".into()
//!     }
//! }
//!
//! let reply = llm.run(&CustomerSupportAgent, "I was charged twice").await?;
//! ```

use crate::ToolRegistry;

/// A declarative agent: instructions plus, optionally, a model, tools, and a
/// tool-call budget. Only [`instructions`](Agent::instructions) is required;
/// everything else has a sensible default, so simple agents are a few lines and
/// richer ones override just what they need.
///
/// Tools use the unit context (`ToolRegistry<()>`): each tool owns its own
/// state (captured by value), which keeps agent definitions self-contained.
/// For a shared, typed tool context, drop to the low-level
/// [`Llm::execute_tool_session`](crate::Llm::execute_tool_session) API.
pub trait Agent: Send + Sync {
    /// System instructions describing the agent's role and behaviour.
    fn instructions(&self) -> String;

    /// Which model to use. Empty (the default) means "use the bound
    /// [`Llm`](crate::Llm)'s default tier", so an agent need not hardcode one.
    fn model(&self) -> String {
        String::new()
    }

    /// Tools the agent can call. Defaults to none.
    fn tools(&self) -> ToolRegistry<()> {
        ToolRegistry::new()
    }

    /// Maximum tool calls dispatched in a single turn.
    fn max_tool_calls(&self) -> usize {
        8
    }
}
