//! Agents-as-tools: let one agent delegate to another.
//!
//! Mirrors Laravel AI's pattern where an agent's `tools()` can include
//! `new RefundsAgent` — a specialised sub-agent exposed to the parent as a
//! callable tool. [`AgentTool`] wraps an [`Agent`] so it satisfies the
//! [`Tool`] contract: the parent model calls it with a self-contained `task`
//! string, the sub-agent answers in isolation (no access to the parent's
//! conversation), and its reply is returned as the tool result.

use std::marker::PhantomData;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::json;

use crate::{Agent, ChatMessage, NullEventSink, Tool, ToolCall, ToolDefinition, ToolOutput};

/// A sub-agent exposed to a parent agent as a tool.
///
/// The sub-agent runs in isolation: it receives only the `task` the parent
/// passes, runs against its own model + instructions, and returns its text
/// answer. Register it in the parent's [`ToolRegistry`](crate::ToolRegistry)
/// like any other tool.
pub struct AgentTool<C> {
    definition: ToolDefinition,
    agent: Agent,
    model: String,
    instructions: String,
    _marker: PhantomData<fn() -> C>,
}

impl<C> AgentTool<C> {
    /// Wrap `agent` as a delegatable tool. `name`/`description` are what the
    /// parent model sees; `model`/`instructions` drive the sub-agent.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: Agent,
        model: impl Into<String>,
        instructions: impl Into<String>,
    ) -> Self {
        let definition = ToolDefinition {
            name: name.into(),
            description: description.into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "A clear, self-contained task for the sub-agent. \
                                        It has no access to the parent conversation."
                    }
                },
                "required": ["task"],
            }),
        };
        Self {
            definition,
            agent,
            model: model.into(),
            instructions: instructions.into(),
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<C> Tool<C> for AgentTool<C>
where
    C: Send + Sync + 'static,
{
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _ctx: C, call: &ToolCall) -> Result<ToolOutput> {
        let task = call
            .arguments
            .get("task")
            .and_then(|value| value.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| call.arguments.to_string());

        let mut sink = NullEventSink;
        let response = self
            .agent
            .stream_message(
                &self.model,
                &self.instructions,
                &[ChatMessage::user(task)],
                &mut sink,
            )
            .await
            .with_context(|| {
                format!("sub-agent delegation via tool '{}' failed", self.definition.name)
            })?;

        Ok(ToolOutput {
            content: json!({ "response": response }),
        })
    }
}
