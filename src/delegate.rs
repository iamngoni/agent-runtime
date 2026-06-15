//! Agents-as-tools: let one agent delegate to another.
//!
//! Mirrors Laravel AI's pattern where an agent's `tools()` can include
//! `new RefundsAgent` — a specialised sub-agent exposed to the parent as a
//! callable tool. [`AgentTool`] wraps a declarative [`Agent`] together with the
//! [`Llm`] that runs it: the parent model calls it with a self-contained
//! `task`, the sub-agent runs its **own full tool session** in isolation (no
//! access to the parent conversation), and its reply is returned as the tool
//! result.

use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::json;

use crate::{Agent, Llm, Tool, ToolCall, ToolDefinition, ToolOutput};

/// A sub-agent exposed to a parent agent as a tool.
///
/// Register it in the parent's [`ToolRegistry`](crate::ToolRegistry) like any
/// other tool. When the parent calls it, the wrapped agent runs via
/// [`Llm::run`], so the sub-agent's own tools execute too.
pub struct AgentTool<C> {
    definition: ToolDefinition,
    llm: Llm,
    agent: Arc<dyn Agent>,
    _marker: PhantomData<fn() -> C>,
}

impl<C> AgentTool<C> {
    /// Wrap `agent` as a delegatable tool. `name`/`description` are what the
    /// parent model sees; `llm` runs the sub-agent.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        llm: Llm,
        agent: impl Agent + 'static,
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
            llm,
            agent: Arc::new(agent),
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

        let response = self
            .llm
            .run(self.agent.as_ref(), task)
            .await
            .with_context(|| {
                format!("sub-agent delegation via tool '{}' failed", self.definition.name)
            })?;

        Ok(ToolOutput {
            content: json!({ "response": response }),
        })
    }
}
