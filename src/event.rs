use anyhow::Result;
use async_trait::async_trait;

use crate::{ToolCall, ToolOutput};

#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    AssistantStarted { model: String },
    AssistantDelta { delta: String },
    ToolStarted { call: ToolCall },
    ToolCompleted { call: ToolCall, output: ToolOutput },
}

#[async_trait]
pub trait EventSink: Send {
    async fn emit(&mut self, event: RuntimeEvent) -> Result<()>;
}

pub struct NullEventSink;

#[async_trait]
impl EventSink for NullEventSink {
    async fn emit(&mut self, _event: RuntimeEvent) -> Result<()> {
        Ok(())
    }
}
