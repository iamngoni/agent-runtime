use anyhow::Result;
use async_trait::async_trait;

use crate::{ToolCall, ToolOutput};

#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    AssistantStarted {
        model: String,
    },
    AssistantDelta {
        delta: String,
    },
    /// The assistant stream ended. `complete` is false when the provider
    /// stopped without its protocol's end marker (`[DONE]`,
    /// `message_stop`), so a caller can show the answer as cut off instead
    /// of presenting a truncated reply as finished.
    AssistantEnded {
        complete: bool,
    },
    ToolStarted {
        call: ToolCall,
    },
    ToolCompleted {
        call: ToolCall,
        output: ToolOutput,
    },
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
