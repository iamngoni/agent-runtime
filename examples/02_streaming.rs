//! Stream assistant tokens to stdout as they arrive via an `EventSink`.
//!
//! Run with: `OPENAI_API_KEY=sk-... cargo run --example 02_streaming`

use std::io::Write;

use agent_runtime::{Agent, AgentProviderKind, EventSink, Llm, RuntimeEvent};
use async_trait::async_trait;

struct Poet;

impl Agent for Poet {
    fn instructions(&self) -> String {
        "You write short, vivid poems.".into()
    }
}

/// A sink that prints `AssistantDelta` tokens as they stream in.
struct PrintSink;

#[async_trait]
impl EventSink for PrintSink {
    async fn emit(&mut self, event: RuntimeEvent) -> anyhow::Result<()> {
        if let RuntimeEvent::AssistantDelta { delta } = event {
            print!("{delta}");
            std::io::stdout().flush().ok();
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = Llm::builder()
        .provider(AgentProviderKind::OpenAi)
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .build()?;

    llm.run_stream(&Poet, &[], "Write a haiku about Rust.", &mut PrintSink)
        .await?;
    println!();
    Ok(())
}
