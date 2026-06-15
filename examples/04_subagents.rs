//! Agents-as-tools: a support agent that delegates refunds to a sub-agent.
//!
//! The sub-agent runs its *own* full session (its own tools, if any) when the
//! parent delegates to it.
//!
//! Run with: `OPENAI_API_KEY=sk-... cargo run --example 04_subagents`

use agent_runtime::{Agent, AgentProviderKind, AgentTool, Llm, ToolRegistry};

/// A specialist sub-agent.
struct RefundsAgent;

impl Agent for RefundsAgent {
    fn instructions(&self) -> String {
        "You process customer refunds. Confirm the order id and issue the refund.".into()
    }
}

/// The parent agent holds an `Llm` so it can run its sub-agents.
struct SupportAgent {
    llm: Llm,
}

impl Agent for SupportAgent {
    fn instructions(&self) -> String {
        "You help customers. For anything refund-related, delegate to the \
         `refunds` tool."
            .into()
    }

    fn tools(&self) -> ToolRegistry<()> {
        let mut r = ToolRegistry::new();
        r.register(AgentTool::new(
            "refunds",
            "Delegates refund handling to a specialist sub-agent",
            self.llm.clone(),
            RefundsAgent,
        ));
        r
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = Llm::builder()
        .provider(AgentProviderKind::OpenAi)
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .build()?;

    let support = SupportAgent { llm: llm.clone() };
    let reply = llm
        .run(&support, "I'd like a refund for order #42, it arrived damaged.")
        .await?;
    println!("{reply}");
    Ok(())
}
