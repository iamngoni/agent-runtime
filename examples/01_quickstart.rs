//! Smallest possible agent: an LLM handle + a one-line declarative agent.
//!
//! Run with: `OPENAI_API_KEY=sk-... cargo run --example 01_quickstart`

use agent_runtime::{Agent, AgentProviderKind, Llm};

/// An agent is just instructions (everything else has a default).
struct Helpful;

impl Agent for Helpful {
    fn instructions(&self) -> String {
        "You are concise and helpful. Answer in one sentence.".into()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = Llm::builder()
        .provider(AgentProviderKind::OpenAi) // or Anthropic, Groq, Cohere, Gemini…
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .build()?;

    // The whole tool session is hidden behind `run`. With no model set on the
    // agent, the LLM's default tier is used.
    let reply = llm.run(&Helpful, "What is the capital of Zimbabwe?").await?;
    println!("{reply}");
    Ok(())
}
