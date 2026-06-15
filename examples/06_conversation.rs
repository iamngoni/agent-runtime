//! Multi-turn memory via a `ConversationStore`.
//!
//! The in-memory store here is for demos only — in production implement
//! `ConversationStore` over Cloudflare KV / Durable Objects, Postgres, Redis…
//!
//! Run with: `OPENAI_API_KEY=sk-... cargo run --example 06_conversation`

use agent_runtime::{
    Agent, AgentProviderKind, ChatMessage, ConversationStore, InMemoryConversationStore, Llm,
};

struct Assistant;

impl Agent for Assistant {
    fn instructions(&self) -> String {
        "You are a friendly assistant. Use what the user told you earlier.".into()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = Llm::builder()
        .provider(AgentProviderKind::OpenAi)
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .build()?;

    let store = InMemoryConversationStore::new();
    let convo = store.create_conversation(Some("user-1"), "demo").await?;

    // Helper: one remembered turn — load history, run, persist both messages.
    async fn turn(
        llm: &Llm,
        store: &InMemoryConversationStore,
        convo: &str,
        input: &str,
    ) -> anyhow::Result<String> {
        let history = store.latest_messages(convo, 100).await?;
        let reply = llm.run_with_history(&Assistant, &history, input).await?;
        store.append_message(convo, &ChatMessage::user(input)).await?;
        store
            .append_message(convo, &ChatMessage::assistant(&reply))
            .await?;
        Ok(reply)
    }

    println!("{}", turn(&llm, &store, &convo, "My name is Sam.").await?);
    println!("{}", turn(&llm, &store, &convo, "What's my name?").await?);
    Ok(())
}
