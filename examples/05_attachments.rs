//! Multimodal input: attach an image to a message.
//!
//! Attachments ride on a `ChatMessage`, so this uses `run_message` (the
//! high-level runner variant that takes a full message) rather than the
//! string-only `run`. Each provider maps the attachment to its own multimodal
//! format.
//!
//! Run with: `OPENAI_API_KEY=sk-... cargo run --example 05_attachments`

use agent_runtime::{Agent, AgentProviderKind, Attachment, ChatMessage, Llm};

struct Vision;

impl Agent for Vision {
    fn instructions(&self) -> String {
        "Describe images briefly.".into()
    }
    fn model(&self) -> String {
        "gpt-4o".into()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = Llm::builder()
        .provider(AgentProviderKind::OpenAi)
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .build()?;

    let message = ChatMessage::user_with_attachments(
        "What's in this image?",
        vec![Attachment::image_url(
            "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
        )],
        // or inline base64: Attachment::image_base64("image/png", b64)
        // or a document:     Attachment::document_base64("application/pdf", b64)
    );

    // Attachments flow through the high-level API via `run_message`.
    let reply = llm.run_message(&Vision, &[], message).await?;
    println!("{reply}");
    Ok(())
}
