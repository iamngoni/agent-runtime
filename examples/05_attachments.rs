//! Multimodal input: attach an image to a message.
//!
//! Attachments ride on a `ChatMessage`, so this uses the lower-level
//! `stream_message` rather than the string-only `run`. Each provider maps the
//! attachment to its own multimodal format.
//!
//! Run with: `OPENAI_API_KEY=sk-... cargo run --example 05_attachments`

use agent_runtime::{
    AgentProviderKind, Attachment, ChatMessage, EventSink, Llm, RuntimeEvent,
};
use async_trait::async_trait;

struct PrintSink;

#[async_trait]
impl EventSink for PrintSink {
    async fn emit(&mut self, event: RuntimeEvent) -> anyhow::Result<()> {
        if let RuntimeEvent::AssistantDelta { delta } = event {
            print!("{delta}");
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

    let message = ChatMessage::user_with_attachments(
        "What's in this image?",
        vec![Attachment::image_url(
            "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
        )],
        // or inline base64: Attachment::image_base64("image/png", b64)
        // or a document:     Attachment::document_base64("application/pdf", b64)
    );

    llm.stream_message("gpt-4o", "Describe images briefly.", &[message], &mut PrintSink)
        .await?;
    println!();
    Ok(())
}
