//! Conversation persistence — the seam for remembering history across turns.
//!
//! Adapted from Laravel AI's `ConversationStore` contract. The runtime itself
//! stays storage-agnostic: it defines the [`ConversationStore`] trait and ships
//! one in-memory implementation for tests and single-process development.
//!
//! **Production should plug in its own impl** — Cloudflare KV / Durable
//! Objects, Postgres, Redis — exactly like a custom
//! [`HttpClient`](crate::HttpClient). The bundled [`InMemoryConversationStore`]
//! does **not** survive a restart and is **not** shared across instances or
//! across Worker isolates, so it must not be used for real persistence.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use async_trait::async_trait;

use crate::ChatMessage;

/// Pluggable persistence for multi-turn conversation history.
///
/// A typical "remembered" turn: load the latest messages as `history`, run the
/// agent, then append the new user and assistant messages.
#[async_trait]
pub trait ConversationStore: Send + Sync {
    /// The most recently created conversation for a user, if any. Lets a caller
    /// resume "the last conversation" without tracking ids itself.
    async fn latest_conversation_id(&self, user_id: &str) -> Result<Option<String>>;

    /// Create a new conversation (optionally owned by a user) and return its id.
    async fn create_conversation(&self, user_id: Option<&str>, title: &str) -> Result<String>;

    /// Append a message to a conversation in arrival order.
    async fn append_message(&self, conversation_id: &str, message: &ChatMessage) -> Result<()>;

    /// The latest `limit` messages for a conversation, oldest-first, ready to
    /// pass straight back as `history`.
    async fn latest_messages(&self, conversation_id: &str, limit: usize)
    -> Result<Vec<ChatMessage>>;
}

#[derive(Default)]
struct Conversation {
    seq: u64,
    user_id: Option<String>,
    #[allow(dead_code)]
    title: String,
    messages: Vec<ChatMessage>,
}

/// In-memory [`ConversationStore`] for tests and single-process dev only.
///
/// Not durable, not shared across instances/isolates. See the module docs.
#[derive(Default)]
pub struct InMemoryConversationStore {
    conversations: Mutex<HashMap<String, Conversation>>,
    next_seq: AtomicU64,
}

impl InMemoryConversationStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ConversationStore for InMemoryConversationStore {
    async fn latest_conversation_id(&self, user_id: &str) -> Result<Option<String>> {
        let conversations = self.conversations.lock().expect("store mutex poisoned");
        let latest = conversations
            .iter()
            .filter(|(_, c)| c.user_id.as_deref() == Some(user_id))
            .max_by_key(|(_, c)| c.seq)
            .map(|(id, _)| id.clone());
        Ok(latest)
    }

    async fn create_conversation(&self, user_id: Option<&str>, title: &str) -> Result<String> {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let id = format!("conv-{seq}");
        let conversation = Conversation {
            seq,
            user_id: user_id.map(str::to_string),
            title: title.to_string(),
            messages: Vec::new(),
        };
        self.conversations
            .lock()
            .expect("store mutex poisoned")
            .insert(id.clone(), conversation);
        Ok(id)
    }

    async fn append_message(&self, conversation_id: &str, message: &ChatMessage) -> Result<()> {
        let mut conversations = self.conversations.lock().expect("store mutex poisoned");
        let conversation = conversations.entry(conversation_id.to_string()).or_default();
        conversation.messages.push(message.clone());
        Ok(())
    }

    async fn latest_messages(
        &self,
        conversation_id: &str,
        limit: usize,
    ) -> Result<Vec<ChatMessage>> {
        let conversations = self.conversations.lock().expect("store mutex poisoned");
        let Some(conversation) = conversations.get(conversation_id) else {
            return Ok(Vec::new());
        };
        let start = conversation.messages.len().saturating_sub(limit);
        Ok(conversation.messages[start..].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn round_trips_messages_in_order() -> Result<()> {
        let store = InMemoryConversationStore::new();
        let id = store.create_conversation(Some("user-1"), "support").await?;

        store
            .append_message(&id, &ChatMessage::user("hi"))
            .await?;
        store
            .append_message(&id, &ChatMessage::assistant("hello"))
            .await?;

        let messages = store.latest_messages(&id, 100).await?;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content.as_deref(), Some("hi"));
        assert_eq!(messages[1].content.as_deref(), Some("hello"));
        Ok(())
    }

    #[tokio::test]
    async fn latest_messages_respects_limit() -> Result<()> {
        let store = InMemoryConversationStore::new();
        let id = store.create_conversation(None, "t").await?;
        for i in 0..5 {
            store
                .append_message(&id, &ChatMessage::user(format!("m{i}")))
                .await?;
        }
        let messages = store.latest_messages(&id, 2).await?;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content.as_deref(), Some("m3"));
        assert_eq!(messages[1].content.as_deref(), Some("m4"));
        Ok(())
    }

    #[tokio::test]
    async fn tracks_latest_conversation_per_user() -> Result<()> {
        let store = InMemoryConversationStore::new();
        let _first = store.create_conversation(Some("user-1"), "a").await?;
        let second = store.create_conversation(Some("user-1"), "b").await?;
        store.create_conversation(Some("user-2"), "c").await?;

        assert_eq!(
            store.latest_conversation_id("user-1").await?,
            Some(second)
        );
        assert_eq!(store.latest_conversation_id("user-3").await?, None);
        Ok(())
    }
}
