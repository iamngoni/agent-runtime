use serde::{Deserialize, Serialize};

use crate::ToolCall;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// What kind of media an [`Attachment`] carries. Providers map these to their
/// own content-block types (image vs document/file).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AttachmentKind {
    Image,
    Document,
}

/// Where an attachment's bytes come from: inlined base64, or a remote URL the
/// provider fetches itself.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentSource {
    /// Base64-encoded bytes inlined into the request.
    Base64(String),
    /// A URL the provider retrieves. Not every provider supports every media
    /// type by URL (e.g. some only accept inline documents).
    Url(String),
}

/// A non-text input riding alongside a [`ChatMessage`] — an image or document
/// for vision/document-capable models. Mirrors Laravel AI's
/// `attachments: [...]` on a prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Attachment {
    pub kind: AttachmentKind,
    /// MIME type, e.g. `image/png` or `application/pdf`. May be empty for URL
    /// images where the provider infers it.
    pub media_type: String,
    pub source: AttachmentSource,
}

impl Attachment {
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            kind: AttachmentKind::Image,
            media_type: media_type.into(),
            source: AttachmentSource::Base64(data.into()),
        }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self {
            kind: AttachmentKind::Image,
            media_type: String::new(),
            source: AttachmentSource::Url(url.into()),
        }
    }

    pub fn document_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            kind: AttachmentKind::Document,
            media_type: media_type.into(),
            source: AttachmentSource::Base64(data.into()),
        }
    }

    pub fn document_url(url: impl Into<String>) -> Self {
        Self {
            kind: AttachmentKind::Document,
            media_type: String::new(),
            source: AttachmentSource::Url(url.into()),
        }
    }

    /// A `data:` URI for this attachment when it is base64-backed; used by
    /// providers (e.g. OpenAI) that take inline media as a data URL.
    pub fn data_uri(&self) -> Option<String> {
        match &self.source {
            AttachmentSource::Base64(data) => Some(format!("data:{};base64,{}", self.media_type, data)),
            AttachmentSource::Url(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: Option<String>,
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Media inputs (images/documents) attached to this message. Only
    /// meaningful on user messages; ignored by providers that don't support
    /// the given media type.
    #[serde(default)]
    pub attachments: Vec<Attachment>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: Some(content.into()),
            tool_call_id: None,
            tool_calls: Vec::new(),
            attachments: Vec::new(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: Some(content.into()),
            tool_call_id: None,
            tool_calls: Vec::new(),
            attachments: Vec::new(),
        }
    }

    /// A user message carrying one or more attachments (images/documents)
    /// alongside its text.
    pub fn user_with_attachments(
        content: impl Into<String>,
        attachments: Vec<Attachment>,
    ) -> Self {
        Self {
            role: MessageRole::User,
            content: Some(content.into()),
            tool_call_id: None,
            tool_calls: Vec::new(),
            attachments,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: Some(content.into()),
            tool_call_id: None,
            tool_calls: Vec::new(),
            attachments: Vec::new(),
        }
    }

    pub fn assistant_with_tools(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content,
            tool_call_id: None,
            tool_calls,
            attachments: Vec::new(),
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: Some(content.into()),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: Vec::new(),
            attachments: Vec::new(),
        }
    }

    /// Attach media to an existing message (builder style).
    pub fn with_attachment(mut self, attachment: Attachment) -> Self {
        self.attachments.push(attachment);
        self
    }

    /// Whether this message carries any media attachments.
    pub fn has_attachments(&self) -> bool {
        !self.attachments.is_empty()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssistantTurn {
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}
