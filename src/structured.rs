//! Structured output: constrain a model's answer to a JSON Schema.
//!
//! Free-form tool calling (the default) lets a model decide *whether* to call a
//! tool. Structured output is the opposite guarantee — the model is *required*
//! to return JSON matching a schema. Both the Anthropic and OpenAI-compatible
//! backends implement this by forcing a single synthetic tool call whose input
//! schema is the requested shape, so it works across every provider that
//! supports forced tool choice (no provider-specific `response_format` needed).
//!
//! Drive it from the high level with
//! [`Llm::run_structured`](crate::Llm::run_structured), or pass a
//! [`ResponseFormat`] straight to
//! [`TextProvider::request_structured`](crate::TextProvider::request_structured).

use schemars::JsonSchema;
use serde_json::Value;

/// A required output schema. When supplied to a structured request, the provider
/// is constrained to return JSON matching `schema`.
#[derive(Debug, Clone)]
pub struct ResponseFormat {
    /// Schema name surfaced to the provider as the forced tool name. Must match
    /// `^[a-zA-Z0-9_-]{1,64}$` (Anthropic/OpenAI tool-name rules); derived names
    /// are sanitized to satisfy this.
    pub name: String,
    /// Human-readable description handed to the model alongside the schema.
    pub description: String,
    /// The JSON Schema the response must conform to.
    pub schema: Value,
}

impl ResponseFormat {
    /// Build a format from an explicit name and JSON Schema.
    pub fn new(name: impl Into<String>, schema: Value) -> Self {
        Self {
            name: sanitize_schema_name(&name.into()),
            description: "Structured response payload.".to_string(),
            schema,
        }
    }

    /// Override the description handed to the model.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Derive a format from a type's [`JsonSchema`] derivation — the ergonomic
    /// path used by [`Llm::run_structured`](crate::Llm::run_structured).
    pub fn for_type<T: JsonSchema>() -> Self {
        let (name, schema) = response_schema_for::<T>();
        Self {
            name,
            description: "Structured response payload.".to_string(),
            schema,
        }
    }
}

/// Generate a JSON Schema for `T`, returning a provider-safe name (derived from
/// the type's schema title) and the schema object with the top-level `$schema`
/// and `title` keys stripped (Anthropic rejects `$schema` in tool input).
pub fn response_schema_for<T: JsonSchema>() -> (String, Value) {
    let mut schema =
        serde_json::to_value(schemars::schema_for!(T)).expect("response schema should serialize");
    let mut name = "response".to_string();
    if let Some(object) = schema.as_object_mut() {
        if let Some(title) = object.get("title").and_then(Value::as_str) {
            name = sanitize_schema_name(title);
        }
        object.remove("$schema");
        object.remove("title");
    }
    (name, schema)
}

/// Coerce an arbitrary label into a valid tool name (`[a-zA-Z0-9_-]`, non-empty).
fn sanitize_schema_name(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches('_');
    if trimmed.is_empty() {
        "response".to_string()
    } else {
        trimmed.chars().take(64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, JsonSchema)]
    struct Reply {
        #[allow(dead_code)]
        text: String,
    }

    #[test]
    fn derives_name_from_type_and_strips_meta_keys() {
        let (name, schema) = response_schema_for::<Reply>();
        assert_eq!(name, "Reply");
        let obj = schema.as_object().unwrap();
        assert!(!obj.contains_key("$schema"));
        assert!(!obj.contains_key("title"));
        assert_eq!(obj["type"], serde_json::json!("object"));
    }

    #[test]
    fn sanitizes_invalid_name_characters() {
        assert_eq!(sanitize_schema_name("My Reply!"), "My_Reply");
        assert_eq!(sanitize_schema_name("***"), "response");
    }
}
