//! Repair for tool-call arguments that are not valid JSON.
//!
//! Adapted from aster (https://github.com/Zfinix/aster,
//! `crates/aster-ai/src/tool_args.rs`), licensed under the Apache License 2.0.
//! Changed to return a `serde_json::Value` instead of rewriting strings in place.
//!
//! A truncated stream or a double-encoded string would otherwise fail the whole
//! turn. Unrepairable arguments become `{}`, so the tool reports its missing
//! parameters and the model retries.

use serde_json::{Map, Value};

/// A JSON object for `raw`, whatever shape it arrived in.
pub fn heal(raw: &str) -> Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Value::Object(Map::new());
    }
    as_object(trimmed)
        .or_else(|| close_open_scopes(trimmed).as_deref().and_then(as_object))
        .unwrap_or_else(|| Value::Object(Map::new()))
}

/// Accepts an object, and unwraps the double-encoded string some endpoints send
/// (`"{\"path\":\"a\"}"` rather than `{"path":"a"}`).
fn as_object(text: &str) -> Option<Value> {
    match serde_json::from_str::<Value>(text).ok()? {
        value @ Value::Object(_) => Some(value),
        Value::String(inner) => as_object(inner.trim()),
        _ => None,
    }
}

/// Shuts unterminated strings and brackets, in the order they were opened.
fn close_open_scopes(text: &str) -> Option<String> {
    let mut scopes = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for ch in text.chars() {
        if in_string {
            match ch {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => scopes.push('}'),
            '[' => scopes.push(']'),
            '}' | ']' if scopes.pop() != Some(ch) => return None,
            _ => {}
        }
    }
    if scopes.is_empty() && !in_string {
        return None;
    }
    let mut healed = text.to_string();
    if in_string {
        if escaped {
            healed.pop();
        }
        healed.push('"');
    }
    // A truncated key or a dangling comma leaves the object mid-pair; dropping
    // back to the last complete one is better than inventing a value.
    while scopes.last() == Some(&'}') && needs_value(&healed) {
        match healed.rfind(',') {
            Some(comma) => healed.truncate(comma),
            None => healed.truncate(healed.find('{')? + 1),
        }
    }
    while let Some(closer) = scopes.pop() {
        healed.push(closer);
    }
    Some(healed)
}

/// True when the object's last pair stops before its value, as in `{"a":1,"b":`.
fn needs_value(text: &str) -> bool {
    let tail = text.trim_end();
    tail.ends_with(':') || tail.ends_with(',') || {
        let after_comma = tail.rsplit(['{', ',']).next().unwrap_or("").trim();
        !after_comma.is_empty() && !after_comma.contains(':')
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::heal;

    #[test]
    fn repairs_the_shapes_models_send() {
        assert_eq!(heal(r#"{"path":"a.rs"}"#), json!({ "path": "a.rs" }));
        assert_eq!(heal(r#""{\"path\":\"a.rs\"}""#), json!({ "path": "a.rs" }));
        assert_eq!(
            heal(r#"{"path":"a.rs","content":"fn ma"#),
            json!({ "path": "a.rs", "content": "fn ma" })
        );
        assert_eq!(heal(r#"{"path":"a.rs","cont"#), json!({ "path": "a.rs" }));
        assert_eq!(heal(r#"{"items":[1,2"#), json!({ "items": [1, 2] }));
        assert_eq!(heal("not json"), json!({}));
        assert_eq!(heal(""), json!({}));
    }
}
