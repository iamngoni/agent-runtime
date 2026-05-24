use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    pub fn from_json_args(
        id: impl Into<String>,
        name: impl Into<String>,
        raw_arguments: &str,
    ) -> Result<Self> {
        let trimmed = raw_arguments.trim();
        let arguments = if trimmed.is_empty() {
            Value::Object(Default::default())
        } else {
            serde_json::from_str(trimmed).context("failed to decode tool arguments as JSON")?
        };

        Ok(Self {
            id: id.into(),
            name: name.into(),
            arguments,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: Value,
}

#[async_trait]
pub trait Tool<C>: Send + Sync {
    fn definition(&self) -> &ToolDefinition;

    async fn execute(&self, ctx: C, call: &ToolCall) -> Result<ToolOutput>;
}

pub struct ToolRegistry<C> {
    tools: HashMap<String, Arc<dyn Tool<C>>>,
}

impl<C> ToolRegistry<C> {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register<T>(&mut self, tool: T) -> &mut Self
    where
        T: Tool<C> + 'static,
    {
        let tool = Arc::new(tool);
        let name = tool.definition().name.clone();
        self.tools.insert(name, tool);
        self
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.definition().clone())
            .collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub async fn execute(&self, ctx: C, call: &ToolCall) -> Result<Option<ToolOutput>> {
        match self.tools.get(&call.name) {
            Some(tool) => tool.execute(ctx, call).await.map(Some),
            None => Ok(None),
        }
    }
}

impl<C> Default for ToolRegistry<C> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct JsonTool<C, Args, F> {
    definition: ToolDefinition,
    executor: F,
    _marker: PhantomData<fn() -> (C, Args)>,
}

impl<C, Args, F> JsonTool<C, Args, F>
where
    Args: DeserializeOwned + JsonSchema,
{
    pub fn new(name: impl Into<String>, description: impl Into<String>, executor: F) -> Self {
        Self {
            definition: ToolDefinition {
                name: name.into(),
                description: description.into(),
                input_schema: json_schema_for::<Args>(),
            },
            executor,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<C, Args, F, Fut> Tool<C> for JsonTool<C, Args, F>
where
    C: Send + Sync + 'static,
    Args: DeserializeOwned + JsonSchema + Send + Sync + 'static,
    F: Fn(C, Args) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Value>> + Send + 'static,
{
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, ctx: C, call: &ToolCall) -> Result<ToolOutput> {
        let arguments: Args =
            serde_json::from_value(call.arguments.clone()).with_context(|| {
                format!(
                    "failed to parse arguments for tool '{}'",
                    self.definition.name
                )
            })?;
        let content = (self.executor)(ctx, arguments).await?;
        Ok(ToolOutput { content })
    }
}

fn json_schema_for<T>() -> Value
where
    T: JsonSchema,
{
    let mut schema =
        serde_json::to_value(schemars::schema_for!(T)).expect("tool schema should serialize");
    if let Some(object) = schema.as_object_mut() {
        object.remove("$schema");
        object.remove("title");
    }
    schema
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde_json::json;

    use super::ToolCall;

    #[test]
    fn parses_json_string_arguments() -> Result<()> {
        let tool_call = ToolCall::from_json_args(
            "call_123",
            "search_products",
            "{\"query\":\"lagavulin\",\"products_max_count\":5}",
        )?;

        assert_eq!(tool_call.id, "call_123");
        assert_eq!(tool_call.name, "search_products");
        assert_eq!(
            tool_call.arguments,
            json!({
                "query": "lagavulin",
                "products_max_count": 5
            })
        );
        Ok(())
    }

    #[test]
    fn serializes_provider_neutral_shape() -> Result<()> {
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            name: "search_products".to_string(),
            arguments: json!({
                "query": "lagavulin",
                "products_max_count": 5
            }),
        };

        let value = serde_json::to_value(&tool_call)?;
        assert_eq!(
            value,
            json!({
                "id": "call_123",
                "name": "search_products",
                "arguments": {
                    "query": "lagavulin",
                    "products_max_count": 5
                }
            })
        );
        Ok(())
    }
}
