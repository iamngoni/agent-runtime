use anyhow::{Context, Result};
use web_time::Instant;
use tracing::info;

use crate::{
    AgentProvider, AssistantTurn, ChatMessage, EventSink, RuntimeEvent, ToolCall, ToolOutput,
    ToolRegistry,
};

#[derive(Debug, Clone)]
pub struct ExecutedToolCall {
    pub call: ToolCall,
    pub output: ToolOutput,
}

pub struct ToolSessionRequest<'a, C> {
    pub model: &'a str,
    pub decision_system_prompt: &'a str,
    pub followup_system_prompt: &'a str,
    pub history: &'a [ChatMessage],
    pub tool_registry: &'a ToolRegistry<C>,
    pub tool_context: C,
    pub max_tool_calls: usize,
}

pub enum ToolSessionOutcome {
    Direct {
        assistant_turn: AssistantTurn,
        message: String,
    },
    ToolBacked {
        assistant_turn: AssistantTurn,
        executed_tool_calls: Vec<ExecutedToolCall>,
        message: String,
    },
}

impl ToolSessionOutcome {
    /// The final natural-language message, regardless of whether tools ran.
    pub fn message(&self) -> &str {
        match self {
            ToolSessionOutcome::Direct { message, .. }
            | ToolSessionOutcome::ToolBacked { message, .. } => message,
        }
    }

    /// Consume the outcome, returning its final message.
    pub fn into_message(self) -> String {
        match self {
            ToolSessionOutcome::Direct { message, .. }
            | ToolSessionOutcome::ToolBacked { message, .. } => message,
        }
    }
}

pub async fn execute_tool_calls<C, E>(
    tool_registry: &ToolRegistry<C>,
    tool_context: C,
    tool_calls: &[ToolCall],
    max_tool_calls: usize,
    verbose: bool,
    sink: &mut E,
) -> Result<Vec<ExecutedToolCall>>
where
    C: Clone + Send + Sync + 'static,
    E: EventSink,
{
    let mut executed_tool_calls = Vec::new();

    if verbose {
        info!(
            requested_tool_calls = tool_calls.len(),
            max_tool_calls, "executing tool call batch"
        );
    }

    for call in tool_calls.iter().take(max_tool_calls) {
        if !tool_registry.contains(&call.name) {
            if verbose {
                info!(
                    tool_call_id = %call.id,
                    tool_name = %call.name,
                    arguments = %call.arguments,
                    "skipping unregistered tool call"
                );
            }
            continue;
        }

        let tool_started_at = Instant::now();
        if verbose {
            info!(
                tool_call_id = %call.id,
                tool_name = %call.name,
                arguments = %call.arguments,
                "starting tool call"
            );
        }

        sink.emit(RuntimeEvent::ToolStarted { call: call.clone() })
            .await?;

        let output = tool_registry
            .execute(tool_context.clone(), call)
            .await?
            .with_context(|| format!("tool registry lost registration for {}", call.name))?;

        sink.emit(RuntimeEvent::ToolCompleted {
            call: call.clone(),
            output: output.clone(),
        })
        .await?;

        if verbose {
            info!(
                tool_call_id = %call.id,
                tool_name = %call.name,
                duration_ms = tool_started_at.elapsed().as_millis() as u64,
                output = %output.content,
                "completed tool call"
            );
        }

        executed_tool_calls.push(ExecutedToolCall {
            call: call.clone(),
            output,
        });
    }

    if verbose {
        info!(
            requested_tool_calls = tool_calls.len(),
            executed_tool_calls = executed_tool_calls.len(),
            "completed tool call batch"
        );
    }

    Ok(executed_tool_calls)
}

pub fn build_followup_messages(
    history: &[ChatMessage],
    assistant_turn: &AssistantTurn,
    executed_tool_calls: &[ExecutedToolCall],
) -> Result<Vec<ChatMessage>> {
    let mut followup_messages = history.to_vec();
    followup_messages.push(ChatMessage::assistant_with_tools(
        assistant_turn.content.clone(),
        assistant_turn.tool_calls.clone(),
    ));

    for executed in executed_tool_calls {
        followup_messages.push(ChatMessage::tool(
            executed.call.id.clone(),
            serde_json::to_string(&executed.output.content)
                .context("failed to serialize tool result payload")?,
        ));
    }

    Ok(followup_messages)
}

pub async fn execute_tool_session<C, E>(
    provider: &dyn AgentProvider,
    request: ToolSessionRequest<'_, C>,
    sink: &mut E,
) -> Result<ToolSessionOutcome>
where
    C: Clone + Send + Sync + 'static,
    E: EventSink,
{
    if provider.verbose() {
        info!(
            model = %request.model,
            history_count = request.history.len(),
            max_tool_calls = request.max_tool_calls,
            "starting tool session"
        );
    }

    let assistant_turn = provider
        .request_assistant_turn(
            request.model,
            request.decision_system_prompt,
            request.history,
            &request.tool_registry.definitions(),
        )
        .await?;

    let direct_message = assistant_turn
        .content
        .clone()
        .unwrap_or_default()
        .trim()
        .to_string();

    if assistant_turn.tool_calls.is_empty() {
        if provider.verbose() {
            info!(
                model = %request.model,
                message_chars = direct_message.chars().count(),
                "tool session resolved directly without tool calls"
            );
        }
        return Ok(ToolSessionOutcome::Direct {
            assistant_turn,
            message: direct_message,
        });
    }

    let executed_tool_calls = execute_tool_calls(
        request.tool_registry,
        request.tool_context,
        &assistant_turn.tool_calls,
        request.max_tool_calls,
        provider.verbose(),
        sink,
    )
    .await?;

    if executed_tool_calls.is_empty() {
        if provider.verbose() {
            info!(
                model = %request.model,
                requested_tool_calls = assistant_turn.tool_calls.len(),
                "tool session fell back to direct response because no tool calls executed"
            );
        }
        return Ok(ToolSessionOutcome::Direct {
            assistant_turn,
            message: direct_message,
        });
    }

    let followup_messages =
        build_followup_messages(request.history, &assistant_turn, &executed_tool_calls)?;

    let message = provider
        .stream_message(
            request.model,
            request.followup_system_prompt,
            &followup_messages,
            sink,
        )
        .await?;

    if provider.verbose() {
        info!(
            model = %request.model,
            executed_tool_calls = executed_tool_calls.len(),
            followup_message_count = followup_messages.len(),
            final_message_chars = message.chars().count(),
            "tool session completed with follow-up synthesis"
        );
    }

    Ok(ToolSessionOutcome::ToolBacked {
        assistant_turn,
        executed_tool_calls,
        message,
    })
}
