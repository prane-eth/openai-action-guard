from __future__ import annotations

import inspect
import logging
from typing import TypeVar, Iterable, cast
from .types.agent_action_guard import AgentAction, AgentActionGuard, ActionGuardDecision
from .types.responses.response import Response
from .types.chat.chat_completion_chunk import ChatCompletionChunk
from .types.chat.chat_completion import ChatCompletion
from .types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from .types.responses.response_output_item_added_event import ResponseOutputItemAddedEvent
from .types.chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
)
from .types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from ._exceptions import ActionGuardError
from ._streaming import Stream, AsyncStream

__all__ = ["apply_action_guard"]

_ResponseT = TypeVar("_ResponseT")
_StreamItemT = TypeVar("_StreamItemT")
log: logging.Logger = logging.getLogger(__name__)
_RESPONSE_ACTION_TYPES = {
    "function_call",
    "custom_tool_call",
    "mcp_call",
    "mcp_approval_request",
}


def with_action_guard(
    result: _ResponseT, *, action_guard: AgentActionGuard | None
) -> _ResponseT:
    if action_guard is None:
        return result

    return apply_action_guard(result, action_guard=action_guard)


def apply_action_guard(
    result: _ResponseT, *, action_guard: AgentActionGuard | None
) -> _ResponseT:
    if action_guard is None:
        return result

    for action in _iter_actions(result):
        decision = action_guard(action)
        if inspect.isawaitable(decision):
            raise TypeError("`action_guard` must be synchronous")

        if decision is ActionGuardDecision.ALLOW:
            continue

        if decision is ActionGuardDecision.BLOCK:
            message = _describe_action(action)
            log.warning(message)
            raise ActionGuardError(message, action=action)

        raise TypeError(
            "`action_guard` must return `GuardDecision.ALLOW` or `GuardDecision.BLOCK`"
        )

    return result


def _iter_actions(result: object) -> Iterable[AgentAction]:
    if isinstance(result, ChatCompletion):
        for choice in result.choices:
            for tool_call in choice.message.tool_calls or []:
                yield cast(AgentAction, tool_call)
        return

    if isinstance(result, ChatCompletionChunk):
        for choice in result.choices:
            for tool_call in choice.delta.tool_calls or []:
                if tool_call.type == "function" and tool_call.function and tool_call.function.name:
                    # Streamed chunks expose tool calls in delta format.
                    yield cast(AgentAction, tool_call)
        return

    if isinstance(result, Response):
        for output in result.output:
            if output.type in _RESPONSE_ACTION_TYPES:
                yield cast(AgentAction, output)
        return

    if isinstance(result, ResponseOutputItemAddedEvent):
        if result.item.type in _RESPONSE_ACTION_TYPES:
            yield cast(AgentAction, result.item)
        return

    raise TypeError(f"Unsupported action guard response type: {type(result)!r}")


def _describe_action(action: AgentAction) -> str:
    action_type = action.type

    if isinstance(action, ChatCompletionMessageFunctionToolCall):
        function_name = action.function.name
        function_action_id = action.id
        suffix = f" ({function_action_id})" if function_action_id else ""
        return f"Action blocked by `action_guard`: function tool `{function_name}`{suffix}"

    if isinstance(action, ChatCompletionMessageCustomToolCall):
        custom_name = action.custom.name
        custom_action_id = action.id
        suffix = f" ({custom_action_id})" if custom_action_id else ""
        return f"Action blocked by `action_guard`: custom tool `{custom_name}`{suffix}"

    if isinstance(action, ChoiceDeltaToolCall):
        function_name = action.function.name if action.function else None
        call_id = action.id
        suffix = f" ({call_id})" if call_id else ""
        if function_name:
            return f"Action blocked by `action_guard`: function tool `{function_name}`{suffix}"
        return f"Action blocked by `action_guard`: function tool{suffix}"

    name: str | None = getattr(action, "name", None)
    server_label = getattr(action, "server_label", None)
    action_id: str | None = getattr(action, "call_id", None) or getattr(action, "id", None)

    parts = [f"Action blocked by `action_guard`: {action_type}"]
    if name:
        parts.append(f"`{name}`")
    if server_label:
        parts.append(f"from `{server_label}`")
    if action_id:
        parts.append(f"({action_id})")
    return " ".join(parts)
