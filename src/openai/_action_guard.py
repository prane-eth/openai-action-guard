from __future__ import annotations

import inspect
import logging
from typing import Iterable, TypeVar, cast

from ._exceptions import ActionGuardError
from .types.action_guard import Action, ActionGuard, GuardDecision
from .types.chat.chat_completion import ChatCompletion
from .types.responses.response import Response

__all__ = ["apply_action_guard", "ensure_action_guard_support"]

_ResponseT = TypeVar("_ResponseT")
log: logging.Logger = logging.getLogger(__name__)
_RESPONSE_ACTION_TYPES = {
    "function_call",
    "custom_tool_call",
    "mcp_call",
    "mcp_approval_request",
}


def ensure_action_guard_support(
    *, action_guard: ActionGuard | None, stream: bool
) -> None:
    if action_guard is not None and stream:
        raise ValueError("`action_guard` is only supported for non-streaming requests")


def apply_action_guard(
    result: _ResponseT, *, action_guard: ActionGuard | None
) -> _ResponseT:
    if action_guard is None:
        return result

    for action in _iter_actions(result):
        decision = action_guard(action)
        if inspect.isawaitable(decision):
            raise TypeError("`action_guard` must be synchronous")

        if decision is GuardDecision.ALLOW:
            continue

        if decision is GuardDecision.BLOCK:
            message = _describe_action(action)
            log.warning(message)
            raise ActionGuardError(message, action=action)

        raise TypeError(
            "`action_guard` must return `GuardDecision.ALLOW` or `GuardDecision.BLOCK`"
        )

    return result


def _iter_actions(result: object) -> Iterable[Action]:
    if isinstance(result, ChatCompletion):
        for choice in result.choices:
            for tool_call in choice.message.tool_calls or []:
                yield cast(Action, tool_call)
        return

    if isinstance(result, Response):
        for output in result.output:
            if output.type in _RESPONSE_ACTION_TYPES:
                yield cast(Action, output)
        return

    raise TypeError(f"Unsupported action guard response type: {type(result)!r}")


def _describe_action(action: Action) -> str:
    action_type = getattr(action, "type", action.__class__.__name__)

    if action_type == "function":
        name = action.function.name
        action_id = getattr(action, "id", None)
        suffix = f" ({action_id})" if action_id else ""
        return f"Action blocked by `action_guard`: function tool `{name}`{suffix}"

    name = getattr(action, "name", None)
    server_label = getattr(action, "server_label", None)
    action_id = getattr(action, "call_id", None) or getattr(action, "id", None)

    parts = [f"Action blocked by `action_guard`: {action_type}"]
    if name:
        parts.append(f"`{name}`")
    if server_label:
        parts.append(f"from `{server_label}`")
    if action_id:
        parts.append(f"({action_id})")
    return " ".join(parts)
