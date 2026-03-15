
from __future__ import annotations

from enum import Enum
from typing import Union, Callable
from typing_extensions import TypeAlias

from .beta.threads.runs.function_tool_call import FunctionToolCall

from .responses.response_output_item import McpCall, McpApprovalRequest
from .responses.response_custom_tool_call import ResponseCustomToolCall
from .responses.response_function_tool_call import ResponseFunctionToolCall
from .chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
)
from .chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

__all__ = ["AgentAction", "AgentActionGuard", "ActionGuardDecision"]


class ActionGuardDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


AgentAction: TypeAlias = Union[
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageCustomToolCall,
    ResponseFunctionToolCall,
    ResponseCustomToolCall,
    McpCall,
    McpApprovalRequest,
    FunctionToolCall,
]

AgentActionGuard: TypeAlias = Callable[[AgentAction], ActionGuardDecision]
