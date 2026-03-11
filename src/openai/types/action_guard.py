
from __future__ import annotations

from enum import Enum
from typing import Union, Callable
from typing_extensions import TypeAlias

from .responses.response_output_item import McpCall, McpApprovalRequest
from .responses.response_custom_tool_call import ResponseCustomToolCall
from .responses.response_function_tool_call import ResponseFunctionToolCall
from .chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
)
from .chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

__all__ = ["Action", "ActionGuard", "GuardDecision"]


class GuardDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


Action: TypeAlias = Union[
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageCustomToolCall,
    ResponseFunctionToolCall,
    ResponseCustomToolCall,
    McpCall,
    McpApprovalRequest,
]

ActionGuard: TypeAlias = Callable[[Action], GuardDecision]
