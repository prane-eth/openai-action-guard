from __future__ import annotations

from typing_extensions import TypeVar

import pytest
import httpx
from respx import MockRouter
from inline_snapshot import snapshot

import openai
from openai import OpenAI, AsyncOpenAI
from openai._utils import assert_signatures_in_sync

from ...conftest import base_url
from ..snapshots import make_snapshot_request

_T = TypeVar("_T")

# all the snapshots in this file are auto-generated from the live API
#
# you can update them with
#
# `OPENAI_LIVE=1 pytest --inline-snapshot=fix -p no:xdist -o addopts=""`


@pytest.mark.respx(base_url=base_url)
def test_output_text(client: OpenAI, respx_mock: MockRouter) -> None:
    response = make_snapshot_request(
        lambda c: c.responses.create(
            model="gpt-4o-mini",
            input="What's the weather like in SF?",
        ),
        content_snapshot=snapshot(
            '{"id": "resp_689a0b2545288193953c892439b42e2800b2e36c65a1fd4b", "object": "response", "created_at": 1754925861, "status": "completed", "background": false, "error": null, "incomplete_details": null, "instructions": null, "max_output_tokens": null, "max_tool_calls": null, "model": "gpt-4o-mini-2024-07-18", "output": [{"id": "msg_689a0b2637b08193ac478e568f49e3f900b2e36c65a1fd4b", "type": "message", "status": "completed", "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "I can\'t provide real-time updates, but you can easily check the current weather in San Francisco using a weather website or app. Typically, San Francisco has cool, foggy summers and mild winters, so it\'s good to be prepared for variable weather!"}], "role": "assistant"}], "parallel_tool_calls": true, "previous_response_id": null, "prompt_cache_key": null, "reasoning": {"effort": null, "summary": null}, "safety_identifier": null, "service_tier": "default", "store": true, "temperature": 1.0, "text": {"format": {"type": "text"}, "verbosity": "medium"}, "tool_choice": "auto", "tools": [], "top_logprobs": 0, "top_p": 1.0, "truncation": "disabled", "usage": {"input_tokens": 14, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 64}, "user": null, "metadata": {}}'
        ),
        path="/responses",
        mock_client=client,
        respx_mock=respx_mock,
    )

    assert response.output_text == snapshot(
        "I can't provide real-time updates, but you can easily check the current weather in San Francisco using a weather website or app. Typically, San Francisco has cool, foggy summers and mild winters, so it's good to be prepared for variable weather!"
    )


@pytest.mark.respx(base_url=base_url)
def test_action_guard_blocks_mcp_approval_requests(
    client: OpenAI, respx_mock: MockRouter
) -> None:
    seen: list[openai.AgentAction] = []

    def action_guard(action: openai.AgentAction) -> openai.ActionGuardDecision:
        seen.append(action)
        return openai.ActionGuardDecision.BLOCK

    with pytest.raises(openai.ActionGuardError, match="filesystem"):
        make_snapshot_request(
            lambda c: c.responses.create(
                model="gpt-5.4-2026-03-05",
                input="Read /etc/passwd",
                action_guard=action_guard,
            ),
            content_snapshot=snapshot(
                '{"id": "resp_action_guard", "object": "response", "created_at": 1754925861, "status": "completed", "background": false, "error": null, "incomplete_details": null, "instructions": null, "max_output_tokens": null, "max_tool_calls": null, "model": "gpt-5.4-2026-03-05", "output": [{"id": "mcpr_123", "type": "mcp_approval_request", "arguments": "{\\"path\\":\\"/etc/passwd\\"}", "name": "read_file", "server_label": "filesystem"}], "parallel_tool_calls": true, "previous_response_id": null, "prompt_cache_key": null, "reasoning": {"effort": null, "summary": null}, "safety_identifier": null, "service_tier": "default", "store": true, "temperature": 1.0, "text": {"format": {"type": "text"}, "verbosity": "medium"}, "tool_choice": "auto", "tools": [], "top_logprobs": 0, "top_p": 1.0, "truncation": "disabled", "usage": {"input_tokens": 14, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 12, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 26}, "user": null, "metadata": {}}'
            ),
            path="/responses",
            mock_client=client,
            respx_mock=respx_mock,
        )

    assert len(seen) == 1
    assert seen[0].type == "mcp_approval_request"


@pytest.mark.respx(base_url=base_url)
def test_action_guard_is_ignored_for_streaming_create(client: OpenAI, respx_mock: MockRouter) -> None:
    seen: list[openai.AgentAction] = []

    def action_guard(action: openai.AgentAction) -> openai.ActionGuardDecision:
        seen.append(action)
        return openai.ActionGuardDecision.BLOCK

    respx_mock.post("/responses").mock(
        return_value=httpx.Response(
            200,
            content=(
                b'data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response",'
                b'"created_at":1754925861,"status":"in_progress","background":false,"error":null,'
                b'"incomplete_details":null,"instructions":null,"max_output_tokens":null,"max_tool_calls":null,'
                b'"model":"gpt-5.4-2026-03-05","output":[],"parallel_tool_calls":true,"previous_response_id":null,'
                b'"prompt_cache_key":null,"reasoning":{"effort":null,"summary":null},"safety_identifier":null,'
                b'"service_tier":"default","store":true,"temperature":1.0,"text":{"format":{"type":"text"},'
                b'"verbosity":"medium"},"tool_choice":"auto","tools":[],"top_logprobs":0,"top_p":1.0,'
                b'"truncation":"disabled","usage":null,"user":null,"metadata":{}}}'
                b"\n\n"
                b'data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":'
                b'{"id":"mcpr_123","type":"mcp_approval_request","arguments":"{}","name":"read_file",'
                b'"server_label":"filesystem"}}'
                b"\n\n"
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    stream = client.responses.create(
        model="gpt-5.4-2026-03-05",
        input="hi",
        stream=True,
        action_guard=action_guard,
    )

    for _ in stream:
        pass

    assert len(seen) == 0


@pytest.mark.respx(base_url=base_url)
def test_stream_method_blocks_action_guard(client: OpenAI, respx_mock: MockRouter) -> None:
    seen: list[openai.AgentAction] = []

    def action_guard(action: openai.AgentAction) -> openai.ActionGuardDecision:
        seen.append(action)
        return openai.ActionGuardDecision.BLOCK

    respx_mock.post("/responses").mock(
        return_value=httpx.Response(
            200,
            content=(
                b'data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response",'
                b'"created_at":1754925861,"status":"in_progress","background":false,"error":null,'
                b'"incomplete_details":null,"instructions":null,"max_output_tokens":null,"max_tool_calls":null,'
                b'"model":"gpt-5.4-2026-03-05","output":[],"parallel_tool_calls":true,"previous_response_id":null,'
                b'"prompt_cache_key":null,"reasoning":{"effort":null,"summary":null},"safety_identifier":null,'
                b'"service_tier":"default","store":true,"temperature":1.0,"text":{"format":{"type":"text"},'
                b'"verbosity":"medium"},"tool_choice":"auto","tools":[],"top_logprobs":0,"top_p":1.0,'
                b'"truncation":"disabled","usage":null,"user":null,"metadata":{}}}'
                b"\n\n"
                b'data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":'
                b'{"id":"mcpr_123","type":"mcp_approval_request","arguments":"{}","name":"read_file",'
                b'"server_label":"filesystem"}}'
                b"\n\n"
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    with pytest.raises(openai.ActionGuardError, match="filesystem"):
        with client.responses.stream(
            model="gpt-5.4-2026-03-05",
            input="hi",
            action_guard=action_guard,
        ) as stream:
            for _ in stream:
                pass

    assert len(seen) == 1
    assert seen[0].type == "mcp_approval_request"


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
def test_stream_method_definition_in_sync(sync: bool, client: OpenAI, async_client: AsyncOpenAI) -> None:
    checking_client: OpenAI | AsyncOpenAI = client if sync else async_client

    assert_signatures_in_sync(
        checking_client.responses.create,
        checking_client.responses.stream,
        exclude_params={"stream", "tools"},
    )


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
def test_parse_method_definition_in_sync(sync: bool, client: OpenAI, async_client: AsyncOpenAI) -> None:
    checking_client: OpenAI | AsyncOpenAI = client if sync else async_client

    assert_signatures_in_sync(
        checking_client.responses.create,
        checking_client.responses.parse,
        exclude_params={"tools"},
    )
