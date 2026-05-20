# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rollout session and proxy for external agent loops.

Provides an OpenAI-compatible HTTP API that transparently captures logprobs,
handles partial rollout, and tracks per-episode state for training data assembly.

Architecture:
    ExternalAgent (standard OpenAI client)
        ↕  HTTP (OpenAI chat/completions)
    RolloutSessionProxy (per-worker aiohttp server)
        → routes by session_id in URL path
        → delegates to RolloutSession
    RolloutSession (per-turn state)
        → tokenize messages via chat template
        → server_manager.generate() (with logprobs, partial rollout)
        → decode → OpenAI response format
        → logs per-turn snapshots: (prompt_ids, response_ids) pairs
"""
import json
import logging
import os
import time
from typing import Any, Optional
from uuid import uuid4

from aiohttp import web

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AsyncLLMServerManager,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.chat_template import apply_chat_template
from verl.utils.ray_utils import get_event_loop
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RolloutSession:
    """Per-turn session that logs (prompt_ids, response_ids) for each generation.

    Each call to handle_chat_completion tokenizes the full messages, generates a
    response, and logs a per-turn snapshot. The session does not accumulate state
    across turns — every turn is processed the same way.
    """

    def __init__(
        self,
        session_id: str,
        server_manager: AsyncLLMServerManager,
        tokenizer,
        processor,
        rollout_config,
        tool_parser: Optional[ToolParser],
        tool_schemas: list[dict],
        apply_chat_template_kwargs: dict,
        system_prompt: list[int],
        default_sampling_params: dict,
        tool_format: Optional[str] = None,
    ):
        self.session_id = session_id
        self.request_id = uuid4().hex  # For sticky routing / prefix caching
        self._server_manager = server_manager
        self._tokenizer = tokenizer
        self._processor = processor
        self._rollout_config = rollout_config
        self._tool_parser = tool_parser
        self._tool_schemas = tool_schemas
        self._apply_chat_template_kwargs = apply_chat_template_kwargs
        self._system_prompt = system_prompt
        self._default_sampling_params = default_sampling_params
        self._tool_format = tool_format
        self._loop = get_event_loop()

        # Per-turn snapshots — the core output
        self._per_turn_snapshots: list[AgentLoopOutput] = []

        # Trajectory-level state
        self._extra_fields: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {"generate_sequences": 0.0, "tool_calls": 0.0}
        self._assistant_turns: int = 0
        self._user_turns: int = 0

        # Tool schema parsing (for tool call extraction)
        self._tool_schemas_parsed: Optional[list[OpenAIFunctionToolSchema]] = None
        if self._tool_schemas:
            try:
                self._tool_schemas_parsed = [OpenAIFunctionToolSchema(**s) for s in self._tool_schemas]
            except Exception:
                self._tool_schemas_parsed = None

        # External metadata (set by agent loop after completion)
        self._tool_rewards: list[float] = []
        self._turn_scores: list[float] = []
        self._extra_output_fields: dict[str, Any] = {}

    async def handle_chat_completion(self, body: dict) -> dict:
        """Handle an OpenAI chat/completions request.

        The session is a dumb tokenize-and-generate service. Budget tracking,
        max_tokens computation, and termination decisions belong to the agent
        loop. The session's only invariants are:

        1. Tokenize the full messages to ``prompt_ids``, or use
           ``body["prompt_ids"]`` verbatim when the caller has pre-tokenized.
        2. Raise if ``len(prompt_ids) > prompt_length``.
        3. Raise if ``max_tokens <= 0`` — the agent is responsible for
           breaking out of its loop before making a request with no budget.
        4. Otherwise, generate with the requested sampling params and log a
           per-turn snapshot.
        5. Report ``prompt_tokens`` / ``completion_tokens`` in the response
           so the agent can reconcile its budget state from ``response.usage``.
           The response also includes ``response_token_ids`` for callers that
           maintain their own rolling buffer.
        """
        messages = body["messages"]
        tools_from_request = body.get("tools")
        prompt_ids_from_request = body.get("prompt_ids")

        # Build sampling params from request + defaults
        sampling_params = dict(self._default_sampling_params)
        for key in ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens"):
            if key in body:
                sampling_params[key] = body[key]

        # --- Tokenize full messages ---
        # Callers that pre-tokenize themselves (to match the reference
        # per-turn loop's raw-token concatenation byte-for-byte) can pass
        # ``prompt_ids`` directly in the request body and skip this path.
        if prompt_ids_from_request is not None:
            prompt_ids = list(prompt_ids_from_request)
        else:
            tools_for_template = tools_from_request or self._tool_schemas or None
            prompt_ids = await self._apply_chat_template(messages, tools=tools_for_template)

        # Hard error on prompt overflow. The agent loop is responsible for
        # truncating tool/interaction responses (via max_tool_response_length
        # / max_interaction_response_length) and sizing prompt_length to cover
        # init_prompt + response_length + one-turn tool/interaction slack.
        # If this fires, the configuration is inconsistent and silently
        # accepting it would produce truncated training data.
        if len(prompt_ids) > self._rollout_config.prompt_length:
            raise RuntimeError(
                f"Session {self.session_id}: tokenized prompt length "
                f"{len(prompt_ids)} exceeds configured prompt_length "
                f"{self._rollout_config.prompt_length}. Check "
                f"max_tool_response_length, max_interaction_response_length, "
                f"and the prompt_length/response_length budget."
            )

        # Hard error on non-positive max_tokens. The agent is responsible for
        # checking its own remaining budget and terminating the loop before
        # making a request with no budget to spend. Reaching this branch
        # indicates an agent-side bug.
        requested_max_tokens = sampling_params.get("max_tokens")
        if requested_max_tokens is not None and requested_max_tokens <= 0:
            raise RuntimeError(
                f"Session {self.session_id}: received max_tokens="
                f"{requested_max_tokens} which is non-positive. The agent "
                f"must break out of its loop before requesting a generation "
                f"with an exhausted budget."
            )

        # --- Generate ---
        t0 = time.monotonic()
        output: TokenOutput = await self._server_manager.generate(
            request_id=self.request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        self._metrics["generate_sequences"] += time.monotonic() - t0

        # Track preemption metrics
        if self._metrics.get("num_preempted") is None:
            self._metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            self._metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        # Track weight version
        if not self._extra_fields:
            self._extra_fields.update(output.extra_fields)
        else:
            max_global_steps = output.extra_fields.get("max_global_steps")
            if max_global_steps:
                self._extra_fields["max_global_steps"] = max_global_steps

        # --- Log per-turn snapshot ---
        response_ids = list(output.token_ids)
        self._per_turn_snapshots.append(
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=[1] * len(response_ids),
                response_logprobs=list(output.log_probs) if output.log_probs else None,
                num_turns=self._user_turns + self._assistant_turns + 1,
                metrics=AgentLoopMetrics(**self._metrics),
                routed_experts=output.routed_experts,
                extra_fields=dict(self._extra_fields),
            )
        )
        self._assistant_turns += 1

        # --- Decode and parse tool calls ---
        response_text = await self._loop.run_in_executor(
            None, lambda ids=response_ids: self._tokenizer.decode(ids, skip_special_tokens=True)
        )

        tool_calls_openai = None
        finish_reason = "stop"
        if output.stop_reason == "length" or output.stop_reason == "aborted":
            finish_reason = "length"

        if self._tool_parser:
            content, parsed_calls = await self._tool_parser.extract_tool_calls(
                response_ids, self._tool_schemas_parsed
            )
            if parsed_calls:
                tool_calls_openai = []
                for tc in parsed_calls:
                    tool_calls_openai.append(
                        {
                            "id": f"call_{uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                    )
                finish_reason = "tool_calls"
                response_text = content  # Use content without tool call tags

        message: dict[str, Any] = {"role": "assistant", "content": response_text}
        if tool_calls_openai:
            message["tool_calls"] = tool_calls_openai
        return {
            "id": f"chatcmpl-{uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": "default",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(response_ids),
                "total_tokens": len(prompt_ids) + len(response_ids),
            },
            # Raw model output tokens. The OpenAI ChatCompletion pydantic
            # model uses ``extra='allow'``, so clients can read this as
            # ``response.response_token_ids``. Callers that maintain their
            # own rolling buffer feed these back in to match the reference
            # per-turn loop byte-for-byte.
            "response_token_ids": list(response_ids),
        }

    def _normalize_tool_call_arguments(self, messages: list[dict]) -> list[dict]:
        """Parse assistant tool_call arguments from JSON string to dict for XML-style templates.

        The Qwen3.5 / qwen3_coder chat template iterates
        ``tool_call.arguments|items`` (see ``chat_template.jinja`` line 120) and
        raises ``TypeError: Can only get item pairs from a mapping`` when
        ``arguments`` is the OpenAI-spec JSON *string*. The OpenAI client on the
        agent side always sends arguments as a string, so the proxy has to
        parse it back into a dict before handing the messages to the template.
        Hermes-style templates embed the full tool_call as JSON and don't need
        this, so only apply when the configured tool format is the XML variant.
        """
        if self._tool_format != "qwen3_coder":
            return messages
        normalized: list[dict] = []
        for msg in messages:
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                normalized.append(msg)
                continue
            new_tool_calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except Exception:
                        parsed = args
                    fn = {**fn, "arguments": parsed}
                new_tool_calls.append({**tc, "function": fn})
            normalized.append({**msg, "tool_calls": new_tool_calls})
        return normalized

    async def _apply_chat_template(
        self, messages: list[dict], tools: list[dict] = None, remove_system_prompt: bool = False
    ) -> list[int]:
        """Apply chat template to messages, reusing the same logic as AgentLoopBase."""
        messages = self._normalize_tool_call_arguments(messages)
        if self._processor is not None:
            raw_prompt = await self._loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self._processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self._apply_chat_template_kwargs,
                ),
            )
            model_inputs = self._processor(
                text=[raw_prompt],
                return_tensors="pt",
                do_sample_frames=False,
            )
            prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        else:
            tokenized = await self._loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self._tokenizer,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self._apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self._system_prompt) :]

        return prompt_ids

    def to_per_turn_outputs(self) -> list[AgentLoopOutput]:
        """Get per-turn training data (one AgentLoopOutput per assistant turn).

        Asserts that every snapshot fits the configured prompt_length /
        response_length — silent truncation is not allowed.
        """
        prompt_length = self._rollout_config.prompt_length
        response_length = self._rollout_config.response_length
        for i, output in enumerate(self._per_turn_snapshots):
            if len(output.prompt_ids) > prompt_length:
                raise RuntimeError(
                    f"Session {self.session_id}: per-turn snapshot {i} prompt_ids "
                    f"length {len(output.prompt_ids)} exceeds prompt_length "
                    f"{prompt_length}. This indicates tool/interaction output "
                    f"exceeded the configured budget."
                )
            if len(output.response_ids) > response_length:
                raise RuntimeError(
                    f"Session {self.session_id}: per-turn snapshot {i} response_ids "
                    f"length {len(output.response_ids)} exceeds response_length "
                    f"{response_length}."
                )
            output.num_turns = self._user_turns + self._assistant_turns + 1
            output.metrics = AgentLoopMetrics(**self._metrics)
            output.extra_fields.update(
                {
                    "turn_scores": self._turn_scores,
                    "tool_rewards": self._tool_rewards,
                }
            )
            output.extra_fields.update(self._extra_output_fields)
        return self._per_turn_snapshots


class RolloutSessionProxy:
    """Per-worker HTTP proxy that serves OpenAI-compatible API with session tracking.

    One proxy per AgentLoopWorker, shared across all concurrent episodes.
    Each session gets a unique base_url via path-based routing:
        http://localhost:{port}/sessions/{session_id}/v1/chat/completions

    The external agent uses this base_url with a standard OpenAI client and
    is completely unaware of verl internals.
    """

    def __init__(
        self,
        server_manager: AsyncLLMServerManager,
        tokenizer,
        processor,
        rollout_config,
        tool_parser: Optional[ToolParser],
        tool_schemas: list[dict],
        apply_chat_template_kwargs: dict,
        system_prompt: list[int],
        default_sampling_params: dict,
        tool_format: Optional[str] = None,
    ):
        self._server_manager = server_manager
        self._tokenizer = tokenizer
        self._processor = processor
        self._rollout_config = rollout_config
        self._tool_parser = tool_parser
        self._tool_schemas = tool_schemas
        self._apply_chat_template_kwargs = apply_chat_template_kwargs
        self._system_prompt = system_prompt
        self._default_sampling_params = default_sampling_params
        self._tool_format = tool_format

        self._sessions: dict[str, RolloutSession] = {}
        self._port: Optional[int] = None
        self._runner: Optional[web.AppRunner] = None

    async def start(self):
        """Start the proxy on an ephemeral port."""
        app = web.Application()
        app.router.add_post("/sessions/{session_id}/v1/chat/completions", self._handle_chat_completions)
        # Also support /models endpoint for OpenAI client compatibility
        app.router.add_get("/sessions/{session_id}/v1/models", self._handle_models)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 0)
        await site.start()
        # Get the ephemeral port from the socket
        for sock in site._server.sockets:
            self._port = sock.getsockname()[1]
            break
        self._runner = runner
        logger.info(f"RolloutSessionProxy started on port {self._port}")

    async def stop(self):
        """Stop the proxy."""
        if self._runner:
            await self._runner.cleanup()

    @property
    def port(self) -> int:
        assert self._port is not None, "Proxy not started"
        return self._port

    def create_session(self, default_sampling_params: Optional[dict] = None) -> tuple[str, str]:
        """Create a new session and return (session_id, base_url).

        Args:
            default_sampling_params: Per-session sampling params override.
                If provided, the session uses these instead of the proxy's cached defaults.
                This is important because sampling params differ between training
                (e.g., top_p=1.0) and validation (e.g., top_p=0.6), and the proxy
                is cached across both.
        """
        session_id = uuid4().hex
        session = RolloutSession(
            session_id=session_id,
            server_manager=self._server_manager,
            tokenizer=self._tokenizer,
            processor=self._processor,
            rollout_config=self._rollout_config,
            tool_parser=self._tool_parser,
            tool_schemas=self._tool_schemas,
            apply_chat_template_kwargs=self._apply_chat_template_kwargs,
            system_prompt=self._system_prompt,
            default_sampling_params=default_sampling_params if default_sampling_params is not None else self._default_sampling_params,
            tool_format=self._tool_format,
        )
        self._sessions[session_id] = session
        base_url = f"http://localhost:{self._port}/sessions/{session_id}/v1"
        return session_id, base_url

    def get_session(self, session_id: str) -> RolloutSession:
        """Get session by ID."""
        return self._sessions[session_id]

    def remove_session(self, session_id: str):
        """Remove session after episode completes."""
        self._sessions.pop(session_id, None)

    async def _handle_chat_completions(self, request: web.Request) -> web.Response:
        """Handle POST /sessions/{session_id}/v1/chat/completions."""
        session_id = request.match_info["session_id"]
        if session_id not in self._sessions:
            return web.json_response({"error": {"message": f"Session {session_id} not found"}}, status=404)

        session = self._sessions[session_id]
        try:
            body = await request.json()
            response = await session.handle_chat_completion(body)
            return web.json_response(response)
        except Exception as e:
            logger.exception(f"Error handling chat completion for session {session_id}")
            return web.json_response({"error": {"message": str(e)}}, status=500)

    async def _handle_models(self, request: web.Request) -> web.Response:
        """Handle GET /sessions/{session_id}/v1/models for client compatibility."""
        return web.json_response(
            {
                "data": [{"id": "default", "object": "model"}],
                "object": "list",
            }
        )


# Module-level cache: one proxy per server_manager instance
_proxy_cache: dict[int, RolloutSessionProxy] = {}


async def get_or_create_proxy(
    server_manager: AsyncLLMServerManager,
    tokenizer,
    processor,
    rollout_config,
    tool_parser: Optional[ToolParser],
    tool_schemas: list[dict],
    apply_chat_template_kwargs: dict,
    system_prompt: list[int],
    default_sampling_params: dict,
    tool_format: Optional[str] = None,
) -> RolloutSessionProxy:
    """Get or create a shared proxy for the given server_manager."""
    key = id(server_manager)
    if key not in _proxy_cache:
        proxy = RolloutSessionProxy(
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=processor,
            rollout_config=rollout_config,
            tool_parser=tool_parser,
            tool_schemas=tool_schemas,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            system_prompt=system_prompt,
            default_sampling_params=default_sampling_params,
            tool_format=tool_format,
        )
        await proxy.start()
        _proxy_cache[key] = proxy
    return _proxy_cache[key]
