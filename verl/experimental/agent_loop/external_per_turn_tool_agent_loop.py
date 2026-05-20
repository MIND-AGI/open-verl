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
"""External per-turn tool agent loop using OpenAI-compatible API.

The agent function uses a standard ``openai.AsyncOpenAI`` client to talk to
a proxy that transparently captures logprobs, handles partial rollout, and
tracks per-episode state. The agent is completely unaware of verl internals.

Example usage with config:
    actor_rollout_ref.rollout.agent.default_agent_loop=external_per_turn_tool_agent

To use a custom agent function:
    actor_rollout_ref.rollout.agent.external_agent_fn=my_module:my_agent_fn
"""
import asyncio
import importlib
import json
import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from openai import AsyncOpenAI

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.rollout_session import get_or_create_proxy
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.chat_template import initialize_system_prompt

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Process-wide cache of open local trace file handles, keyed by
# ``(service_name, pid)``. ``ExternalPerTurnToolAgentLoop`` is
# re-instantiated for every trajectory (see ``agent_loop.py``
# ``hydra.utils.instantiate``), so without memoization the second and
# subsequent instances in a single Ray worker would open the same
# trace file twice — which HDFS FUSE rejects with EBUSY because it
# cannot hold two concurrent append leases on the same file.
_LOCAL_TRACE_FILES: dict[tuple[str, int], Any] = {}


def _import_agent_fn(dotted_path: str) -> Callable:
    """Import an agent function from a dotted path like 'my_module:my_fn'."""
    module_path, _, fn_name = dotted_path.rpartition(":")
    if not module_path:
        module_path, _, fn_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def _create_logfire_logger(service_name: str):
    """Lazily initialize a logfire logger.

    Returns the ``logfire`` module when ``LOGFIRE_KEY`` is set and
    ``logfire`` imports cleanly, otherwise ``None`` (callers must handle
    the None case as a no-op).
    """
    try:
        token = os.getenv("LOGFIRE_KEY")
        if not token:
            return None
        import logfire

        logfire.configure(
            token=token,
            service_name=service_name,
            service_version="v1.0.0",
            scrubbing=False,
        )
        logger.info(f"Logfire initialized for external agent loop: {service_name}")
        return logfire
    except Exception as e:
        logger.warning(f"Failed to initialize Logfire: {e}")
    return None


def _create_local_trace_file(service_name: str, trace_dir: Optional[str]):
    """Lazily open a JSONL file mirroring logfire events to local disk.

    Each sampled trajectory is appended as a single JSON line to
    ``{trace_dir}/{service_name}_{pid}_{uuid}.jsonl``, carrying the
    same structured data that ``_log_per_turn_trajectory`` sends to
    logfire (span attributes + per-turn prompt/response decoded text
    and token counts). Returns ``None`` when ``trace_dir`` is unset
    (no-op).

    Filenames include a per-process uuid, not just the PID, because
    HDFS FUSE mounts hold an append lease on a file for up to an hour
    after the original writer exits. Ray worker PIDs tend to cluster
    in a narrow range, so successive re-runs of the same experiment
    frequently land on the same ``{service_name}_{pid}.jsonl`` path
    and the new worker's very first ``open(..., "a")`` call gets
    EBUSY because HDFS still thinks the prior run's lease is live.
    A fresh uuid sidesteps this entirely.

    Within a process, asyncio's cooperative scheduling plus Python's
    GIL guarantees line-atomic appends across concurrent coroutines:
    ``file.write`` of a complete line with no interleaved awaits runs
    to completion before another coroutine can observe the file.

    The open handle is memoized in ``_LOCAL_TRACE_FILES`` keyed by
    ``(service_name, pid)`` because ``ExternalPerTurnToolAgentLoop``
    is re-instantiated for every trajectory within a single Ray
    worker process. HDFS FUSE rejects opening two append handles on
    the same file from a single process, so every instance after the
    first must reuse the handle created by the first. The cache key
    is process-scoped (not path-scoped) so the uuid is generated
    exactly once per process.
    """
    if not trace_dir:
        return None
    key = (service_name, os.getpid())
    cached = _LOCAL_TRACE_FILES.get(key)
    if cached is not None and not cached.closed:
        return cached
    try:
        base = Path(trace_dir).expanduser()
        if not base.is_absolute():
            base = Path(os.getcwd()) / base
        base.mkdir(parents=True, exist_ok=True)
        uniq = uuid.uuid4().hex[:8]
        file_path = base / f"{service_name}_{os.getpid()}_{uniq}.jsonl"
        f = open(file_path, "a", buffering=1, encoding="utf-8")
        _LOCAL_TRACE_FILES[key] = f
        logger.info(f"Local trace file opened for external agent loop: {file_path}")
        return f
    except Exception as e:
        logger.warning(f"Failed to open local trace file: {e}")
    return None


def _truncate_response(text: str, max_length: int, truncate_side: str) -> str:
    """Truncate a tool or interaction response string to ``max_length`` chars.

    Mirrors the semantics of ``PerTurnToolAgentLoop``'s tool-response
    truncation. ``max_length`` is measured in characters, which is a
    conservative upper bound on tokens for the prompt_length budget
    assertion elsewhere in this file.
    """
    if len(text) <= max_length:
        return text
    if truncate_side == "left":
        return text[:max_length] + "...(truncated)"
    if truncate_side == "right":
        return "(truncated)..." + text[-max_length:]
    length = max_length // 2
    return text[:length] + "...(truncated)..." + text[-length:]


async def default_tool_agent(
    base_url: str,
    messages: list[dict],
    tools: list[dict],
    max_assistant_turns: int = 16,
    max_user_turns: int = 16,
    execute_tool: Optional[Callable] = None,
    interact: Optional[Callable] = None,
    _metadata: Optional[dict] = None,
    response_length: Optional[int] = None,
    budget: Optional[dict] = None,
    build_prompt_ids: Optional[Callable] = None,
    absorb_response_tokens: Optional[Callable] = None,
    **kwargs,
):
    """Default tool agent that replicates PerTurnToolAgentLoop behavior using OpenAI API.

    This demonstrates how an external agent interacts with verl's training
    infrastructure through a standard OpenAI-compatible API. The agent is
    unaware of logprobs, partial rollout, or any training concerns.

    Because this agent is append-only, it opts into byte-level alignment
    with the reference loop: when the loop class injects ``build_prompt_ids``
    / ``absorb_response_tokens`` closures, we feed pre-tokenized
    ``prompt_ids`` to the session via ``extra_body`` and thread the raw
    response tokens back into the rolling buffer after each call.

    Budget enforcement: this function directly accumulates every token added
    to the response region.

    - After each API call, add ``response.usage.completion_tokens`` (the
      assistant tokens just generated).
    - The ``execute_tool`` and ``interact`` callbacks tokenize the text they
      produce and update the same ``budget`` dict, so tool and interaction
      tokens are credited the moment they're appended to ``messages``.

    ``max_tokens`` on the next call is then simply
    ``response_length - budget["used_response_tokens"]``, and the loop
    terminates with ``terminated_reason="max_response_length"`` when the
    budget is exhausted.

    Args:
        base_url: OpenAI-compatible API endpoint (provided by RolloutSessionProxy).
        messages: Initial conversation messages from the dataset.
        tools: OpenAI tool schemas.
        max_assistant_turns: Maximum number of assistant turns.
        max_user_turns: Maximum number of user/interaction turns.
        execute_tool: Async callback to execute a tool: (name, arguments_str) -> str.
            Tool responses are truncated to ``max_tool_response_length`` and
            their token count is added to ``budget`` inside the callback.
        interact: Async callback for interaction evaluation after non-tool responses:
            (messages) -> (should_terminate, feedback_text). Interaction
            feedback is truncated to ``max_interaction_response_length`` and
            its token count is added to ``budget`` inside the callback.
        _metadata: Optional mutable dict for tracking metadata like terminated_reason.
            Set by the agent loop and also written to by the interact callback.
        response_length: Per-episode response-region token budget.
        budget: Shared mutable dict with key ``used_response_tokens``. The
            agent updates it with assistant tokens; the tool/interaction
            callbacks update it with their respective tokenized output.
        build_prompt_ids: Optional ``async (messages, tools) -> list[int]``.
            When provided, used to pre-tokenize prompts for byte alignment
            with the reference loop (see class docstring above).
        absorb_response_tokens: Optional sync
            ``(response_token_ids, message_count) -> None`` paired with
            ``build_prompt_ids``.
        **kwargs: Additional dataset fields (ignored by default agent).
    """
    client = AsyncOpenAI(base_url=base_url, api_key="dummy")

    assistant_turns = 0
    tool_turns = 0
    interaction_turns = 0

    while True:
        # Call the model via standard OpenAI API
        create_kwargs = {
            "model": "default",
            "messages": messages,
        }
        if tools:
            create_kwargs["tools"] = tools

        # Set ``max_tokens`` for this call from the remaining response-region
        # budget. The session does not clamp; the agent is responsible.
        if response_length is not None and budget is not None:
            remaining = response_length - budget["used_response_tokens"]
            if remaining <= 0:
                if _metadata is not None and _metadata.get("terminated_reason") is None:
                    _metadata["terminated_reason"] = "max_response_length"
                break
            create_kwargs["max_tokens"] = remaining

        # Pre-tokenize via the rolling buffer (if provided) and send
        # prompt_ids directly in extra_body so the session skips its
        # chat-template path. Absent the builder, the session tokenizes
        # ``messages`` via chat template.
        if build_prompt_ids is not None:
            prompt_ids = await build_prompt_ids(messages, tools)
            create_kwargs["extra_body"] = {"prompt_ids": prompt_ids}

        try:
            response = await client.chat.completions.create(**create_kwargs)
        except Exception as e:
            logger.warning(f"Chat completion failed: {e}")
            if _metadata is not None and _metadata.get("terminated_reason") is None:
                _metadata["terminated_reason"] = "api_error"
            break

        # Credit assistant tokens to the shared budget.
        if budget is not None and response.usage is not None:
            budget["used_response_tokens"] += response.usage.completion_tokens

        choice = response.choices[0]
        assistant_message = choice.message.model_dump(exclude_unset=True)

        # Feed raw response tokens back into the rolling buffer. We use
        # ``len(messages)`` (before the append) as the watermark so the
        # next call's ``messages[watermark]`` is exactly the assistant
        # we're about to append below.
        if absorb_response_tokens is not None:
            response_token_ids = getattr(response, "response_token_ids", None)
            if response_token_ids is not None:
                absorb_response_tokens(list(response_token_ids), len(messages))

        messages.append(assistant_message)
        assistant_turns += 1

        # Check termination: assistant turns limit
        if assistant_turns >= max_assistant_turns:
            if _metadata is not None and _metadata.get("terminated_reason") is None:
                _metadata["terminated_reason"] = "max_assistant_turns"
            break

        # Check termination: response length exhausted
        if choice.finish_reason == "length":
            if _metadata is not None and _metadata.get("terminated_reason") is None:
                _metadata["terminated_reason"] = "max_response_length"
            break

        # Handle tool calls
        if choice.message.tool_calls and execute_tool:
            for tool_call in choice.message.tool_calls:
                fn = tool_call.function
                try:
                    tool_result = await execute_tool(fn.name, fn.arguments)
                except Exception as e:
                    tool_result = f"Error executing tool: {e}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )
            tool_turns += 1
        elif choice.finish_reason == "tool_calls":
            # Tool calls present but no execute_tool callback -- stop
            if _metadata is not None and _metadata.get("terminated_reason") is None:
                _metadata["terminated_reason"] = "no_tool_call"
            break
        elif interact:
            # No tool calls, evaluate via interaction (multi-attempt)
            should_terminate, feedback = await interact(messages)
            interaction_turns += 1

            if should_terminate:
                # terminated_reason already set by interact callback
                break

            # Check interaction turns limit
            if interaction_turns >= max_user_turns:
                if _metadata is not None and _metadata.get("terminated_reason") is None:
                    _metadata["terminated_reason"] = "max_interaction_turns"
                break

            # Add feedback as user message for next attempt
            if feedback:
                messages.append({"role": "user", "content": feedback})
        else:
            # No tool calls, generation complete
            if _metadata is not None and _metadata.get("terminated_reason") is None:
                _metadata["terminated_reason"] = "no_tool_call"
            break

    # Write turn counts to metadata so the caller can propagate them to metrics
    if _metadata is not None:
        _metadata["num_tool_turns"] = tool_turns
        _metadata["num_assistant_turns"] = assistant_turns
        _metadata["num_interaction_turns"] = interaction_turns


@register("external_per_turn_tool_agent")
class ExternalPerTurnToolAgentLoop(AgentLoopBase):
    """Per-turn agent loop that delegates to an external agent function using OpenAI API.

    The external agent receives a ``base_url`` and uses a standard ``openai.AsyncOpenAI``
    client. The proxy behind the base_url transparently captures logprobs, handles
    partial rollout, and tracks per-episode state.

    The agent function signature is:
        async def agent_fn(base_url: str, messages: list[dict], **kwargs)

    Config keys:
        rollout.agent.external_agent_fn: dotted path to custom agent function
        rollout.multi_turn.tool_config_path: tool config for default agent
        rollout.multi_turn.max_assistant_turns: max assistant turns
        rollout.multi_turn.max_user_turns: max user turns
        rollout.multi_turn.format: tool call format (hermes, gpt-oss, etc.)
        rollout.multi_turn.interaction_config_path: interaction config for multi-episode
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        self.max_interaction_response_length = self.rollout_config.multi_turn.max_interaction_response_length
        self.interaction_response_truncate_side = (
            self.rollout_config.multi_turn.interaction_response_truncate_side
        )

        # Initialize tool parser for response parsing in the proxy
        self.tool_format = self.rollout_config.multi_turn.format
        self.tool_parser = ToolParser.get_tool_parser(self.tool_format, self.tokenizer) if self.tool_format else None

        # Initialize tools for the default agent function
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

        # Initialize interactions from config file (for multi-episode training)
        interaction_config_path = getattr(self.rollout_config.multi_turn, "interaction_config_path", None)
        if interaction_config_path:
            from verl.interactions.utils.interaction_registry import initialize_interactions_from_config

            self.interaction_map = initialize_interactions_from_config(interaction_config_path)
        else:
            self.interaction_map = {}

        # Load custom agent function or use default
        external_agent_fn = getattr(self.rollout_config.agent, "external_agent_fn", None)
        if external_agent_fn:
            self.agent_fn = _import_agent_fn(external_agent_fn)
        else:
            self.agent_fn = default_tool_agent

        # Slack beyond ``response_length`` for the single non-model response
        # that fires on the last turn, after the assistant has already
        # exhausted the budget. Only ONE non-model response can push past
        # response_length (the one after the terminal assistant turn), so
        # we take the max of a full parallel tool-call fan-out and a single
        # interaction feedback — not the sum, and not turn-multiplied.
        # Measured in characters (conservative upper bound on tokens).
        # The real shape check (``init_prompt_len + response_length + slack
        # <= prompt_length``) lives in ``run()`` because ``init_prompt_len``
        # is only known per sample.
        _tool_slack = max(1, self.max_parallel_calls or 1) * self.max_tool_response_length
        self._single_turn_nonmodel_slack = max(_tool_slack, self.max_interaction_response_length)

        # Optional trajectory logging. Two independent sinks, both sampled
        # at ``VERL_LOGFIRE_SAMPLE_RATE`` (default 0.01):
        # - Logfire (remote): enabled by setting ``LOGFIRE_KEY``.
        # - Local JSONL file: writes to ``{trainer.default_local_dir}/traces``
        #   so trajectory logs live next to the experiment's checkpoints.
        # A single ``random.random()`` roll per trajectory drives both
        # sinks, so when both are enabled they see exactly the same
        # trajectories. When neither sink can open, all logging is a no-op.
        service_name = getattr(
            getattr(self.rollout_config, "trace", None),
            "experiment_name",
            None,
        ) or "verl-external-agent-loop"
        self._trace_service_name = service_name
        self._logfire = _create_logfire_logger(service_name=service_name)
        default_local_dir = getattr(
            getattr(self.config, "trainer", None),
            "default_local_dir",
            None,
        )
        self._local_trace_file = _create_local_trace_file(
            service_name=service_name,
            trace_dir=os.path.join(default_local_dir, "traces") if default_local_dir else None,
        )
        try:
            self._trace_sample_rate = float(os.getenv("VERL_LOGFIRE_SAMPLE_RATE", "0.01"))
        except ValueError:
            self._trace_sample_rate = 0.01

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        messages = list(kwargs["raw_prompt"])

        # Runtime per-sample shape assertion: the initial prompt plus
        # response_length plus a single-turn tool/interaction slack must fit
        # in prompt_length. The session keeps the assistant + non-model
        # tokens within ``response_length`` across all but the terminal
        # turn; only the terminal turn can push the trajectory past that
        # budget by exactly one tool or interaction response.
        initial_prompt_ids = await self.apply_chat_template(messages)
        init_prompt_len = len(initial_prompt_ids)
        worst_case_prompt_len = (
            init_prompt_len + self.response_length + self._single_turn_nonmodel_slack
        )
        if worst_case_prompt_len > self.prompt_length:
            raise ValueError(
                f"Sample initial prompt ({init_prompt_len} tokens) + "
                f"response_length ({self.response_length}) + single-turn "
                f"tool/interaction slack ({self._single_turn_nonmodel_slack}) "
                f"= {worst_case_prompt_len} exceeds prompt_length "
                f"({self.prompt_length}). The sample cannot safely run "
                f"through a full multi-turn episode without overflow."
            )

        # Get or create shared proxy for this worker
        proxy = await get_or_create_proxy(
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            rollout_config=self.rollout_config,
            tool_parser=self.tool_parser,
            tool_schemas=self.tool_schemas,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
            system_prompt=self.system_prompt,
            default_sampling_params=sampling_params,
            tool_format=self.tool_format,
        )

        # Create session for this episode with per-call sampling params
        # (the proxy caches defaults from the first call, which may be validation params)
        session_id, base_url = proxy.create_session(default_sampling_params=sampling_params)

        # Per-trajectory rolling-token state lives in these closures
        # (NOT in the session). Only agents that know they are
        # append-only opt into this by calling ``build_prompt_ids``; on
        # a watermark mismatch (e.g. self-summarize clearing messages)
        # the builder transparently resets via a full re-render.
        rolling_token_buffer: list[int] = []
        rolling_watermark: list[int] = [-1]  # boxed for closure write

        async def build_prompt_ids(
            new_messages: list[dict], new_tools: Optional[list[dict]]
        ) -> list[int]:
            wm = rolling_watermark[0]
            can_incremental = (
                len(rolling_token_buffer) > 0
                and 0 <= wm < len(new_messages)
                and new_messages[wm].get("role") == "assistant"
            )
            if can_incremental:
                tail = new_messages[wm + 1 :]
                if tail:
                    tail_ids = await self.apply_chat_template(tail, remove_system_prompt=True)
                    rolling_token_buffer.extend(tail_ids)
                return list(rolling_token_buffer)
            ids = await self.apply_chat_template(new_messages, tools=new_tools)
            rolling_token_buffer.clear()
            rolling_token_buffer.extend(ids)
            return list(ids)

        def absorb_response_tokens(response_token_ids: list[int], message_count: int) -> None:
            rolling_token_buffer.extend(response_token_ids)
            rolling_watermark[0] = message_count

        try:
            # Track tool rewards across tool executions
            tool_rewards: list[float] = []

            # Track interaction state
            turn_scores: list[float] = []
            metadata: dict[str, Any] = {
                "terminated_reason": None,
                "reward_extra_info": {},
            }

            # Initialize interaction if configured
            interaction = None
            interaction_kwargs: dict[str, Any] = {}
            if self.interaction_map:
                interaction_kwargs = kwargs.get("extra_info", {}).get("interaction_kwargs", {})
                if "name" in interaction_kwargs:
                    interaction_name = interaction_kwargs["name"]
                    if interaction_name not in self.interaction_map:
                        raise ValueError(
                            f"Interaction '{interaction_name}' not found in interaction_map. "
                            f"Available interactions: {list(self.interaction_map.keys())}"
                        )
                    interaction = self.interaction_map[interaction_name]
                    request_id = uuid4().hex
                    await interaction.start_interaction(request_id, **interaction_kwargs)

            # Shared budget counter: the agent function credits assistant
            # tokens after each API call, and the tool/interaction callbacks
            # credit the tokens they add to ``messages`` when they return.
            budget: dict[str, int] = {"used_response_tokens": 0}

            agent_kwargs = {
                "base_url": base_url,
                "messages": messages,
                "tools": self.tool_schemas,
                "max_assistant_turns": self.max_assistant_turns,
                "max_user_turns": self.max_user_turns,
                "execute_tool": self._make_tool_executor(
                    kwargs.get("tools_kwargs", {}), tool_rewards, budget
                ),
                "_metadata": metadata,
                "response_length": self.response_length,
                "budget": budget,
                "build_prompt_ids": build_prompt_ids,
                "absorb_response_tokens": absorb_response_tokens,
            }

            # Add interaction callback if configured
            if interaction:
                agent_kwargs["interact"] = self._make_interaction_callback(
                    interaction, request_id, interaction_kwargs, turn_scores, metadata, budget
                )

            # Subclass hook: extra kwargs for the agent function (e.g.,
            # summarization parameters).
            agent_kwargs.update(self._build_extra_agent_kwargs())

            # Pass through dataset fields
            agent_kwargs.update(kwargs)

            # Delegate to external agent
            await self.agent_fn(**agent_kwargs)

            # Collect per-turn training data from session
            session = proxy.get_session(session_id)
            session._tool_rewards = tool_rewards
            session._turn_scores = turn_scores
            session._extra_output_fields = {
                "reward_extra_info": metadata["reward_extra_info"],
                "terminated_reason": metadata["terminated_reason"],
                **self._build_extra_output_fields(metadata),
            }
            # Propagate turn counts from agent function into session metrics
            for key in ("num_tool_turns", "num_assistant_turns", "num_interaction_turns"):
                if key in metadata:
                    session._metrics[key] = metadata[key]
            outputs = session.to_per_turn_outputs()

            # Randomly sample a fraction of real trajectories and log decoded
            # per-turn prompt/response pairs to enabled trace sinks (logfire
            # and/or a local JSONL file under the experiment's checkpoint dir).
            if (
                outputs
                and (self._logfire is not None or self._local_trace_file is not None)
                and random.random() < self._trace_sample_rate
            ):
                self._log_per_turn_trajectory(outputs, metadata)

            if not outputs:
                logger.warning(f"Session {session_id} produced no outputs — agent may not have called the model")
                # Return a minimal output to avoid downstream errors
                prompt_ids = await self.apply_chat_template(messages)
                outputs = [
                    AgentLoopOutput(
                        prompt_ids=prompt_ids,
                        response_ids=[self.tokenizer.eos_token_id],
                        response_mask=[1],
                        num_turns=1,
                        metrics={"generate_sequences": 0.0, "tool_calls": 0.0, "num_preempted": -1},
                        extra_fields={"turn_scores": [], "tool_rewards": []},
                    )
                ]

            return outputs
        finally:
            proxy.remove_session(session_id)

    def _log_per_turn_trajectory(self, outputs, metadata: dict[str, Any]) -> None:
        """Decode per-turn snapshots and dispatch to enabled trace sinks.

        Both sinks see the exact same structured data: span-level
        attributes (``num_turns``, ``terminated_reason``, tool/interaction/
        summarization counts) and a nested list of per-turn events with
        decoded prompt/response text and token counts. Logfire sees it as
        a span with child info events; the local file sees it as one JSON
        object per line with a nested ``turns`` array. Decoding happens
        once and is shared across sinks so the two representations cannot
        drift.
        """
        decode_kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}

        turns: list[dict[str, Any]] = []
        for turn_id, output in enumerate(outputs):
            try:
                prompt_text = self.tokenizer.decode(output.prompt_ids, **decode_kwargs)
                response_text = self.tokenizer.decode(output.response_ids, **decode_kwargs)
            except Exception as e:
                logger.warning(f"per-turn decode failed at turn {turn_id}: {e}")
                continue
            turns.append(
                {
                    "turn_number": turn_id,
                    "prompt": prompt_text,
                    "response": response_text,
                    "prompt_tokens": len(output.prompt_ids),
                    "response_tokens": len(output.response_ids),
                }
            )

        span_attrs = {
            "num_turns": len(outputs),
            "terminated_reason": metadata.get("terminated_reason") or "null",
            "num_tool_turns": metadata.get("num_tool_turns"),
            "num_interaction_turns": metadata.get("num_interaction_turns"),
            "num_summarizations": metadata.get("num_summarizations"),
        }

        if self._logfire is not None:
            with self._logfire.span(
                "external_per_turn — {num_turns} turns — {terminated_reason}",
                **span_attrs,
            ):
                for turn in turns:
                    self._logfire.info("🔄 Turn {turn_number}", **turn)

        if self._local_trace_file is not None:
            try:
                record = {
                    "timestamp": time.time(),
                    "service_name": self._trace_service_name,
                    **span_attrs,
                    "turns": turns,
                }
                self._local_trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write local trace record: {e}")

    def _build_extra_agent_kwargs(self) -> dict[str, Any]:
        """Hook for subclasses to add extra kwargs passed to the agent function.

        Default: no extra kwargs. Subclasses (e.g., the self-summarize loop)
        override this to inject configuration that only applies to their
        variant.
        """
        return {}

    def _build_extra_output_fields(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Hook for subclasses to add extra fields to ``session._extra_output_fields``.

        Default: no extra fields. Subclasses override to surface variant-
        specific data (e.g., ``num_summarizations`` for the self-summarize loop).
        """
        return {}

    def _make_tool_executor(
        self, tools_kwargs: dict, tool_rewards: list[float], budget: dict
    ) -> Callable:
        """Create an async tool execution callback for the agent function.

        The callback truncates tool output to ``max_tool_response_length`` and
        credits the tokenized length of the (possibly-truncated) result to
        ``budget["used_response_tokens"]`` before returning.

        Args:
            tools_kwargs: Per-tool kwargs from the dataset.
            tool_rewards: Mutable list to accumulate tool rewards.
            budget: Shared mutable dict tracking response-region token usage.
        """

        async def execute_tool(name: str, arguments_str: str) -> str:
            """Execute a tool by name with JSON arguments string."""
            tool = self.tools.get(name)
            if tool is None:
                result = f"Unknown tool: {name}"
                budget["used_response_tokens"] += len(
                    self.tokenizer.encode(result, add_special_tokens=False)
                )
                return result

            instance_id = None
            try:
                args = json.loads(arguments_str)
                kwargs = tools_kwargs.get(name, {})
                instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
                tool_response, tool_reward, _ = await tool.execute(instance_id, args)

                if tool_reward is not None:
                    tool_rewards.append(tool_reward)

                result = _truncate_response(
                    tool_response.text or "",
                    self.max_tool_response_length,
                    self.tool_response_truncate_side,
                )
            except Exception as e:
                logger.warning(f"Error executing tool {name}: {e}")
                result = f"Error executing tool: {e}"
            finally:
                if tool and instance_id:
                    await tool.release(instance_id)

            budget["used_response_tokens"] += len(
                self.tokenizer.encode(result, add_special_tokens=False)
            )
            return result

        return execute_tool

    def _make_interaction_callback(
        self,
        interaction,
        request_id: str,
        interaction_kwargs: dict,
        turn_scores: list[float],
        metadata: dict[str, Any],
        budget: dict,
    ) -> Callable:
        """Create an async interaction callback for multi-episode evaluation.

        The callback evaluates the model's response using the configured
        interaction (e.g., math_reward checks if the boxed answer is correct),
        truncates the feedback to ``max_interaction_response_length``, and
        credits the tokenized length to ``budget["used_response_tokens"]``
        before returning.

        Args:
            interaction: BaseInteraction instance (e.g., MultiTurnMathInteraction).
            request_id: Unique identifier for this episode.
            interaction_kwargs: Kwargs passed to interaction.generate_response (e.g., ground_truth).
            turn_scores: Mutable list to accumulate per-turn scores.
            metadata: Mutable dict for terminated_reason and reward_extra_info.
            budget: Shared mutable dict tracking response-region token usage.
        """

        async def interact(messages: list[dict]) -> tuple[bool, str]:
            """Evaluate the model's answer and return (should_terminate, feedback_text)."""
            t0 = time.monotonic()
            should_terminate, response, score, metrics = await interaction.generate_response(
                request_id, messages, **interaction_kwargs
            )
            metadata.setdefault("compute_rewards_time", 0.0)
            metadata["compute_rewards_time"] += time.monotonic() - t0

            if score is not None:
                turn_scores.append(score)
                metadata["reward_extra_info"] = metrics

            if should_terminate:
                metadata["terminated_reason"] = "correct_answer"

            if response:
                response = _truncate_response(
                    response,
                    self.max_interaction_response_length,
                    self.interaction_response_truncate_side,
                )
                budget["used_response_tokens"] += len(
                    self.tokenizer.encode(response, add_special_tokens=False)
                )

            return should_terminate, response

        return interact
