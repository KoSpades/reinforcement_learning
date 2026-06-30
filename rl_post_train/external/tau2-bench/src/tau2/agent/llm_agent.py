import json
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Generic, List, Optional, TypeVar

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import (
    HalfDuplexAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import Action, Task
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


class LLMAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


LLMAgentStateType = TypeVar("LLMAgentStateType", bound="LLMAgentState")


class LLMAgent(
    LLMConfigMixin, HalfDuplexAgent[LLMAgentStateType], Generic[LLMAgentStateType]
):
    """
    A half-duplex LLM agent for turn-based conversations.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the LLMAgent.
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentStateType:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentStateType
    ) -> tuple[AssistantMessage, LLMAgentStateType]:
        """
        Respond to a user or tool message.
        """
        assistant_message = self._generate_next_message(message, state)
        state.messages.append(assistant_message)
        return assistant_message, state

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentStateType
    ) -> AssistantMessage:
        """
        Generate the next message from a user or tool message.
        """
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio. Use VoiceLLMAgent instead.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )
        return assistant_message


AGENT_GT_INSTRUCTION = """
You are testing that our user simulator is working correctly.
User simulator will have an issue for you to solve.
You must behave according to the <policy> provided below.
To make following the policy easier, we give you the list of resolution steps you are expected to take.
These steps involve either taking an action or asking the user to take an action.

In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT_GT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<resolution_steps>
{resolution_steps}
</resolution_steps>
""".strip()


class LLMGTAgent(
    LLMConfigMixin, HalfDuplexAgent[LLMAgentStateType], Generic[LLMAgentStateType]
):
    """
    A GroundTruth agent that can be used to solve a task.
    This agent will receive the expected actions.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: str,
        llm_args: Optional[dict] = None,
        provide_function_args: bool = True,
    ):
        """
        Initialize the LLMAgent.
        If provide_function_args is True, the resolution steps will include the function arguments.
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        assert self.check_valid_task(task), (
            f"Task {task.id} is not valid. Cannot run GT agent."
        )
        self.task = task
        self.provide_function_args = provide_function_args

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        """
        Check if the task is valid.
        Only the tasks that require at least one action are valid.
        """
        if task.evaluation_criteria is None:
            return False
        expected_actions = task.evaluation_criteria.actions or []
        if len(expected_actions) == 0:
            return False
        return True

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_GT.format(
            agent_instruction=AGENT_GT_INSTRUCTION,
            domain_policy=self.domain_policy,
            resolution_steps=self.make_agent_instructions_from_actions(),
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentStateType:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentStateType
    ) -> tuple[AssistantMessage, LLMAgentStateType]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_gt_response",
            **self.llm_args,
        )
        state.messages.append(assistant_message)
        return assistant_message, state

    def make_agent_instructions_from_actions(self) -> str:
        """
        Make agent instructions from a list of actions
        """
        lines = []
        for i, action in enumerate(self.task.evaluation_criteria.actions):
            lines.append(
                f"[Step {i + 1}] {self.make_agent_instructions_from_action(action=action, include_function_args=self.provide_function_args)}"
            )
        return "\n".join(lines)

    @classmethod
    def make_agent_instructions_from_action(
        cls, action: Action, include_function_args: bool = False
    ) -> str:
        """
        Make agent instructions from an action.
        If the action is a user action, returns instructions for the agent to give to the user.
        If the action is an agent action, returns instructions for the agent to perform the action.
        """
        if action.requestor == "user":
            if include_function_args:
                return f"Instruct the user to perform the following action: {action.get_func_format()}."
            else:
                return f"User action: {action.name}."
        elif action.requestor == "assistant":
            if include_function_args:
                return f"Perform the following action: {action.get_func_format()}."
            else:
                return f"Assistant action: {action.name}."
        else:
            raise ValueError(f"Unknown action requestor: {action.requestor}")


AGENT_SOLO_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
You will be provided with a ticket that contains the user's request.
You will need to plan and call the appropriate tools to solve the ticket.

You cannot communicate with the user, only make tool calls.
Stop when you consider that you have solved the ticket.
To do so, send a message containing a single tool call to the `{stop_function_name}` tool. Do not include any other tool calls in this last message.

Always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT_SOLO = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<ticket>
{ticket}
</ticket>
""".strip()


class LLMSoloAgent(
    LLMConfigMixin, HalfDuplexAgent[LLMAgentStateType], Generic[LLMAgentStateType]
):
    """
    An LLM agent that can be used to solve a task without any interaction with the customer.
    The task need to specify a ticket format.
    """

    STOP_FUNCTION_NAME = "done"
    TRANSFER_TOOL_NAME = "transfer_to_human_agents"
    STOP_TOKEN = "###STOP###"

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the LLMAgent.
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        assert self.check_valid_task(task), (
            f"Task {task.id} is not valid. Cannot run GT agent."
        )
        self.task = task
        self.add_stop_tool()
        self.validate_tools()

    def add_stop_tool(self) -> None:
        """Add the stop tool to the tools."""

        def done() -> str:
            """Call this function when you are done with the task."""
            return self.STOP_TOKEN

        self.tools.append(as_tool(done))

    def validate_tools(self) -> None:
        """Check if the tools are valid."""
        tool_names = {tool.name for tool in self.tools}
        if self.TRANSFER_TOOL_NAME not in tool_names:
            logger.warning(
                f"Tool {self.TRANSFER_TOOL_NAME} not found in tools. This tool is required for the agent to transfer the user to a human agent."
            )
        if self.STOP_FUNCTION_NAME not in tool_names:
            raise ValueError(f"Tool {self.STOP_FUNCTION_NAME} not found in tools.")

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        """
        Check if the task is valid.
        Task should contain a ticket and evaluation criteria.
        If the task contains an initial state, the message history should only contain tool calls and responses.
        """
        if task.initial_state is not None:
            message_history = task.initial_state.message_history or []
            for message in message_history:
                if isinstance(message, UserMessage):
                    return False
                if isinstance(message, AssistantMessage) and not message.is_tool_call():
                    return False
            return True
        if task.ticket is None:
            return False
        if task.evaluation_criteria is None:
            return False
        expected_actions = task.evaluation_criteria.actions or []
        if len(expected_actions) == 0:
            return False
        return True

    @property
    def system_prompt(self) -> str:
        agent_instruction = AGENT_SOLO_INSTRUCTION.format(
            stop_function_name=self.STOP_FUNCTION_NAME,
            stop_token=self.STOP_TOKEN,
        )
        return SYSTEM_PROMPT_SOLO.format(
            agent_instruction=agent_instruction,
            domain_policy=self.domain_policy,
            ticket=self.task.ticket,
        )

    def _check_if_stop_toolcall(self, message: AssistantMessage) -> AssistantMessage:
        """Check if the message is a stop message.
        If the message contains a tool call with the name STOP_FUNCTION_NAME, then the message is a stop message.
        """
        is_stop = False
        for tool_call in message.tool_calls:
            if tool_call.name == self.STOP_FUNCTION_NAME:
                is_stop = True
                break
        if is_stop:
            message.content = self.STOP_TOKEN
            message.tool_calls = None
        return message

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        """Check if the message is a stop message."""
        if message.content is None:
            return False
        return cls.STOP_TOKEN in message.content

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentStateType:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: Optional[ValidAgentInputMessage], state: LLMAgentStateType
    ) -> tuple[AssistantMessage, LLMAgentStateType]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, UserMessage):
            raise ValueError("LLMSoloAgent does not support user messages.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is None:
            assert len(state.messages) == 0, "Message history should be empty"
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            tool_choice="required",
            call_name="agent_solo_response",
            **self.llm_args,
        )
        if not assistant_message.is_tool_call():
            raise ValueError("LLMSoloAgent only supports tool calls.")
        message = self._check_if_stop_toolcall(assistant_message)
        state.messages.append(assistant_message)
        return assistant_message, state


# =============================================================================
# GUARDED LLM AGENT
# =============================================================================
#
# A drop-in replacement for LLMAgent that keeps the same proposer (the existing
# `generate()` call, with the full policy in context and all tools available)
# but wraps it in a chain of ordered guards. Each guard inspects the candidate
# action and returns one of three verdicts:
#
#   - approve         : pass the candidate through unchanged.
#   - revise(reason)  : reject and bounce the candidate back to the proposer with
#                       the critique appended, for a bounded number of retries.
#   - replace(action) : deterministically swap the candidate for a different
#                       AssistantMessage.
#
# The pipeline still emits exactly one AssistantMessage per turn (the half-duplex
# contract). Multi-step tool sequences happen across multiple calls to
# `generate_next_message`; the guards just sit on each call.
#
# Two guards address the two core failure modes diagnosed across the airline
# tasks (see diary.md):
#
#   1. PolicyGuard (verify-and-revise, mandatory citation) attacks non-adherence
#      and fabrication: the proposer often reasons correctly then acts against
#      its own reasoning, or invents coverage that isn't in the policy. The guard
#      is a *grounded* verification call: it must quote the policy clause that
#      permits a write/cancel/transfer, or it returns a violation. There is no
#      clause to quote for an invented rule, so forcing a citation kills
#      fabrication. This is post-hoc verification, precisely because the model
#      ignores policy it already has in context.
#
#   2. ConfirmationGuard (mostly deterministic, replace-not-revise) attacks the
#      "wrote to the DB without explicit user confirmation" failure mode. Read vs
#      write is a static lookup (no LLM call for reads). For an unconfirmed write
#      it stores a normalized signature of the action and replaces the candidate
#      with a confirmation message listing the details. It approves the write
#      only when the re-emitted action's signature matches what was confirmed and
#      the latest user turn is an affirmation.
#
# Guard order is policy-before-confirmation: don't bother confirming an action
# that is already illegal.


class GuardDecision(str, Enum):
    """The three verdicts a guard can return."""

    APPROVE = "approve"
    REVISE = "revise"
    REPLACE = "replace"


@dataclass
class Verdict:
    """A guard's verdict on a candidate action."""

    decision: GuardDecision
    reason: Optional[str] = None  # populated for REVISE
    replacement: Optional[AssistantMessage] = None  # populated for REPLACE

    @classmethod
    def approve(cls) -> "Verdict":
        return cls(GuardDecision.APPROVE)

    @classmethod
    def revise(cls, reason: str) -> "Verdict":
        return cls(GuardDecision.REVISE, reason=reason)

    @classmethod
    def replace(cls, action: AssistantMessage) -> "Verdict":
        return cls(GuardDecision.REPLACE, replacement=action)


@dataclass
class GuardContext:
    """Everything a guard needs to review a candidate, without touching the
    canonical persisted transcript directly for scratch work."""

    state: "GuardedLLMAgentState"
    latest_user_message: Optional[UserMessage]
    transcript: list  # read-only view of state.messages
    domain_policy: str
    tools: List[Tool]
    llm: str
    llm_args: dict


class GuardedLLMAgentState(LLMAgentState):
    """LLMAgentState plus the stateful bookkeeping the confirmation guard needs.

    `pending_action` holds the normalized signature (a sorted list of
    "tool_name::canonical_args" strings) of a write the agent has surfaced to the
    user and is waiting for confirmation on. `pending_tool_calls` keeps the exact
    already-vetted tool calls so an affirmative user reply can deterministically
    execute what was confirmed, even if the proposer emits text instead of
    reconstructing the call.
    """

    pending_action: Optional[list] = None
    pending_tool_calls: Optional[list[ToolCall]] = None


# Tools that mutate the backend and therefore require explicit user confirmation
# *and* a policy citation before they may be executed. Kept as a class attribute
# on the agent so other domains can plug in their own write set.
AIRLINE_WRITE_TOOLS = frozenset(
    {
        "book_reservation",
        "cancel_reservation",
        "send_certificate",
        "update_reservation_baggages",
        "update_reservation_passengers",
        "update_reservation_flights",
    }
)

# Conservative defaults for all registered domains. Anything that is not clearly
# read-only is treated as requiring policy verification and user confirmation.
READ_ONLY_TOOL_PREFIXES = (
    "get_",
    "list_",
    "find_",
    "calculate",
    "assert_",
    "query_",
)
READ_ONLY_TOOL_NAMES = frozenset(
    {
        "get_current_time",
        "get_details_by_id",
        "get_id",
    }
)

_AFFIRMATIONS = (
    "yes",
    "yep",
    "yeah",
    "yup",
    "sure",
    "ok",
    "okay",
    "confirm",
    "confirmed",
    "go ahead",
    "proceed",
    "sounds good",
    "do it",
    "please do",
    "go for it",
    "that works",
    "looks good",
    "correct",
)
_NEGATIONS = (
    "no",
    "don't",
    "do not",
    "wait",
    "stop",
    "hold on",
    "not yet",
    "cancel that",
    "nevermind",
    "never mind",
    "actually",
)
_CHANGE_MARKERS = (
    " but ",
    " instead",
    " change",
    " different",
    " update",
    " rather",
)


def _is_consequential(tool_call: ToolCall, write_tools: frozenset) -> bool:
    """A tool call is consequential if it writes to the DB or transfers to a
    human. These are the actions the policy guard must verify."""
    return tool_call.name in write_tools or tool_call.name.startswith("transfer_to_")


def _infer_write_tools(tools: List[Tool]) -> frozenset:
    """Infer mutating tools from the tool set.

    The guarded agent is registered for every domain, so an airline-only allowlist
    is unsafe. This inference intentionally errs on the side of confirmation:
    tools that are not obvious reads/assertions are treated as writes, while human
    transfers are handled separately by the policy guard and do not require user
    confirmation.
    """
    write_tools = set()
    for tool in tools:
        name = tool.name
        if name.startswith("transfer_to_"):
            continue
        if name in READ_ONLY_TOOL_NAMES:
            continue
        if any(name.startswith(prefix) for prefix in READ_ONLY_TOOL_PREFIXES):
            continue
        write_tools.add(name)
    return frozenset(write_tools)


def _extract_json_object(text: Optional[str]) -> Optional[dict]:
    """Extract the first balanced JSON object from a model response, tolerating
    surrounding prose or <think> blocks that small models emit."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _describe_tool_call(tool_call: ToolCall) -> str:
    """Render a tool call as a readable one-liner for confirmation messages."""
    args = ", ".join(f"{k}={v}" for k, v in tool_call.arguments.items())
    return f"{tool_call.name}({args})"


def _render_transcript(messages: list) -> str:
    """Render the conversation transcript for a guard's verification call."""
    return "\n\n".join(str(m) for m in messages)


class Guard:
    """Base class for a guard. A guard inspects a candidate AssistantMessage and
    returns a Verdict."""

    def review(self, candidate: AssistantMessage, ctx: GuardContext) -> Verdict:
        raise NotImplementedError


class PolicyGuard(Guard):
    """Grounded, verify-and-revise policy compliance guard.

    Only engages on consequential tool calls (writes / cancel / transfer). For
    everything else — talking to the user, read-only tool calls — it approves
    immediately with no LLM call. When it does engage, it makes a separate
    grounded `generate()` call that must quote the exact policy clause permitting
    the action given the facts established in the transcript. No citation, or a
    citation whose conditions are not satisfied by verified facts, is a violation
    and yields a `revise`.
    """

    SYSTEM_PROMPT = """
You are a strict policy-compliance verifier for a customer-service agent.
You are given the agent's POLICY, the conversation TRANSCRIPT, and a single ACTION
(a tool call that writes to the database, cancels, or transfers to a human) that
the agent proposes to take next.

Decide whether the policy EXPLICITLY permits this exact action given ONLY the facts
already established in the transcript (via the user and via tool results).

Hard rules:
- To PERMIT an action you MUST quote the exact sentence or clause from the policy
  that authorizes it. If no such clause exists, the verdict is "violation".
- Every condition in that clause must be satisfied by facts that are actually
  present in the transcript. If a required fact (e.g. membership level, insurance,
  booking time, price, reason for cancellation) has not been verified through a
  tool result or stated by the user, the verdict is "violation" — do not assume it.
- Do not invent, paraphrase loosely, or extend the policy. If you cannot find a
  clause to quote, it is a violation.
- A transfer to a human is permitted only if the request genuinely cannot be
  handled with the available actions; if the agent could still help (e.g. by
  asking the user for missing information), it is a violation.

Respond with ONLY a JSON object, no other text:
{"verdict": "permit" | "violation", "citation": "<exact quoted policy clause, empty if violation>", "reason": "<one short sentence>"}
""".strip()

    USER_TEMPLATE = """
<policy>
{policy}
</policy>

<transcript>
{transcript}
</transcript>

<proposed_action>
{action}
</proposed_action>

Verify the proposed action now. Remember: quote a permitting clause or return a violation.
""".strip()

    def __init__(self, write_tools: frozenset):
        self.write_tools = write_tools

    def review(self, candidate: AssistantMessage, ctx: GuardContext) -> Verdict:
        if not candidate.is_tool_call():
            return Verdict.approve()
        consequential = [
            tc
            for tc in candidate.tool_calls
            if _is_consequential(tc, self.write_tools)
        ]
        if not consequential:
            # Read-only tool calls are safe and idempotent: no verification.
            return Verdict.approve()

        action_text = "\n".join(_describe_tool_call(tc) for tc in consequential)
        user_content = self.USER_TEMPLATE.format(
            policy=ctx.domain_policy,
            transcript=_render_transcript(ctx.transcript),
            action=action_text,
        )
        messages = [
            SystemMessage(role="system", content=self.SYSTEM_PROMPT),
            UserMessage(role="user", content=user_content),
        ]
        try:
            response = generate(
                model=ctx.llm,
                messages=messages,
                tools=None,
                call_name="policy_guard",
                **{**ctx.llm_args, "temperature": 0.0},
            )
        except Exception as e:  # noqa: BLE001 - infra failure should not break the turn
            logger.warning(f"PolicyGuard verification call failed, failing closed: {e}")
            return Verdict.revise(
                "The policy verifier failed before it could confirm this action is "
                "permitted. Ask the user for clarification or try a read-only step "
                "instead of taking a consequential action."
            )

        verdict_obj = _extract_json_object(response.content)
        if verdict_obj is None:
            # The verifier ran but produced no parseable verdict. Fail closed on
            # the safety decision: ask the proposer to justify with a citation.
            return Verdict.revise(
                "The policy verifier could not confirm this action is permitted. "
                "Re-justify it by quoting the exact policy clause that authorizes "
                "it, or take a different action."
            )

        verdict = str(verdict_obj.get("verdict", "")).strip().lower()
        citation = str(verdict_obj.get("citation", "")).strip()
        reason = str(verdict_obj.get("reason", "")).strip()
        if verdict == "permit" and citation:
            return Verdict.approve()
        return Verdict.revise(
            reason
            or "This action is not permitted by the policy given the verified facts."
        )


class ConfirmationGuard(Guard):
    """Deterministic confirmation guard.

    Read/write classification is a static lookup. A candidate with no write tool
    calls is approved immediately (no LLM call). For a candidate that contains a
    write, the guard compares a normalized signature of the action against the
    `pending_action` recorded in state:

    - If there is a matching pending action AND the latest user turn is an
      affirmation, the write was confirmed: approve and clear the pending action.
    - Otherwise, store the signature as the new pending action and REPLACE the
      candidate with a confirmation message that lists the action details, so the
      user can answer on the next turn.

    Signature matching matters: a write that differs from what the user confirmed
    must be re-confirmed rather than slipped through.
    """

    def __init__(self, write_tools: frozenset):
        self.write_tools = write_tools

    @staticmethod
    def is_affirmation(message: Optional[UserMessage]) -> bool:
        """Cheap heuristic for 'did the user say yes'. Signature matching makes
        this safe to keep loose: a 'yes, but change X' answer produces a
        different write whose signature won't match the pending action, so it
        gets re-confirmed rather than executed."""
        if message is None or not message.has_text_content():
            return False
        text = message.content.strip().lower()
        if any(neg in text for neg in _NEGATIONS):
            return False
        padded_text = f" {text} "
        if any(marker in padded_text for marker in _CHANGE_MARKERS):
            return False
        return any(aff in text for aff in _AFFIRMATIONS)

    @staticmethod
    def is_negation(message: Optional[UserMessage]) -> bool:
        if message is None or not message.has_text_content():
            return False
        text = message.content.strip().lower()
        return any(neg in text for neg in _NEGATIONS)

    def _signature(self, write_calls: list) -> list:
        return sorted(
            f"{tc.name}::{json.dumps(tc.arguments, sort_keys=True, default=str)}"
            for tc in write_calls
        )

    def _confirmation_message(self, write_calls: list) -> AssistantMessage:
        lines = ["Before I proceed, please confirm the following action(s):"]
        for tc in write_calls:
            lines.append(f"  - {_describe_tool_call(tc)}")
        lines.append("Shall I go ahead? (yes/no)")
        return AssistantMessage.text("\n".join(lines))

    @staticmethod
    def _confirmed_action_message(write_calls: list[ToolCall]) -> AssistantMessage:
        return AssistantMessage(role="assistant", content=None, tool_calls=write_calls)

    @staticmethod
    def _clear_pending(ctx: GuardContext) -> None:
        ctx.state.pending_action = None
        ctx.state.pending_tool_calls = None

    def review(self, candidate: AssistantMessage, ctx: GuardContext) -> Verdict:
        pending = ctx.state.pending_action
        pending_tool_calls = ctx.state.pending_tool_calls
        if pending and self.is_negation(ctx.latest_user_message):
            self._clear_pending(ctx)
            pending = None
            pending_tool_calls = None

        if pending and pending_tool_calls and self.is_affirmation(ctx.latest_user_message):
            if candidate.is_tool_call():
                candidate_write_calls = [
                    tc for tc in candidate.tool_calls if tc.name in self.write_tools
                ]
                if self._signature(candidate_write_calls) == pending:
                    self._clear_pending(ctx)
                    return Verdict.approve()
            self._clear_pending(ctx)
            return Verdict.replace(self._confirmed_action_message(pending_tool_calls))

        if not candidate.is_tool_call():
            return Verdict.approve()
        write_calls = [
            tc for tc in candidate.tool_calls if tc.name in self.write_tools
        ]
        if not write_calls:
            # All reads (or a transfer, which the policy guard already vetted).
            return Verdict.approve()

        signature = self._signature(write_calls)
        # Unconfirmed write (new, changed, or not yet affirmed): surface it.
        ctx.state.pending_action = signature
        ctx.state.pending_tool_calls = write_calls
        return Verdict.replace(self._confirmation_message(write_calls))


class GuardedLLMAgent(LLMAgent[LLMAgentStateType], Generic[LLMAgentStateType]):
    """An LLMAgent whose proposer is wrapped in an ordered chain of guards.

    The proposer is the inherited `generate()` call (full policy in context, all
    tools available). On each turn the proposer produces one candidate
    AssistantMessage; the guards run in order and the first non-approve verdict
    drives what happens next:

      - revise  -> the candidate is bounced back to the proposer with the
                   critique appended to a throwaway buffer (never persisted), up
                   to `max_proposer_attempts` total tries.
      - replace -> the replacement AssistantMessage is emitted immediately.

    On retry exhaustion the agent fails safe to a clarifying message rather than
    emitting a guessed write or a reflexive transfer.
    """

    DEFAULT_WRITE_TOOLS = AIRLINE_WRITE_TOOLS

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        max_proposer_attempts: int = 3,
        write_tools: Optional[frozenset] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.max_proposer_attempts = max_proposer_attempts
        self.write_tools = (
            write_tools if write_tools is not None else _infer_write_tools(tools)
        )
        self.guards: List[Guard] = [
            PolicyGuard(self.write_tools),
            ConfirmationGuard(self.write_tools),
        ]

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentStateType:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return GuardedLLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    @staticmethod
    def _latest_user_message(messages: list) -> Optional[UserMessage]:
        for m in reversed(messages):
            if isinstance(m, UserMessage):
                return m
        return None

    def _propose(self, base_messages: list, scratch: list) -> AssistantMessage:
        """Run the proposer over the canonical context plus a throwaway scratch
        buffer of rejected candidates + critiques."""
        return generate(
            model=self.llm,
            tools=self.tools,
            messages=base_messages + scratch,
            call_name="agent_proposer",
            **self.llm_args,
        )

    def _revision_feedback(
        self, candidate: AssistantMessage, reason: str
    ) -> list[Message]:
        """Build throwaway feedback that bounces a rejected candidate back to the
        proposer. The critique is delivered as a tool result (the format the
        model is trained to react to), so a rejected tool call looks like a tool
        error explaining why it was blocked."""
        critique = (
            "[POLICY GUARD] This action was REJECTED and NOT executed. "
            f"Reason: {reason} "
            "Either propose a corrected action that you can justify by quoting the "
            "exact policy clause that permits it, or send a message to the user "
            "instead."
        )
        feedback: list[Message] = [candidate]
        for tc in candidate.tool_calls or []:
            if not tc.id:
                tc.id = f"guard_{uuid.uuid4().hex[:8]}"
            feedback.append(
                ToolMessage(
                    id=tc.id,
                    role="tool",
                    content=critique,
                    requestor="assistant",
                    error=True,
                )
            )
        return feedback

    def _fail_safe_message(self, reason: Optional[str]) -> AssistantMessage:
        """When the proposer cannot produce a policy-compliant action within the
        retry budget, ask the user to clarify rather than guess a write or
        reflexively transfer."""
        logger.warning(
            f"GuardedLLMAgent exhausted proposer retries; failing safe. "
            f"Last reason: {reason}"
        )
        return AssistantMessage.text(
            "I'm sorry, but I'm not able to complete that request as described, "
            "because it doesn't appear to be permitted by our policy. Could you "
            "clarify what you'd like to do, or let me know if there's anything "
            "else I can help with?"
        )

    def _run_guards(
        self, candidate: AssistantMessage, ctx: GuardContext
    ) -> Verdict:
        """Run the guards in order; return the first non-approve verdict."""
        for guard in self.guards:
            verdict = guard.review(candidate, ctx)
            if verdict.decision != GuardDecision.APPROVE:
                return verdict
        return Verdict.approve()

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentStateType
    ) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio. Use VoiceLLMAgent instead.")
        # Ingest the incoming message into the canonical, persisted transcript.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        base_messages = state.system_messages + state.messages
        ctx = GuardContext(
            state=state,
            latest_user_message=self._latest_user_message(state.messages),
            transcript=state.messages,
            domain_policy=self.domain_policy,
            tools=self.tools,
            llm=self.llm,
            llm_args=self.llm_args,
        )

        scratch: list[Message] = []  # throwaway revision buffer; never persisted
        candidate = self._propose(base_messages, scratch)
        last_reason: Optional[str] = None
        for attempt in range(1, self.max_proposer_attempts + 1):
            verdict = self._run_guards(candidate, ctx)
            if verdict.decision == GuardDecision.APPROVE:
                return candidate
            if verdict.decision == GuardDecision.REPLACE:
                return verdict.replacement
            # REVISE: bounce back to the proposer unless we are out of attempts.
            last_reason = verdict.reason
            if attempt >= self.max_proposer_attempts:
                break
            scratch.extend(self._revision_feedback(candidate, verdict.reason))
            candidate = self._propose(base_messages, scratch)

        return self._fail_safe_message(last_reason)


# =============================================================================
# AGENT FACTORY FUNCTIONS
# =============================================================================


def create_llm_agent(tools, domain_policy, **kwargs):
    """Factory function for LLMAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
    """
    return LLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


def create_guarded_llm_agent(tools, domain_policy, **kwargs):
    """Factory function for GuardedLLMAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
            - max_proposer_attempts (int): Total proposer tries per turn.
            - write_tools (frozenset): Tools treated as DB writes (override per domain).
    """
    extra = {}
    if kwargs.get("max_proposer_attempts") is not None:
        extra["max_proposer_attempts"] = kwargs["max_proposer_attempts"]
    if kwargs.get("write_tools") is not None:
        extra["write_tools"] = kwargs["write_tools"]
    return GuardedLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        **extra,
    )


def create_llm_gt_agent(tools, domain_policy, **kwargs):
    """Factory function for LLMGTAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
            - task (Task): The task to solve (required for GT agent).
    """
    return LLMGTAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
    )


def create_llm_solo_agent(tools, domain_policy, **kwargs):
    """Factory function for LLMSoloAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
            - task (Task): The task to solve (required for solo agent).
    """
    return LLMSoloAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
    )
