import pytest

from tau2.agent.llm_agent import (
    GuardedLLMAgent,
    LLMAgent,
    LLMSoloAgent,
    _infer_write_tools,
)
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.environment.tool import as_tool


@pytest.fixture
def agent(get_environment) -> LLMAgent:
    return LLMAgent(
        llm="gpt-4o-mini",
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
    )


@pytest.fixture
def solo_agent(get_environment, base_task) -> LLMSoloAgent:
    return LLMSoloAgent(
        llm="gpt-4o-mini",
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
        task=base_task,
    )


@pytest.fixture
def first_user_message():
    return UserMessage(content="Hello can you help me create a task?", role="user")


def test_agent(agent: LLMAgent, first_user_message: UserMessage):
    agent_state = agent.get_init_state()
    assert agent_state is not None
    agent_msg, agent_state = agent.generate_next_message(
        first_user_message, agent_state
    )
    # Check the response is an assistant message
    assert isinstance(agent_msg, AssistantMessage)
    # Check the state is updated
    assert agent_state is not None
    assert len(agent_state.messages) == 2
    # Check the messages are of the correct type
    assert isinstance(agent_state.messages[0], UserMessage)
    assert isinstance(agent_state.messages[1], AssistantMessage)
    assert agent_state.messages[0].content == first_user_message.content
    assert agent_state.messages[1].content == agent_msg.content


def test_agent_set_state(agent: LLMAgent, first_user_message: UserMessage):
    _ = agent.get_init_state(
        message_history=[
            UserMessage(content="Hello, can you help me find a flight?", role="user"),
            AssistantMessage(
                content="Hello, I can help you find a flight.", role="assistant"
            ),
        ]
    )


def test_solo_agent(solo_agent: LLMSoloAgent):
    agent_state = solo_agent.get_init_state()
    assert agent_state is not None
    agent_msg, agent_state = solo_agent.generate_next_message(None, agent_state)
    assert isinstance(agent_msg, AssistantMessage)
    assert agent_state is not None
    assert len(agent_state.messages) == 1


def create_task(user_id: str, title: str) -> str:
    """Create a task.

    Args:
        user_id: User id.
        title: Task title.
    """
    return f"{user_id}:{title}"


def get_users() -> list[str]:
    """Get users."""
    return ["user-1"]


def search_direct_flight(origin: str, destination: str, date: str) -> list[str]:
    """Search direct flights.

    Args:
        origin: Origin airport.
        destination: Destination airport.
        date: Flight date.
    """
    return [f"{origin}-{destination}-{date}"]


def test_guarded_agent_infers_non_airline_write_tools():
    write_tools = _infer_write_tools(
        [as_tool(create_task), as_tool(get_users), as_tool(search_direct_flight)]
    )

    assert "create_task" in write_tools
    assert "get_users" not in write_tools
    assert "search_direct_flight" not in write_tools


def test_guarded_agent_fails_closed_when_policy_guard_errors(monkeypatch):
    tool_call = ToolCall(
        id="call_1",
        name="create_task",
        arguments={"user_id": "user-1", "title": "Check billing"},
    )

    def fake_generate(*, tools=None, **kwargs):
        if tools is None:
            raise RuntimeError("verifier unavailable")
        return AssistantMessage(role="assistant", content=None, tool_calls=[tool_call])

    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate)
    agent = GuardedLLMAgent(
        tools=[as_tool(create_task)],
        domain_policy="Only create a task when policy permits it.",
        llm="test-model",
        max_proposer_attempts=1,
    )

    msg, state = agent.generate_next_message(
        UserMessage(role="user", content="Create a billing task."),
        agent.get_init_state(),
    )

    assert isinstance(msg, AssistantMessage)
    assert msg.tool_calls is None
    assert "not able to complete" in msg.content
    assert state.pending_action is None


def test_guarded_agent_executes_confirmed_pending_action(monkeypatch):
    tool_call = ToolCall(
        id="call_1",
        name="create_task",
        arguments={"user_id": "user-1", "title": "Check billing"},
    )
    calls = []

    def fake_generate(*, tools=None, **kwargs):
        calls.append(tools)
        if tools is None:
            return AssistantMessage(
                role="assistant",
                content='{"verdict": "permit", "citation": "Agents may create tasks.", "reason": "Permitted."}',
            )
        if len(calls) == 1:
            return AssistantMessage(role="assistant", content=None, tool_calls=[tool_call])
        return AssistantMessage(role="assistant", content="Yes, I can do that.")

    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate)
    agent = GuardedLLMAgent(
        tools=[as_tool(create_task)],
        domain_policy="Agents may create tasks.",
        llm="test-model",
    )
    state = agent.get_init_state()

    first_msg, state = agent.generate_next_message(
        UserMessage(role="user", content="Create a billing task."),
        state,
    )
    assert first_msg.tool_calls is None
    assert "please confirm" in first_msg.content.lower()
    assert state.pending_action is not None

    second_msg, state = agent.generate_next_message(
        UserMessage(role="user", content="Yes, go ahead."),
        state,
    )
    assert second_msg.tool_calls == [tool_call]
    assert state.pending_action is None
    assert state.pending_tool_calls is None
