"""Unit tests for the OpenCode agent run loop."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from slop_code.agent_runner.agents.opencode import OpenCodeAgent
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentError
from slop_code.common.llms import APIPricing
from slop_code.common.llms import TokenUsage
from slop_code.execution.runtime import RuntimeEvent
from slop_code.execution.runtime import RuntimeResult


class FakeRuntime:
    """Minimal runtime stub used to feed deterministic events to the agent."""

    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []
        self.cleaned = False
        self.last_stream_args: tuple[tuple, dict] | None = None
        self.kill_calls = 0

    def stream(
        self,
        command: str,
        env: dict,
        timeout: float | None,
    ) -> Iterable[RuntimeEvent]:
        self.last_stream_args = ((command, env, timeout), {})
        yield from self.events

    def cleanup(self) -> None:
        self.cleaned = True

    def kill(self) -> None:
        self.kill_calls += 1


@dataclass
class FakeSession:
    runtime: FakeRuntime
    working_dir: str

    spec: object | None = None
    spawn_kwargs: dict[str, object] | None = None

    def spawn(
        self, **kwargs: object
    ) -> FakeRuntime:  # pragma: no cover - trivial
        self.spawn_kwargs = kwargs
        return self.runtime


@pytest.fixture
def make_agent(tmp_path_factory, request):
    def _make(
        *,
        cost_limits: AgentCostLimits = AgentCostLimits(
            step_limit=0,
            cost_limit=100.0,
            net_cost_limit=200.0,
        ),
    ) -> tuple[OpenCodeAgent, FakeRuntime]:
        tmp_dir = tmp_path_factory.mktemp("opencode_agent")
        runtime = FakeRuntime()
        session = FakeSession(runtime=runtime, working_dir=str(tmp_dir))

        agent = OpenCodeAgent(
            problem_name="sample-problem",
            verbose=False,
            cost_limits=cost_limits,
            pricing=APIPricing(
                input=0.6,
                output=2.2,
                cache_read=0.11,
            ),
            credential=None,
            model_id="glm-4.6",
            provider="zai-coding-plan",
            opencode_config={},
            env={},
            thinking=None,
        )
        agent.setup(session)
        request.addfinalizer(agent.cleanup)
        return agent, runtime

    return _make


def _token_usage_from_part(part: dict[str, dict]) -> TokenUsage:
    tokens = part["tokens"]
    return TokenUsage(
        input=tokens["input"],
        output=tokens["output"],
        cache_read=tokens["cache"]["read"],
        cache_write=tokens["cache"].get("write", 0),
        reasoning=tokens["reasoning"],
    )


def _runtime_events_from_stdout_chunks(
    chunks: list[str],
    *,
    exit_code: int = 0,
    stderr: str = "",
) -> list[RuntimeEvent]:
    events = [
        RuntimeEvent(kind="stdout", text=chunk, result=None) for chunk in chunks
    ]
    events.append(
        RuntimeEvent(
            kind="finished",
            text=None,
            result=RuntimeResult(
                exit_code=exit_code,
                stdout="".join(chunks),
                stderr=stderr,
                elapsed=1.0,
                timed_out=False,
                setup_stdout="",
                setup_stderr="",
            ),
        )
    )
    return events


class TestInjectThinkingConfig:
    """Tests for _inject_thinking_config and _make_opencode_config."""

    def _make_bare_agent(self, tmp_path, **overrides):
        defaults = dict(
            problem_name="test",
            verbose=False,
            cost_limits=AgentCostLimits(
                step_limit=0, cost_limit=100.0, net_cost_limit=200.0
            ),
            pricing=None,
            credential=None,
            model_id="some-org/some-model",
            provider="openrouter",
            opencode_config={},
            env={},
            thinking=None,
        )
        defaults.update(overrides)
        agent = OpenCodeAgent(**defaults)
        # Set up tmp_dir manually so _make_opencode_config works
        import tempfile

        agent._tmp_dir = tempfile.TemporaryDirectory(dir=str(tmp_path))
        return agent

    def test_openrouter_injects_at_provider_model_level(self, tmp_path):
        agent = self._make_bare_agent(
            tmp_path,
            provider="openrouter",
            model_id="moonshotai/kimi-k2-thinking",
            thinking="high",
        )
        agent._make_opencode_config()

        cfg = agent.open_code_config
        model_opts = cfg["provider"]["openrouter"]["models"][
            "moonshotai/kimi-k2-thinking"
        ]["options"]
        assert model_opts["reasoningEffort"] == "high"
        assert "agent" not in cfg

    def test_non_openrouter_uses_agent_build(self, tmp_path):
        agent = self._make_bare_agent(
            tmp_path,
            provider="anthropic",
            thinking="medium",
        )
        agent._make_opencode_config()

        cfg = agent.open_code_config
        assert cfg["agent"]["build"]["reasonEffort"] == "medium"
        assert "provider" not in cfg

    def test_existing_config_takes_precedence(self, tmp_path):
        agent = self._make_bare_agent(
            tmp_path,
            provider="openrouter",
            model_id="some-org/some-model",
            thinking="low",
            opencode_config={
                "provider": {
                    "openrouter": {
                        "models": {
                            "some-org/some-model": {
                                "options": {
                                    "thinking": {
                                        "type": "enabled",
                                        "budgetTokens": 16000,
                                    }
                                }
                            }
                        }
                    }
                }
            },
        )
        agent._make_opencode_config()

        opts = agent.open_code_config["provider"]["openrouter"]["models"][
            "some-org/some-model"
        ]["options"]
        # Explicit thinking config preserved
        assert opts["thinking"] == {
            "type": "enabled",
            "budgetTokens": 16000,
        }
        # Auto-injected reasoningEffort also present (deep merge adds it)
        assert opts["reasoningEffort"] == "low"

    def test_no_injection_when_thinking_is_none(self, tmp_path):
        agent = self._make_bare_agent(
            tmp_path,
            provider="openrouter",
            thinking=None,
        )
        agent._make_opencode_config()

        cfg = agent.open_code_config
        assert "provider" not in cfg
        assert "agent" not in cfg

    def test_does_not_overwrite_existing_agent_config(self, tmp_path):
        agent = self._make_bare_agent(
            tmp_path,
            provider="anthropic",
            thinking="high",
            opencode_config={
                "agent": {
                    "build": {"someKey": "value"},
                    "maxSteps": 200,
                }
            },
        )
        agent._make_opencode_config()

        cfg = agent.open_code_config
        assert cfg["agent"]["build"]["reasonEffort"] == "high"
        assert cfg["agent"]["build"]["someKey"] == "value"
        assert cfg["agent"]["maxSteps"] == 200


def test_run_collects_messages_and_updates_usage(make_agent):
    agent, runtime = make_agent()

    tool_use = {"type": "tool_use", "part": {"tool": "search"}}
    first_step = {
        "type": "step_finish",
        "part": {
            "reason": "tool-calls",
            "cost": 0.3,
            "tokens": {
                "input": 10,
                "output": 4,
                "reasoning": 1,
                "cache": {"read": 2, "write": 0},
            },
        },
    }
    final_step = {
        "type": "step_finish",
        "part": {
            "reason": "stop",
            "cost": 1.0,
            "tokens": {
                "input": 5,
                "output": 6,
                "reasoning": 2,
                "cache": {"read": 4, "write": 0},
            },
        },
    }

    chunk_1 = json.dumps(tool_use) + "\n" + json.dumps(first_step)[:20]
    chunk_2 = json.dumps(first_step)[20:] + "\n" + json.dumps(final_step)
    runtime.events = _runtime_events_from_stdout_chunks([chunk_1, chunk_2])

    agent.run("build something cool")
    messages = agent.messages

    assert messages[0] == tool_use
    assert messages[1] == first_step
    assert messages[2] == final_step

    assert agent.usage.steps == 2
    expected_cost = first_step["part"]["cost"] + final_step["part"]["cost"]
    assert agent.usage.cost == pytest.approx(expected_cost)
    assert (
        agent.usage.current_tokens.input
        == final_step["part"]["tokens"]["input"]
    )
    assert (
        agent.usage.current_tokens.output
        == final_step["part"]["tokens"]["output"]
    )
    assert agent.usage.net_tokens.reasoning == 3
    assert (
        agent.usage.current_tokens.cache_read
        == final_step["part"]["tokens"]["cache"]["read"]
    )
    assert agent.continue_on_run is False


def test_build_command_matches_harbor_shape(make_agent):
    agent, _ = make_agent()

    command = agent._build_opencode_command("build something cool")

    assert command == (
        "opencode --model=zai-coding-plan/glm-4.6 run --format=json "
        "--thinking --dangerously-skip-permissions -- "
        "'build something cool'"
    )


def test_setup_sets_fake_vcs_env(make_agent):
    agent, _ = make_agent()

    assert isinstance(agent.session, FakeSession)
    assert agent.session.spawn_kwargs is not None
    assert agent.session.spawn_kwargs["env_vars"]["OPENCODE_FAKE_VCS"] == "git"


def test_run_skips_invalid_json_lines(make_agent):
    agent, runtime = make_agent()

    good_step = {
        "type": "step_finish",
        "part": {
            "reason": "stop",
            "cost": 0.25,
            "tokens": {
                "input": 3,
                "output": 2,
                "reasoning": 0,
                "cache": {"read": 1, "write": 0},
            },
        },
    }

    chunk = "this is not json\n" + json.dumps(good_step)
    runtime.events = _runtime_events_from_stdout_chunks([chunk])

    agent.run("ignore malformed lines")
    messages = agent.messages

    assert messages == [good_step]
    assert agent.usage.steps == 1
    assert agent.usage.cost == pytest.approx(good_step["part"]["cost"])


def test_run_raises_when_finished_event_missing(make_agent):
    agent, runtime = make_agent()

    unfinished_step = {
        "type": "step_finish",
        "part": {
            "reason": "tool-calls",
            "cost": 0.1,
            "tokens": {
                "input": 1,
                "output": 1,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
        },
    }
    runtime.events = [
        RuntimeEvent(
            kind="stdout",
            text=json.dumps(unfinished_step),
            result=None,
        ),
    ]

    with pytest.raises(AgentError):
        agent.run("no finished event")


def test_run_raises_on_opencode_error_event(make_agent):
    agent, runtime = make_agent()

    error_message = {
        "type": "error",
        "error": {"message": "Model not found: zai-coding-plan/glm-5.1."},
    }
    runtime.events = _runtime_events_from_stdout_chunks(
        [json.dumps(error_message)]
    )

    with pytest.raises(AgentError, match="Model not found"):
        agent.run("trigger error")


def test_run_raises_when_no_step_finish_messages(make_agent):
    agent, runtime = make_agent()

    non_step_message = {
        "type": "tool_use",
        "part": {"tool": "read_file"},
    }
    runtime.events = _runtime_events_from_stdout_chunks(
        [json.dumps(non_step_message)]
    )

    with pytest.raises(AgentError, match="step_finish"):
        agent.run("no step finish")


def test_run_respects_step_limit(make_agent):
    limits = AgentCostLimits(
        step_limit=1,
        cost_limit=100.0,
        net_cost_limit=200.0,
    )
    agent, runtime = make_agent(cost_limits=limits)

    step = {
        "type": "step_finish",
        "part": {
            "reason": "tool-calls",
            "cost": 0.1,
            "tokens": {
                "input": 1,
                "output": 2,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
        },
    }
    runtime.events = _runtime_events_from_stdout_chunks([json.dumps(step)])

    agent.run("should stop by step")
    assert agent.messages[0] == step
    assert agent.cost_limits.is_above_limits(
        agent.usage,
        prior_cost=agent.prior_cost,
    )
    assert agent.continue_on_run is False
    assert agent.usage.steps == limits.step_limit


def test_run_respects_cost_limit(make_agent):
    limits = AgentCostLimits(
        step_limit=0,
        cost_limit=1e-6,
        net_cost_limit=200.0,
    )
    agent, runtime = make_agent(cost_limits=limits)

    expensive_step = {
        "type": "step_finish",
        "part": {
            "reason": "tool-calls",
            "cost": 0.75,
            "tokens": {
                "input": 2,
                "output": 3,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
        },
    }
    runtime.events = _runtime_events_from_stdout_chunks(
        [json.dumps(expensive_step)]
    )

    agent.run("too expensive")
    assert agent.messages[0] == expensive_step
    assert agent.cost_limits.is_above_limits(
        agent.usage,
        prior_cost=agent.prior_cost,
    )
    assert agent.continue_on_run is False
    assert agent.usage.cost > limits.cost_limit
