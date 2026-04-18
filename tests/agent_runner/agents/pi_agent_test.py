"""Unit tests for the PI agent."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import yaml

from slop_code.agent_runner.agents.pi import PiAgent
from slop_code.agent_runner.agents.pi import PiConfig
from slop_code.agent_runner.agents.utils import HOME_PATH
from slop_code.agent_runner.credentials import CredentialType
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentError
from slop_code.agent_runner.registry import build_agent_config
from slop_code.common.llms import APIPricing
from slop_code.common.llms import ModelDefinition
from slop_code.execution import DockerConfig
from slop_code.execution import DockerEnvironmentSpec
from slop_code.execution.runtime import RuntimeEvent
from slop_code.execution.runtime import RuntimeResult

if TYPE_CHECKING:
    from slop_code.execution import Session


class FakeRuntime:
    """Minimal runtime stub for testing."""

    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []
        self.cleaned = False
        self.last_stream_args: tuple[tuple, dict] | None = None

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


@dataclass
class FakeSession:
    """Fake session for testing."""

    runtime: FakeRuntime
    working_dir: Path
    spec: DockerEnvironmentSpec | None = None
    last_spawn_env_vars: dict[str, str] | None = None
    last_spawn_mounts: dict[str, dict[str, str] | str] | None = None

    def spawn(self, **_: object) -> FakeRuntime:
        env_vars = cast("dict[str, str] | None", _.get("env_vars"))
        mounts = cast(
            "dict[str, dict[str, str] | str] | None",
            _.get("mounts"),
        )
        self.last_spawn_env_vars = dict(env_vars or {})
        self.last_spawn_mounts = dict(mounts or {})
        return self.runtime


@pytest.fixture
def mock_pricing() -> APIPricing:
    return APIPricing(
        input=0.15,
        output=0.60,
        cache_read=0.03,
        cache_write=0.60,
    )


@pytest.fixture
def mock_cost_limits() -> AgentCostLimits:
    return AgentCostLimits(
        step_limit=10,
        cost_limit=100.0,
        net_cost_limit=200.0,
    )


@pytest.fixture
def mock_model_def(mock_pricing: APIPricing) -> ModelDefinition:
    return ModelDefinition(
        internal_name="gpt-5.2-codex",
        provider="openai",
        pricing=mock_pricing,
        provider_slugs={
            "openai": "gpt-5.2-codex",
            "openai-codex": "gpt-5.2-codex",
        },
    )


class TestPiConfig:
    """Tests for PiConfig."""

    def test_version_is_required(
        self, mock_cost_limits: AgentCostLimits
    ) -> None:
        with pytest.raises(Exception):
            PiConfig(type="pi", cost_limits=mock_cost_limits)

    def test_config_defaults(self, mock_cost_limits: AgentCostLimits) -> None:
        config = PiConfig(
            type="pi",
            version="0.45.7",
            cost_limits=mock_cost_limits,
        )
        assert config.binary == "pi"
        assert config.timeout is None
        assert config.extra_args == []
        assert config.env == {}
        assert config.provider is None
        assert config.thinking is None

    def test_config_loads_from_default_yaml(self) -> None:
        data = yaml.safe_load(Path("configs/agents/pi.yaml").read_text())
        config = build_agent_config(data)
        assert isinstance(config, PiConfig)
        assert config.type == "pi"

    def test_get_docker_file_renders_version(
        self, mock_cost_limits: AgentCostLimits
    ) -> None:
        config = PiConfig(
            type="pi",
            version="0.45.7",
            cost_limits=mock_cost_limits,
        )
        dockerfile = config.get_docker_file("base-image:latest")
        assert dockerfile is not None
        assert "base-image:latest" in dockerfile
        assert "@mariozechner/pi-coding-agent@0.45.7" in dockerfile

    def test_get_docker_file_configures_user_npm_path(
        self, mock_cost_limits: AgentCostLimits
    ) -> None:
        config = PiConfig(
            type="pi",
            version="0.45.7",
            cost_limits=mock_cost_limits,
        )
        dockerfile = config.get_docker_file("base-image:latest")
        assert dockerfile is not None
        assert 'ENV NPM_CONFIG_PREFIX="$HOME/.npm-global"' in dockerfile
        assert 'ENV PATH="$HOME/.npm-global/bin:$PATH"' in dockerfile
        assert "USER agent" in dockerfile


class TestPiAgent:
    """Tests for PiAgent."""

    def test_from_config_creates_agent(
        self,
        mock_cost_limits: AgentCostLimits,
        mock_model_def: ModelDefinition,
    ) -> None:
        config = PiConfig(
            type="pi",
            version="0.45.7",
            cost_limits=mock_cost_limits,
        )
        credential = ProviderCredential(
            provider="openai",
            credential_type=CredentialType.ENV_VAR,
            value="test-api-key",
            source="OPENAI_API_KEY",
            destination_key="OPENAI_API_KEY",
        )

        agent = PiAgent._from_config(
            config=config,
            model=mock_model_def,
            credential=credential,
            problem_name="test-problem",
            verbose=False,
            image="test-image",
        )

        assert isinstance(agent, PiAgent)
        assert agent.binary == "pi"
        assert agent.provider == "openai"
        assert agent.model == "gpt-5.2-codex"

    def test_from_config_requires_image(
        self,
        mock_cost_limits: AgentCostLimits,
        mock_model_def: ModelDefinition,
    ) -> None:
        config = PiConfig(
            type="pi",
            version="0.45.7",
            cost_limits=mock_cost_limits,
        )
        credential = ProviderCredential(
            provider="openai",
            credential_type=CredentialType.ENV_VAR,
            value="test-api-key",
            source="OPENAI_API_KEY",
            destination_key="OPENAI_API_KEY",
        )

        with pytest.raises(ValueError, match="requires an image"):
            PiAgent._from_config(
                config=config,
                model=mock_model_def,
                credential=credential,
                problem_name="test-problem",
                verbose=False,
                image=None,
            )

    @pytest.mark.parametrize(
        ("scb_provider", "pi_provider"),
        [
            ("openai", "openai"),
            ("codex_auth", "openai-codex"),
            ("anthropic", "anthropic"),
            ("google", "google"),
            ("openrouter", "openrouter"),
            ("groq", "groq"),
            ("mistral", "mistral"),
            ("xai", "xai"),
            ("bedrock", "amazon-bedrock"),
            ("zhipu", "zai"),
            ("zai", "zai"),
            ("ai_gateway", "vercel-ai-gateway"),
            ("vercel-ai-gateway", "vercel-ai-gateway"),
            ("minimax", "minimax"),
        ],
    )
    def test_resolve_pi_provider(
        self,
        scb_provider: str,
        pi_provider: str,
    ) -> None:
        assert PiAgent._resolve_pi_provider(scb_provider) == pi_provider

    def test_resolve_pi_provider_rejects_unsupported(self) -> None:
        with pytest.raises(ValueError, match="Unsupported PI provider mapping"):
            PiAgent._resolve_pi_provider("unknown_provider")

    def test_build_command_has_required_flags(
        self,
        mock_cost_limits: AgentCostLimits,
        mock_pricing: APIPricing,
    ) -> None:
        agent = PiAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="pi",
            provider="openai",
            model="gpt-5.2-codex",
            timeout=None,
            thinking=None,
            extra_args=[],
            env={},
        )

        command = agent._build_command("do something")
        assert command[0] == "pi"
        assert "--print" in command
        assert "--mode" in command
        assert "json" in command
        assert "--no-session" in command
        assert "--provider" in command
        assert "openai" in command
        assert "--model" in command
        assert "gpt-5.2-codex" in command

    def test_build_command_with_minimal_thinking(
        self,
        mock_cost_limits: AgentCostLimits,
        mock_pricing: APIPricing,
    ) -> None:
        agent = PiAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="pi",
            provider="openai",
            model="gpt-5.2-codex",
            timeout=None,
            thinking="minimal",
            extra_args=[],
            env={},
        )

        command = agent._build_command("do something")
        assert "--thinking" in command
        thinking_index = command.index("--thinking")
        assert command[thinking_index + 1] == "minimal"

    def test_resolve_pi_thinking_none_preset_does_not_force_off(self) -> None:
        thinking = PiAgent._resolve_pi_thinking(
            config_thinking=None,
            thinking_preset="none",
            thinking_max_tokens=None,
        )
        assert thinking is None

    def test_resolve_pi_thinking_disabled_preset_does_not_force_off(
        self,
    ) -> None:
        thinking = PiAgent._resolve_pi_thinking(
            config_thinking=None,
            thinking_preset="disabled",
            thinking_max_tokens=None,
        )
        assert thinking is None

    def test_build_command_rejects_forbidden_extra_args(
        self,
        mock_cost_limits: AgentCostLimits,
        mock_pricing: APIPricing,
    ) -> None:
        agent = PiAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="pi",
            provider="openai",
            model="gpt-5.2-codex",
            timeout=None,
            thinking=None,
            extra_args=["--mode", "text"],
            env={},
        )

        with pytest.raises(
            AgentError, match="extra_args contains protected PI flag"
        ):
            agent._build_command("do something")

    def test_prepare_runtime_execution_does_not_embed_api_key_in_command(
        self,
        mock_cost_limits: AgentCostLimits,
        mock_pricing: APIPricing,
    ) -> None:
        credential = ProviderCredential(
            provider="openai",
            credential_type=CredentialType.ENV_VAR,
            value="super-secret-key",
            source="OPENAI_API_KEY",
            destination_key="OPENAI_API_KEY",
        )
        agent = PiAgent(
            problem_name="test-problem",
            verbose=False,
            image="test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=credential,
            binary="pi",
            provider="openai",
            model="gpt-5.2-codex",
            timeout=None,
            thinking=None,
            extra_args=[],
            env={},
        )

        command, env_overrides = agent._prepare_runtime_execution(
            "do something"
        )
        joined = " ".join(command)
        assert "super-secret-key" not in joined
        assert env_overrides["OPENAI_API_KEY"] == "super-secret-key"

    def test_codex_auth_conversion_supports_codex_cli_format(self) -> None:
        codex_auth = {
            "tokens": {
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "account_id": "account-id",
            }
        }
        converted = PiAgent._convert_codex_auth_payload(codex_auth)
        assert converted == {
            "openai-codex": {
                "type": "oauth",
                "access": "access-token",
                "refresh": "refresh-token",
                "expires": 0,
                "accountId": "account-id",
            }
        }

    def test_codex_auth_conversion_rejects_invalid_payload(self) -> None:
        with pytest.raises(AgentError, match="Invalid Codex auth payload"):
            PiAgent._convert_codex_auth_payload({"tokens": {}})

    def test_setup_with_codex_auth_uses_temp_pi_auth_only(
        self,
        tmp_path: Path,
        mock_cost_limits: AgentCostLimits,
        mock_pricing: APIPricing,
    ) -> None:
        auth_source = tmp_path / "auth.json"
        auth_source.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": "a",
                        "refresh_token": "r",
                        "account_id": "acct",
                    }
                }
            )
        )
        credential = ProviderCredential(
            provider="codex_auth",
            credential_type=CredentialType.FILE,
            value=auth_source.read_text(),
            source=str(auth_source),
            destination_key="",
        )
        spec = DockerEnvironmentSpec(
            name="docker-test",
            docker=DockerConfig(image="pi-test-image"),
        )
        runtime = FakeRuntime()
        session = FakeSession(runtime=runtime, working_dir=tmp_path, spec=spec)

        agent = PiAgent(
            problem_name="test-problem",
            verbose=False,
            image="pi-test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=credential,
            binary="pi",
            provider="openai-codex",
            model="gpt-5.2-codex",
            timeout=None,
            thinking=None,
            extra_args=[],
            env={},
        )

        agent.setup(session)
        assert session.last_spawn_env_vars is not None
        assert session.last_spawn_env_vars["HOME"] == HOME_PATH
        assert (
            session.last_spawn_env_vars["PI_CODING_AGENT_DIR"]
            == f"{HOME_PATH}/.pi/agent"
        )
        assert session.last_spawn_mounts is not None
        assert any(
            isinstance(value, dict)
            and value.get("bind") == f"{HOME_PATH}/.pi/agent"
            for value in session.last_spawn_mounts.values()
        )
        assert agent._pi_auth_dir is not None
        converted_path = agent._pi_auth_dir / "auth.json"
        assert converted_path.exists()
        converted = json.loads(converted_path.read_text())
        assert "openai-codex" in converted
        assert converted["openai-codex"]["access"] == "a"
        assert auth_source.read_text() != converted_path.read_text()

    def test_setup_with_invalid_codex_auth_raises_clear_error(
        self,
        tmp_path: Path,
        mock_cost_limits: AgentCostLimits,
        mock_pricing: APIPricing,
    ) -> None:
        credential = ProviderCredential(
            provider="codex_auth",
            credential_type=CredentialType.FILE,
            value="{invalid-json",
            source=str(tmp_path / "missing.json"),
            destination_key="",
        )
        runtime = FakeRuntime()
        session = FakeSession(runtime=runtime, working_dir=tmp_path)

        agent = PiAgent(
            problem_name="test-problem",
            verbose=False,
            image="pi-test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=credential,
            binary="pi",
            provider="openai-codex",
            model="gpt-5.2-codex",
            timeout=None,
            thinking=None,
            extra_args=[],
            env={},
        )

        with pytest.raises(AgentError, match="Failed to parse Codex auth JSON"):
            agent.setup(session)

    def test_parse_line_message_end_usage(
        self, mock_pricing: APIPricing
    ) -> None:
        line = json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done"}],
                    "usage": {
                        "input": 100,
                        "output": 10,
                        "cacheRead": 20,
                        "cacheWrite": 5,
                        "cost": {"total": 0.0123},
                    },
                },
            }
        )
        cost, tokens, payload = PiAgent.parse_line(line, pricing=mock_pricing)
        assert cost == pytest.approx(0.0123)
        assert tokens is not None
        assert tokens.input == 100
        assert tokens.output == 10
        assert tokens.cache_read == 20
        assert tokens.cache_write == 5
        assert payload is not None

    def test_parse_line_does_not_reprice_usage_without_reported_cost(
        self,
        mock_pricing: APIPricing,
    ) -> None:
        line = json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done"}],
                    "usage": {
                        "input": 1_000_000,
                        "output": 1_000_000,
                        "cacheRead": 1_000_000,
                        "cacheWrite": 1_000_000,
                    },
                },
            }
        )

        cost, tokens, payload = PiAgent.parse_line(line, pricing=mock_pricing)

        assert cost == 0.0
        assert tokens is not None
        assert tokens.input == 1_000_000
        assert tokens.output == 1_000_000
        assert tokens.cache_read == 1_000_000
        assert tokens.cache_write == 1_000_000
        assert payload is not None

    def test_parse_line_ignores_non_message_end(
        self, mock_pricing: APIPricing
    ) -> None:
        line = json.dumps({"type": "tool_execution_start", "toolName": "bash"})
        cost, tokens, payload = PiAgent.parse_line(line, pricing=mock_pricing)
        assert cost is None
        assert tokens is None
        assert payload is not None

    def test_run_does_not_reprice_pi_usage_when_reported_cost_is_absent(
        self,
        tmp_path: Path,
        mock_cost_limits: AgentCostLimits,
        mock_pricing: APIPricing,
    ) -> None:
        runtime = FakeRuntime()
        usage_line = json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done"}],
                    "usage": {
                        "input": 1_000_000,
                        "output": 1_000_000,
                        "cacheRead": 1_000_000,
                        "cacheWrite": 1_000_000,
                    },
                },
            }
        )
        runtime.events = [
            RuntimeEvent(kind="stdout", text=f"{usage_line}\n"),
            RuntimeEvent(
                kind="finished",
                result=RuntimeResult(
                    exit_code=0,
                    stdout=f"{usage_line}\n",
                    stderr="",
                    setup_stdout="",
                    setup_stderr="",
                    elapsed=0.1,
                    timed_out=False,
                ),
            ),
        ]
        spec = DockerEnvironmentSpec(
            name="docker-test",
            docker=DockerConfig(image="pi-test-image"),
        )
        session = FakeSession(runtime=runtime, working_dir=tmp_path, spec=spec)
        agent = PiAgent(
            problem_name="test-problem",
            verbose=False,
            image="pi-test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="pi",
            provider="openai",
            model="gpt-5.2-codex",
            timeout=None,
            thinking=None,
            extra_args=[],
            env={},
        )

        agent.setup(cast("Session", session))
        agent.run("do something")

        assert agent.usage.cost == 0.0
        assert agent.usage.net_tokens.input == 1_000_000
        assert agent.usage.net_tokens.output == 1_000_000
        assert agent.usage.net_tokens.cache_read == 1_000_000
        assert agent.usage.net_tokens.cache_write == 1_000_000

    def test_run_raises_when_pi_returns_message_error_with_zero_exit(
        self,
        tmp_path: Path,
        mock_cost_limits: AgentCostLimits,
        mock_pricing: APIPricing,
    ) -> None:
        runtime = FakeRuntime()
        error_line = json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "stopReason": "error",
                    "errorMessage": "400 Reasoning is mandatory for this endpoint and cannot be disabled.",
                    "usage": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                    },
                },
            }
        )
        runtime.events = [
            RuntimeEvent(kind="stdout", text=f"{error_line}\n"),
            RuntimeEvent(
                kind="finished",
                result=RuntimeResult(
                    exit_code=0,
                    stdout=f"{error_line}\n",
                    stderr="",
                    setup_stdout="",
                    setup_stderr="",
                    elapsed=0.1,
                    timed_out=False,
                ),
            ),
        ]
        spec = DockerEnvironmentSpec(
            name="docker-test",
            docker=DockerConfig(image="pi-test-image"),
        )
        session = FakeSession(runtime=runtime, working_dir=tmp_path, spec=spec)
        agent = PiAgent(
            problem_name="test-problem",
            verbose=False,
            image="pi-test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="pi",
            provider="openrouter",
            model="minimax/minimax-m2.7",
            timeout=None,
            thinking=None,
            extra_args=[],
            env={},
        )

        agent.setup(session)
        with pytest.raises(AgentError, match="Reasoning is mandatory"):
            agent.run("do something")

    def test_run_allows_sigterm_exit_to_preserve_partial_solution(
        self,
        tmp_path: Path,
        mock_cost_limits: AgentCostLimits,
        mock_pricing: APIPricing,
    ) -> None:
        runtime = FakeRuntime()
        runtime.events = [
            RuntimeEvent(
                kind="finished",
                result=RuntimeResult(
                    exit_code=143,
                    stdout="",
                    stderr="",
                    setup_stdout="",
                    setup_stderr="",
                    elapsed=0.1,
                    timed_out=False,
                ),
            ),
        ]
        spec = DockerEnvironmentSpec(
            name="docker-test",
            docker=DockerConfig(image="pi-test-image"),
        )
        session = FakeSession(runtime=runtime, working_dir=tmp_path, spec=spec)
        agent = PiAgent(
            problem_name="test-problem",
            verbose=False,
            image="pi-test-image",
            cost_limits=mock_cost_limits,
            pricing=mock_pricing,
            credential=None,
            binary="pi",
            provider="openrouter",
            model="moonshotai/kimi-k2.5",
            timeout=None,
            thinking="high",
            extra_args=[],
            env={},
        )

        agent.setup(cast("Session", session))
        agent.run("build a datagate.py server")

        assert agent._last_command is not None
        assert agent._last_command.result is not None
        assert agent._last_command.result.exit_code == 143


class TestPiAgentRegistration:
    """Tests for agent registration."""

    def test_agent_is_registered(self) -> None:
        from slop_code.agent_runner.registry import available_agent_types
        from slop_code.agent_runner.registry import get_agent_cls

        assert "pi" in available_agent_types()
        assert get_agent_cls("pi") is PiAgent
