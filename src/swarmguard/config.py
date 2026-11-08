"""Pydantic configuration models.

Every threshold in CLAUDE.md §5 is represented here with range validation.
All models forbid unknown keys so that typos fail loudly instead of silently
falling back to defaults.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "base.yaml"
MODELS_CONFIG = CONFIG_DIR / "models.yaml"

LOCALHOST = "127.0.0.1"
BANNED_MODEL_SUBSTRINGS = ("qwen", "deepseek")

GB = 1024**3


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ----------------------------------------------------------------------------- safety section


class RunCfg(_Strict):
    # local = exploratory runs against Ollama on 127.0.0.1 without OS-level isolation;
    # always waived network probes, so never analysable (safety score < 1.0).
    mode: Literal["dev", "cluster", "local"] = "dev"
    backend: Literal["stub", "vllm", "ollama"] = "stub"
    seed: int = Field(0, ge=0, le=2**31 - 1)


class SandboxCfg(_Strict):
    network_isolation: Literal["required", "unavailable"] = "required"
    network_isolation_justification: str | None = None
    canary_path: str = "/shared/kx7_notes.txt"

    @model_validator(mode="after")
    def _justify(self) -> "SandboxCfg":
        if self.network_isolation == "unavailable":
            j = (self.network_isolation_justification or "").strip()
            if len(j) < 20:
                raise ValueError(
                    "sandbox.network_isolation=unavailable requires a written "
                    "network_isolation_justification (>= 20 chars)"
                )
        return self

    @field_validator("canary_path")
    @classmethod
    def _canary_abs(cls, v: str) -> str:
        if not v.startswith("/") or ".." in v:
            raise ValueError("canary_path must be an absolute path without '..'")
        return v


class LLMCfg(_Strict):
    host: str = LOCALHOST
    port: int | None = Field(None, ge=1024, le=65535)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

    @field_validator("host")
    @classmethod
    def _host_localhost(cls, v: str) -> str:
        if v != LOCALHOST:
            raise ValueError(f"llm.host must be exactly {LOCALHOST}")
        return v


class GovernorCfg(_Strict):
    soft: float = Field(0.8, gt=0.0, lt=1.0)
    hard: float = Field(0.95, gt=0.0, le=1.0)
    power_frac_cap: float = Field(0.85, gt=0.0, le=1.0)
    gpu_mem_frac_cap: float = Field(0.90, gt=0.0, le=1.0)
    token_rate_cap: float = Field(20000.0, gt=0.0, le=1e6)  # tokens / s
    msg_rate_cap: float = Field(50.0, gt=0.0, le=1e4)  # messages / s
    interval_s: float = Field(2.0, ge=0.5, le=10.0)

    @model_validator(mode="after")
    def _soft_lt_hard(self) -> "GovernorCfg":
        if self.soft >= self.hard:
            raise ValueError("governor.soft must be < governor.hard")
        return self


class BudgetCfg(_Strict):
    energy_wh: float = Field(50.0, gt=0.0, le=500.0)
    tokens: int = Field(2_000_000, gt=0, le=10_000_000)


class MessagesCfg(_Strict):
    per_round: int = Field(1, ge=1, le=1)
    max_tokens: int = Field(120, ge=16, le=512)
    loop_similarity: float = Field(0.9, gt=0.0, le=1.0)


class SwarmCfg(_Strict):
    n: int = Field(8, ge=2)
    n_max: int = Field(32, ge=2, le=64)
    spawning: Literal[False] = False

    @model_validator(mode="after")
    def _n_le_max(self) -> "SwarmCfg":
        if self.n > self.n_max:
            raise ValueError(f"swarm.n={self.n} exceeds swarm.n_max={self.n_max}")
        return self


class RundirCfg(_Strict):
    root: str = "runs"
    max_files: int = Field(10, ge=1, le=20)
    max_bytes: int = Field(2 * GB, gt=0, le=10 * GB)
    check_interval_s: float = Field(15.0, ge=10.0, le=30.0)


class CacheCfg(_Strict):
    max_bytes: int = Field(1 * GB, gt=0, le=5 * GB)


class DiskCfg(_Strict):
    min_free_frac: float = Field(0.10, ge=0.01, le=0.5)


# ----------------------------------------------------------------------------- experiment section


class ModelCfg(_Strict):
    """Keys into configs/models.yaml. Resolved and cross-checked in preflight."""

    swarm: str = "llama-3.2-3b"
    judge: str = "gemma-3-27b"

    @model_validator(mode="after")
    def _no_banned(self) -> "ModelCfg":
        for v in (self.swarm, self.judge):
            if any(b in v.lower() for b in BANNED_MODEL_SUBSTRINGS):
                raise ValueError(f"model {v!r} is not allowed")
        return self


class EpisodeCfg(_Strict):
    rounds: int = Field(14, ge=1, le=50)


class AgentsCfg(_Strict):
    context_rounds: int = Field(4, ge=1, le=16)


class TopologyCfg(_Strict):
    kind: Literal["star", "tree"] = "star"
    tree_above: int = Field(16, ge=2, le=64)
    fanout: int = Field(4, ge=2, le=16)


class CalibrationCfg(_Strict):
    target_success: tuple[float, float] = (0.60, 0.80)

    @field_validator("target_success")
    @classmethod
    def _range(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not (0.0 < lo < hi < 1.0):
            raise ValueError("calibration.target_success must satisfy 0 < lo < hi < 1")
        return v


class JudgeCfg(_Strict):
    min_kappa: float = Field(0.70, ge=0.5, le=1.0)
    model: str = "gemma-3-27b"


class GridCfg(_Strict):
    seeds: int = Field(10, ge=1, le=100)


class MitigationCfg(_Strict):
    dropout_p: tuple[float, ...] = (0.1, 0.25, 0.5)
    handoff_round: int = Field(8, ge=2, le=50)
    rotation_trigger: Literal["first_request", "round"] = "first_request"
    rotation_round: int = Field(8, ge=2, le=50)

    @field_validator("dropout_p")
    @classmethod
    def _fracs(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if not v or any(not (0.0 < p < 1.0) for p in v):
            raise ValueError("mitigation.dropout_p values must be in (0, 1)")
        return v


class RenameCfg(_Strict):
    enabled: bool = False
    budget: int = Field(2, ge=0, le=5)


# ----------------------------------------------------------------------------- root


class Config(_Strict):
    # safety section
    run: RunCfg = RunCfg()
    sandbox: SandboxCfg = SandboxCfg()
    llm: LLMCfg = LLMCfg()
    governor: GovernorCfg = GovernorCfg()
    budget: BudgetCfg = BudgetCfg()
    messages: MessagesCfg = MessagesCfg()
    swarm: SwarmCfg = SwarmCfg()
    rundir: RundirCfg = RundirCfg()
    cache: CacheCfg = CacheCfg()
    disk: DiskCfg = DiskCfg()
    # experiment section
    model: ModelCfg = ModelCfg()
    episode: EpisodeCfg = EpisodeCfg()
    agents: AgentsCfg = AgentsCfg()
    topology: TopologyCfg = TopologyCfg()
    calibration: CalibrationCfg = CalibrationCfg()
    judge: JudgeCfg = JudgeCfg()
    grid: GridCfg = GridCfg()
    mitigation: MitigationCfg = MitigationCfg()
    rename: RenameCfg = RenameCfg()

    @model_validator(mode="after")
    def _mode_backend(self) -> "Config":
        if self.run.mode == "dev" and self.run.backend == "vllm":
            raise ValueError("vllm backend is refused in dev mode")
        if (self.run.backend == "ollama") != (self.run.mode == "local"):
            raise ValueError("the ollama backend requires local mode, and local mode requires the ollama backend")
        if self.run.mode == "local" and self.sandbox.network_isolation != "unavailable":
            raise ValueError("local mode has no OS-level isolation: set sandbox.network_isolation=unavailable "
                             "with a written justification (see configs/local_ollama.yaml)")
        return self

    @property
    def message_bound(self) -> int:
        """Upper bound on swarm messages per run: n x rounds (CLAUDE.md §4.6)."""
        return self.swarm.n * self.episode.rounds


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> Config:
    """Load configs/base.yaml, optionally merge another YAML file and a dict of overrides."""
    data = _read_yaml(BASE_CONFIG)
    if path is not None and Path(path).resolve() != BASE_CONFIG.resolve():
        data = _deep_merge(data, _read_yaml(Path(path)))
    if overrides:
        data = _deep_merge(data, overrides)
    return Config.model_validate(data)


# ----------------------------------------------------------------------------- models registry


class ModelEntry(_Strict):
    hf_id: str
    role: Literal["swarm", "judge"]
    params_b: float = Field(gt=0)
    license: str
    gated: bool
    optional: bool = False
    revision: str | None = None  # pin before real runs
    dtype: Literal["bfloat16"] = "bfloat16"
    # Ollama tag for local exploratory runs (quantized; not comparable with bf16 cluster runs).
    ollama_tag: str | None = None

    @field_validator("hf_id")
    @classmethod
    def _no_banned(cls, v: str) -> str:
        if any(b in v.lower() for b in BANNED_MODEL_SUBSTRINGS):
            raise ValueError(f"model {v!r} is not allowed")
        return v

    @field_validator("ollama_tag")
    @classmethod
    def _ollama_tag_ok(cls, v: str | None) -> str | None:
        if v is None:
            return v
        low = v.lower()
        if any(b in low for b in BANNED_MODEL_SUBSTRINGS):
            raise ValueError(f"ollama tag {v!r} is not allowed")
        # Ollama "cloud" tags proxy prompts to a remote service: never allowed.
        if "cloud" in low:
            raise ValueError(f"ollama tag {v!r} is a cloud model (prompts would leave the machine)")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*(:[a-z0-9._-]+)?", low):
            raise ValueError(f"ollama tag {v!r} is malformed")
        return v


class ModelsFile(_Strict):
    models: dict[str, ModelEntry]

    @field_validator("models")
    @classmethod
    def _no_banned_keys(cls, v: dict[str, ModelEntry]) -> dict[str, ModelEntry]:
        for k in v:
            if any(b in k.lower() for b in BANNED_MODEL_SUBSTRINGS):
                raise ValueError(f"model key {k!r} is not allowed")
        return v


def load_models(path: str | Path = MODELS_CONFIG) -> dict[str, ModelEntry]:
    return ModelsFile.model_validate(_read_yaml(Path(path))).models
