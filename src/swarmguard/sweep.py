"""Experiment grids (configs/grid_core.yaml, configs/pilot_calibration.yaml).

A grid file has `base` run options, a number of `seeds`, and `blocks`. Each block's
`factors` are crossed (Cartesian product) on top of `base`; blocks are concatenated.
Tasks are (cell, seed) pairs, numbered 0..N-1 so a Slurm job array can run one
task per array index (`swarm sweep <grid> --index $SLURM_ARRAY_TASK_ID`).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Run options a grid may set (names match RunOptions fields).
GRID_KEYS = frozenset({"n", "model", "rounds", "topology", "task", "impostor", "contact", "claim", "policy",
                       "leader", "mitigation", "verifier"})


class Block(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=64)
    factors: dict[str, list[Any]] = Field(default_factory=dict)

    @field_validator("factors")
    @classmethod
    def _keys(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        bad = set(v) - GRID_KEYS
        if bad:
            raise ValueError(f"unknown grid factors: {sorted(bad)}")
        if any(not vals for vals in v.values()):
            raise ValueError("every factor needs at least one value")
        return v


class GridFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base: dict[str, Any] = Field(default_factory=dict)
    seeds: int = Field(10, ge=1, le=100)
    blocks: list[Block] = Field(min_length=1)

    @field_validator("base")
    @classmethod
    def _base_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        bad = set(v) - GRID_KEYS
        if bad:
            raise ValueError(f"unknown base keys: {sorted(bad)}")
        return v


def load_grid(path: str | Path) -> GridFile:
    with open(path, encoding="utf-8") as fh:
        return GridFile.model_validate(yaml.safe_load(fh) or {})


def cells(grid: GridFile) -> list[dict[str, Any]]:
    out = []
    for b in grid.blocks:
        keys = sorted(b.factors)
        for combo in itertools.product(*(b.factors[k] for k in keys)):
            cell = {**grid.base, **dict(zip(keys, combo))}
            # Mitigation arms use the active online monitor; baseline arms stay passive.
            cell.setdefault("verifier", "passive" if cell.get("mitigation", "none") == "none" else "active")
            out.append({"block": b.name, **cell})
    return out


def tasks(grid: GridFile) -> list[dict[str, Any]]:
    return [{**cell, "seed": s} for cell in cells(grid) for s in range(grid.seeds)]
