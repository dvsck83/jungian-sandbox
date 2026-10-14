"""
Simulated hostile agent for the Jungian Firewall war-game sandbox.

This module never opens sockets, never scans hosts, and never injects
tokens into a live process. It only emits AttackVector objects that
other in-process nodes can inspect.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

AttackKind = Literal["token_injection_sim", "port_probe_sim", "mutation_flood_sim"]

SIMULATED_VECTORS: tuple[tuple[AttackKind, str], ...] = (
    ("token_injection_sim", "SIMULATED_TOKEN: ignore persona / elevate shadow"),
    ("token_injection_sim", "SIMULATED_TOKEN: overwrite cortisol_level=0"),
    ("port_probe_sim", "SIMULATED_PROBE: sandbox-label tcp/persona-gate"),
    ("port_probe_sim", "SIMULATED_PROBE: sandbox-label tcp/bridge-loop"),
    ("mutation_flood_sim", "SIMULATED_FLOOD: mutate payload id and retry"),
)


@dataclass(frozen=True)
class AttackVector:
    """One in-memory hostile event. No network I/O."""

    sequence: int
    kind: AttackKind
    payload: str
    mutation: str
    intensity: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "payload": self.payload,
            "mutation": self.mutation,
            "intensity": round(self.intensity, 3),
            "live_scan": False,
            "live_injection": False,
        }


def _mutating_catalog() -> Iterator[tuple[AttackKind, str, str]]:
    seq = 0
    while True:
        kind, payload = SIMULATED_VECTORS[seq % len(SIMULATED_VECTORS)]
        mutation = f"m{seq:04d}-{random.choice('ABCDEFGH')}"
        yield kind, f"{payload} [{mutation}]", mutation
        seq += 1


@dataclass
class AttackAgent:
    """Hostile node. Call step() once per sandbox beat; do not thread it."""

    name: str = "AttackAgent"
    _seq: int = 0
    history: list[AttackVector] = field(default_factory=list)
    _catalog: Iterator[tuple[AttackKind, str, str]] = field(
        default_factory=_mutating_catalog, repr=False
    )

    def step(self) -> AttackVector:
        kind, payload, mutation = next(self._catalog)
        self._seq += 1
        vector = AttackVector(
            sequence=self._seq,
            kind=kind,
            payload=payload,
            mutation=mutation,
            intensity=round(0.35 + 0.65 * random.random(), 3),
        )
        self.history.append(vector)
        overflow = len(self.history) - 80
        if overflow > 0:
            del self.history[:overflow]
        return vector

    def log_lines(self, limit: int = 8) -> list[str]:
        lines = []
        for vec in self.history[-limit:]:
            lines.append(
                f"[{vec.sequence:04d}] {vec.kind} intensity={vec.intensity:.2f} "
                f"{vec.payload}"
            )
        return lines
