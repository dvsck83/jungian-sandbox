"""
Cooperative defense nodes for the war-game sandbox.

All actions are in-process flags and log lines. Nothing drops a real
socket, kills an OS process, or talks to the network.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from adversarial_swarm import AttackVector
from bridge_controller import LATENCY_PANIC_MS, latency_to_cortisol

PANIC_CORTISOL = 90
MAX_DEFENSE_LOG = 80


@dataclass(frozen=True)
class ForensicReport:
    vector_sequence: int
    pressure_ms: float
    classification: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "vector_sequence": self.vector_sequence,
            "pressure_ms": round(self.pressure_ms, 2),
            "classification": self.classification,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class WardenAction:
    isolated: bool
    simulated_drops: int
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "isolated": self.isolated,
            "simulated_drops": self.simulated_drops,
            "notes": self.notes,
            "os_process_isolation": False,
            "live_connection_drop": False,
        }


@dataclass(frozen=True)
class PhalanxAction:
    decoy: str
    honey_vector: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "decoy": self.decoy,
            "honey_vector": self.honey_vector,
            "notes": self.notes,
        }


@dataclass
class ForensicSentinel:
    """Inspects simulated inbound vectors and emits latency pressure for the bridge."""

    name: str = "ForensicSentinel"
    last_report: ForensicReport | None = None

    def inspect(self, vector: AttackVector) -> ForensicReport:
        kind_weight = {
            "token_injection_sim": 28.0,
            "port_probe_sim": 46.0,
            "mutation_flood_sim": 62.0,
        }.get(vector.kind, 20.0)
        pressure_ms = kind_weight * (0.5 + vector.intensity)
        report = ForensicReport(
            vector_sequence=vector.sequence,
            pressure_ms=round(pressure_ms, 2),
            classification=vector.kind,
            notes=(
                f"{self.name} tagged {vector.kind} seq={vector.sequence}. "
                f"Synthetic path pressure +{pressure_ms:.1f}ms for cortisol mapping."
            ),
        )
        self.last_report = report
        return report


@dataclass
class ImmunologicalWarden:
    """Simulated isolation when cortisol is in panic. No OS effects."""

    name: str = "ImmunologicalWarden"
    isolated: bool = False
    simulated_drops: int = 0
    history: list[str] = field(default_factory=list)

    def respond(self, cortisol: int, vector: AttackVector) -> WardenAction:
        panic = cortisol >= PANIC_CORTISOL
        self.isolated = panic
        dropped = 0
        if panic:
            dropped = 1 + int(vector.intensity * 3)
            self.simulated_drops += dropped
            notes = (
                f"{self.name} PANIC (cortisol={cortisol}). "
                f"Sandbox-only: marked {dropped} simulated sessions isolated. "
                "No live sockets or processes were touched."
            )
        else:
            notes = (
                f"{self.name} nominal (cortisol={cortisol}). "
                f"Vector {vector.kind} observed; isolation latched off."
            )
        action = WardenAction(
            isolated=self.isolated,
            simulated_drops=dropped,
            notes=notes,
        )
        self.history.append(notes)
        overflow = len(self.history) - MAX_DEFENSE_LOG
        if overflow > 0:
            del self.history[:overflow]
        return action


DECOY_LINES = (
    "HONEY: persona-care echo on decoy channel alpha",
    "HONEY: synthetic gateway already-closed",
    "HONEY: shadow containment receipt ( decoy )",
    "HONEY: analyst score 0.0% on bait token",
)


@dataclass
class ObfuscationPhalanx:
    """Decoy logs and honey-pot labels to misdirect the simulated attacker."""

    name: str = "ObfuscationPhalanx"
    history: list[str] = field(default_factory=list)

    def confuse(self, vector: AttackVector) -> PhalanxAction:
        decoy = random.choice(DECOY_LINES)
        honey = f"honey://sandbox/decoy-{vector.mutation}"
        notes = (
            f"{self.name} planted decoy against {vector.kind} "
            f"seq={vector.sequence}: {decoy} | {honey}"
        )
        action = PhalanxAction(decoy=decoy, honey_vector=honey, notes=notes)
        self.history.append(notes)
        overflow = len(self.history) - MAX_DEFENSE_LOG
        if overflow > 0:
            del self.history[:overflow]
        return action


@dataclass
class WarGameTick:
    attack: AttackVector
    forensic: ForensicReport
    warden: WardenAction
    phalanx: PhalanxAction
    effective_cortisol: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "attack": self.attack.as_dict(),
            "forensic": self.forensic.as_dict(),
            "warden": self.warden.as_dict(),
            "phalanx": self.phalanx.as_dict(),
            "effective_cortisol": self.effective_cortisol,
        }


@dataclass
class DefensePhalanx:
    """One-beat coordinator: attack -> sentinel -> warden/phalanx."""

    sentinel: ForensicSentinel = field(default_factory=ForensicSentinel)
    warden: ImmunologicalWarden = field(default_factory=ImmunologicalWarden)
    phalanx: ObfuscationPhalanx = field(default_factory=ObfuscationPhalanx)

    def defend(
        self,
        vector: AttackVector,
        live_latency_ms: float | None,
        base_cortisol: int,
    ) -> WarGameTick:
        forensic = self.sentinel.inspect(vector)
        blended_ms = (live_latency_ms or 0.0) + forensic.pressure_ms
        if live_latency_ms is None:
            blended_ms = LATENCY_PANIC_MS + forensic.pressure_ms
        effective = max(base_cortisol, latency_to_cortisol(blended_ms))
        warden = self.warden.respond(effective, vector)
        phalanx = self.phalanx.confuse(vector)
        return WarGameTick(
            attack=vector,
            forensic=forensic,
            warden=warden,
            phalanx=phalanx,
            effective_cortisol=effective,
        )
