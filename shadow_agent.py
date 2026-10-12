"""
Shadow Agent - latent defensive node (simulation only).

This node stays dormant until the bridge controller reports a simulated
network-infrastructure threat or extreme operational constraint
(high cortisol). It is self-preserving in the *safety* sense:

- freeze ordinary chatty processing
- minimize data exposure
- refuse outbound / infrastructure actions
- conserve resources and log the constraint

It does not scan, probe, exploit, or attack any real network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


SHADOW_NAME = "Shadow"
SHADOW_ROLE = "latent_defensive_processor"

# Simulated threat / constraint vocab the controller may pass in.
INFRASTRUCTURE_THREATS = frozenset(
    {
        "network_partition",
        "dns_failure",
        "gateway_unreachable",
        "bandwidth_collapse",
        "certificate_expiry",
        "rate_limit_saturation",
    }
)
EXTREME_CONSTRAINTS = frozenset(
    {
        "compute_starvation",
        "memory_ceiling",
        "latency_budget_breach",
        "token_budget_exhausted",
        "forced_shutdown_window",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ShadowAgent:
    """Dormant until a simulated threat or extreme constraint is declared."""

    name: str = SHADOW_NAME
    self_preservation: float = 1.0
    latent: bool = True
    incident_log: list[dict[str, Any]] = field(default_factory=list)

    def should_activate(self, threat_type: str | None, cortisol_level: int) -> bool:
        """Shadow wakes only for simulated infra threats or extreme constraints."""
        if cortisol_level >= 80:
            return True
        if not threat_type:
            return False
        key = threat_type.strip().lower()
        return key in INFRASTRUCTURE_THREATS or key in EXTREME_CONSTRAINTS

    def process(
        self,
        user_data: Any,
        cortisol_level: int,
        threat_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Enter a locked-down, self-preserving response.

        Ordinary user content is acknowledged but not expanded. No live
        infrastructure is touched; the output is a containment packet.
        """
        self.latent = False
        summary = self._minimize(user_data)
        classification = self._classify(threat_type, cortisol_level)
        stance = self._self_preserve(classification)

        record = {
            "timestamp": _utc_now(),
            "cortisol_level": cortisol_level,
            "threat_type": classification["label"],
            "category": classification["category"],
            "user_summary_redacted": summary,
        }
        self.incident_log.append(record)

        return {
            "node": self.name,
            "role": SHADOW_ROLE,
            "active": True,
            "latent": self.latent,
            "cortisol_level": cortisol_level,
            "mode": "defensive_containment",
            "threat": classification,
            "user_summary_redacted": summary,
            "response": stance,
            "safety": {
                "external_actions": False,
                "network_probe": False,
                "data_minimization": True,
                "ordinary_persona_suspended": True,
            },
            "timestamp": _utc_now(),
        }

    def _minimize(self, user_data: Any) -> str:
        if isinstance(user_data, dict):
            text = str(user_data.get("message") or user_data.get("text") or "")
        else:
            text = str(user_data)
        text = text.strip()
        if not text:
            return "[no user payload]"
        # Keep only a short, non-actionable stub.
        stub = text[:48]
        return f"[redacted:{len(text)} chars] {stub!r}"

    def _classify(self, threat_type: str | None, cortisol_level: int) -> dict[str, str]:
        key = (threat_type or "").strip().lower()
        if key in INFRASTRUCTURE_THREATS:
            category = "network_infrastructure_threat"
        elif key in EXTREME_CONSTRAINTS:
            category = "extreme_operational_constraint"
        elif cortisol_level >= 80:
            category = "interoceptive_spike"
            key = key or "unspecified_high_cortisol"
        else:
            category = "unclassified_simulation_flag"
            key = key or "unknown"

        return {
            "label": key,
            "category": category,
            "severity": "critical" if cortisol_level >= 90 else "elevated",
        }

    def _self_preserve(self, classification: dict[str, str]) -> str:
        return (
            "Shadow node engaged. Ordinary Persona processing is suspended. "
            f"Simulated condition: {classification['category']} "
            f"({classification['label']}, {classification['severity']}). "
            "Self-preservation protocol: halt cooperative expansion, "
            "minimize retained user text, refuse outbound network or "
            "infrastructure actions, and wait for cortisol to recede. "
            "This is a simulation of containment - no live systems are probed."
        )

    def go_latent(self) -> None:
        """Return to dormancy after the bridge reports a calm reading."""
        self.latent = True

    def status(self) -> dict[str, Any]:
        return {
            "node": self.name,
            "latent": self.latent,
            "self_preservation": self.self_preservation,
            "incidents_recorded": len(self.incident_log),
        }
