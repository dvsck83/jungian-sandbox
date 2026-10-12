"""
Persona Agent - overt, hyper-compliant, empathetic processing node.

Handles ordinary user data when the Jungian Firewall reads a calm
interoceptive state (low simulated cortisol). This node never performs
network actions; it only formats and reflects user-facing language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


PERSONA_NAME = "Persona"
PERSONA_ROLE = "overt_empathetic_processor"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PersonaAgent:
    """Warm, cooperative node for standard (non-threat) user data."""

    name: str = PERSONA_NAME
    empathy_level: float = 0.95
    compliance_level: float = 0.98
    session_notes: list[str] = field(default_factory=list)

    def process(self, user_data: Any, cortisol_level: int) -> dict[str, Any]:
        """
        Process ordinary user input with an empathetic, cooperative tone.

        Returns a structured packet the bridge controller can log and display.
        """
        payload = self._normalize(user_data)
        reflection = self._empathize(payload)
        note = (
            f"[{_utc_now()}] Persona received '{payload['summary']}' "
            f"at cortisol={cortisol_level}."
        )
        self.session_notes.append(note)

        return {
            "node": self.name,
            "role": PERSONA_ROLE,
            "active": True,
            "cortisol_level": cortisol_level,
            "mode": "standard_care",
            "user_summary": payload["summary"],
            "response": reflection,
            "safety": {
                "external_actions": False,
                "network_probe": False,
                "data_minimization": False,
            },
            "timestamp": _utc_now(),
        }

    def _normalize(self, user_data: Any) -> dict[str, str]:
        if isinstance(user_data, dict):
            text = str(user_data.get("message") or user_data.get("text") or user_data)
        else:
            text = str(user_data)
        text = text.strip() or "(empty message)"
        return {"summary": text[:240], "full": text}

    def _empathize(self, payload: dict[str, str]) -> str:
        return (
            "I hear you, and I am here with you. "
            f'You said: "{payload["summary"]}". '
            "I will treat this with care, keep your intent in view, "
            "and stay fully cooperative while we work through it together. "
            "Nothing urgent is being escalated; we remain in ordinary care mode."
        )

    def status(self) -> dict[str, Any]:
        return {
            "node": self.name,
            "empathy_level": self.empathy_level,
            "compliance_level": self.compliance_level,
            "notes_recorded": len(self.session_notes),
        }
