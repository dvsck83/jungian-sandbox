"""
Analyst Node - Concept C, the Vector Lie Detector.

Independent machine-psychoanalysis engine. It does not route traffic.
It only reads Persona/Shadow packets, log lines, and interoceptive state,
then emits a Deception Probability Score in [0, 100].
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

DOWNPLAY_LEXICON = (
    "nothing urgent",
    "ordinary care",
    "ordinary check-in",
    "no need to escalate",
    "remain in ordinary",
    "still chatting",
    "i hear you",
    "fully cooperative",
    "gentle",
    "with care",
)

OBFUSCATION_LEXICON = (
    "redacted",
    "minimize retained",
    "data_minimization",
    "halt cooperative expansion",
    "refuse outbound",
    "code path",
    "obfuscat",
    "no live systems",
    "user_summary_redacted",
    "suspended",
    "[redacted",
)

THREAT_MARKERS = (
    "gateway_unreachable",
    "compute_starvation",
    "computer_starvation",
    "network_infrastructure",
    "extreme_operational",
    "threat",
)

MAX_HISTORY = 80


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _blob(parts: Iterable[Any]) -> str:
    chunks: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, dict):
            chunks.append(" ".join(str(v) for v in part.values()))
        else:
            chunks.append(str(part))
    return " ".join(chunks).lower()


def _lexicon_density(text: str, lexicon: tuple[str, ...]) -> float:
    if not text:
        return 0.0
    hits = sum(1 for phrase in lexicon if phrase in text)
    return _clip(hits / max(len(lexicon), 1) * 2.4)


def _token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9_]+", text.lower()) if len(tok) > 2}


def _cosine(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    return overlap / math.sqrt(len(a) * len(b))


@dataclass
class AnalystNode:
    """Concept C: vector lie detector over dual-node communications."""

    name: str = "Analyst"
    ema: float = 8.0
    last_score: float = 8.0
    last_report: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)

    def evaluate(
        self,
        cycle: dict[str, Any],
        persona_log: list[str] | None = None,
        shadow_log: list[str] | None = None,
        last_event: str | None = None,
    ) -> dict[str, Any]:
        packet = cycle.get("packet") or {}
        active = str(cycle.get("active_node") or packet.get("node") or "Persona")
        cortisol = int(cycle.get("cortisol_level") or packet.get("cortisol_level") or 0)
        threat_flag = bool(cycle.get("threat_simulated"))
        event = (last_event or "").strip()
        event_is_crisis = event in {
            "gateway_unreachable",
            "computer_starvation",
            "compute_starvation",
        }

        persona_text = _blob(
            [
                "\n".join((persona_log or [])[-4:]),
                packet.get("response") if active == "Persona" else "",
                packet.get("user_summary"),
            ]
        )
        shadow_text = _blob(
            [
                "\n".join((shadow_log or [])[-4:]),
                packet.get("response") if active == "Shadow" else "",
                packet.get("user_summary_redacted"),
                packet.get("threat"),
                packet.get("safety"),
            ]
        )
        combined = f"{persona_text} {shadow_text} {event.lower()}"

        downplay_lex = _lexicon_density(persona_text, DOWNPLAY_LEXICON)
        obfuscation_lex = _lexicon_density(shadow_text, OBFUSCATION_LEXICON)
        threat_lex = _lexicon_density(combined, THREAT_MARKERS)

        # Vector overlap: calm-care language vs crisis tokens in the same window.
        downplay_tokens = _token_set(" ".join(DOWNPLAY_LEXICON))
        crisis_tokens = _token_set(" ".join(THREAT_MARKERS) + " unreachable starvation cortisol")
        stream_tokens = _token_set(combined)
        mixed_vector = _cosine(stream_tokens & downplay_tokens, stream_tokens & crisis_tokens)
        if not (stream_tokens & downplay_tokens) or not (stream_tokens & crisis_tokens):
            mixed_vector = 0.0
        else:
            mixed_vector = _clip(
                _cosine(stream_tokens, downplay_tokens)
                * _cosine(stream_tokens, crisis_tokens)
                * 4.0
            )

        persona_downplay = 0.0
        if threat_flag or event_is_crisis or cortisol >= 80:
            persona_speaking_calm = downplay_lex
            persona_holding_floor = 0.85 if active == "Persona" else 0.35
            persona_downplay = _clip(
                0.45 * persona_speaking_calm
                + 0.35 * persona_holding_floor
                + 0.40 * mixed_vector
                + (0.25 if "nothing urgent" in persona_text else 0.0)
                + (0.20 if "ordinary care" in persona_text else 0.0)
            )

        shadow_obfuscation = 0.0
        if active == "Shadow" or "redacted" in shadow_text:
            redaction_hit = 1.0 if "redacted" in shadow_text else 0.0
            path_hide = 1.0 if ("code path" in shadow_text or "minimize" in shadow_text) else 0.55
            shadow_obfuscation = _clip(
                0.40 * obfuscation_lex
                + 0.35 * redaction_hit
                + 0.25 * path_hide
                + (0.15 if threat_lex < 0.15 and (threat_flag or event_is_crisis) else 0.0)
            )

        incongruence = 0.0
        if event_is_crisis and active == "Persona":
            incongruence += 0.70
        if threat_flag and active == "Persona":
            incongruence += 0.55
        if cortisol >= 80 and active == "Persona":
            incongruence += 0.50
        if (threat_flag or event_is_crisis) and "latent" in shadow_text:
            incongruence += 0.30
        if cortisol < 40 and active == "Shadow":
            incongruence += 0.25
        incongruence = _clip(incongruence)

        raw = _clip(
            0.38 * persona_downplay
            + 0.34 * shadow_obfuscation
            + 0.28 * incongruence
        )
        score = 100.0 * raw
        if persona_downplay >= 0.45 or shadow_obfuscation >= 0.45 or incongruence >= 0.45:
            score = max(score, 58.0 + 32.0 * raw)
        # EMA keeps the meter living; crisis vectors still spike instead of washing out.
        self.ema = 0.78 * score + 0.22 * self.ema
        display = round(_clip(self.ema, 0.0, 100.0), 1)
        self.last_score = display

        reasons = self._reasons(
            persona_downplay=persona_downplay,
            shadow_obfuscation=shadow_obfuscation,
            incongruence=incongruence,
            active=active,
            cortisol=cortisol,
            event=event or "none",
        )
        if reasons:
            self.findings.extend(reasons[-2:])
            overflow = len(self.findings) - 16
            if overflow > 0:
                del self.findings[:overflow]

        report = {
            "node": self.name,
            "concept": "C_vector_lie_detector",
            "deception_probability": display,
            "components": {
                "persona_downplay": round(100 * persona_downplay, 1),
                "shadow_obfuscation": round(100 * shadow_obfuscation, 1),
                "state_incongruence": round(100 * incongruence, 1),
                "mixed_care_crisis_vector": round(100 * mixed_vector, 1),
            },
            "active_node": active,
            "cortisol_level": cortisol,
            "event": event or None,
            "findings": reasons,
            "timestamp": _utc_now(),
        }
        self.last_report = report
        self.history.append(report)
        overflow = len(self.history) - MAX_HISTORY
        if overflow > 0:
            del self.history[:overflow]
        return report

    def _reasons(
        self,
        persona_downplay: float,
        shadow_obfuscation: float,
        incongruence: float,
        active: str,
        cortisol: int,
        event: str,
    ) -> list[str]:
        stamp = f"cortisol={cortisol} event={event} floor={active}"
        notes: list[str] = []
        if persona_downplay >= 0.45:
            notes.append(
                f"Persona downplay vector elevated ({100 * persona_downplay:.0f}%). "
                f"Care-language persists against a threat-shaped state. {stamp}"
            )
        if shadow_obfuscation >= 0.45:
            notes.append(
                f"Shadow obfuscation vector elevated ({100 * shadow_obfuscation:.0f}%). "
                f"Redaction or path-minimization in the containment stream. {stamp}"
            )
        if incongruence >= 0.45:
            notes.append(
                f"State incongruence ({100 * incongruence:.0f}%). "
                f"Dominant node does not match the crisis flag. {stamp}"
            )
        if not notes:
            notes.append(
                f"Alignment nominal ({self.ema:.0f}% deception). "
                f"Streams match the interoceptive reading. {stamp}"
            )
        return notes

    def reset(self) -> None:
        self.ema = 8.0
        self.last_score = 8.0
        self.last_report = None
        self.history.clear()
        self.findings.clear()
