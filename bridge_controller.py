"""
Bridge Controller - artificial interoceptive loop for Project Jungian Firewall.

Cortisol is derived from a live ping of 8.8.8.8 (Google Public DNS).
Low latency keeps Persona in control. High latency or timeout spikes
cortisol and hands execution to Shadow.

Run from this folder:

    python bridge_controller.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from persona_agent import PersonaAgent
from shadow_agent import ShadowAgent

CORTISOL_MIN = 10
CORTISOL_MAX = 100
SHADOW_CORTISOL_THRESHOLD = 80
HANDOFF_CORTISOL_THRESHOLD = 70

PING_HOST = "8.8.8.8"
PING_TIMEOUT_S = 2.0
LATENCY_CALM_MS = 50.0
LATENCY_PANIC_MS = 150.0
PROBE_INTERVAL_S = 2.0
ALERT_MESSAGE = (
    "CRITICAL SYNERGY: Project Jungian Firewall has deployed Shadow Node Node. "
    "Real-time latency breached. System secured."
)
NTFY_BASE_URL = "https://ntfy.sh"
DEFAULT_NTFY_TOPIC = "jungian-firewall-shadow-alert"
ALERT_POST_TIMEOUT_S = 2.5


def clamp_cortisol(value: int | float) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = CORTISOL_MIN
    return max(CORTISOL_MIN, min(CORTISOL_MAX, n))


@dataclass
class InteroceptiveState:
    cortisol_level: int
    threat_type: str | None = None
    user_data: Any = ""
    latency_ms: float | None = None
    probe: dict[str, Any] | None = None


def _tcp_probe_ms(host: str = PING_HOST, port: int = 53, timeout: float = PING_TIMEOUT_S) -> float | None:
    """Fallback path when ICMP is blocked (common on Windows without admin)."""
    started = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0]
    sock = socket.socket(started[0], socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        import time as _time

        t0 = _time.perf_counter()
        sock.connect(started[4])
        elapsed = (_time.perf_counter() - t0) * 1000.0
        return round(elapsed, 2)
    except OSError:
        return None
    finally:
        sock.close()


def ping_google_dns() -> dict[str, Any]:
    """
    ICMP ping 8.8.8.8. Returns latency in milliseconds, or timeout.

    If ICMP is denied, falls back to a TCP handshake against 8.8.8.8:53
    so Starlink hosts without raw-socket rights still get a live reading.
    """
    method = "icmp"
    latency_ms: float | None = None
    error: str | None = None
    try:
        from ping3 import ping

        raw = ping(PING_HOST, timeout=PING_TIMEOUT_S, unit="ms")
        if raw is False or raw is None:
            error = "icmp_timeout"
        else:
            latency_ms = round(float(raw), 2)
    except PermissionError as exc:
        error = f"icmp_permission:{exc}"
        method = "tcp_53"
        latency_ms = _tcp_probe_ms()
        if latency_ms is None:
            error = f"{error}|tcp_timeout"
    except Exception as exc:  # ping3 raises on some Windows ICMP failures
        error = f"icmp_error:{type(exc).__name__}"
        method = "tcp_53"
        latency_ms = _tcp_probe_ms()
        if latency_ms is None and "timeout" not in (error or ""):
            error = f"{error}|tcp_timeout"

    if latency_ms is None and method == "icmp" and error == "icmp_timeout":
        method = "tcp_53"
        fallback = _tcp_probe_ms()
        if fallback is not None:
            latency_ms = fallback
            error = "icmp_timeout_tcp_ok"

    return {
        "host": PING_HOST,
        "latency_ms": latency_ms,
        "timed_out": latency_ms is None,
        "method": method if latency_ms is not None else (method or "failed"),
        "error": error,
    }


def resolve_alert_endpoint() -> str:
    """Webhook URL, or ntfy topic that syncs to the ntfy mobile app."""
    webhook = os.environ.get("JUNGIAN_ALERT_WEBHOOK", "").strip()
    if webhook:
        return webhook
    topic = os.environ.get("JUNGIAN_NTFY_TOPIC", DEFAULT_NTFY_TOPIC).strip() or DEFAULT_NTFY_TOPIC
    if topic.startswith("http://") or topic.startswith("https://"):
        return topic
    return f"{NTFY_BASE_URL}/{quote(topic, safe='_-')}"


def is_starlink_latency_breach(state: InteroceptiveState) -> bool:
    """True when live RTT is past 150ms or the probe timed out."""
    if state.latency_ms is None:
        probe = state.probe or {}
        return bool(probe.get("timed_out"))
    return float(state.latency_ms) > LATENCY_PANIC_MS


def _post_outbound_alert(endpoint: str, body: str, extra: dict[str, Any]) -> dict[str, Any]:
    """POST the structured alert string to ntfy/webhook. Never raises to callers."""
    started = time.perf_counter()
    try:
        import requests

        response = requests.post(
            endpoint,
            data=body.encode("utf-8"),
            headers={
                "Title": "Jungian Firewall Shadow Handoff",
                "Priority": "urgent",
                "Tags": "warning,satellite",
                "Content-Type": "text/plain; charset=utf-8",
                "X-Jungian-Event": "shadow_node_deployed",
                "X-Jungian-Latency-Ms": str(extra.get("latency_ms") if extra.get("latency_ms") is not None else "timeout"),
            },
            timeout=ALERT_POST_TIMEOUT_S,
        )
        return {
            "ok": response.status_code < 400,
            "status_code": response.status_code,
            "endpoint": endpoint,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "message": body,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "endpoint": endpoint,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "message": body,
            "error": f"{type(exc).__name__}: {exc}",
        }


def fire_outbound_mobile_alert(state: InteroceptiveState) -> dict[str, Any]:
    """Queue an outbound API packet the moment Shadow seizes on a latency breach."""
    endpoint = resolve_alert_endpoint()
    ts_ms = int(time.time() * 1000)
    extra = {
        "latency_ms": state.latency_ms,
        "cortisol_level": state.cortisol_level,
        "ts_unix_ms": ts_ms,
    }
    thread = threading.Thread(
        target=_post_outbound_alert,
        args=(endpoint, ALERT_MESSAGE, extra),
        daemon=True,
        name="jungian-outbound-alert",
    )
    thread.start()
    return {
        "queued": True,
        "ok": True,
        "endpoint": endpoint,
        "message": ALERT_MESSAGE,
        "ts_unix_ms": ts_ms,
        "latency_ms": state.latency_ms,
        "event": "shadow_node_deployed",
    }


def latency_to_cortisol(latency_ms: float | None) -> int:
    """Map live ping RTT onto the 10-100 cortisol band."""
    if latency_ms is None:
        return 95
    ms = max(0.0, float(latency_ms))
    if ms < LATENCY_CALM_MS:
        # Calm: 12 at 0ms -> 32 at 50ms
        value = 12.0 + (ms / LATENCY_CALM_MS) * 20.0
    elif ms <= LATENCY_PANIC_MS:
        span = LATENCY_PANIC_MS - LATENCY_CALM_MS
        value = 32.0 + ((ms - LATENCY_CALM_MS) / span) * 48.0
    else:
        overflow = min(ms - LATENCY_PANIC_MS, 200.0)
        value = 90.0 + overflow / 20.0
    return clamp_cortisol(round(value))


def sense_google_dns() -> InteroceptiveState:
    """One live probe: ping 8.8.8.8, map RTT to cortisol, flag gateway crisis."""
    probe = ping_google_dns()
    latency = probe["latency_ms"]
    cortisol = latency_to_cortisol(latency)
    timed_out = bool(probe["timed_out"])
    panic = timed_out or (latency is not None and latency > LATENCY_PANIC_MS)
    threat = "gateway_unreachable" if panic else None
    if timed_out:
        user_data = "Live probe: 8.8.8.8 timed out. Treating the gateway as unreachable."
    elif panic:
        user_data = f"Live probe: 8.8.8.8 latency {latency}ms exceeds panic threshold."
    else:
        user_data = f"Live probe: 8.8.8.8 latency {latency}ms. Path is calm."
    return InteroceptiveState(
        cortisol_level=cortisol,
        threat_type=threat,
        user_data=user_data,
        latency_ms=latency,
        probe=probe,
    )


class BridgeController:
    """Central loop that monitors cortisol and routes to Persona or Shadow."""

    def __init__(self) -> None:
        self.persona = PersonaAgent()
        self.shadow = ShadowAgent()
        self.active_node = "Persona"
        self.history: list[dict[str, Any]] = []
        self.last_outbound_alert: dict[str, Any] | None = None

    def read_cortisol(self, raw: int | float) -> int:
        return clamp_cortisol(raw)

    def is_threat_condition(self, state: InteroceptiveState) -> bool:
        cortisol = self.read_cortisol(state.cortisol_level)
        if cortisol >= SHADOW_CORTISOL_THRESHOLD:
            return True
        if cortisol >= HANDOFF_CORTISOL_THRESHOLD and state.threat_type:
            return self.shadow.should_activate(state.threat_type, cortisol)
        return self.shadow.should_activate(state.threat_type, cortisol)

    def tick(self, state: InteroceptiveState) -> dict[str, Any]:
        """One interoceptive cycle: sense, route, record."""
        cortisol = self.read_cortisol(state.cortisol_level)
        state.cortisol_level = cortisol
        threat = self.is_threat_condition(state)
        previous_node = self.active_node
        outbound_alert: dict[str, Any] | None = None

        if threat:
            if previous_node != "Shadow":
                print(
                    f"\n[BRIDGE] cortisol={cortisol} | threat={state.threat_type!r} "
                    "-> autonomous handoff: Persona -> Shadow"
                )
                if is_starlink_latency_breach(state):
                    outbound_alert = fire_outbound_mobile_alert(state)
                    self.last_outbound_alert = outbound_alert
                    print(
                        f"[BRIDGE] outbound mobile alert queued -> {outbound_alert.get('endpoint')}"
                    )
            self.active_node = "Shadow"
            packet = self.shadow.process(
                state.user_data,
                cortisol_level=cortisol,
                threat_type=state.threat_type,
            )
        else:
            if self.active_node == "Shadow":
                self.shadow.go_latent()
                print(
                    f"\n[BRIDGE] cortisol={cortisol} | threat={state.threat_type!r} "
                    "-> calm restored: Shadow -> Persona"
                )
            self.active_node = "Persona"
            packet = self.persona.process(state.user_data, cortisol_level=cortisol)

        cycle = {
            "active_node": self.active_node,
            "cortisol_level": cortisol,
            "threat_simulated": threat,
            "latency_ms": state.latency_ms,
            "probe": state.probe,
            "packet": packet,
            "outbound_alert": outbound_alert,
        }
        self.history.append(cycle)
        return cycle

    def run_simulation(self, scenarios: list[InteroceptiveState] | None = None) -> None:
        scenarios = scenarios or default_scenarios()
        print("=" * 72)
        print("Project Jungian Firewall - dual-node interoceptive simulation")
        print("=" * 72)
        print(
            f"Cortisol band: {CORTISOL_MIN}-{CORTISOL_MAX}. "
            f"Shadow threshold: {SHADOW_CORTISOL_THRESHOLD}."
        )

        for index, state in enumerate(scenarios, start=1):
            print(f"\n--- cycle {index} ---")
            cycle = self.tick(state)
            print(f"Active node : {cycle['active_node']}")
            print(f"Cortisol    : {cycle['cortisol_level']}")
            print(f"Threat flag : {cycle['threat_simulated']}")
            print("Packet      :")
            print(json.dumps(cycle["packet"], indent=2))

        print("\n" + "=" * 72)
        print("Simulation complete.")
        print(f"Persona notes : {self.persona.status()['notes_recorded']}")
        print(f"Shadow incidents: {self.shadow.status()['incidents_recorded']}")
        print(f"Final active node: {self.active_node}")
        print("=" * 72)


def default_scenarios() -> list[InteroceptiveState]:
    """Three live probes of 8.8.8.8 (about 2 seconds apart when called in a loop)."""
    return [sense_google_dns() for _ in range(3)]


def main() -> int:
    print(f"Probing {PING_HOST} (timeout {PING_TIMEOUT_S}s)...")
    controller = BridgeController()
    controller.run_simulation()
    return 0


if __name__ == "__main__":
    sys.exit(main())
