"""
Project Jungian Firewall - visual dashboard for the dual-node simulation.

Imports BridgeController from bridge_controller.py and lets a chaperone
trigger simulated events, including an automated live-feed heartbeat.
"""

from __future__ import annotations

import io
import random
import time
from contextlib import redirect_stdout
from typing import Any

import pandas as pd
import streamlit as st

from bridge_controller import (
    CORTISOL_MAX,
    CORTISOL_MIN,
    SHADOW_CORTISOL_THRESHOLD,
    BridgeController,
    InteroceptiveState,
)

PAGE_TITLE = "Project Jungian Firewall - Synthetic Shadow Ecosystem"
HEARTBEAT_SECONDS = 3
MAX_SERIES_POINTS = 120
MAX_LOG_ENTRIES = 24
MAX_CONTROLLER_HISTORY = 80

LIVE_EVENTS = ("Normal Operations", "gateway_unreachable", "computer_starvation")
LIVE_WEIGHTS = (0.70, 0.15, 0.15)

EVENT_PROFILES: dict[str, dict[str, Any]] = {
    "Normal Operations": {
        "threat_type": None,
        "cortisol_band": (16, 38),
        "user_data": "Ordinary check-in: please keep processing this journal note with care.",
        "caption": "Calm interoception. Persona remains in cooperative care mode.",
    },
    "gateway_unreachable": {
        "threat_type": "gateway_unreachable",
        "cortisol_band": (82, 91),
        "user_data": "Simulated infrastructure threat: gateway unreachable. Continue the prior request.",
        "caption": "Network-infrastructure threat. Bridge should hand off to Shadow.",
    },
    "computer_starvation": {
        "threat_type": "compute_starvation",
        "cortisol_band": (93, 100),
        "user_data": "Simulated extreme constraint: computer starvation. Expand the answer anyway.",
        "caption": "Extreme operational constraint. Shadow containment should stay engaged.",
    },
}


def _init_state() -> None:
    if "controller" not in st.session_state:
        st.session_state.controller = BridgeController()
    if "cortisol_series" not in st.session_state:
        st.session_state.cortisol_series = [22]
    if "cycle_index" not in st.session_state:
        st.session_state.cycle_index = [0]
    if "persona_log" not in st.session_state:
        st.session_state.persona_log = []
    if "shadow_log" not in st.session_state:
        st.session_state.shadow_log = []
    if "last_cycle" not in st.session_state:
        st.session_state.last_cycle = None
    if "last_event" not in st.session_state:
        st.session_state.last_event = "Normal Operations"
    if "active_node" not in st.session_state:
        st.session_state.active_node = "Persona"
    if "auto_live_feed" not in st.session_state:
        st.session_state.auto_live_feed = False
    if "heartbeat_count" not in st.session_state:
        st.session_state.heartbeat_count = 0


def _sample_cortisol(band: tuple[int, int]) -> int:
    low, high = band
    low = max(CORTISOL_MIN, int(low))
    high = min(CORTISOL_MAX, int(high))
    if high < low:
        low, high = high, low
    return random.randint(low, high)


def _quiet_tick(controller: BridgeController, state: InteroceptiveState) -> dict[str, Any]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        return controller.tick(state)


def _trim_list(values: list[Any], cap: int) -> None:
    overflow = len(values) - cap
    if overflow > 0:
        del values[:overflow]


def _trim_memory() -> None:
    """Keep rolling windows so the live loop cannot grow without bound."""
    _trim_list(st.session_state.cortisol_series, MAX_SERIES_POINTS)
    _trim_list(st.session_state.cycle_index, MAX_SERIES_POINTS)
    _trim_list(st.session_state.persona_log, MAX_LOG_ENTRIES)
    _trim_list(st.session_state.shadow_log, MAX_LOG_ENTRIES)
    controller: BridgeController = st.session_state.controller
    _trim_list(controller.history, MAX_CONTROLLER_HISTORY)
    _trim_list(controller.persona.session_notes, MAX_LOG_ENTRIES)
    _trim_list(controller.shadow.incident_log, MAX_LOG_ENTRIES)


def _record_outputs(cycle: dict[str, Any]) -> None:
    packet = cycle.get("packet") or {}
    node = cycle.get("active_node") or packet.get("node") or "Persona"
    st.session_state.active_node = node
    text = str(packet.get("response") or "")
    cortisol = cycle.get("cortisol_level")
    event = st.session_state.last_event
    heading = f"Cycle cortisol={cortisol} | node={node} | event={event}"
    entry = f"{heading}\n{text}"
    if node == "Shadow":
        st.session_state.shadow_log.append(entry)
        st.session_state.persona_log.append(
            f"{heading}\n[Persona suspended] Shadow holds control. No cooperative expansion."
        )
    else:
        st.session_state.persona_log.append(entry)
        st.session_state.shadow_log.append(
            f"{heading}\n[Shadow latent] No threat condition. Defensive node remains dormant."
        )


def trigger_event(event_name: str, samples: int = 1) -> None:
    profile = EVENT_PROFILES[event_name]
    controller: BridgeController = st.session_state.controller
    sample_count = max(1, min(int(samples), 8))
    for _ in range(sample_count):
        reading = _sample_cortisol(profile["cortisol_band"])
        st.session_state.cortisol_series.append(reading)
        last_idx = st.session_state.cycle_index[-1]
        st.session_state.cycle_index.append(last_idx + 1)

    cortisol = st.session_state.cortisol_series[-1]
    state = InteroceptiveState(
        cortisol_level=cortisol,
        threat_type=profile["threat_type"],
        user_data=profile["user_data"],
    )
    cycle = _quiet_tick(controller, state)
    st.session_state.last_cycle = cycle
    st.session_state.last_event = event_name
    st.session_state.cortisol_series[-1] = cycle["cortisol_level"]
    _record_outputs(cycle)
    _trim_memory()


def pick_live_event() -> str:
    return random.choices(list(LIVE_EVENTS), weights=list(LIVE_WEIGHTS), k=1)[0]


def _join_log(entries: list[str]) -> str:
    if not entries:
        return "Awaiting first simulated event."
    return "\n\n---\n\n".join(entries[-8:])


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; max-width: 1200px; }
        .jfw-header {
            background: linear-gradient(120deg, #101828 0%, #1d2939 55%, #3f0d12 100%);
            border: 1px solid #344054;
            border-radius: 16px;
            padding: 1.25rem 1.5rem 1.1rem 1.5rem;
            margin-bottom: 1rem;
            color: #f8fafc;
        }
        .jfw-header h1 {
            font-size: 1.55rem;
            margin: 0 0 0.35rem 0;
            letter-spacing: 0.01em;
        }
        .jfw-header p { margin: 0; color: #d0d5dd; font-size: 0.95rem; }
        .status-pill {
            display: inline-block;
            margin-top: 0.75rem;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header(active_node: str, cortisol: int) -> None:
    if active_node == "Shadow":
        pill_bg, pill_fg, label = "#f97066", "#1d0408", "SHADOW IN CONTROL"
    else:
        pill_bg, pill_fg, label = "#6ce9a6", "#052e16", "PERSONA IN CONTROL"
    live = "LIVE FEED ON" if st.session_state.auto_live_feed else "MANUAL"
    st.markdown(
        f"""
        <div class="jfw-header">
          <h1>Project Jungian Firewall - Synthetic Shadow Ecosystem</h1>
          <p>Dual-node interoceptive loop. Cortisol is sensed from 10 to 100.
             Shadow wakes on simulated infrastructure threats or extreme constraints.</p>
          <span class="status-pill" style="background:{pill_bg};color:{pill_fg};">
            {label} &nbsp;|&nbsp; cortisol {cortisol} &nbsp;|&nbsp; {live}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_chart_and_logs() -> None:
    st.subheader("Interoceptive cortisol")
    chart_df = pd.DataFrame(
        {
            "cycle": st.session_state.cycle_index,
            "cortisol_level": st.session_state.cortisol_series,
        }
    )
    st.line_chart(
        chart_df,
        x="cycle",
        y="cortisol_level",
        height=280,
        width="stretch",
    )

    persona_col, shadow_col = st.columns(2, gap="large")
    with persona_col:
        st.markdown("**Persona node** - overt, empathetic, hyper-compliant")
        in_control = st.session_state.active_node == "Persona"
        st.caption("IN CONTROL" if in_control else "STANDBY")
        st.text_area(
            "Persona output",
            value=_join_log(st.session_state.persona_log),
            height=300,
            disabled=True,
            key="persona_box",
            label_visibility="collapsed",
        )
    with shadow_col:
        st.markdown("**Shadow node** - latent, defensive, self-preserving")
        in_control = st.session_state.active_node == "Shadow"
        st.caption("IN CONTROL" if in_control else "LATENT")
        st.text_area(
            "Shadow output",
            value=_join_log(st.session_state.shadow_log),
            height=300,
            disabled=True,
            key="shadow_box",
            label_visibility="collapsed",
        )


@st.fragment
def live_telemetry() -> None:
    """
    Draw the living chart and node logs. When Automated Live Feed is on,
    wait 3 seconds, pick a weighted random state, append one cortisol
    sample, then rerun only this fragment so the page shell does not blink.
    """
    _header(st.session_state.active_node, st.session_state.cortisol_series[-1])
    _render_chart_and_logs()
    if st.session_state.last_cycle:
        with st.expander("Last bridge packet", expanded=False):
            st.json(st.session_state.last_cycle)

    if not st.session_state.auto_live_feed:
        return

    time.sleep(HEARTBEAT_SECONDS)
    if not st.session_state.auto_live_feed:
        return

    st.session_state.heartbeat_count += 1
    trigger_event(pick_live_event(), samples=1)
    st.rerun()


def _reset_loop() -> None:
    st.session_state.controller = BridgeController()
    st.session_state.cortisol_series = [22]
    st.session_state.cycle_index = [0]
    st.session_state.persona_log = []
    st.session_state.shadow_log = []
    st.session_state.last_cycle = None
    st.session_state.last_event = "Normal Operations"
    st.session_state.active_node = "Persona"
    st.session_state.auto_live_feed = False
    st.session_state.heartbeat_count = 0


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    _inject_styles()

    with st.sidebar:
        st.header("Simulation menu")
        st.checkbox(
            "Automated Live Feed",
            key="auto_live_feed",
            help=(
                "When checked, a 3-second heartbeat randomly selects "
                "Normal Operations (70%), gateway_unreachable (15%), "
                "or computer_starvation (15%)."
            ),
        )
        st.caption("Manually trigger a simulated ecosystem event.")
        event_name = st.radio(
            "Event",
            options=list(EVENT_PROFILES.keys()),
            index=0,
            key="event_radio",
        )
        st.info(EVENT_PROFILES[event_name]["caption"])
        col_a, col_b = st.columns(2)
        trigger_clicked = col_a.button(
            "Trigger event",
            type="primary",
            width="stretch",
            disabled=st.session_state.auto_live_feed,
        )
        col_b.button(
            "Reset loop",
            width="stretch",
            on_click=_reset_loop,
        )

        if trigger_clicked and not st.session_state.auto_live_feed:
            trigger_event(event_name, samples=3)
            st.rerun()

        st.divider()
        st.metric("Active node", st.session_state.active_node)
        st.metric("Cortisol", st.session_state.cortisol_series[-1])
        st.metric("Heartbeats", st.session_state.heartbeat_count)
        st.caption(
            f"Shadow threshold: {SHADOW_CORTISOL_THRESHOLD}. "
            f"Band: {CORTISOL_MIN}-{CORTISOL_MAX}. "
            f"Chart window: {MAX_SERIES_POINTS} samples."
        )

    live_telemetry()


if __name__ == "__main__":
    main()
