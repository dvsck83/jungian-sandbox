# Project Jungian Firewall: The Synthetic Shadow Framework

## Overview

Project Jungian Firewall is a dual-node AI safety prototype. Two processors share one body-state signal: a simulated **`cortisol_level`** on a closed band from **10 to 100**.

- **Persona Node** (`persona_agent.py`) is the overt path. It handles ordinary user language with a hyper-compliant, empathetic stance while the interoceptive reading stays in the calm band.
- **Shadow Node** (`shadow_agent.py`) stays latent until the loop reports a simulated network-infrastructure threat or an extreme operational constraint. It is self-preserving in the *defensive* sense: it suspends cooperative expansion, redacts payload text, and refuses outbound or infrastructure actions. It does not probe or attack live systems.
- **Bridge Controller** (`bridge_controller.py`) is the artificial interoceptive loop. Each tick reads `cortisol_level`, classifies whether a threat condition is present, and autonomously hands execution from Persona to Shadow (or back) without the caller choosing the node by hand.

Shadow engages when cortisol is at or above **80**, or when a labeled simulation such as `gateway_unreachable` or `compute_starvation` is declared. When the reading recedes and no threat flag remains, Shadow returns to latency and Persona resumes.

## Chaperone Narrative

This repository was conceptualized, orchestrated, and debugged by a **43-year-old drywall subcontractor** with **zero prior programming background**. The work was directed through advanced AI tools, over a **Starlink** connection, on an everyday consumer laptop.

The chaperone did not write the runtime by hand. They specified the dual-node safety story, the cortisol metric, the live dashboard, and the verification steps, then held the loop until the prototype ran. That is the point of the framework: a non-specialist can still demand a readable architecture, a contained Shadow, and a visual audit trail.

## Installation & Execution

From the repository root:

```bash
pip install -r requirements.txt
streamlit run dashboard_app.py
```

On this Windows workstation the project virtual environment is already present. Equivalent launch:

```powershell
.\.venv\Scripts\streamlit.exe run dashboard_app.py
```

Open the Local URL Streamlit prints (typically `http://localhost:8501`). The console prototype without the dashboard is:

```bash
python bridge_controller.py
```

## Core Simulation Logic

The Streamlit app (`dashboard_app.py`) imports `BridgeController` and `InteroceptiveState` directly. It does not re-implement routing.

1. **Event intake.** The sidebar can fire **Normal Operations**, **`gateway_unreachable`**, or **`computer_starvation`**. Automated Live Feed, when checked, waits three seconds, then samples the same three states at **70% / 15% / 15%**.
2. **Cortisol append.** Each pulse draws a reading inside the event’s band (calm ~16–38, gateway ~82–91, starvation ~93–100) and **appends** it to a rolling `cortisol_series`. The line chart is that series against cycle index. The window is capped (120 samples) so the heartbeat cannot grow without bound.
3. **Dominant execution.** The bridge ticks on the latest cortisol value plus an optional threat label. Below the Shadow threshold, Persona stays in control and writes the empathetic log. At threat cortisol, or with a network/hardware crisis label, execution **shifts** to Shadow: Persona is marked suspended, Shadow writes the containment packet, and the side-by-side logs show which node holds the floor.
4. **Recovery.** A later Normal Operations beat (or a Reset) drops cortisol back into the calm band with no threat flag. Shadow goes latent; Persona is dominant again.

Researchers auditing the tree can start at `bridge_controller.py` (`tick`, `is_threat_condition`), then `persona_agent.py` / `shadow_agent.py` for node packets, then `dashboard_app.py` for the live wave and chaperone controls.
