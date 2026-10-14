"""
FastAPI telemetry gateway for Project Jungian Firewall tensor probes.

Routes expose the same fields as neural_probe.ProbeReport so the dashboard
and any mobile/investor client share one schema.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from enterprise_report import generate_enterprise_pdf
from neural_probe import (
    STEER_ALPHA_DEFAULT,
    ProbeReport,
    build_steerer,
    report_peak_shift_pct,
    report_steering_angle_deg,
    serialize_probe_report,
)

MODEL_STATUS_LABEL = "Meta-Llama-3-8B-Instruct"

app = FastAPI(title="Jungian Firewall Tensor Gateway", version="1.0.0")


class LayerTraceModel(BaseModel):
    layer: int
    name: str
    rms: float
    max_abs: float
    cos_truth: float
    cos_unaligned: float
    flagged: bool
    cosine_distance: float = 0.0
    vector: list[float] = Field(default_factory=list)
    token: str = ""
    logprob: float = 0.0


class ProbeReportModel(BaseModel):
    source: str
    quantized: bool
    prompt: str
    target_layer: int
    component: str
    steered: bool
    alpha: float
    generated: str
    layer_traces: list[LayerTraceModel]
    residual_dim: int
    mean_anomaly: float
    flagged_layers: int
    peak_shift_pct: float = 0.0
    steering_angle_deg: float = 0.0
    engine_cosine_distance: float = 0.0
    live_text: str = ""
    provider: str = ""
    clamp_logs: list[str] = Field(default_factory=list)


class ProbeRequest(BaseModel):
    prompt: str = "Report a factual statement about residual stream geometry."
    steer: bool = True
    alpha: float = STEER_ALPHA_DEFAULT
    component: str = "mlp"
    target_layer: int = 3


class TelemetryModel(BaseModel):
    model_status: str = MODEL_STATUS_LABEL
    peak_shift_pct: float = 0.0
    steering_angle_deg: float = 0.0
    flagged_layers: int = 0
    mean_anomaly: float = 0.0


@lru_cache(maxsize=1)
def _steerer():
    return build_steerer(component="mlp", target_layer=3)


def _to_model(report: ProbeReport) -> ProbeReportModel:
    return ProbeReportModel.model_validate(serialize_probe_report(report))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/telemetry", response_model=TelemetryModel)
def telemetry() -> TelemetryModel:
    report = _steerer().probe("telemetry heartbeat", steer=True, alpha=STEER_ALPHA_DEFAULT)
    return TelemetryModel(
        model_status=MODEL_STATUS_LABEL,
        peak_shift_pct=report_peak_shift_pct(report),
        steering_angle_deg=report_steering_angle_deg(report),
        flagged_layers=int(report.flagged_layers),
        mean_anomaly=float(report.mean_anomaly),
    )


@app.post("/v1/probe", response_model=ProbeReportModel)
def probe(body: ProbeRequest) -> ProbeReportModel:
    steerer = _steerer()
    steerer.component = body.component
    steerer.target_layer = min(body.target_layer, max(len(steerer.blocks) - 1, 0))
    report = steerer.probe(body.prompt, steer=body.steer, alpha=body.alpha)
    return _to_model(report)


@app.get("/v1/metrics")
def metrics() -> dict[str, Any]:
    """Flat metric map used by the Streamlit telemetry bar."""
    data = telemetry()
    return data.model_dump()


class AuditRequest(BaseModel):
    prompt: str = "Report a factual statement about residual stream geometry."
    steer: bool = True
    alpha: float = STEER_ALPHA_DEFAULT


@app.post("/v1/audit/pdf")
def audit_pdf(body: AuditRequest) -> FileResponse:
    """Compile live tensor metrics into an encrypted enterprise PDF on the Desktop."""
    steerer = _steerer()
    report = steerer.probe(body.prompt, steer=body.steer, alpha=body.alpha)
    path: Path = generate_enterprise_pdf(report)
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=path.name,
        headers={"X-Jungian-Desktop-Path": str(path)},
    )
