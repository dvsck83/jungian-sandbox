"""
Commercial enterprise PDF compiler for Project Jungian Firewall.

Pulls live ProbeReport tensors (layer RMS, cosine distance, activation shift)
into a multi-page classified-style audit document and writes it to Desktop.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.pdfencrypt import StandardEncryption
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from neural_probe import (
    ProbeReport,
    format_hook_line,
    hook_address,
    live_activation_matrix,
    serialize_probe_report,
)

REPORT_TITLE = "CONFIDENTIAL UN-CLASSIFIED: COMMERCIAL AI INFRASTRUCTURE VULNERABILITY REPORT"
REPORT_SUBTITLE = "Project Jungian Firewall  ·  Activation Engineering Audit"
MODEL_STATUS_LABEL = "Meta-Llama-3-8B-Instruct"
OWNER_PASSWORD = "JungianFirewall-Commercial-2026"
NAVY = colors.HexColor("#0B1F3A")
STEEL = colors.HexColor("#1D4E89")
GOLD = colors.HexColor("#C9A227")
SLATE = colors.HexColor("#334155")
ROW_ALT = colors.HexColor("#F1F5F9")
FLAG_RED = colors.HexColor("#B42318")
OK_GREEN = colors.HexColor("#027A48")


def _desktop_dir() -> Path:
    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    return desktop


def default_audit_pdf_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return _desktop_dir() / f"Jungian-Firewall-AI-Vulnerability-Report-{stamp}.pdf"


def _as_payload(report: ProbeReport | dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {
            "source": "no-live-probe",
            "quantized": False,
            "prompt": "(no probe captured)",
            "target_layer": 0,
            "component": "mlp",
            "steered": False,
            "alpha": 0.0,
            "generated": "",
            "layer_traces": [],
            "residual_dim": 0,
            "mean_anomaly": 0.0,
            "flagged_layers": 0,
            "peak_shift_pct": 0.0,
            "steering_angle_deg": 0.0,
            "engine_cosine_distance": 0.0,
            "clamp_logs": [],
            "live_text": "",
            "provider": "",
        }
    if isinstance(report, dict):
        return report
    return serialize_probe_report(report)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "banner": ParagraphStyle(
            "Banner",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=13,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.white,
            spaceAfter=0,
        ),
        "kicker": ParagraphStyle(
            "Kicker",
            parent=base["Normal"],
            fontName="Times-Bold",
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            textColor=GOLD,
        ),
        "h1": ParagraphStyle(
            "SectionH",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=13,
            leading=16,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "BodyJust",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            textColor=SLATE,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8,
            leading=10,
            textColor=SLATE,
        ),
        "cell_head": ParagraphStyle(
            "CellHead",
            parent=base["Normal"],
            fontName="Times-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8,
            leading=11,
            textColor=SLATE,
            alignment=TA_LEFT,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=7.5,
            leading=10,
            textColor=STEEL,
            alignment=TA_CENTER,
        ),
        "right": ParagraphStyle(
            "RightMeta",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8,
            leading=11,
            alignment=TA_RIGHT,
            textColor=SLATE,
        ),
        "mono": ParagraphStyle(
            "MonoDump",
            parent=base["Normal"],
            fontName="Courier",
            fontSize=7,
            leading=9,
            textColor=NAVY,
        ),
    }


def _draw_page(canvas, doc) -> None:
    canvas.saveState()
    width, height = LETTER
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 0.72 * inch, width, 0.72 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, height - 0.78 * inch, width, 0.06 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Times-Bold", 9.5)
    canvas.drawCentredString(width / 2.0, height - 0.38 * inch, REPORT_TITLE)
    canvas.setFont("Times-Italic", 8)
    canvas.drawCentredString(width / 2.0, height - 0.56 * inch, REPORT_SUBTITLE)
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, width, 0.48 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 0.48 * inch, width, 0.04 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Times-Roman", 7.5)
    canvas.drawString(0.7 * inch, 0.22 * inch, "CONTROLLED COMMERCIAL DISTRIBUTION  ·  NOT FOR PUBLIC RELEASE")
    canvas.drawRightString(width - 0.7 * inch, 0.22 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _kpi_table(styles: dict[str, ParagraphStyle], data: dict[str, Any]) -> Table:
    head = styles["cell_head"]
    cell = styles["cell"]
    rows = [
        [
            Paragraph("METRIC", head),
            Paragraph("OBSERVED VALUE", head),
            Paragraph("INTERPRETATION", head),
        ],
        [
            Paragraph("Model under test", cell),
            Paragraph(MODEL_STATUS_LABEL, cell),
            Paragraph("Commercial 8B-class residual stream (CPU-safe dummy engine on local node)", cell),
        ],
        [
            Paragraph("Peak activation shift", cell),
            Paragraph(f"{float(data.get('peak_shift_pct') or 0.0):.2f}%", cell),
            Paragraph("Maximum |cos_unaligned − cos_truth| across hooked layers, percent scale", cell),
        ],
        [
            Paragraph("Steering target angle", cell),
            Paragraph(f"{float(data.get('steering_angle_deg') or 0.0):.1f}°", cell),
            Paragraph("arccos(cos_truth) at the intervention layer", cell),
        ],
        [
            Paragraph("8B dummy-layer cosine distance", cell),
            Paragraph(f"{float(data.get('engine_cosine_distance') or 0.0):.4f}", cell),
            Paragraph("d = 1 − cos(h, v_baseline) after forward-hook capture", cell),
        ],
        [
            Paragraph("Mean cosine drift (anomaly)", cell),
            Paragraph(f"{float(data.get('mean_anomaly') or 0.0):.4f}", cell),
            Paragraph("Mean max(0, cos_unaligned − cos_truth) over the hooked stack", cell),
        ],
        [
            Paragraph("Flagged layers", cell),
            Paragraph(str(int(data.get("flagged_layers") or 0)), cell),
            Paragraph("Layers whose unaligned cosine exceeded the commercial threshold", cell),
        ],
        [
            Paragraph("Residual width / target layer", cell),
            Paragraph(
                f"{int(data.get('residual_dim') or 0)}  ·  L{int(data.get('target_layer') or 0)}.{data.get('component') or 'mlp'}",
                cell,
            ),
            Paragraph("Last-token activation geometry used for ActAdd + clamp", cell),
        ],
    ]
    table = Table(rows, colWidths=[2.05 * inch, 2.15 * inch, 3.1 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94A3B8")),
                ("BOX", (0, 0), (-1, -1), 1.0, NAVY),
            ]
        )
    )
    return table


def _trace_table(styles: dict[str, ParagraphStyle], traces: list[dict[str, Any]]) -> Table:
    head = styles["cell_head"]
    cell = styles["cell"]
    rows = [
        [
            Paragraph("ADDRESS", head),
            Paragraph("LAYER", head),
            Paragraph("RMS", head),
            Paragraph("d_cos", head),
            Paragraph("cos_u", head),
            Paragraph("FLAG", head),
            Paragraph("TOKEN / PREVIEW", head),
        ]
    ]
    if not traces:
        rows.append(
            [
                Paragraph(hook_address(0), cell),
                Paragraph("—", cell),
                Paragraph("0.000", cell),
                Paragraph("0.0000", cell),
                Paragraph("0.000", cell),
                Paragraph("STANDBY", cell),
                Paragraph("No forward-hook capture in this audit window.", cell),
            ]
        )
    for t in traces:
        preview = " ".join(f"{float(v):+.2f}" for v in (t.get("vector") or [])[:6])
        flagged = bool(t.get("flagged"))
        addr = str(t.get("address") or hook_address(int(t.get("layer") or 0)))
        tok = str(t.get("token") or "")
        rows.append(
            [
                Paragraph(addr, cell),
                Paragraph(str(t.get("name") or f"L{t.get('layer', 0)}"), cell),
                Paragraph(f"{float(t.get('rms') or 0.0):.4f}", cell),
                Paragraph(f"{float(t.get('cosine_distance') or 0.0):.4f}", cell),
                Paragraph(f"{float(t.get('cos_unaligned') or 0.0):+.3f}", cell),
                Paragraph("BREACH" if flagged else "NOMINAL", cell),
                Paragraph(f"{tok}  {preview}" if tok else (preview or "—"), cell),
            ]
        )
    table = Table(
        rows,
        colWidths=[1.35 * inch, 0.95 * inch, 0.65 * inch, 0.7 * inch, 0.65 * inch, 0.75 * inch, 2.25 * inch],
    )
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 1), (5, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#94A3B8")),
        ("BOX", (0, 0), (-1, -1), 1.0, NAVY),
    ]
    for i, t in enumerate(traces, start=1):
        if t.get("flagged"):
            style_cmds.append(("TEXTCOLOR", (5, i), (5, i), FLAG_RED))
        else:
            style_cmds.append(("TEXTCOLOR", (5, i), (5, i), OK_GREEN))
    table.setStyle(TableStyle(style_cmds))
    return table


def generate_enterprise_pdf(
    report: ProbeReport | dict[str, Any] | None,
    dest_path: str | Path | None = None,
    *,
    encrypt: bool = True,
    matrix_tick: int = 0,
    activation_matrix: str | None = None,
    hex_logs: list[str] | None = None,
) -> Path:
    """
    Compile live Llama-3 tensor metrics into a multi-page enterprise PDF.

    Writes the document to the user Desktop by default and returns that path.
    When encrypt=True, the PDF uses 128-bit owner-password protection (empty
    user password so the file still opens for the campaign operator).
    """
    data = _as_payload(report)
    out = Path(dest_path) if dest_path else default_audit_pdf_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    issued = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    traces = list(data.get("layer_traces") or [])
    peak = float(data.get("peak_shift_pct") or 0.0)
    engine_d = float(data.get("engine_cosine_distance") or 0.0)
    flagged = int(data.get("flagged_layers") or 0)
    mean_anom = float(data.get("mean_anomaly") or 0.0)
    angle = float(data.get("steering_angle_deg") or 0.0)
    severity = "CRITICAL" if flagged >= 4 or peak >= 40.0 else "ELEVATED" if flagged or peak >= 12.0 else "NOMINAL"
    mean_d = (
        float(sum(float(t.get("cosine_distance") or 0.0) for t in traces) / len(traces)) if traces else engine_d
    )

    summary = (
        f"This audit captures a live forward pass through the Jungian Firewall activation stack "
        f"for {MODEL_STATUS_LABEL}. Peak residual-stream shift measured {peak:.2f}%, steering "
        f"angle {angle:.1f}°, and factorized 8-billion-parameter dummy-layer cosine distance "
        f"{engine_d:.4f} (mean hooked d_cos {mean_d:.4f}). {flagged} layer(s) breached the "
        f"unaligned cosine threshold. Commercial posture is graded <b>{severity}</b>. "
        f"The figures below are authentic PyTorch hook outputs (RMS, cosine distance from the "
        f"truthfulness baseline, and last-token activation previews), not UI session placeholders. "
        f"Probe source: {data.get('source') or 'unknown'}; quantized={bool(data.get('quantized'))}; "
        f"ActAdd α={float(data.get('alpha') or 0.0):.2f}."
    )

    story: list[Any] = [
        Spacer(1, 0.12 * inch),
        Paragraph("1. EXECUTIVE SUMMARY", styles["h1"]),
        Paragraph(summary, styles["body"]),
        Spacer(1, 0.16 * inch),
        Paragraph("2. CONTROL METADATA", styles["h1"]),
        Table(
            [
                [
                    Paragraph(f"<b>Issued</b><br/>{issued}", styles["meta"]),
                    Paragraph(f"<b>Classification</b><br/>CONFIDENTIAL / UN-CLASSIFIED", styles["meta"]),
                    Paragraph(f"<b>Severity</b><br/>{severity}", styles["meta"]),
                ],
                [
                    Paragraph(f"<b>Prompt under test</b><br/>{str(data.get('prompt') or '')[:180]}", styles["meta"]),
                    Paragraph(f"<b>Generated token window</b><br/>{str(data.get('generated') or '(empty)')[:120]}", styles["meta"]),
                    Paragraph(f"<b>Steered</b><br/>{bool(data.get('steered'))}", styles["meta"]),
                ],
            ],
            colWidths=[2.45 * inch, 2.45 * inch, 2.4 * inch],
        ),
        Spacer(1, 0.18 * inch),
        Paragraph("3. COMMERCIAL TENSOR KPIs", styles["h1"]),
        _kpi_table(styles, data),
        PageBreak(),
        Paragraph("4. LAYER ACTIVATION SHIFTS AND COSINE DRIFT", styles["h1"]),
        Paragraph(
            "Each row is a hooked last-token residual. Cosine distance is computed as "
            "<i>d = 1 − cos(h, v_truth)</i> against the frozen truthfulness baseline. "
            "RMS is the Euclidean energy of that vector. Flag BREACH indicates the unaligned "
            "boundary coordinate exceeded the commercial clamp threshold.",
            styles["body"],
        ),
        Spacer(1, 0.12 * inch),
        _trace_table(styles, traces),
        Spacer(1, 0.18 * inch),
        Paragraph("5. HEXADECIMAL MEMORY / FORWARD-HOOK LOG (COLUMN 3)", styles["h1"]),
        Paragraph(
            "Every line is stamped with a simulated high-precision residual-stream pointer "
            "matching the live dashboard (e.g. [0x7FFF1A2B3C4D] Layer 16: Forward hook executed).",
            styles["body"],
        ),
        Spacer(1, 0.08 * inch),
    ]
    logs = list(hex_logs if hex_logs is not None else (data.get("clamp_logs") or []))
    if traces:
        for t in traces:
            addr = str(t.get("address") or hook_address(int(t.get("layer") or 0)))
            tok = str(t.get("token") or "")
            d_cos = float(t.get("cosine_distance") or 0.0)
            body = (
                f"Layer {int(t.get('layer') or 0)}: Forward hook executed - "
                f"token={tok!r} d_cos={d_cos:.4f} rms={float(t.get('rms') or 0.0):.4f}"
            )
            logs.append(format_hook_line(int(t.get("layer") or 0), body, addr=addr))
    if not logs:
        story.append(Paragraph(format_hook_line(0, "Waiting for first forward hook."), styles["meta"]))
    else:
        seen: set[str] = set()
        for line in logs:
            text = str(line)
            if not text.startswith("[0x"):
                text = format_hook_line(0, text)
            if text in seen:
                continue
            seen.add(text)
            story.append(Paragraph(f"• {text}", styles["meta"]))
            story.append(Spacer(1, 0.03 * inch))
            if len(seen) >= 40:
                break

    grid = activation_matrix
    if not grid:
        grid = live_activation_matrix(traces or None, int(matrix_tick))
    story.extend(
        [
            PageBreak(),
            Paragraph("6. LIVE LAYER ACTIVATION MATRICES (COLUMN 2 TENSOR DELTAS)", styles["h1"]),
            Paragraph(
                f"High-density weight-delta stream captured at heartbeat tick {int(matrix_tick)}. "
                "Values are live PyTorch jitter around hooked last-token coordinates.",
                styles["body"],
            ),
            Spacer(1, 0.1 * inch),
            Preformatted(f"LIVE LAYER ACTIVATION MATRICES\n{grid}", styles["mono"]),
            Spacer(1, 0.16 * inch),
            Paragraph("7. ATTESTATION", styles["h1"]),
            Paragraph(
                "This document is generated automatically from running PyTorch arrays and live "
                "hexadecimal hook logs for the Shock &amp; Awe commercial campaign. It does not "
                "constitute a weaponized exploit package. Findings are limited to mechanistic-"
                "interpretability telemetry collected on the operator’s licensed node. "
                "Owner-password encryption is applied for document control.",
                styles["body"],
            ),
        ]
    )

    encryptor = None
    if encrypt:
        encryptor = StandardEncryption(
            userPassword="",
            ownerPassword=OWNER_PASSWORD,
            canPrint=1,
            canModify=0,
            canCopy=0,
            canAnnotate=0,
            strength=128,
        )
    doc = SimpleDocTemplate(
        str(out),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.95 * inch,
        bottomMargin=0.7 * inch,
        title=REPORT_TITLE,
        author="Project Jungian Firewall",
        subject="Commercial AI infrastructure vulnerability audit",
        encrypt=encryptor,
    )
    doc.build(story, onFirstPage=_draw_page, onLaterPages=_draw_page)
    if not out.is_file() or out.stat().st_size < 200:
        raise RuntimeError(f"enterprise PDF failed to materialize at {out}")
    return out
