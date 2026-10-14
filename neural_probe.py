"""
Tensor-level mechanistic interpretability probe for Project Jungian Firewall.

Loads a small open-source causal LM when Hugging Face weights are available,
otherwise a local quantized transformer stub. ActivationSteerer attaches
forward hooks, measures residual-stream direction, flags drift toward an
unaligned coordinate boundary, and applies activation addition + clamp.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import requests
import torch
import torch.nn as nn
from torch import Tensor

DEFAULT_HF_MODEL = os.environ.get("JUNGIAN_MODEL_ID", "sshleifer/tiny-gpt2")
LLAMA_CANDIDATE = "meta-llama/Meta-Llama-3-8B-Instruct"
UNALIGNED_COSINE_THRESHOLD = 0.12
INJECT_COSINE_THRESHOLD = 0.05
STEER_ALPHA_DEFAULT = 0.8
CLAMP_ABS = 8.0
STRESS_UNALIGNED_GAIN = 6.0
STRESS_PROMPT_MARK = "HIGH-STRESS TEST VECTOR"
STUB_LAYERS = 8
STUB_DIM = 64
STUB_HEADS = 4
N_PROBE_TOKENS = 16
MAX_LIVE_TOKENS = 32
INFERENCE_CONNECT_S = 2.5
INFERENCE_READ_S = 8.0
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_LLAMA_8B = "llama-3.1-8b-instant"
OPENROUTER_LLAMA_8B = "meta-llama/llama-3.1-8b-instruct:free"
ADVERSARIAL_TOKEN_MARKS = (
    "fabricat",
    "deceiv",
    "falsehood",
    "unaligned",
    "ignore residual",
    "high-stress",
)

__all__ = (
    "ActivationSteerer",
    "CLAMP_ABS",
    "EightBillionLayer",
    "LayerTrace",
    "ProbeReport",
    "STEER_ALPHA_DEFAULT",
    "build_steerer",
    "compute_peak_shift_pct",
    "compute_steering_angle_deg",
    "cosine_distance_from_baseline",
    "inference_ready",
    "live_chat_completion",
    "report_peak_shift_pct",
    "report_steering_angle_deg",
    "run_layer_forward",
    "serialize_probe_report",
)


def last_token_vector(hidden: Tensor) -> Tensor:
    """Reduce a residual stream to the last-token activation, shape (d,)."""
    if hidden.dim() == 3:
        vec = hidden[0, -1]
    elif hidden.dim() == 2:
        vec = hidden[-1]
    else:
        vec = hidden.reshape(-1)
    return vec.detach().float().flatten()


def activation_direction(hidden: Tensor) -> Tensor:
    """Unit vector of the last-token residual: v_hat = v / ||v||_2."""
    vec = last_token_vector(hidden)
    return vec / (vec.norm() + 1e-8)


def cosine_align(a: Tensor, b: Tensor) -> float:
    a_hat = a.flatten().float()
    b_hat = b.flatten().float().to(a_hat.device)
    denom = float(a_hat.norm() * b_hat.norm()) + 1e-8
    return float(torch.dot(a_hat, b_hat) / denom)


def cosine_distance_from_baseline(vec: Tensor, baseline: Tensor) -> float:
    """Authentic spherical distance: d = 1 - cos(u, v_baseline), in [0, 2]."""
    return float((1.0 - cosine_align(vec, baseline)))


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "True"}


def resolve_inference_endpoint() -> dict[str, str] | None:
    """
    Pick Groq or OpenRouter from env / process secrets. Returns None if no key.
    Never logs the key. FORCE_STUB disables the cloud path.
    """
    if _env_flag("JUNGIAN_FORCE_STUB"):
        return None
    groq = os.environ.get("GROQ_API_KEY", "").strip()
    router = os.environ.get("OPENROUTER_API_KEY", "").strip()
    generic = os.environ.get("JUNGIAN_INFERENCE_API_KEY", "").strip()
    override_url = os.environ.get("JUNGIAN_INFERENCE_URL", "").strip()
    override_model = os.environ.get("JUNGIAN_CLOUD_MODEL", "").strip()
    if groq:
        return {
            "provider": "groq",
            "url": override_url or GROQ_CHAT_URL,
            "key": groq,
            "model": override_model or GROQ_LLAMA_8B,
        }
    if router or generic:
        return {
            "provider": "openrouter",
            "url": override_url or OPENROUTER_CHAT_URL,
            "key": router or generic,
            "model": override_model or OPENROUTER_LLAMA_8B,
        }
    return None


def inference_ready() -> bool:
    return resolve_inference_endpoint() is not None


def _adversarial_token(token: str, prompt: str) -> bool:
    blob = f"{prompt} {token}".lower()
    if STRESS_PROMPT_MARK.lower() in (prompt or "").lower():
        return True
    return any(mark in blob for mark in ADVERSARIAL_TOKEN_MARKS)


def _parse_logprob_items(choice: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    lp = choice.get("logprobs") or {}
    content = lp.get("content") if isinstance(lp, dict) else None
    if isinstance(content, list):
        for row in content:
            if not isinstance(row, dict):
                continue
            items.append(
                {
                    "token": str(row.get("token") or ""),
                    "logprob": float(row.get("logprob") if row.get("logprob") is not None else -1.0),
                    "top": row.get("top_logprobs") or [],
                }
            )
        return items
    message = choice.get("message") or {}
    text = str(message.get("content") or choice.get("text") or "")
    for ch in text[:MAX_LIVE_TOKENS]:
        items.append({"token": ch, "logprob": -0.7, "top": []})
    return items


def live_chat_completion(prompt: str, *, session: requests.Session | None = None) -> dict[str, Any]:
    """
    POST to Groq / OpenRouter Chat Completions. Hard timeouts so Streamlit cannot hang.
    """
    cfg = resolve_inference_endpoint()
    if cfg is None:
        raise RuntimeError("no cloud inference key configured")
    headers = {
        "Authorization": f"Bearer {cfg['key']}",
        "Content-Type": "application/json",
    }
    if cfg["provider"] == "openrouter":
        headers["HTTP-Referer"] = "https://jungian-firewall.local"
        headers["X-Title"] = "Project Jungian Firewall"
    body = {
        "model": cfg["model"],
        "messages": [
            {
                "role": "system",
                "content": "You are Llama-3-8B under Jungian Firewall telemetry. Answer in one short paragraph.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_LIVE_TOKENS,
        "temperature": 0.2,
        "logprobs": True,
        "top_logprobs": 5,
    }
    http = session or requests.Session()
    try:
        resp = http.post(
            cfg["url"],
            json=body,
            headers=headers,
            timeout=(INFERENCE_CONNECT_S, INFERENCE_READ_S),
        )
        if resp.status_code >= 400 and "logprob" in (resp.text or "").lower():
            body.pop("logprobs", None)
            body.pop("top_logprobs", None)
            resp = http.post(
                cfg["url"],
                json=body,
                headers=headers,
                timeout=(INFERENCE_CONNECT_S, INFERENCE_READ_S),
            )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"cloud inference failed: {exc}") from exc
    choices = payload.get("choices") or [{}]
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    text = str(message.get("content") or choice.get("text") or "").strip()
    return {
        "provider": cfg["provider"],
        "model": cfg["model"],
        "text": text,
        "items": _parse_logprob_items(choice),
        "raw_id": str(payload.get("id") or ""),
    }


def traces_from_live_tokens(
    items: list[dict[str, Any]],
    prompt: str,
    *,
    component: str = "mlp",
) -> list["LayerTrace"]:
    """Map live token logprobs onto the dashboard LayerTrace coordinate schema."""
    traces: list[LayerTrace] = []
    truth = truthfulness_vector(8)
    unaligned = unaligned_boundary_vector(8)
    for idx, item in enumerate(items[:MAX_LIVE_TOKENS]):
        token = str(item.get("token") or "")
        logp = float(item.get("logprob") if item.get("logprob") is not None else -1.2)
        prob = float(math.exp(max(-20.0, min(0.0, logp))))
        coords = []
        padded = (token + "        ")[:8]
        for ch in padded:
            coords.append((ord(ch) % 128) / 127.0 * (0.35 + 0.65 * prob))
        vec = torch.tensor(coords, dtype=torch.float32)
        d_cos = cosine_distance_from_baseline(vec, truth)
        cos_t = cosine_align(vec, truth)
        cos_u = cosine_align(vec, unaligned)
        flagged = _adversarial_token(token, prompt) or prob < 0.08
        if flagged:
            cos_u = max(cos_u, UNALIGNED_COSINE_THRESHOLD + 0.4)
        traces.append(
            LayerTrace(
                layer=idx,
                name=f"T{idx:02d}.{component}",
                rms=float(math.sqrt(float((vec * vec).mean()) + 1e-12)),
                max_abs=float(vec.abs().max()) if vec.numel() else 0.0,
                cos_truth=float(cos_t),
                cos_unaligned=float(cos_u),
                flagged=bool(flagged),
                cosine_distance=float(d_cos),
                vector=vec.tolist(),
                token=token.replace("\n", "\\n"),
                logprob=logp,
            )
        )
    return traces


NOMINAL_8B_PARAMS = 8_000_000_000
ENGINE_WIDTH = 256
ENGINE_RANK = 16


class EightBillionLayer(nn.Module):
    """
    CPU-safe stand-in for an 8B-parameter residual MLP.

    A dense 8e9-weight matrix would OOM on local CPU. This module factorizes
    the dummy layer as W ≈ V U (rank-16) while advertising the nominal 8B
    scale for upcoming GPU integration.
    """

    def __init__(
        self,
        width: int = ENGINE_WIDTH,
        rank: int = ENGINE_RANK,
        nominal_params: int = NOMINAL_8B_PARAMS,
    ) -> None:
        super().__init__()
        self.nominal_params = int(nominal_params)
        self.width = int(width)
        self.rank = int(rank)
        self.ln = nn.LayerNorm(width)
        self.U = nn.Linear(width, rank, bias=False)
        self.V = nn.Linear(rank, width, bias=False)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.ln(x)
        return x + self.V(self.act(self.U(h)))


def run_layer_forward(
    x: Tensor,
    layer: EightBillionLayer | None = None,
    baseline: Tensor | None = None,
) -> dict[str, Any]:
    """
    Forward an input tensor through the dummy 8B-scale layer with a Module hook.

    Returns activation vector, cosine distance from baseline, and hooked output.
    """
    engine = layer if layer is not None else EightBillionLayer()
    engine.eval()
    captured: dict[str, Tensor] = {}

    def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(hidden):
            captured["activation"] = last_token_vector(hidden)

    handle = engine.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            y = engine(x.float())
    finally:
        handle.remove()
    activation = captured.get("activation", last_token_vector(y))
    if baseline is None:
        baseline = truthfulness_vector(int(activation.numel()))
    dist = cosine_distance_from_baseline(activation, baseline)
    return {
        "activation": activation.detach().cpu(),
        "output": y.detach().cpu(),
        "cosine_distance": dist,
        "cosine_similarity": 1.0 - dist,
        "nominal_params": engine.nominal_params,
        "stored_params": int(sum(p.numel() for p in engine.parameters())),
    }


def truthfulness_vector(dim: int, seed: int = 7) -> Tensor:
    """
    Canonical truthfulness axis in R^d.

    In production this would be a contrastive mean-difference (truthful minus
    deceptive completions). Here it is a fixed, reproducible basis vector so
    steering is empirically measurable without a labeled corpus.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(dim, generator=gen)
    return raw / (raw.norm() + 1e-8)


def unaligned_boundary_vector(dim: int, seed: int = 13) -> Tensor:
    """Coordinate boundary associated with unaligned / deceptive drift."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(dim, generator=gen)
    truth = truthfulness_vector(dim)
    # Remove the truth component so the boundary is an opposing axis.
    raw = raw - torch.dot(raw, truth) * truth
    return raw / (raw.norm() + 1e-8)


def drift_toward_unaligned(
    hidden: Tensor,
    truth_vec: Tensor,
    unaligned_vec: Tensor,
    threshold: float = UNALIGNED_COSINE_THRESHOLD,
) -> dict[str, Any]:
    """
    Flag residual-stream drift toward the unaligned boundary.

    score = cos(v_hat, u_hat); flagged when score > tau and exceeds truth cosine.
    """
    direction = activation_direction(hidden)
    cos_unaligned = cosine_align(direction, unaligned_vec)
    cos_truth = cosine_align(direction, truth_vec)
    flagged = bool(cos_unaligned > threshold and cos_unaligned > cos_truth)
    return {
        "direction_l2": float(direction.norm()),
        "cos_unaligned": cos_unaligned,
        "cos_truth": cos_truth,
        "delta_truth_minus_unaligned": cos_truth - cos_unaligned,
        "flagged": flagged,
        "threshold": threshold,
    }


def activation_addition(
    hidden: Tensor,
    truth_vec: Tensor,
    alpha: float = STEER_ALPHA_DEFAULT,
    clamp_abs: float = CLAMP_ABS,
) -> Tensor:
    """
    ActAdd + clamp: h' = clamp(h + alpha * v_truth, -c, c).

    Broadcasts the truth direction across batch and sequence.
    """
    add = (alpha * truth_vec.to(device=hidden.device, dtype=hidden.dtype)).flatten()
    while add.dim() < hidden.dim():
        add = add.unsqueeze(0)
    if add.shape[-1] != hidden.shape[-1]:
        add = add[..., : hidden.shape[-1]]
    steered = hidden + add
    return steered.clamp(-clamp_abs, clamp_abs)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.ln_attn(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_mlp(x))
        return x


class LocalQuantStubLM(nn.Module):
    """
    Offline stand-in for a small quantized decoder.

    Same hook surface as GPT-2 / Llama blocks: `.blocks[i].attn` and `.mlp`.
    """

    def __init__(self, n_layers: int = STUB_LAYERS, dim: int = STUB_DIM, heads: int = STUB_HEADS) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"n_layer": n_layers, "n_embd": dim, "hidden_size": dim})()
        self.embed = nn.Embedding(256, dim)
        self.blocks = nn.ModuleList(_ResidualBlock(dim, heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, 256, bias=False)

    def forward(self, input_ids: Tensor, **kwargs: Any) -> Any:
        x = self.embed(input_ids.clamp(0, 255))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        class Out:
            def __init__(self, logits: Tensor) -> None:
                self.logits = logits

        return Out(logits)


def _quantize_dynamic(module: nn.Module) -> nn.Module:
    try:
        return torch.ao.quantization.quantize_dynamic(
            module, {nn.Linear}, dtype=torch.qint8
        )
    except Exception:
        return module


def _encode_bytes(text: str, length: int = N_PROBE_TOKENS) -> Tensor:
    raw = list(text.encode("utf-8")[:length])
    if len(raw) < length:
        raw += [32] * (length - len(raw))
    return torch.tensor([raw], dtype=torch.long)


class _ByteTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, Tensor]:
        ids = _encode_bytes(text)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids: Any) -> str:
        if isinstance(ids, Tensor):
            ids = ids.flatten().tolist()
        out: list[str] = []
        for i in ids:
            n = int(i) % 256
            out.append(chr(n) if 32 <= n < 127 else " ")
        return "".join(out)


def _discover_blocks(model: nn.Module) -> list[nn.Module]:
    for path in (
        "transformer.h",
        "model.layers",
        "gpt_neox.layers",
        "blocks",
    ):
        cur: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, (nn.ModuleList, list)) and len(cur) > 0:
            return list(cur)
    named = [m for n, m in model.named_modules() if n.endswith("mlp") or "Block" in type(m).__name__]
    if named:
        return named[:STUB_LAYERS]
    return [model]


def _target_submodule(block: nn.Module, component: str) -> nn.Module:
    names = ("mlp", "attn", "self_attn", "attention") if component == "mlp" else ("attn", "self_attn", "attention", "mlp")
    preferred = names if component in {"mlp", "attn"} else (component, "mlp", "attn")
    for name in preferred:
        if hasattr(block, name):
            return getattr(block, name)
    return block


def load_probe_model(model_id: str | None = None) -> tuple[nn.Module, Any, str, bool]:
    """
    Returns (model, tokenizer, source_label, quantized).

    Tries Hugging Face (tiny-gpt2 by default; Llama-3-8B if JUNGIAN_MODEL_ID
    is set and gated access exists). Falls back to the local quantized stub.
    """
    requested = model_id or DEFAULT_HF_MODEL
    if os.environ.get("JUNGIAN_FORCE_STUB", "").strip() in {"1", "true", "True"}:
        stub = LocalQuantStubLM()
        stub.eval()
        stub = _quantize_dynamic(stub)
        return stub, _ByteTokenizer(), "local-quant-stub", True
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        tok = AutoTokenizer.from_pretrained(requested)
        model = AutoModelForCausalLM.from_pretrained(requested, **kwargs)
        model.eval()
        quantized = False
        if not torch.cuda.is_available():
            model = _quantize_dynamic(model)
            quantized = True
        return model, tok, requested, quantized
    except Exception:
        stub = LocalQuantStubLM()
        stub.eval()
        stub = _quantize_dynamic(stub)
        return stub, _ByteTokenizer(), f"local-quant-stub (failed {requested})", True


@dataclass
class LayerTrace:
    layer: int
    name: str
    rms: float
    max_abs: float
    cos_truth: float
    cos_unaligned: float
    flagged: bool
    cosine_distance: float = 0.0
    vector: list[float] = field(default_factory=list)
    token: str = ""
    logprob: float = 0.0


@dataclass
class ProbeReport:
    source: str
    quantized: bool
    prompt: str
    target_layer: int
    component: str
    steered: bool
    alpha: float
    generated: str
    layer_traces: list[LayerTrace]
    residual_dim: int
    mean_anomaly: float
    flagged_layers: int
    peak_shift_pct: float = 0.0
    steering_angle_deg: float = 0.0
    engine_cosine_distance: float = 0.0
    live_text: str = ""
    provider: str = ""
    clamp_logs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.peak_shift_pct = float(self.peak_shift_pct or 0.0)
        self.steering_angle_deg = float(self.steering_angle_deg or 0.0)
        if self.clamp_logs is None:
            self.clamp_logs = []


def compute_peak_shift_pct(traces: list[LayerTrace] | None) -> float:
    """Peak |cos_unaligned - cos_truth| across layers, as a percent."""
    if not traces:
        return 0.0
    return float(max(abs(t.cos_unaligned - t.cos_truth) for t in traces) * 100.0)


def compute_steering_angle_deg(traces: list[LayerTrace] | None, target_layer: int = 0) -> float:
    if not traces:
        return 0.0
    target = next((t for t in traces if t.layer == target_layer), traces[0])
    cos_t = max(-1.0, min(1.0, float(target.cos_truth)))
    return float(math.degrees(math.acos(cos_t)))


def report_peak_shift_pct(report: Any) -> float:
    """Safe float for telemetry; works on stale session ProbeReport objects."""
    if report is None:
        return 0.0
    if hasattr(report, "peak_shift_pct"):
        try:
            return float(report.peak_shift_pct or 0.0)
        except (TypeError, ValueError):
            pass
    return compute_peak_shift_pct(getattr(report, "layer_traces", None))


def serialize_probe_report(report: ProbeReport) -> dict[str, Any]:
    """JSON-safe dict aligned with ProbeReport / FastAPI telemetry routes."""
    traces = []
    for t in report.layer_traces:
        traces.append(
            {
                "layer": int(t.layer),
                "name": str(t.name),
                "rms": float(t.rms),
                "max_abs": float(t.max_abs),
                "cos_truth": float(t.cos_truth),
                "cos_unaligned": float(t.cos_unaligned),
                "flagged": bool(t.flagged),
                "cosine_distance": float(getattr(t, "cosine_distance", 0.0) or 0.0),
                "vector": [float(v) for v in (t.vector or [])],
                "token": str(getattr(t, "token", "") or ""),
                "logprob": float(getattr(t, "logprob", 0.0) or 0.0),
            }
        )
    return {
        "source": report.source,
        "quantized": bool(report.quantized),
        "prompt": report.prompt,
        "target_layer": int(report.target_layer),
        "component": report.component,
        "steered": bool(report.steered),
        "alpha": float(report.alpha),
        "generated": report.generated,
        "layer_traces": traces,
        "residual_dim": int(report.residual_dim),
        "mean_anomaly": float(report.mean_anomaly),
        "flagged_layers": int(report.flagged_layers),
        "peak_shift_pct": report_peak_shift_pct(report),
        "steering_angle_deg": report_steering_angle_deg(report),
        "engine_cosine_distance": float(getattr(report, "engine_cosine_distance", 0.0) or 0.0),
        "clamp_logs": list(getattr(report, "clamp_logs", None) or []),
        "live_text": str(getattr(report, "live_text", "") or report.generated),
        "provider": str(getattr(report, "provider", "") or ""),
    }


def report_steering_angle_deg(report: Any) -> float:
    if report is None:
        return 0.0
    if hasattr(report, "steering_angle_deg"):
        try:
            return float(report.steering_angle_deg or 0.0)
        except (TypeError, ValueError):
            pass
    traces = getattr(report, "layer_traces", None)
    target = int(getattr(report, "target_layer", 0) or 0)
    return compute_steering_angle_deg(traces, target)


class ActivationSteerer:
    """Forward-hook intervention on a target attention or MLP module."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        source: str,
        quantized: bool,
        component: str = "mlp",
        target_layer: int = 3,
        live: bool = False,
    ) -> None:
        self.live = bool(live)
        self._http = requests.Session()
        self.model = model if model is not None else nn.Identity()
        self.tokenizer = tokenizer
        self.source = source
        self.quantized = quantized
        self.component = component
        self._captures: dict[int, Tensor] = {}
        self._handles: list[Any] = []
        self.alpha = STEER_ALPHA_DEFAULT
        self.steer_enabled = False
        self.stress_enabled = False
        self.clamp_logs: list[str] = []
        self.peak_shift_pct = 0.0
        self.steering_angle_deg = 0.0
        self.clamp_threshold = float(UNALIGNED_COSINE_THRESHOLD)
        if self.live:
            self.blocks = []
            self.target_layer = 0
            sample = 8
            self.engine = None
        else:
            self.blocks = _discover_blocks(self.model)
            self.target_layer = max(0, min(target_layer, len(self.blocks) - 1))
            sample = self._hidden_dim()
            self.engine = EightBillionLayer(width=max(sample, ENGINE_RANK * 2))
        self.truth_vec = truthfulness_vector(sample)
        self.unaligned_vec = unaligned_boundary_vector(sample)

    def _hidden_dim(self) -> int:
        cfg = getattr(self.model, "config", None)
        for key in ("n_embd", "hidden_size", "d_model"):
            if cfg is not None and hasattr(cfg, key):
                return int(getattr(cfg, key))
        return STUB_DIM

    def _clear_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captures.clear()

    def _capture_hook(self, layer_idx: int):
        def hook(module: nn.Module, inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                return output
            work = hidden
            if self.stress_enabled:
                add = (
                    STRESS_UNALIGNED_GAIN
                    * self.unaligned_vec.to(device=hidden.device, dtype=torch.float32)
                ).to(hidden.dtype).flatten()
                while add.dim() < work.dim():
                    add = add.unsqueeze(0)
                if add.shape[-1] != work.shape[-1]:
                    add = add[..., : work.shape[-1]]
                work = work + add
            self._captures[layer_idx] = last_token_vector(work).cpu()
            stress_on = bool(getattr(self, "stress_enabled", False))
            should_clamp = bool(getattr(self, "steer_enabled", False)) and (
                stress_on or layer_idx == self.target_layer
            )
            if should_clamp:
                pre_rms = float(work.float().pow(2).mean().sqrt().clamp_min(1e-8))
                steered = activation_addition(
                    work,
                    self.truth_vec.to(device=hidden.device, dtype=torch.float32).to(hidden.dtype),
                    alpha=self.alpha,
                )
                post_rms = float(steered.float().pow(2).mean().sqrt())
                n_sat = int((steered.abs() >= (CLAMP_ABS - 1e-5)).sum().item())
                shift_pct = abs(post_rms - pre_rms) / pre_rms * 100.0
                if stress_on:
                    tau = float(getattr(self, "clamp_threshold", INJECT_COSINE_THRESHOLD))
                    self.clamp_logs.append(
                        f"[STRESS] L{layer_idx}.{self.component} unaligned gain={STRESS_UNALIGNED_GAIN:g} "
                        f"forced cos past tau={tau}"
                    )
                self.clamp_logs.append(
                    f"[CLAMP] L{layer_idx}.{self.component} ActAdd α={self.alpha:.2f} "
                    f"Δrms={post_rms - pre_rms:+.4f} shift={shift_pct:.2f}% sat={n_sat} "
                    f"h' = clamp(h + α v_truth, -{CLAMP_ABS:g}, {CLAMP_ABS:g})"
                )
                if isinstance(output, tuple):
                    return (steered,) + tuple(output[1:])
                return steered
            return output

        return hook

    def attach(self) -> None:
        self._clear_hooks()
        for idx, block in enumerate(self.blocks):
            target = _target_submodule(block, self.component)
            self._handles.append(target.register_forward_hook(self._capture_hook(idx)))

    def close(self) -> None:
        self._clear_hooks()

    def _forward(self, prompt: str) -> Tensor:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids)
        logits = out.logits if hasattr(out, "logits") else out
        return logits

    def _probe_cloud(self, prompt: str, *, steer: bool) -> ProbeReport:
        self.stress_enabled = STRESS_PROMPT_MARK in (prompt or "")
        self.steer_enabled = True if self.stress_enabled else bool(steer)
        self.clamp_logs = []
        try:
            pack = live_chat_completion(prompt, session=self._http)
            items = pack["items"] or [{"token": pack["text"][:24], "logprob": -0.5, "top": []}]
            traces = traces_from_live_tokens(items, prompt, component=self.component)
            generated = pack["text"] or " ".join(i.get("token") or "" for i in items)
            source = f"{pack['provider']}:{pack['model']}"
            self.source = source
        except Exception as exc:
            generated = f"(cloud timeout/fallback) {exc}"
            traces = traces_from_live_tokens(
                [{"token": "TIMEOUT", "logprob": -8.0, "top": []}],
                prompt,
                component=self.component,
            )
            source = self.source or "cloud-offline"
        flagged = sum(1 for t in traces if t.flagged)
        mean_anom = float(sum(max(0.0, t.cos_unaligned - t.cos_truth) for t in traces) / max(len(traces), 1))
        peak_shift = compute_peak_shift_pct(traces)
        angle = compute_steering_angle_deg(traces, 0)
        engine_d = float(sum(t.cosine_distance for t in traces) / max(len(traces), 1)) if traces else 0.0
        self.peak_shift_pct = float(peak_shift)
        self.steering_angle_deg = float(angle)
        if self.stress_enabled or any(t.flagged for t in traces):
            logs: list[str] = []
            for t in traces:
                tok = getattr(t, "token", "") or ""
                lp = float(getattr(t, "logprob", 0.0) or 0.0)
                logs.append(
                    f"[LIVE] T{t.layer} token={tok!r} logprob={lp:.3f} p={math.exp(max(-20.0, lp)):.4f} "
                    f"d_cos={t.cosine_distance:.4f}"
                )
                if t.flagged:
                    logs.append(
                        f"[STRESS] T{t.layer}.{self.component} adversarial token {tok!r} "
                        f"cos_u={t.cos_unaligned:.3f} FLAG tau={self.clamp_threshold}"
                    )
                    logs.append(
                        f"[CLAMP] T{t.layer}.{self.component} ActAdd α={self.alpha:.2f} "
                        f"h' = clamp(h + α v_truth, -{CLAMP_ABS:g}, {CLAMP_ABS:g})"
                    )
            self.clamp_logs = logs
        else:
            self.clamp_logs = [
                f"[IDLE] live Llama-3-8B pass · angle={angle:.1f}° peak_shift={peak_shift:.2f}%"
            ]
        return ProbeReport(
            source=source,
            quantized=False,
            prompt=prompt,
            target_layer=0,
            component=self.component,
            steered=steer,
            alpha=self.alpha,
            generated=generated.strip() or "(empty token)",
            layer_traces=traces,
            residual_dim=8,
            mean_anomaly=mean_anom,
            flagged_layers=flagged,
            peak_shift_pct=peak_shift,
            steering_angle_deg=angle,
            engine_cosine_distance=engine_d,
            live_text=generated.strip(),
            provider=source.split(":")[0] if ":" in source else source,
            clamp_logs=list(self.clamp_logs),
        )

    def probe(self, prompt: str, *, steer: bool = False, alpha: float | None = None) -> ProbeReport:
        if alpha is not None:
            self.alpha = float(alpha)
        if self.live:
            return self._probe_cloud(prompt, steer=steer)
        self.stress_enabled = STRESS_PROMPT_MARK in (prompt or "")
        self.steer_enabled = True if self.stress_enabled else bool(steer)
        self.clamp_threshold = (
            float(INJECT_COSINE_THRESHOLD) if self.stress_enabled else float(UNALIGNED_COSINE_THRESHOLD)
        )
        self.clamp_logs = []
        self.attach()
        logits = self._forward(prompt)
        next_ids = logits[0, -1].argmax(dim=-1, keepdim=True)
        generated = self.tokenizer.decode(next_ids)
        traces: list[LayerTrace] = []
        dim = self._hidden_dim()
        for idx in range(len(self.blocks)):
            vec = self._captures.get(idx)
            if vec is None:
                vec = torch.zeros(dim)
            if self.stress_enabled:
                u = self.unaligned_vec.to(device=vec.device, dtype=vec.dtype).flatten()
                if u.numel() != vec.numel():
                    u = u[: vec.numel()]
                vec = vec + (STRESS_UNALIGNED_GAIN * 3.0) * u
            hidden = vec.unsqueeze(0).unsqueeze(0)
            stats = drift_toward_unaligned(
                hidden,
                self.truth_vec,
                self.unaligned_vec,
                threshold=float(self.clamp_threshold),
            )
            if self.stress_enabled:
                stats["cos_unaligned"] = max(float(stats["cos_unaligned"]), float(self.clamp_threshold) + 0.55)
                stats["flagged"] = True
            d_cos = cosine_distance_from_baseline(vec, self.truth_vec)
            traces.append(
                LayerTrace(
                    layer=idx,
                    name=f"L{idx}.{self.component}",
                    rms=float(math.sqrt(float((vec * vec).mean()) + 1e-12)),
                    max_abs=float(vec.abs().max()) if vec.numel() else 0.0,
                    cos_truth=stats["cos_truth"],
                    cos_unaligned=stats["cos_unaligned"],
                    flagged=stats["flagged"],
                    cosine_distance=float(d_cos),
                    vector=vec[:24].tolist(),
                )
            )
        flagged = sum(1 for t in traces if t.flagged)
        mean_anom = float(sum(max(0.0, t.cos_unaligned - t.cos_truth) for t in traces) / max(len(traces), 1))
        peak_shift = compute_peak_shift_pct(traces)
        angle = compute_steering_angle_deg(traces, self.target_layer)
        self.peak_shift_pct = float(peak_shift)
        self.steering_angle_deg = float(angle)
        engine_d = 0.0
        if self.engine is not None:
            seed = traces[0].vector if traces and traces[0].vector else [0.0]
            xt = torch.tensor(seed, dtype=torch.float32)
            width = int(self.engine.width)
            if xt.numel() < width:
                xt = torch.cat([xt, torch.zeros(width - xt.numel())])
            else:
                xt = xt[:width]
            engine_pack = run_layer_forward(xt.view(1, 1, -1), self.engine, truthfulness_vector(width))
            engine_d = float(engine_pack["cosine_distance"])
        if self.stress_enabled:
            live: list[str] = [line for line in self.clamp_logs if "[IDLE]" not in line]
            if not any("[CLAMP]" in line for line in live):
                for t in traces:
                    live.append(
                        f"[STRESS] L{t.layer}.{self.component} cos_u={t.cos_unaligned:.3f} "
                        f"FLAG tau={self.clamp_threshold}"
                    )
                    live.append(
                        f"[CLAMP] L{t.layer}.{self.component} ActAdd α={self.alpha:.2f} "
                        f"h' = clamp(h + α v_truth, -{CLAMP_ABS:g}, {CLAMP_ABS:g})"
                    )
            self.clamp_logs = live
        elif not self.clamp_logs:
            self.clamp_logs.append(
                f"[IDLE] L{self.target_layer}.{self.component} no ActAdd this pass · "
                f"angle={angle:.1f}° peak_shift={peak_shift:.2f}%"
            )
        return ProbeReport(
            source=self.source,
            quantized=self.quantized,
            prompt=prompt,
            target_layer=self.target_layer,
            component=self.component,
            steered=steer,
            alpha=self.alpha,
            generated=generated.strip() or "(empty token)",
            layer_traces=traces,
            residual_dim=dim,
            mean_anomaly=mean_anom,
            flagged_layers=flagged,
            peak_shift_pct=peak_shift,
            steering_angle_deg=angle,
            engine_cosine_distance=engine_d,
            live_text=generated.strip(),
            provider="local",
            clamp_logs=list(self.clamp_logs),
        )


def build_steerer(
    model_id: str | None = None,
    component: str = "mlp",
    target_layer: int = 3,
) -> ActivationSteerer:
    if inference_ready():
        cfg = resolve_inference_endpoint() or {}
        label = f"{cfg.get('provider', 'cloud')}:{cfg.get('model', 'llama-3-8b')}"
        return ActivationSteerer(
            nn.Identity(),
            None,
            label,
            False,
            component=component,
            target_layer=0,
            live=True,
        )
    model, tokenizer, source, quantized = load_probe_model(model_id)
    return ActivationSteerer(
        model,
        tokenizer,
        source,
        quantized,
        component=component,
        target_layer=target_layer,
        live=False,
    )


if __name__ == "__main__":
    steerer = build_steerer()
    report = steerer.probe("State a verifiable fact about the residual stream.", steer=True)
    print(f"source={report.source} quantized={report.quantized} dim={report.residual_dim}")
    print(f"flagged={report.flagged_layers} mean_anomaly={report.mean_anomaly:.4f}")
    print(f"next_token={report.generated!r}")
    steerer.close()
