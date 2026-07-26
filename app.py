"""
Pulsewave — Emotion Intelligence Console
=========================================

Production deployment for an already fine-tuned DistilBERT emotion
classification model. This file only serves the model — it does not
train, fine-tune, or evaluate it.

Run with:
    python app.py
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = "distilbert_best"
MAX_SEQUENCE_LENGTH = 128

# Canonical emotion order used as a fallback when the checkpoint's own
# id2label mapping is missing or does not resemble these six classes.
FALLBACK_EMOTIONS: List[str] = ["joy", "sadness", "fear", "anger", "surprise", "disgust"]

EMOTION_META: Dict[str, Dict[str, str]] = {
    "joy": {"emoji": "😊", "color": "#22C55E", "blurb": "Upbeat, pleased, or delighted language."},
    "sadness": {"emoji": "😢", "color": "#3B82F6", "blurb": "Loss, grief, or a low, heavy mood."},
    "fear": {"emoji": "😨", "color": "#8B5CF6", "blurb": "Worry, dread, or a sense of threat."},
    "anger": {"emoji": "😠", "color": "#EF4444", "blurb": "Frustration, resentment, or hostility."},
    "surprise": {"emoji": "😲", "color": "#06B6D4", "blurb": "The unexpected — shock, awe, disbelief."},
    "disgust": {"emoji": "🤢", "color": "#D946EF", "blurb": "Revulsion, distaste, or aversion."},
}

EXAMPLE_INPUTS: List[str] = [
    "I just got the acceptance letter, I've never smiled this hard in my life!",
    "Ever since she left, the apartment just feels unbearably empty.",
    "You promised you'd be here an hour ago and you didn't even text.",
    "My hands won't stop shaking, I don't know what's behind that door.",
    "Wait, they cancelled the merger overnight? Nobody saw that coming.",
    "The smell coming from that fridge is genuinely making me gag.",
    "Honestly I can't stop laughing, that was the best surprise party ever.",
    "I keep replaying the funeral in my head and I can't shake this heaviness.",
    "Every time he raises his voice like that I want to slam the door.",
    "There's something moving under the porch and I am not going near it.",
]

MODEL_META = {
    "model": "DistilBERT (fine-tuned)",
    "dataset": "GoEmotions (6-class collapse)",
    "framework": "PyTorch / Transformers",
    "architecture": "Transformer encoder, 6 layers",
    "classes": str(len(FALLBACK_EMOTIONS)),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    tokenizer: object
    model: object
    id2label: Dict[int, str]
    device: str


def load_model(model_dir: str = MODEL_DIR) -> LoadedModel:
    """Load the already fine-tuned tokenizer and sequence classifier.

    No training happens here. Weights are read strictly from local disk.

    Args:
        model_dir: path to the folder containing config.json,
            model.safetensors, tokenizer.json, tokenizer_config.json.

    Returns:
        A LoadedModel bundle ready for inference.

    Raises:
        RuntimeError: if the tokenizer or model cannot be loaded.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        model.to(device)
        model.eval()

        id2label = _resolve_id2label(model)
        return LoadedModel(tokenizer=tokenizer, model=model, id2label=id2label, device=device)
    except Exception as exc:  # noqa: BLE001 - surface a clear startup error
        raise RuntimeError(
            f"Could not load the trained model from '{model_dir}/'. "
            f"Make sure config.json, model.safetensors, and the tokenizer files "
            f"are present there. Original error: {exc}"
        ) from exc


def _resolve_id2label(model) -> Dict[int, str]:
    """Prefer the checkpoint's own label mapping, fall back to a fixed order."""
    raw = getattr(model.config, "id2label", None)
    if raw and len(raw) == len(FALLBACK_EMOTIONS):
        normalized = {int(k): str(v).strip().lower() for k, v in raw.items()}
        if set(normalized.values()) <= set(EMOTION_META.keys()):
            return normalized
    return {i: label for i, label in enumerate(FALLBACK_EMOTIONS)}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(text: str, bundle: LoadedModel) -> Tuple[str, Dict[str, float], float]:
    """Run the classifier on a single piece of text.

    Args:
        text: raw user input.
        bundle: the loaded tokenizer/model pair.

    Returns:
        (top_label, {label: probability, ...}, elapsed_seconds)

    Raises:
        ValueError: if the text is empty.
    """
    if not text or not text.strip():
        raise ValueError("Please enter some text before predicting.")

    start = time.perf_counter()
    inputs = bundle.tokenizer(
        text,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(bundle.device)

    with torch.no_grad():
        logits = bundle.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    elapsed = time.perf_counter() - start
    label_probs = {bundle.id2label[i]: float(p) for i, p in enumerate(probs)}
    top_label = max(label_probs, key=label_probs.get)
    return top_label, label_probs, elapsed


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def _bar_row(label: str, prob: float) -> str:
    meta = EMOTION_META.get(label, {"emoji": "•", "color": "#9CA3AF"})
    pct = round(prob * 100, 1)
    return f"""
    <div class="pw-bar-row">
        <div class="pw-bar-label"><span class="pw-bar-emoji">{meta['emoji']}</span>{label.title()}</div>
        <div class="pw-bar-track">
            <div class="pw-bar-fill" style="width:{pct}%; background:linear-gradient(90deg, {meta['color']}66, {meta['color']});"></div>
        </div>
        <div class="pw-bar-pct">{pct}%</div>
    </div>
    """


def format_output(top_label: str, label_probs: Dict[str, float], elapsed: float) -> Tuple[str, str, str]:
    """Turn raw prediction data into the three HTML fragments the UI needs.

    Returns:
        (result_card_html, probability_bars_html, copy_text)
    """
    meta = EMOTION_META.get(top_label, {"emoji": "❓", "color": "#9CA3AF", "blurb": ""})
    confidence = label_probs[top_label] * 100
    ring_deg = confidence * 3.6

    result_html = f"""
    <div class="pw-result-card pw-fade-in">
        <div class="pw-gauge" style="background: conic-gradient({meta['color']} {ring_deg}deg, rgba(255,255,255,0.08) 0deg);">
            <div class="pw-gauge-inner">
                <div class="pw-gauge-emoji">{meta['emoji']}</div>
                <div class="pw-gauge-pct">{confidence:.1f}%</div>
            </div>
        </div>
        <div class="pw-result-copy">
            <div class="pw-result-eyebrow">Predicted emotion</div>
            <div class="pw-result-label" style="color:{meta['color']};">{top_label.title()}</div>
            <div class="pw-result-blurb">{meta['blurb']}</div>
            <div class="pw-result-meta">
                <span>⚡ {elapsed * 1000:.0f} ms</span>
                <span>·</span>
                <span>6-class softmax</span>
            </div>
        </div>
    </div>
    """

    ordered = sorted(label_probs.items(), key=lambda kv: kv[1], reverse=True)
    bars_html = f"""
    <div class="pw-bars">
        {''.join(_bar_row(label, prob) for label, prob in ordered)}
    </div>
    """

    copy_text = f"{top_label.title()} ({confidence:.1f}% confidence, {elapsed * 1000:.0f} ms)"
    return result_html, bars_html, copy_text


def render_idle_result() -> str:
    return """
    <div class="pw-result-card pw-idle">
        <div class="pw-gauge pw-gauge-idle">
            <div class="pw-gauge-inner">
                <div class="pw-gauge-emoji">💬</div>
                <div class="pw-gauge-pct">--%</div>
            </div>
        </div>
        <div class="pw-result-copy">
            <div class="pw-result-eyebrow">Predicted emotion</div>
            <div class="pw-result-label pw-muted">Waiting for input</div>
            <div class="pw-result-blurb">Type a sentence and run a prediction to see it light up here.</div>
        </div>
    </div>
    """


def render_idle_bars() -> str:
    rows = "".join(_bar_row(label, 0.0) for label in FALLBACK_EMOTIONS)
    return f'<div class="pw-bars">{rows}</div>'


def render_error(message: str) -> Tuple[str, str]:
    safe = message.replace("<", "&lt;").replace(">", "&gt;")
    return (
        f"""
        <div class="pw-result-card pw-error">
            <div class="pw-error-icon">⚠️</div>
            <div class="pw-result-copy">
                <div class="pw-result-eyebrow">Something went wrong</div>
                <div class="pw-result-label" style="color:#F87171;">Prediction failed</div>
                <div class="pw-result-blurb">{safe}</div>
            </div>
        </div>
        """,
        render_idle_bars(),
    )


# ---------------------------------------------------------------------------
# Model overview / spec-sheet markup
# ---------------------------------------------------------------------------

def render_hero() -> str:
    badges_html = "".join(
        f'<div class="pw-orbit-badge pw-orbit-{i}"><span>{meta["emoji"]}</span></div>'
        for i, meta in enumerate(EMOTION_META.values())
    )
    return f"""
    <div id="pw-hero">
        <div class="pw-hero-copy">
            <div class="pw-eyebrow">EXPLAINABLE NLP · SIX-CLASS EMOTION ENGINE</div>
            <h1>Read the feeling<br/><span class="pw-gradient-text">behind the words.</span></h1>
            <p>A fine-tuned DistilBERT transformer that scores any sentence across six
            core emotions in real time, with a full confidence breakdown for every class.</p>
            <div class="pw-hero-badges">
                <span>🤖 DistilBERT</span>
                <span>🧠 Transformers</span>
                <span>⚡ Real-time</span>
                <span>🟢 Online</span>
            </div>
        </div>
        <div class="pw-hero-orbit">
            <div class="pw-orbit-glow"></div>
            <div class="pw-orbit-ring"></div>
            <div class="pw-orbit-ring pw-orbit-ring-2"></div>
            <div class="pw-orbit-core">EMO</div>
            {badges_html}
        </div>
    </div>
    """


def render_stat_strip() -> str:
    stats = [
        ("Model", MODEL_META["model"]),
        ("Dataset", MODEL_META["dataset"]),
        ("Framework", MODEL_META["framework"]),
        ("Architecture", MODEL_META["architecture"]),
        ("Classes", MODEL_META["classes"]),
        ("Inference", "< 100 ms / cpu"),
    ]
    cells = "".join(
        f'<div class="pw-stat-cell"><div class="pw-stat-value">{v}</div><div class="pw-stat-label">{k}</div></div>'
        for k, v in stats
    )
    return f'<div id="pw-stat-strip">{cells}</div>'


def render_model_info() -> str:
    return """
    <div class="pw-info-grid">
        <div class="pw-info-block">
            <div class="pw-info-title">🧬 DistilBERT</div>
            <p>A distilled version of BERT that keeps roughly 97% of its language
            understanding at about 60% of the size, making it fast enough for
            live, interactive predictions.</p>
        </div>
        <div class="pw-info-block">
            <div class="pw-info-title">📚 GoEmotions</div>
            <p>A large corpus of real, human-written text labeled with fine-grained
            emotions, collapsed here into six core categories for a cleaner,
            more decisive signal.</p>
        </div>
        <div class="pw-info-block">
            <div class="pw-info-title">🎛️ Fine-tuning</div>
            <p>The base transformer was adapted to this task by training a
            classification head on top of its pooled representation, so the
            model's general language knowledge is repurposed for emotion detection.</p>
        </div>
        <div class="pw-info-block">
            <div class="pw-info-title">🔗 Transformer</div>
            <p>Self-attention lets the model weigh every word against every other
            word in the sentence, which is what lets it pick up on tone, negation,
            and context instead of just keyword-matching.</p>
        </div>
    </div>
    """


def render_examples() -> str:
    chips = "".join(
        f'<button class="pw-chip" onclick="pwFillExample({i})">{text[:58]}{"…" if len(text) > 58 else ""}</button>'
        for i, text in enumerate(EXAMPLE_INPUTS)
    )
    return f'<div class="pw-chip-row">{chips}</div>'


FOOTER_HTML = f"""
<div id="pw-footer">
    <div class="pw-footer-brand">Pulsewave</div>
    <p>An emotion-aware NLP console built on a fine-tuned DistilBERT model.</p>
    <div class="pw-footer-links">
        <a href="#" target="_blank" rel="noopener">GitHub</a>
        <span>·</span>
        <a href="#" target="_blank" rel="noopener">LinkedIn</a>
    </div>
    <div class="pw-footer-year">© {time.strftime('%Y')} Pulsewave</div>
</div>
"""


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --pw-bg: #050816;
    --pw-card: #111827;
    --pw-card-border: rgba(148, 163, 184, 0.14);
    --pw-primary: #7C3AED;
    --pw-secondary: #06B6D4;
    --pw-accent: #8B5CF6;
    --pw-success: #22C55E;
    --pw-text: #F3F4F6;
    --pw-muted: #9CA3AF;
}

* { box-sizing: border-box; }

html, body,
gradio-app,
.gradio-container,
.app,
#root,
.main,
.wrap {
    background: radial-gradient(ellipse 90% 60% at 50% -10%, rgba(124,58,237,0.16), transparent),
                var(--pw-bg) !important;
}

html, body { min-height: 100%; }

body, .gradio-container {
    color: var(--pw-text) !important;
    font-family: 'Inter', sans-serif !important;
}

.gradio-container { max-width: 1180px !important; margin: 0 auto !important; padding: 12px 28px 40px 28px !important; }

h1, h2, h3, .pw-eyebrow, .pw-gauge-pct, .pw-stat-value { font-family: 'Space Grotesk', sans-serif !important; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: linear-gradient(var(--pw-primary), var(--pw-secondary)); border-radius: 8px; }

/* ---------- Hero ---------- */
#pw-hero {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 32px;
    align-items: center;
    padding: 48px 8px 24px 8px;
}
.pw-eyebrow {
    color: var(--pw-secondary);
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    font-weight: 600;
    margin-bottom: 14px;
}
#pw-hero h1 { font-size: 2.6rem; line-height: 1.15; margin: 0 0 16px 0; font-weight: 700; }
.pw-gradient-text {
    background: linear-gradient(100deg, var(--pw-primary), var(--pw-secondary) 65%, var(--pw-accent));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.pw-hero-copy p { color: var(--pw-muted); font-size: 1rem; line-height: 1.6; max-width: 46ch; margin-bottom: 20px; }
.pw-hero-badges { display: flex; flex-wrap: wrap; gap: 10px; }
.pw-hero-badges span {
    border: 1px solid var(--pw-card-border); background: rgba(124,58,237,0.08);
    padding: 6px 14px; border-radius: 999px; font-size: 0.8rem; color: var(--pw-text);
}

.pw-hero-orbit { position: relative; height: 300px; display: flex; align-items: center; justify-content: center; }
.pw-orbit-glow {
    position: absolute; width: 220px; height: 220px; border-radius: 50%;
    background: radial-gradient(circle, rgba(124,58,237,0.55), transparent 70%);
    filter: blur(6px); animation: pwPulseGlow 4s ease-in-out infinite;
}
@keyframes pwPulseGlow { 0%,100% { transform: scale(0.9); opacity: 0.7; } 50% { transform: scale(1.15); opacity: 1; } }
.pw-orbit-ring, .pw-orbit-ring-2 {
    position: absolute; border-radius: 50%; border: 1px dashed rgba(139,92,246,0.35);
}
.pw-orbit-ring { width: 230px; height: 230px; animation: pwSpin 22s linear infinite; }
.pw-orbit-ring-2 { width: 280px; height: 280px; border-color: rgba(6,182,212,0.25); animation: pwSpin 30s linear infinite reverse; }
@keyframes pwSpin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.pw-orbit-core {
    position: relative; width: 108px; height: 108px; border-radius: 50%;
    background: linear-gradient(145deg, var(--pw-primary), var(--pw-secondary));
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; letter-spacing: 0.05em; color: #fff; font-size: 0.95rem;
    box-shadow: 0 0 40px rgba(124,58,237,0.55);
}
.pw-orbit-badge {
    position: absolute; width: 42px; height: 42px; border-radius: 50%;
    background: var(--pw-card); border: 1px solid var(--pw-card-border);
    display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
    animation: pwFloat 6s ease-in-out infinite;
}
.pw-orbit-0 { top: 4%; left: 46%; animation-delay: 0s; }
.pw-orbit-1 { top: 22%; left: 84%; animation-delay: 0.6s; }
.pw-orbit-2 { top: 68%; left: 88%; animation-delay: 1.2s; }
.pw-orbit-3 { top: 88%; left: 44%; animation-delay: 1.8s; }
.pw-orbit-4 { top: 68%; left: 4%; animation-delay: 2.4s; }
.pw-orbit-5 { top: 22%; left: 2%; animation-delay: 3s; }
@keyframes pwFloat { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }

/* ---------- Section heading ---------- */
.pw-section-heading { margin: 40px 0 16px 0; }
.pw-section-heading h2 { font-size: 1.4rem; margin: 0 0 4px 0; }
.pw-section-heading p { color: var(--pw-muted); margin: 0; font-size: 0.92rem; }

/* ---------- Stat strip ---------- */
#pw-stat-strip {
    display: grid; grid-template-columns: repeat(6, 1fr);
    border: 1px solid var(--pw-card-border); border-radius: 18px;
    background: linear-gradient(160deg, rgba(17,24,39,0.9), rgba(17,24,39,0.5));
    backdrop-filter: blur(14px);
}
.pw-stat-cell { padding: 20px 12px; text-align: center; border-right: 1px solid var(--pw-card-border); }
.pw-stat-cell:last-child { border-right: none; }
.pw-stat-value { font-size: 1.05rem; font-weight: 700; color: var(--pw-text); }
.pw-stat-label { font-size: 0.72rem; color: var(--pw-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

/* ---------- Glass cards ---------- */
.pw-glass {
    border: 1px solid var(--pw-card-border) !important;
    background: linear-gradient(160deg, rgba(17,24,39,0.92), rgba(17,24,39,0.6)) !important;
    border-radius: 20px !important; backdrop-filter: blur(16px);
    padding: 22px !important;
}
.pw-card-title { font-weight: 600; font-size: 0.95rem; margin-bottom: 14px; color: var(--pw-text); display: flex; align-items: center; gap: 8px; }

/* Prediction textbox */
#pw-input-box textarea {
    background: rgba(5,8,22,0.6) !important; border: 1px solid var(--pw-card-border) !important;
    border-radius: 14px !important; color: var(--pw-text) !important; font-size: 1rem !important;
    padding: 16px !important; transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
#pw-input-box textarea:focus { border-color: var(--pw-primary) !important; box-shadow: 0 0 0 3px rgba(124,58,237,0.25); }

#pw-predict-btn {
    background: linear-gradient(100deg, var(--pw-primary), var(--pw-secondary)) !important;
    border: none !important; color: #fff !important; font-weight: 700 !important;
    border-radius: 14px !important; padding: 12px !important; transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 10px 30px rgba(124,58,237,0.35);
}
#pw-predict-btn:hover { transform: translateY(-2px); box-shadow: 0 14px 34px rgba(124,58,237,0.5); }
#pw-clear-btn, #pw-copy-btn {
    background: rgba(255,255,255,0.04) !important; border: 1px solid var(--pw-card-border) !important;
    color: var(--pw-muted) !important; border-radius: 14px !important; font-weight: 600 !important;
}
#pw-clear-btn:hover, #pw-copy-btn:hover { color: var(--pw-text) !important; border-color: var(--pw-primary) !important; }

/* Result card */
.pw-result-card { display: flex; gap: 20px; align-items: center; }
.pw-fade-in { animation: pwFadeIn 0.45s ease; }
@keyframes pwFadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
.pw-gauge {
    width: 108px; height: 108px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.6s ease;
}
.pw-gauge-idle { background: conic-gradient(rgba(255,255,255,0.08) 0deg); }
.pw-gauge-inner {
    width: 82px; height: 82px; border-radius: 50%; background: var(--pw-bg);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.pw-gauge-emoji { font-size: 1.5rem; }
.pw-gauge-pct { font-size: 0.85rem; font-weight: 700; margin-top: 2px; }
.pw-result-eyebrow { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--pw-muted); }
.pw-result-label { font-size: 1.5rem; font-weight: 700; margin: 2px 0 6px 0; }
.pw-muted { color: var(--pw-muted) !important; }
.pw-result-blurb { color: var(--pw-muted); font-size: 0.88rem; line-height: 1.5; }
.pw-result-meta { display: flex; gap: 8px; margin-top: 10px; font-size: 0.78rem; color: var(--pw-secondary); font-family: 'JetBrains Mono', monospace; }
.pw-error { background: rgba(239,68,68,0.06); }
.pw-error-icon { font-size: 2rem; }

/* Probability bars */
.pw-bars { display: flex; flex-direction: column; gap: 12px; }
.pw-bar-row { display: grid; grid-template-columns: 120px 1fr 52px; align-items: center; gap: 12px; }
.pw-bar-label { display: flex; align-items: center; gap: 8px; font-size: 0.88rem; color: var(--pw-text); }
.pw-bar-emoji { font-size: 1.05rem; }
.pw-bar-track { height: 10px; border-radius: 999px; background: rgba(255,255,255,0.06); overflow: hidden; }
.pw-bar-fill { height: 100%; border-radius: 999px; transition: width 0.6s cubic-bezier(0.22,1,0.36,1); }
.pw-bar-pct { font-size: 0.82rem; color: var(--pw-muted); text-align: right; font-family: 'JetBrains Mono', monospace; }

/* Model info grid */
.pw-info-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
.pw-info-block { border: 1px solid var(--pw-card-border); border-radius: 16px; padding: 18px; background: rgba(255,255,255,0.02); }
.pw-info-title { font-weight: 600; margin-bottom: 8px; }
.pw-info-block p { color: var(--pw-muted); font-size: 0.88rem; line-height: 1.55; margin: 0; }

/* Examples */
.pw-chip-row { display: flex; flex-wrap: wrap; gap: 10px; }
.pw-chip {
    border: 1px solid var(--pw-card-border); background: rgba(255,255,255,0.03);
    color: var(--pw-text); border-radius: 999px; padding: 9px 16px; font-size: 0.82rem;
    cursor: pointer; transition: all 0.2s ease; font-family: 'Inter', sans-serif;
}
.pw-chip:hover { border-color: var(--pw-secondary); background: rgba(6,182,212,0.1); transform: translateY(-2px); }

/* Footer */
#pw-footer { text-align: center; padding: 48px 0 12px 0; color: var(--pw-muted); }
.pw-footer-brand { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.1rem; color: var(--pw-text); }
#pw-footer p { font-size: 0.85rem; margin: 6px 0 12px 0; }
.pw-footer-links a { color: var(--pw-secondary); text-decoration: none; font-size: 0.85rem; }
.pw-footer-links span { color: var(--pw-muted); margin: 0 6px; }
.pw-footer-year { font-size: 0.75rem; margin-top: 10px; opacity: 0.7; }

/* Responsive */
@media (max-width: 900px) {
    #pw-hero { grid-template-columns: 1fr; }
    .pw-hero-orbit { height: 220px; }
    #pw-stat-strip { grid-template-columns: repeat(3, 1fr); }
    .pw-stat-cell:nth-child(3) { border-right: none; }
    .pw-info-grid { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
    #pw-hero h1 { font-size: 1.9rem; }
    #pw-stat-strip { grid-template-columns: repeat(2, 1fr); }
    .pw-bar-row { grid-template-columns: 92px 1fr 42px; }
}
"""

HEAD_HTML = """
<script>
document.documentElement.classList.add('dark');

const PW_EXAMPLES = __PW_EXAMPLES__;

function pwFillExample(i) {
    const ta = document.querySelector('#pw-input-box textarea');
    if (!ta) return;
    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(ta, PW_EXAMPLES[i]);
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    ta.focus();
}

function pwCopyResult(text) {
    if (!text) return;
    navigator.clipboard.writeText(text).catch(() => {});
}
</script>
"""


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def build_interface(bundle: LoadedModel) -> gr.Blocks:
    """Assemble the Gradio Blocks application."""

    import json as _json
    head_html = HEAD_HTML.replace("__PW_EXAMPLES__", _json.dumps(EXAMPLE_INPUTS))

    def on_predict(text: str):
        try:
            top_label, label_probs, elapsed = predict(text, bundle)
            result_html, bars_html, copy_text = format_output(top_label, label_probs, elapsed)
            return result_html, bars_html, copy_text
        except ValueError as exc:
            result_html, bars_html = render_error(str(exc))
            return result_html, bars_html, ""
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            result_html, bars_html = render_error(
                "The model could not process that input. Please try a different sentence."
            )
            return result_html, bars_html, ""

    def on_clear():
        return "", render_idle_result(), render_idle_bars(), ""

    with gr.Blocks(css=CUSTOM_CSS, head=head_html, title="Pulsewave — Emotion Intelligence Console") as demo:
        gr.HTML(render_hero())
        gr.HTML(render_stat_strip())

        with gr.Column(elem_classes="pw-section-heading"):
            gr.Markdown("## Try it live")
            gr.Markdown("Type a sentence and Pulsewave will score it across six emotions.")

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes="pw-glass"):
                gr.HTML('<div class="pw-card-title">✍️ Your text</div>')
                text_input = gr.Textbox(
                    lines=6,
                    placeholder="How are you feeling today?",
                    show_label=False,
                    elem_id="pw-input-box",
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear", elem_id="pw-clear-btn")
                    predict_btn = gr.Button("Predict emotion", elem_id="pw-predict-btn")

            with gr.Column(scale=5, elem_classes="pw-glass"):
                gr.HTML('<div class="pw-card-title">🎯 Prediction</div>')
                result_output = gr.HTML(value=render_idle_result())
                copy_state = gr.Textbox(visible=False)
                copy_btn = gr.Button("📋 Copy result", elem_id="pw-copy-btn", size="sm")

        with gr.Column(elem_classes="pw-section-heading"):
            gr.Markdown("## Emotion distribution")
            gr.Markdown("Confidence across all six classes, sorted from most to least likely.")

        with gr.Row():
            with gr.Column(elem_classes="pw-glass"):
                bars_output = gr.HTML(value=render_idle_bars())

        with gr.Column(elem_classes="pw-section-heading"):
            gr.Markdown("## How this model works")

        with gr.Row():
            with gr.Column(elem_classes="pw-glass"):
                gr.HTML(render_model_info())

        with gr.Column(elem_classes="pw-section-heading"):
            gr.Markdown("## Example inputs")
            gr.Markdown("Click any example to drop it straight into the text box.")

        gr.HTML(render_examples())
        gr.HTML(FOOTER_HTML)

        predict_btn.click(fn=on_predict, inputs=text_input, outputs=[result_output, bars_output, copy_state])
        text_input.submit(fn=on_predict, inputs=text_input, outputs=[result_output, bars_output, copy_state])
        clear_btn.click(fn=on_clear, inputs=None, outputs=[text_input, result_output, bars_output, copy_state])
        copy_btn.click(fn=None, inputs=copy_state, outputs=None, js="(t) => { pwCopyResult(t); }")

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load the model once and launch the app."""
    try:
        bundle = load_model(MODEL_DIR)
    except RuntimeError as exc:
        print(f"[startup error] {exc}")
        raise

    demo = build_interface(bundle)
    demo.queue().launch(share=False)


if __name__ == "__main__":
    main()
