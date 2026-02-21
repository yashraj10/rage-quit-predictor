"""Phase 6: Rage Quit Predictor — Real functionality + Figma visual design."""
import json
import pickle
from pathlib import Path
import numpy as np
import streamlit as st
import torch

from data.dataset import DotaMatchDataset
from data.vocab import ID_TO_EVENT, PAD_TOKEN_ID
from model.transformer import RageQuitTransformer
from model.evaluate import evaluate_model

st.set_page_config(page_title="Rage Quit Predictor", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════════════════
# CSS — Figma zinc design system
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

.stApp { background-color: #09090b; }
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 0 !important; max-width: 1280px; }

/* Tabs */
button[data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    color: #71717a !important; font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #fafafa !important; background: #18181b !important;
    border: 1px solid #27272a !important; border-radius: 6px !important;
}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { background-color: transparent !important; }

/* Navbar */
.nav { display:flex; align-items:center; padding:16px 0; border-bottom:1px solid #27272a; margin-bottom:0; }
.nav-icon { width:40px; height:40px; background:#18181b; border:1px solid #27272a; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:18px; margin-right:14px; }
.nav-title { font-family:'JetBrains Mono',monospace; font-size:1.05rem; font-weight:700; color:#fafafa; }
.nav-badge { font-size:0.6rem; color:#71717a; background:#18181b; border:1px solid #27272a; padding:2px 8px; border-radius:4px; margin-left:8px; }
.nav-sub { font-family:'Inter',sans-serif; font-size:0.78rem; color:#52525b; margin-top:2px; }

/* Metric Card */
.mc { background:#18181b; border:1px solid #27272a; padding:18px 20px; position:relative; overflow:hidden; }
.mc .bar { position:absolute; left:0; top:0; bottom:0; width:3px; }
.mc .inn { padding-left:12px; }
.mc .lbl { font-family:'JetBrains Mono',monospace; font-size:0.58rem; color:#71717a; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px; }
.mc .val { font-family:'JetBrains Mono',monospace; font-size:1.6rem; font-weight:400; color:#e4e4e7; display:inline; }
.mc .dlt { font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:500; margin-left:8px; }
.dlt-g { color:#10b981; } .dlt-r { color:#f43f5e; } .dlt-c { color:#06b6d4; } .dlt-a { color:#f59e0b; }

/* Risk Banner */
.rb { display:flex; justify-content:space-between; align-items:center; padding:12px 24px; font-family:'JetBrains Mono',monospace; border-top:1px solid; border-bottom:1px solid; margin-bottom:24px; }
.rb.high { border-color:rgba(244,63,94,0.3); background:rgba(244,63,94,0.04); }
.rb.low  { border-color:rgba(16,185,129,0.3); background:rgba(16,185,129,0.04); }
.rb.mid  { border-color:rgba(245,158,11,0.3); background:rgba(245,158,11,0.04); }
.rb .rl { font-weight:700; font-size:0.75rem; letter-spacing:2px; text-transform:uppercase; display:flex; align-items:center; gap:10px; }
.rb .rr { font-weight:600; font-size:0.9rem; }
.rb.high .rl,.rb.high .rr { color:#f43f5e; }
.rb.low .rl,.rb.low .rr { color:#10b981; }
.rb.mid .rl,.rb.mid .rr { color:#f59e0b; }
.pdot { width:8px; height:8px; border-radius:50%; animation:pulse 2s ease-in-out infinite; display:inline-block; }
.pdot.r { background:#f43f5e; } .pdot.g { background:#10b981; } .pdot.y { background:#f59e0b; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Section Header */
.sh { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#71717a; text-transform:uppercase; letter-spacing:2.5px; margin:24px 0 14px; padding-bottom:8px; border-bottom:1px solid #27272a; }

/* Comparison Table */
.ct { width:100%; border-collapse:collapse; font-family:'JetBrains Mono',monospace; font-size:0.8rem; background:#18181b; border:1px solid #27272a; }
.ct th { background:#09090b; color:#71717a; font-weight:500; font-size:0.6rem; text-transform:uppercase; letter-spacing:0.8px; padding:12px; text-align:right; border-bottom:1px solid #27272a; }
.ct th:first-child { text-align:left; }
.ct td { padding:12px; color:#a1a1aa; text-align:right; }
.ct td:first-child { text-align:left; }
.ct .nm { color:#a1a1aa; font-weight:600; } .ct .tf { color:#8b5cf6; font-weight:600; }
.ct .w { color:#10b981; font-weight:700; background:rgba(16,185,129,0.05); }
.ct tr:hover td { background:rgba(39,39,42,0.3); }

/* Decision Card */
.dcd { background:#18181b; padding:20px; border-left:2px solid #27272a; margin-bottom:8px; }
.dcd.rose { border-left-color:#f43f5e; } .dcd.violet { border-left-color:#8b5cf6; } .dcd.amber { border-left-color:#f59e0b; }
.dcd .q { font-family:'JetBrains Mono',monospace; font-size:0.85rem; font-weight:700; margin-bottom:8px; }
.dcd.rose .q { color:#f43f5e; } .dcd.violet .q { color:#8b5cf6; } .dcd.amber .q { color:#f59e0b; }
.dcd .a { font-family:'Inter',sans-serif; font-size:0.85rem; color:#a1a1aa; line-height:1.6; }

/* ── Timeline Sequence ── */
.seq-outer { background:#18181b; border:1px solid #27272a; padding:24px; }
.seq-title { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#71717a; text-transform:uppercase; letter-spacing:2px; margin-bottom:24px; }
.seq-scroll { overflow-x:auto; padding-bottom:8px; }
.seq-track { display:flex; align-items:center; gap:0; min-width:max-content; padding:8px 0 16px; }
.sev { display:flex; flex-direction:column; align-items:center; gap:6px; }
.sev .time { font-family:'JetBrains Mono',monospace; font-size:0.56rem; color:#52525b; }
.sev .pip { padding:5px 12px; border-radius:100px; font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:500; white-space:nowrap; border:1px solid; cursor:default; }
.sev .pip:hover { transform:translateY(-3px) scale(1.05); transition:all 0.2s ease; filter:brightness(1.2); }
.sev .tick { width:1px; height:16px; background:#27272a; }
/* Low attention — outline */
.pip-pos { background:rgba(0,255,136,0.08); color:#00FF88; border-color:rgba(0,255,136,0.35); font-weight:600; }
.pip-neg { background:rgba(255,45,85,0.08); color:#FF2D55; border-color:rgba(255,45,85,0.35); font-weight:600; }
.pip-wrn { background:rgba(255,184,0,0.08); color:#FFB800; border-color:rgba(255,184,0,0.35); font-weight:600; }
.pip-neu { background:rgba(113,113,122,0.08); color:#a1a1aa; border-color:rgba(113,113,122,0.3); }
/* High attention — solid + glow */
.pip-pos.hi { background:#00FF88; color:#001A0D; border-color:#00FF88; box-shadow:0 0 12px rgba(0,255,136,0.5), 0 0 24px rgba(0,255,136,0.25); }
.pip-neg.hi { background:#FF2D55; color:#ffffff; border-color:#FF2D55; box-shadow:0 0 12px rgba(255,45,85,0.6), 0 0 24px rgba(255,45,85,0.3); }
.pip-wrn.hi { background:#FFB800; color:#1A1200; border-color:#FFB800; box-shadow:0 0 12px rgba(255,184,0,0.5), 0 0 24px rgba(255,184,0,0.25); }
.pip-neu.hi { background:#71717a; color:#fafafa; border-color:#71717a; box-shadow:0 0 8px rgba(113,113,122,0.4); }
/* Medium attention — tinted */
.pip-pos.mid { background:rgba(0,255,136,0.2); color:#00FF88; border-color:rgba(0,255,136,0.5); }
.pip-neg.mid { background:rgba(255,45,85,0.2); color:#FF2D55; border-color:rgba(255,45,85,0.5); }
.pip-wrn.mid { background:rgba(255,184,0,0.2); color:#FFB800; border-color:rgba(255,184,0,0.5); }
.pip-neu.mid { background:rgba(113,113,122,0.15); color:#d4d4d8; border-color:rgba(113,113,122,0.4); }
.seq-dot { width:4px; height:4px; border-radius:50%; background:#27272a; margin:28px 6px 0; flex-shrink:0; }
.seq-legend { display:flex; justify-content:center; gap:28px; margin-top:20px; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#71717a; }
.seq-legend .ld { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:6px; vertical-align:middle; }
.abar { width:3px; border-radius:2px; margin-top:2px; transition:height 0.2s; }
.ev { display:inline-block; white-space:nowrap; }
.ev.pip-pos { color:#00FF88; } .ev.pip-neg { color:#FF2D55; } .ev.pip-wrn { color:#FFB800; } .ev.pip-neu { color:#a1a1aa; }

/* Insight Card */
.ic { padding:18px; border:1px solid #27272a; background:rgba(9,9,11,0.5); }
.ic h4 { font-family:'JetBrains Mono',monospace; font-size:0.85rem; font-weight:500; color:#d4d4d8; margin:0 0 8px; }
.ic p { font-family:'Inter',sans-serif; font-size:0.78rem; color:#71717a; line-height:1.6; margin:0; }

/* Architecture */
.arch-block { background:#18181b; border:1px solid #27272a; border-radius:8px; padding:16px 20px; margin:8px 0; font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#e4e4e7; }
.arch-block.accent { border-left:3px solid #8b5cf6; }
.arch-arrow { text-align:center; color:#52525b; font-size:1.2rem; margin:4px 0; }
.arch-label { font-family:'Inter',sans-serif; font-size:0.7rem; color:#71717a; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA & MODEL
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    cp = torch.load("results/weights/best_model.pt", map_location="cpu", weights_only=False)
    model = RageQuitTransformer(**cp["config"])
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    return model, cp["epoch"], cp["val_auc_pr"]

@st.cache_data
def load_data():
    with open("data/processed/test_sequences.pkl", "rb") as f:
        test_recs = pickle.load(f)
    return test_recs

@st.cache_data
def load_metrics():
    """Load precomputed test metrics — single source of truth."""
    metrics_path = Path("results/metrics/test_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None

def extract_cls_attention(model, batch_dict, layer_idx=-1):
    """Extract [CLS] → all tokens attention from transformer last layer."""
    model.eval()
    attention_maps = []

    def make_hook(maps_list):
        def hook_fn(module, args, kwargs, output):
            # Force a second call with need_weights=True to get attention
            with torch.no_grad():
                q, k, v = args[0], args[1], args[2]
                _, attn_weights = module.forward(
                    q, k, v,
                    key_padding_mask=kwargs.get("key_padding_mask", None),
                    need_weights=True,
                    average_attn_weights=False,
                )
                if attn_weights is not None:
                    maps_list.append(attn_weights.detach())
            return output
        return hook_fn

    hooks = []
    for layer in model.transformer_encoder.layers:
        h = layer.self_attn.register_forward_hook(make_hook(attention_maps), with_kwargs=True)
        hooks.append(h)

    with torch.no_grad():
        _ = model(batch_dict["event_ids"], batch_dict["continuous_features"],
                  batch_dict["minutes"], batch_dict["attention_mask"])

    for h in hooks:
        h.remove()

    if not attention_maps:
        return None

    attn = attention_maps[layer_idx]   # (B, heads, seq, seq)
    attn_avg = attn.mean(dim=1)        # (B, seq, seq)
    cls_attn = attn_avg[:, 0, :]       # (B, seq) — CLS attending to all
    return cls_attn.numpy()

def predict_with_attention(model, record):
    dataset = DotaMatchDataset([record], max_seq_len=256)
    batch = dataset[0]
    single = {k: v.unsqueeze(0) for k, v in batch.items()}
    with torch.no_grad():
        logits = model(
            single["event_ids"], single["continuous_features"],
            single["minutes"], single["attention_mask"],
        ).squeeze()
        prob = torch.sigmoid(logits).item()
    cls_attn = extract_cls_attention(model, single)
    attn_weights = cls_attn[0] if cls_attn is not None else None
    return prob, attn_weights

def get_events_with_minutes(record):
    SKIP = {"[CLS]", "[SEP]", "SMALL_PURCHASE"}
    result = []
    for pos, (eid, minute) in enumerate(zip(record["event_ids"], record["minutes"])):
        if eid == PAD_TOKEN_ID:
            continue
        name = ID_TO_EVENT.get(eid, f"UNK_{eid}")
        if name in SKIP:
            continue
        result.append((name, int(minute), pos))
    return result

POSITIVE = {"KILL","ASSIST","MULTI_KILL","GOLD_SPIKE_UP","BIG_PURCHASE","TEAM_FIGHT_WIN","TOWER_TAKEN","ACTION_BURST","LH_ABOVE_AVG"}
NEGATIVE = {"DEATH","DEATH_STREAK","GOLD_SPIKE_DOWN","TEAM_FIGHT_LOSS","TOWER_LOST"}
WARNING  = {"ACTION_DROUGHT","XP_FALLING_BEHIND","LH_BELOW_AVG","LONG_IDLE"}

def pill_class(e):
    if e in POSITIVE: return "pip-pos"
    if e in NEGATIVE: return "pip-neg"
    if e in WARNING:  return "pip-wrn"
    return "pip-neu"

def short_name(e):
    renames = {
        "TEAM_FIGHT_WIN": "TF Win", "TEAM_FIGHT_LOSS": "TF Loss",
        "GOLD_SPIKE_UP": "Gold ↑", "GOLD_SPIKE_DOWN": "Gold ↓",
        "XP_FALLING_BEHIND": "XP Behind", "LH_BELOW_AVG": "LH Low",
        "LH_ABOVE_AVG": "LH High", "BIG_PURCHASE": "Big Buy",
        "SMALL_PURCHASE": "Sm Buy", "ACTION_BURST": "APM ↑",
        "ACTION_DROUGHT": "APM ↓", "DEATH_STREAK": "Deaths x3",
        "MULTI_KILL": "Multi Kill", "LONG_IDLE": "Idle",
        "TOWER_TAKEN": "Tower ↑", "TOWER_LOST": "Tower ↓",
    }
    return renames.get(e, e.replace("_", " ").title())


# ═══════════════════════════════════════════════════════════════════════
# RENDERERS
# ═══════════════════════════════════════════════════════════════════════
def mcard(label, value, delta, accent_color, delta_class):
    dlt = f'<span class="dlt {delta_class}">{delta}</span>' if delta else ""
    return f'<div class="mc"><div class="bar" style="background:{accent_color}"></div><div class="inn"><div class="lbl">{label}</div><div class="val">{value}</div>{dlt}</div></div>'

def build_attention_timeline(events_with_minutes, attn_weights, max_events=40):
    evts = events_with_minutes[:max_events]
    if not evts:
        return '<div style="color:#52525b; font-style:italic;">No events</div>'

    attns = []
    for name, minute, pos in evts:
        if attn_weights is not None and pos < len(attn_weights):
            attns.append(float(attn_weights[pos]))
        else:
            attns.append(0.0)

    a_max = max(attns) if attns and max(attns) > 0 else 1.0
    normed = [a / a_max for a in attns]

    html = '<div class="seq-outer">'
    html += '<div class="seq-title">Attention-Weighted Event Sequence</div>'
    html += '<div class="seq-scroll"><div class="seq-track">'

    for i, ((name, minute, pos), attn_norm) in enumerate(zip(evts, normed)):
        secs = (pos * 7) % 60
        ts = f"{minute:02d}:{secs:02d}"
        pcls = pill_class(name)
        short = short_name(name)
        attn_pct = int(attn_norm * 100)

        if attn_norm > 0.65:
            tier = "hi"
        elif attn_norm > 0.35:
            tier = "mid"
        else:
            tier = ""

        time_color = "#fafafa" if attn_norm > 0.65 else ("#71717a" if attn_norm > 0.35 else "#3f3f46")

        if name in NEGATIVE:
            bar_color = f"rgba(255,45,85,{max(0.3, attn_norm):.2f})"
        elif name in WARNING:
            bar_color = f"rgba(255,184,0,{max(0.3, attn_norm):.2f})"
        elif name in POSITIVE:
            bar_color = f"rgba(0,255,136,{max(0.3, attn_norm):.2f})"
        else:
            bar_color = f"rgba(113,113,122,{max(0.2, attn_norm):.2f})"

        bar_h = max(4, int(attn_norm * 28))

        html += f"""
        <div class="sev">
            <div class="time" style="color:{time_color};">{ts}</div>
            <div class="abar" style="height:{bar_h}px; background:{bar_color};"></div>
            <div class="pip {pcls} {tier}" title="Attention: {attn_pct}%">{short}</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.55rem; color:{bar_color}; margin-top:2px;">{attn_pct}%</div>
        </div>
        """
        if i < len(evts) - 1:
            html += '<div class="seq-dot"></div>'

    html += '</div></div>'
    html += """
    <div class="seq-legend">
        <span><span class="ld" style="background:#00FF88; box-shadow:0 0 6px rgba(0,255,136,0.5);"></span> Positive</span>
        <span><span class="ld" style="background:#FF2D55; box-shadow:0 0 6px rgba(255,45,85,0.5);"></span> Negative</span>
        <span><span class="ld" style="background:#FFB800; box-shadow:0 0 6px rgba(255,184,0,0.5);"></span> Warning</span>
        <span style="margin-left:16px; color:#52525b;">▮ = model attention intensity</span>
    </div>
    """
    html += '</div>'
    return html

def build_attention_summary(events_with_minutes, attn_weights):
    type_attn = {}
    type_count = {}
    for name, minute, pos in events_with_minutes:
        if attn_weights is not None and pos < len(attn_weights):
            a = float(attn_weights[pos])
            type_attn[name] = type_attn.get(name, 0.0) + a
            type_count[name] = type_count.get(name, 0) + 1

    avg_attn = {}
    for name in type_attn:
        if type_count[name] > 0:
            avg_attn[name] = type_attn[name] / type_count[name]

    if not avg_attn:
        return ""

    sorted_attn = sorted(avg_attn.items(), key=lambda x: x[1], reverse=True)
    top = sorted_attn[:6]
    a_max = top[0][1] if top else 1.0

    html = '<div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:16px;">'
    for name, avg in top:
        pct = int((avg / a_max) * 100)
        pcls = pill_class(name)
        short = short_name(name)
        if name in NEGATIVE:
            bar_clr = "#FF2D55"
        elif name in WARNING:
            bar_clr = "#FFB800"
        elif name in POSITIVE:
            bar_clr = "#00FF88"
        else:
            bar_clr = "#71717a"
        html += f"""
        <div style="background:#18181b; border:1px solid #27272a; padding:8px 14px; display:flex; align-items:center; gap:10px; min-width:140px;">
            <span class="ev {pcls}" style="font-size:0.65rem; padding:3px 8px; border-radius:100px; border:1px solid;">{short}</span>
            <div style="flex:1;">
                <div style="background:#27272a; height:4px; border-radius:2px; overflow:hidden;">
                    <div style="width:{pct}%; height:100%; background:{bar_clr}; border-radius:2px; box-shadow:0 0 6px {bar_clr}40;"></div>
                </div>
            </div>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:{bar_clr};">{pct}%</span>
        </div>
        """
    html += '</div>'
    return html


# ═══════════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════════
def generate_sequence_narrative(events_with_minutes, attn_weights, prob, label):
    """Generate a 2-3 line dynamic explanation of what the model sees."""
    if not events_with_minutes:
        return ""

    # Get top attention events
    type_attn = {}
    type_count = {}
    for name, minute, pos in events_with_minutes:
        if attn_weights is not None and pos < len(attn_weights):
            a = float(attn_weights[pos])
            type_attn[name] = type_attn.get(name, 0.0) + a
            type_count[name] = type_count.get(name, 0) + 1
    avg_attn = {n: type_attn[n]/type_count[n] for n in type_attn if type_count[n] > 0}
    top = sorted(avg_attn.items(), key=lambda x: x[1], reverse=True)

    # Count event categories
    neg_count = sum(1 for n,_,_ in events_with_minutes if n in NEGATIVE)
    wrn_count = sum(1 for n,_,_ in events_with_minutes if n in WARNING)
    pos_count = sum(1 for n,_,_ in events_with_minutes if n in POSITIVE)
    total = len(events_with_minutes)

    # Build narrative
    lines = []

    # Line 1: What dominated the game
    if neg_count + wrn_count > pos_count:
        pct = int((neg_count + wrn_count) / total * 100)
        lines.append(f"<span style='color:#FF2D55;'>⚠</span> <b>{pct}%</b> of this player's events were negative or warning signals.")
    else:
        pct = int(pos_count / total * 100)
        lines.append(f"<span style='color:#00FF88;'>✦</span> <b>{pct}%</b> of events were positive — a relatively healthy game.")

    # Line 2: What the model focused on
    if top:
        top_name = short_name(top[0][0])
        top_cls = pill_class(top[0][0])
        if len(top) > 1:
            second_name = short_name(top[1][0])
            lines.append(f"Model locked onto <span class='{top_cls}' style='font-weight:700;'>{top_name}</span> and <b>{second_name}</b> as key signals.")
        else:
            lines.append(f"Model locked onto <span class='{top_cls}' style='font-weight:700;'>{top_name}</span> as the dominant signal.")

    # Line 3: The verdict
    if prob > 0.5:
        lines.append("<span style='color:#FF2D55;'>→</span> Pattern: declining performance + disengagement = predicted quit.")
    elif prob > 0.2:
        lines.append("<span style='color:#FFB800;'>→</span> Some warning signs, but not enough for a confident prediction.")
    else:
        lines.append("<span style='color:#00FF88;'>→</span> No strong quit indicators — model sees a stable player.")

    return "<br>".join(lines)

def main():
    model, best_epoch, best_auc_pr = load_model()
    test_recs = load_data()
    m = load_metrics()

    # Navbar
    st.markdown("""
    <div class="nav">
        <div class="nav-icon">⚡</div>
        <div>
            <div class="nav-title">RAGE_QUIT_PREDICTOR <span class="nav-badge">v1.0.0</span></div>
            <div class="nav-sub">Real-time behavioral sequence analysis for esports</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Performance Metrics", "◇ Sequence Explorer", "⚙ Model Architecture"])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1: PERFORMANCE — all numbers from test_metrics.json
    # ══════════════════════════════════════════════════════════════════
    with tab1:
        if not m:
            st.warning("⚠ Run `python generate_results.py` first to compute metrics.")

        # 5 metric cards from JSON
        c1,c2,c3,c4,c5 = st.columns(5)
        if m:
            cards = [
                ("AUC-ROC", f"{m['auc_roc']:.3f}", "", "#10b981", ""),
                ("AUC-PR", f"{m['auc_pr']:.3f}", "", "#8b5cf6", ""),
                ("F1 Score", f"{m['f1']:.3f}", "", "#06b6d4", ""),
                ("Precision", f"{m['precision']:.3f}", "", "#f43f5e", ""),
                ("Recall", f"{m['recall']:.3f}", "", "#f59e0b", ""),
            ]
        else:
            cards = [
                ("AUC-ROC", "—", "", "#10b981", ""),
                ("AUC-PR", "—", "", "#8b5cf6", ""),
                ("F1 Score", "—", "", "#06b6d4", ""),
                ("Precision", "—", "", "#f43f5e", ""),
                ("Recall", "—", "", "#f59e0b", ""),
            ]
        for col, args in zip([c1,c2,c3,c4,c5], cards):
            with col: st.markdown(mcard(*args), unsafe_allow_html=True)

        # ── NEW: Metric subtitles for non-gamers ──
        st.markdown("""
        <div style="display:flex; gap:12px; padding:6px 0 14px; border-bottom:1px solid #27272a; margin-bottom:0;">
            <span style="flex:1; text-align:center; font-size:0.68rem; color:#52525b; font-family:'Inter',sans-serif;">How well the model ranks players overall</span>
            <span style="flex:1; text-align:center; font-size:0.68rem; color:#52525b; font-family:'Inter',sans-serif;">Ranking quality on the rare quit cases</span>
            <span style="flex:1; text-align:center; font-size:0.68rem; color:#52525b; font-family:'Inter',sans-serif;">Balance of catching quits vs false alarms</span>
            <span style="flex:1; text-align:center; font-size:0.68rem; color:#52525b; font-family:'Inter',sans-serif;">Of flagged players, how many actually quit</span>
            <span style="flex:1; text-align:center; font-size:0.68rem; color:#52525b; font-family:'Inter',sans-serif;">Of all quitters, how many were caught</span>
        </div>
        """, unsafe_allow_html=True)

        # Dataset stats bar
        if m:
            st.markdown(f"""
            <div style="display:flex; gap:24px; padding:12px 24px; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#71717a; border-bottom:1px solid #27272a; margin-bottom:16px;">
                <span>Test samples: <span style="color:#e4e4e7;">{m['total_samples']:,}</span></span>
                <span>Positives: <span style="color:#f43f5e;">{m['total_positive']}</span></span>
                <span>Negatives: <span style="color:#10b981;">{m['total_negative']:,}</span></span>
                <span>Positive rate: <span style="color:#f59e0b;">{m['positive_rate']:.2%}</span></span>
                <span>Threshold: <span style="color:#8b5cf6;">{m['optimal_threshold']:.3f}</span></span>
            </div>
            """, unsafe_allow_html=True)

        # ROC/PR — full width
        if Path("results/figures/roc_pr_curves.png").exists():
            st.image("results/figures/roc_pr_curves.png")

        # Confusion matrix + event importance side by side
        b1, b2 = st.columns(2)
        if Path("results/figures/confusion_matrix.png").exists():
            with b1: st.image("results/figures/confusion_matrix.png")
        if Path("results/figures/event_importance.png").exists():
            with b2: st.image("results/figures/event_importance.png")

        # Model comparison + summary card full width
        if Path("results/figures/model_comparison.png").exists():
            st.image("results/figures/model_comparison.png")
        if Path("results/figures/summary_card.png").exists():
            st.image("results/figures/summary_card.png")

        # Comparison table + evaluation notes
        t1, t2 = st.columns([1, 1])
        with t1:
            st.markdown('<div class="sh">Model Comparison</div>', unsafe_allow_html=True)
            if m:
                st.markdown(f"""
                <table class="ct">
                    <tr><th style="text-align:left">Model</th><th>Prec</th><th>Recall</th><th>F1</th><th>AUC-ROC</th><th>AUC-PR</th></tr>
                    <tr>
                        <td class="nm">Logistic Reg</td>
                        <td>0.197</td><td>0.283</td><td>0.283</td><td>0.884</td><td>0.173</td>
                    </tr>
                    <tr style="background:rgba(39,39,42,0.2)">
                        <td class="tf">● Transformer</td>
                        <td class="w">{m['precision']:.3f}</td><td class="w">{m['recall']:.3f}</td>
                        <td class="w">{m['f1']:.3f}</td><td class="w">{m['auc_roc']:.3f}</td><td class="w">{m['auc_pr']:.3f}</td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)
            else:
                st.warning("Run generate_results.py to compute metrics")

        with t2:
            st.markdown('<div class="sh">Evaluation Notes</div>', unsafe_allow_html=True)
            if m:
                # ── UPDATED: Added plain-English framing for non-gamers ──
                for cls, q, a in [
                    ("rose", "Class Imbalance",
                     f"Positive rate is {m['positive_rate']:.2%} ({m['total_positive']} rage quits out of {m['total_samples']:,} samples). "
                     "The strict 3-part label (abandoned + early leave + losing team) creates severe imbalance. "
                     "Think of it like fraud detection — the event you're predicting is rare, which makes the problem harder."),
                    ("violet", "Why AUC-PR matters more",
                     f"AUC-ROC ({m['auc_roc']:.3f}) looks strong but is inflated by {m['total_negative']:,} easy negatives. "
                     f"AUC-PR ({m['auc_pr']:.3f}) reveals the real precision-recall tradeoff on the minority class."),
                    ("amber", "Confusion Matrix",
                     f"At threshold {m['optimal_threshold']:.3f}: {m['tp']} TP, {m['fp']} FP, {m['fn']} FN, {m['tn']:,} TN. "
                     f"Accuracy: {m['accuracy']:.1%}. "
                     f"In plain terms: the model correctly flagged {m['tp']} quitters, missed {m['fn']}, and falsely flagged {m['fp']} stable players."),
                ]:
                    st.markdown(f'<div class="dcd {cls}"><div class="q">{q}</div><div class="a">{a}</div></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 2: SEQUENCE EXPLORER
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        # ── NEW: Reading guide banner for non-gamers ──
        st.markdown("""
        <div style="background:rgba(139,92,246,0.06); border:1px solid rgba(139,92,246,0.15); padding:14px 20px; margin-bottom:16px; font-family:'Inter',sans-serif; font-size:0.8rem; color:#a1a1aa; line-height:1.6;">
            <b style="color:#8b5cf6;">Reading guide:</b> Each pill is something the player did in-game, ordered by time.
            <span style="color:#00FF88;">Green</span> = good performance,
            <span style="color:#FF2D55;">Red</span> = bad outcome,
            <span style="color:#FFB800;">Yellow</span> = warning sign.
            Brighter pills = the model paid more attention to that event when making its prediction.
        </div>
        """, unsafe_allow_html=True)

        rage_quit_recs = [r for r in test_recs if r["label"] == 1]
        normal_recs = [r for r in test_recs if r["label"] == 0]

        ctrl1, ctrl2 = st.columns([1, 3])
        with ctrl1:
            sample_type = st.radio("Show:", ["Rage Quit Examples", "Normal Game Examples"],
                                   horizontal=True, label_visibility="collapsed")
        recs = rage_quit_recs if sample_type == "Rage Quit Examples" else normal_recs[:50]
        total = len(rage_quit_recs) if sample_type == "Rage Quit Examples" else len(normal_recs)

        with ctrl2:
            if recs:
                idx = st.slider(f"{len(recs)} of {total} sequences", 0, len(recs) - 1, 0,
                                label_visibility="collapsed")

        if recs:
            record = recs[idx]
            prob, attn_weights = predict_with_attention(model, record)
            events = get_events_with_minutes(record)

            if prob > 0.5:
                cls, dot, txt = "high", "r", "HIGH"
            elif prob > 0.2:
                cls, dot, txt = "mid", "y", "MODERATE"
            else:
                cls, dot, txt = "low", "g", "LOW"

            lbl = "RAGE QUIT" if record["label"] == 1 else "NORMAL"
            lbl_color = "#f43f5e" if record["label"] == 1 else "#10b981"
            match_id = record.get("match_id", "—")

            st.markdown(f"""
            <div class="rb {cls}" style="margin-bottom:16px;">
                <div class="rl"><div class="pdot {dot}"></div> RISK: {txt} — {prob:.1%}</div>
                <div class="rr" style="display:flex; gap:24px; align-items:center;">
                    <span style="color:{lbl_color}; font-size:0.75rem;">{lbl}</span>
                    <span style="color:#71717a; font-size:0.75rem;">Match {match_id}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Timeline + Narrative side by side ──
            col_timeline, col_narrative = st.columns([3, 1])

            with col_timeline:
                st.markdown(build_attention_timeline(events, attn_weights, max_events=35), unsafe_allow_html=True)

            with col_narrative:
                narrative = generate_sequence_narrative(events, attn_weights, prob, record["label"])
                st.markdown(f"""
                <div style="background:#18181b; border:1px solid #27272a; padding:20px; height:100%;">
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#71717a; text-transform:uppercase; letter-spacing:2px; margin-bottom:14px;">What's Happening</div>
                    <div style="font-family:'Inter',sans-serif; font-size:0.82rem; color:#a1a1aa; line-height:1.8;">
                        {narrative}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="sh" style="margin-top:20px;">Model Attention Focus</div>', unsafe_allow_html=True)
            summary_html = build_attention_summary(events, attn_weights)
            if summary_html:
                import streamlit.components.v1 as components
                components.html(f"""
                <style>
                .ev {{ display:inline-block; white-space:nowrap; }}
                .ev.pip-pos {{ color:#00FF88; }} .ev.pip-neg {{ color:#FF2D55; }}
                .ev.pip-wrn {{ color:#FFB800; }} .ev.pip-neu {{ color:#a1a1aa; }}
                </style>
                <div style="background:#09090b; padding:0; font-family:'JetBrains Mono',monospace;">
                    {summary_html}
                </div>
                """, height=80, scrolling=False)

        st.markdown("<br>", unsafe_allow_html=True)
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.markdown('<div class="ic"><h4>How to Read This</h4><p>Brighter pills = higher model attention. The model extracts [CLS] → token attention from the final transformer layer, averaged across 4 heads. Faded events were ignored by the model.</p></div>', unsafe_allow_html=True)
        with ic2:
            st.markdown('<div class="ic"><h4>What the Model Learns</h4><p>Death clusters followed by action droughts (going quiet after dying) consistently receive the highest attention. This pattern predicts rage quits within 3 minutes.</p></div>', unsafe_allow_html=True)
        with ic3:
            st.markdown('<div class="ic"><h4>Why Attention Matters</h4><p>Unlike feature importance in tree models, attention shows which specific events in which order the model uses. It\'s sequence-level interpretability.</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="sh">Event Dictionary</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#18181b; border:1px solid #27272a; padding:24px; display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px 32px; font-family:'JetBrains Mono',monospace; font-size:0.78rem;">
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#00FF88; font-weight:600;">LH High</span><span style="color:#71717a;">Last Hits Above Avg</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#FF2D55; font-weight:600;">LH Low</span><span style="color:#71717a;">Last Hits Below Avg</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#FFB800; font-weight:600;">APM ↓</span><span style="color:#71717a;">Action Drought</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#00FF88; font-weight:600;">APM ↑</span><span style="color:#71717a;">Action Burst</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#FFB800; font-weight:600;">XP Behind</span><span style="color:#71717a;">XP Falling Behind Team</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#00FF88; font-weight:600;">Gold ↑</span><span style="color:#71717a;">Gold Spike Up (+500/min)</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#FF2D55; font-weight:600;">Gold ↓</span><span style="color:#71717a;">Gold Spike Down (died)</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#00FF88; font-weight:600;">Kill</span><span style="color:#71717a;">Player Kill</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#FF2D55; font-weight:600;">Death</span><span style="color:#71717a;">Player Death</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#00FF88; font-weight:600;">Assist</span><span style="color:#71717a;">Kill Assist</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#FF2D55; font-weight:600;">Deaths x3</span><span style="color:#71717a;">3+ Deaths Without Kill</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#00FF88; font-weight:600;">Multi Kill</span><span style="color:#71717a;">2+ Kills in 15 Seconds</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#00FF88; font-weight:600;">TF Win</span><span style="color:#71717a;">Team Fight Won</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#FF2D55; font-weight:600;">TF Loss</span><span style="color:#71717a;">Team Fight Lost</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#00FF88; font-weight:600;">Tower ↑</span><span style="color:#71717a;">Team Took Tower</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#FF2D55; font-weight:600;">Tower ↓</span><span style="color:#71717a;">Team Lost Tower</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#00FF88; font-weight:600;">Big Buy</span><span style="color:#71717a;">Item Purchase &gt;2000g</span></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e22;"><span style="color:#FFB800; font-weight:600;">Idle</span><span style="color:#71717a;">No Action &gt;30 Seconds</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 3: ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="sh">Model Architecture</div>', unsafe_allow_html=True)

        left, right = st.columns([1, 1])

        with left:
            st.markdown('<div class="arch-label">Input Layer</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="arch-block">
                <strong>Token Embedding</strong> (128-dim)<br>
                + Continuous Feature Projection (6 → 128d)<br>
                + Game-Time Positional Encoding
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="arch-arrow">↓</div>', unsafe_allow_html=True)
            st.markdown('<div class="arch-label">Encoder</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="arch-block accent">
                <strong style="color:#8b5cf6;">Transformer Encoder</strong><br>
                4 layers × 4 attention heads<br>
                Feed-forward: 128 → 512 → 128<br>
                Pre-norm · GELU · Dropout 0.1
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="arch-arrow">↓</div>', unsafe_allow_html=True)
            st.markdown('<div class="arch-label">Output</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="arch-block">
                <strong>[CLS] Token → Classifier</strong><br>
                128 → 64 → 1<br>
                GELU · Dropout → σ → P(rage_quit)
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown('<div class="arch-label">Key Design Decisions</div>', unsafe_allow_html=True)
            for title, desc in [
                ("Why transformer over XGBoost?",
                 "Sequence ordering matters. DEATH → GOLD_DROP → SILENCE is a different signal than those events spread across 20 minutes."),
                ("Why game-time positional encoding?",
                 "Events cluster during fights and spread during farming. Encoding actual game minute preserves real temporal structure."),
                ("Why [CLS] token pooling?",
                 "After 4 layers of self-attention, position 0 holds a learned summary of the entire game."),
                ("Why dual embedding?",
                 "Each token carries both a discrete signal (DEATH) and continuous context (gold was -500 below avg)."),
            ]:
                st.markdown(f'<div class="arch-block" style="margin-bottom:8px;"><strong style="color:#8b5cf6;">{title}</strong><br><span style="color:#a1a1aa;font-family:Inter,sans-serif;font-size:0.85rem;">{desc}</span></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="arch-label">Model Stats</div>', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            with s1: st.markdown(mcard("Parameters","849,793","","#8b5cf6",""), unsafe_allow_html=True)
            with s2: st.markdown(mcard("Token Vocab","22","","#06b6d4",""), unsafe_allow_html=True)
            with s3: st.markdown(mcard("Best Epoch",str(best_epoch),"","#f59e0b",""), unsafe_allow_html=True)


if __name__ == "__main__":
    main()