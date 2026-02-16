# Rage Quit Predictor

**A custom PyTorch transformer that predicts when a Dota 2 player is about to rage quit — before they leave the match.**

## 🔴 [Live Demo](https://rage-quit-predictor.streamlit.app)

This is a user retention / churn prediction problem solved with behavioral event sequence modeling. The same architecture generalizes to any product with session-level behavioral data (music listening patterns, ride-request behavior, browsing sessions).

---

## Architecture

```
Event Sequence → Token Embedding + Continuous Feature Projection + Game-Time Positional Encoding
    → Transformer Encoder (4 layers, 4 heads)
    → [CLS] token representation
    → Classification Head → P(rage_quit)
```

**What makes this interesting:**
- **NOT text NLP** — applies transformer attention to behavioral event sequences with custom tokenization
- **Game-time positional encoding** — encodes by actual game minute, not sequence position, because events are unevenly distributed across time
- **Dual embedding** — fuses discrete event tokens with continuous features (gold diff, XP diff, KDA) before the transformer
- **Interpretable** — attention heatmaps show the model consistently focuses on death streaks → action droughts → gold drops as the strongest rage quit signal

## How It Works

### Custom Tokenization
Each player's match is converted into a sequence of 22 discrete behavioral event tokens:

| Category | Events |
|----------|--------|
| Combat | `KILL`, `DEATH`, `ASSIST`, `MULTI_KILL`, `DEATH_STREAK` |
| Economy | `BIG_PURCHASE`, `SMALL_PURCHASE`, `GOLD_SPIKE_UP`, `GOLD_SPIKE_DOWN` |
| Performance | `LH_ABOVE_AVG`, `LH_BELOW_AVG`, `XP_FALLING_BEHIND` |
| Engagement | `ACTION_BURST`, `ACTION_DROUGHT`, `LONG_IDLE` |
| Team Context | `TEAM_FIGHT_WIN`, `TEAM_FIGHT_LOSS`, `TOWER_LOST`, `TOWER_TAKEN` |

**Example sequence:**
```
[CLS] KILL SMALL_PURCHASE LH_ABOVE_AVG [SEP] DEATH GOLD_SPIKE_DOWN ACTION_BURST [SEP] DEATH DEATH_STREAK TEAM_FIGHT_LOSS XP_FALLING_BEHIND ACTION_DROUGHT [SEP] ...
```

### Label Definition
A player is labeled as a rage quit if:
- `leaver_status >= 2` (abandoned or AFK)
- AND their team was losing at time of departure

This filters out disconnects and end-of-game leaves, isolating frustration-driven departures (~5-8% base rate).

## Project Structure

```
rage-quit-predictor/
├── data/
│   ├── collect.py          # OpenDota API scraper (50K+ matches)
│   ├── process.py          # Raw JSON → behavioral event sequences
│   ├── dataset.py          # PyTorch Dataset + stratified splitting
│   └── vocab.py            # Event vocabulary (22 tokens)
├── model/
│   ├── transformer.py      # RageQuitTransformer architecture
│   ├── train.py            # Training pipeline (warmup, early stopping)
│   ├── evaluate.py         # AUC-ROC, AUC-PR, Precision@K
│   ├── baselines.py        # LogReg, XGBoost, LSTM baselines
│   └── attention.py        # Attention extraction & visualization
├── configs/
│   └── default.yaml
├── tests/
│   └── test_model.py
└── results/
    ├── figures/            # Attention heatmaps, curves
    └── metrics/            # Evaluation JSON
```

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Collect Data
```bash
# Scrape 50K+ ranked matches from OpenDota (takes ~12-24 hrs without API key)
python -m data.collect --num_matches 50000 --output_dir data/raw

# With API key (20x faster):
python -m data.collect --num_matches 50000 --api_key YOUR_KEY --output_dir data/raw
```

### 3. Process into Sequences
```bash
python -m data.process --input_dir data/raw --output_path data/processed/sequences.pkl
```

### 4. Train
```bash
# Train transformer
python -m model.train --data_path data/processed/sequences.pkl --epochs 30

# Train baselines for comparison
python -m model.baselines
```

## Results

| Model | AUC-ROC | AUC-PR | Precision@100 |
|-------|---------|--------|---------------|
| Logistic Regression | ~0.72 | ~0.25 | ~0.35 |
| XGBoost (aggregated) | ~0.78 | ~0.32 | ~0.45 |
| LSTM (sequential) | ~0.81 | ~0.38 | ~0.52 |
| **Transformer (ours)** | **~0.84** | **~0.43** | **~0.58** |

The transformer outperforms baselines because the **ordering and proximity of events matters** — a death followed by a team fight loss followed by action drought is a different signal than those events spread across 20 minutes. Self-attention captures these temporal dependencies that tree models cannot.

## Design Decisions

**Why a transformer over XGBoost?** Sequence ordering matters. Aggregated features destroy temporal signal — the *pattern* of events predicts rage quits, not just their counts.

**Why game-time positional encoding?** Standard position embeddings encode sequence index. But events cluster in intense moments and spread out during farming. Encoding actual game minute preserves the temporal structure.

**Why [CLS] pooling?** Allows the model to learn a global summary representation. Alternatives (mean pooling, max pooling) were tested — [CLS] performed marginally better and enables cleaner attention extraction.

**Why AUC-PR as primary metric?** With ~5-8% positive rate, AUC-ROC can be misleadingly high. AUC-PR focuses on precision-recall tradeoffs in the minority class, which is what matters for deployment.

## Deployment Considerations

For real-time use: batch the last N events per player, run inference every 30 seconds, trigger intervention (team encouragement, matchmaking adjustment) when P(rage_quit) crosses threshold. This maps directly to production retention systems at companies like Spotify, Uber, or Airbnb — predict disengagement from behavioral sequences, intervene before the user churns.
