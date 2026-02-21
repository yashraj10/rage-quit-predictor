# Rage Quit Predictor

**A custom PyTorch transformer that predicts when a Dota 2 player is about to rage quit — before they leave the match.**

## 🔴 [Live Demo](https://rage-quit-predictor.streamlit.app)

This is a user retention / churn prediction problem solved with behavioral event sequence modeling. The same architecture generalizes to any product with session-level behavioral data (music listening patterns, ride-request behavior, browsing sessions).

---

## Key Results

| Model | AUC-PR ★ | AUC-ROC | F1 | Precision | Recall |
|-------|----------|---------|------|-----------|--------|
| Logistic Regression | 0.173 | 0.884 | 0.283 | 0.197 | 0.283 |
| **Transformer (ours)** | **0.192** | **0.882** | **0.356** | **0.289** | **0.464** |

**★ AUC-PR is the primary metric.** With a 0.62% positive rate (28 rage quits in 4,520 test samples), AUC-ROC is inflated by easy negatives. AUC-PR reveals the real precision-recall tradeoff on the minority class — similar to fraud detection or rare disease screening.

**Why the transformer wins:** Not on aggregate metrics (the 0.019 AUC-PR lift isn't statistically significant on 28 positives), but on **interpretability**. Attention heatmaps show *which events in which order* predict rage quits — something logistic regression on aggregated features fundamentally cannot do.

---

## Architecture

```
Event Sequence → Token Embedding + Continuous Feature Projection + Game-Time Positional Encoding
    → Transformer Encoder (4 layers, 4 heads, 128-dim)
    → [CLS] token representation
    → Classification Head (128 → 64 → 1) → P(rage_quit)
```

**849,793 parameters · 22 event tokens · 6 continuous features per token**

**What makes this interesting:**
- **NOT text NLP** — applies transformer attention to behavioral event sequences with custom tokenization
- **Game-time positional encoding** — encodes by actual game minute, not sequence position, because events are unevenly distributed across time
- **Dual embedding** — fuses discrete event tokens with continuous features (gold diff, XP diff, KDA) before the transformer via concatenation + projection
- **Interpretable** — attention analysis reveals APM drops (action drought) and XP deficit signals as the strongest rage quit predictors

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
| Meta | `[PAD]`, `[CLS]`, `[SEP]` |

Each token also carries 6 continuous features: gold diff, XP diff, KDA ratio, team gold diff, net worth rank, and game minute.

**Example sequence:**
```
[CLS] KILL SMALL_PURCHASE LH_ABOVE_AVG [SEP] DEATH GOLD_SPIKE_DOWN ACTION_BURST [SEP] DEATH DEATH_STREAK TEAM_FIGHT_LOSS XP_FALLING_BEHIND ACTION_DROUGHT [SEP] ...
```

### Label Definition
A player is labeled as a rage quit if:
- `leaver_status >= 2` (abandoned or AFK)
- AND their team was losing at time of departure

This strict 3-part filter isolates frustration-driven departures but creates severe class imbalance: **0.62% positive rate** (vs the ~5% you'd get with just `leaver_status >= 2`). This is deliberately aggressive — cleaner labels at the cost of fewer examples. The tradeoff is similar to fraud detection: the event you're predicting is rare, which makes evaluation fragile.

### What the Model Learns
Attention analysis across correctly predicted rage quits reveals a consistent pattern:
- **APM drops** (ACTION_DROUGHT) receive the highest attention — the player going quiet
- **XP deficit** (XP_FALLING_BEHIND) is the second strongest signal — falling behind the team
- The combination of declining performance + disengagement is the strongest predictor

This matches game design intuition: players who are losing AND stop trying are the most likely to abandon.

## Streamlit Demo

The live app includes three views:

### Performance Metrics
- AUC-PR leads as primary metric with ★ badge
- ROC and Precision-Recall curves, confusion matrix, event importance chart
- Model comparison (Transformer vs Logistic Regression)
- Evaluation notes explaining class imbalance, metric choices, and probability calibration

### Sequence Explorer
- Interactive attention-weighted timeline showing what the model focuses on
- Three-tier visualization: solid glow (high attention), tinted fill (medium), outline (low)
- Color coding: green (positive events), red (negative), yellow (warning signals)
- "What's Happening" narrative panel explaining each prediction in plain English
- Rage quit examples sorted by model confidence — true positives first
- Reading guide banner for non-technical viewers

### Model Architecture
- Visual architecture diagram with design decision explanations
- Model stats: 849K parameters, 22 tokens, training details

## Project Structure

```
rage-quit-predictor/
├── app.py                  # Streamlit demo (all 3 views)
├── data/
│   ├── collect.py          # OpenDota API scraper (50K+ matches)
│   ├── process.py          # Raw JSON → behavioral event sequences
│   ├── dataset.py          # PyTorch Dataset + stratified splitting
│   └── vocab.py            # Event vocabulary (22 tokens)
├── model/
│   ├── transformer.py      # RageQuitTransformer (849K params)
│   ├── train.py            # Training pipeline (warmup, early stopping)
│   ├── evaluate.py         # AUC-ROC, AUC-PR, F1, confusion matrix
│   ├── baselines.py        # Logistic Regression baseline
│   └── attention.py        # Attention extraction & visualization
├── generate_results.py     # Compute all metrics + figures from test set
├── configs/
│   └── default.yaml
└── results/
    ├── figures/            # ROC/PR curves, confusion matrix, event importance
    ├── metrics/            # test_metrics.json (single source of truth)
    └── weights/            # best_model.pt
```

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Collect Data
```bash
# Scrape 50K+ ranked matches from OpenDota
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
python -m model.train --data_path data/processed/sequences.pkl --epochs 30
```

### 5. Generate Results
```bash
python generate_results.py  # computes all metrics + figures from test set
```

### 6. Run Demo
```bash
streamlit run app.py
```

## Design Decisions

**Why a transformer over XGBoost?** Sequence ordering matters. Aggregated features destroy temporal signal — the *pattern* of events predicts rage quits, not just their counts. A death → gold drop → going silent is a different signal than those events spread across 20 minutes.

**Why game-time positional encoding?** Standard position embeddings encode sequence index. Events cluster during fights and spread during farming. Encoding actual game minute preserves real temporal structure.

**Why [CLS] pooling?** Allows the model to learn a global summary representation and enables clean attention extraction for interpretability.

**Why AUC-PR as primary metric?** With 0.62% positive rate, AUC-ROC is inflated by 4,492 easy negatives. AUC-PR focuses on the minority class, which is what matters for deployment decisions.

**Why the strict label filter?** The 3-part filter (abandoned + early leave + losing team) prioritizes label quality over quantity. The tradeoff: only 28 test positives, making evaluation fragile. With more data collection, this would be the first thing to revisit — either relaxing the losing-team requirement or making it a soft feature.

## Known Limitations & Next Steps

**Probability calibration:** `pos_weight ≈ 160` compresses probabilities into [0.999, 1.0], causing the F1-optimal threshold to land at 0.999990. The model *ranks* correctly but probabilities need post-hoc calibration (Platt scaling or temperature scaling).

**Small positive test set:** 28 positives means metrics have wide confidence intervals (~±0.05 on AUC-PR). Bootstrap CIs would improve evaluation credibility. K-fold stratified CV on the full dataset would give more robust estimates.

**Truncation direction:** Currently truncates late-game events when sequences exceed 256 tokens. For rage quit prediction, the most recent events matter most — truncating from the start (keeping the last 256 events) would likely improve recall.

**FFN dimension:** Used the standard 4x BERT ratio (128 → 512) without tuning. With only 22 tokens, FFN dim 256 or 128 would cut parameters by ~30-40% and likely reduce overfitting.

**Confound with losing:** XP_FALLING_BEHIND and ACTION_DROUGHT are correlated with losing, and the label requires the losing team. Evaluating on the losing-team-only subset would prove the model learns disengagement beyond team outcome.

## Deployment Considerations

For real-time use: batch the last N events per player, run inference every 30 seconds, trigger intervention when P(rage_quit) crosses threshold. With F1 of 0.356 (29% precision), this is suitable for **soft interventions** — team encouragement messages, matchmaking priority adjustments — where the cost of a false positive is near zero. Hard interventions would require F1 > 0.50 with precision > 0.40, achievable with more data and calibration.

This maps directly to production retention systems at companies like Spotify, Uber, or Airbnb — predict disengagement from behavioral sequences, intervene before the user churns.