"""Pre-push verification — run all tests before deploying."""
import json
import os
import pickle
from pathlib import Path

import torch

from data.dataset import DotaMatchDataset, split_by_match_id
from model.transformer import RageQuitTransformer

PASS = 0
FAIL = 0

def check(condition, msg_pass, msg_fail=None):
    global PASS, FAIL
    if condition:
        print(f"  ✅ {msg_pass}")
        PASS += 1
    else:
        print(f"  ❌ {msg_fail or msg_pass}")
        FAIL += 1

def section(title):
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print(f"{'═'*55}")

# ── 1. Data ──────────────────────────────────────────────────────────
section("1. DATA CONSISTENCY")

with open("data/processed/sequences.pkl", "rb") as f:
    records = pickle.load(f)

_, _, test1 = split_by_match_id(records)
_, _, test2 = split_by_match_id(records)
check(len(test1) == len(test2), f"Split is deterministic: {len(test1)} test records",
      "Split is non-deterministic!")

ids1 = set(r["match_id"] for r in test1)
ids2 = set(r["match_id"] for r in test2)
check(ids1 == ids2, "Same match IDs across splits", "Different match IDs across splits!")

train1, _, _ = split_by_match_id(records)
train_ids = set(r["match_id"] for r in train1)
test_ids = set(r["match_id"] for r in test1)
overlap = train_ids & test_ids
check(len(overlap) == 0, "No data leakage between train and test",
      f"DATA LEAKAGE: {len(overlap)} match IDs shared!")

pos = sum(r["label"] for r in test1)
check(pos > 100, f"{pos} positives in test set (statistically meaningful)",
      f"Only {pos} positives — too few!")

# ── 2. Metrics ───────────────────────────────────────────────────────
section("2. METRICS CONSISTENCY")

with open("results/metrics/test_metrics.json") as f:
    tm = json.load(f)
with open("results/metrics/baseline_results.json") as f:
    bl = json.load(f)
with open("results/metrics/lstm_results.json") as f:
    lstm = json.load(f)

check(bl["logistic_regression"]["num_samples"] == tm["total_samples"],
      f"All models on same test set ({tm['total_samples']} samples)",
      "Test set size mismatch between models!")
check(bl["logistic_regression"]["num_positive"] == tm["total_positive"],
      f"All models see same {tm['total_positive']} positives",
      "Positive count mismatch between models!")

check(tm["auc_pr"] > bl["logistic_regression"]["auc_pr"],
      f"Transformer AUC-PR ({tm['auc_pr']:.4f}) > LR ({bl['logistic_regression']['auc_pr']:.4f})",
      "Transformer loses to LR on AUC-PR!")
check(tm["auc_pr"] > bl["xgboost"]["auc_pr"],
      f"Transformer AUC-PR ({tm['auc_pr']:.4f}) > XGBoost ({bl['xgboost']['auc_pr']:.4f})",
      "Transformer loses to XGBoost on AUC-PR!")
check(tm["auc_pr"] > lstm["auc_pr"],
      f"Transformer AUC-PR ({tm['auc_pr']:.4f}) > LSTM ({lstm['auc_pr']:.4f})",
      "Transformer loses to LSTM on AUC-PR!")

cm_total = tm["tp"] + tm["fp"] + tm["fn"] + tm["tn"]
check(cm_total == tm["total_samples"],
      "Confusion matrix adds up to total samples",
      f"Confusion matrix sums to {cm_total}, not {tm['total_samples']}!")
check(tm["tp"] + tm["fn"] == tm["total_positive"],
      "TP + FN == total positives",
      "TP + FN mismatch!")

check(0 < tm["auc_pr"] < 1, f"AUC-PR in valid range: {tm['auc_pr']:.4f}")
check(0 < tm["auc_roc"] < 1, f"AUC-ROC in valid range: {tm['auc_roc']:.4f}")
check(tm["positive_rate"] < 0.05, f"Positive rate is low as expected: {tm['positive_rate']:.2%}")

# ── 3. Model ─────────────────────────────────────────────────────────
section("3. MODEL INFERENCE")

cp = torch.load("results/weights/best_model.pt", map_location="cpu", weights_only=False)
model = RageQuitTransformer(**cp["config"])
model.load_state_dict(cp["model_state_dict"])
model.eval()
check(True, f"Model loaded from epoch {cp['epoch']} (val AUC-PR: {cp['val_auc_pr']:.4f})")

ds = DotaMatchDataset(test1[:20], max_seq_len=256)
probs = []
for i in range(20):
    batch = ds[i]
    single = {k: v.unsqueeze(0) for k, v in batch.items()}
    with torch.no_grad():
        prob = torch.sigmoid(
            model(single["event_ids"], single["continuous_features"],
                  single["minutes"], single["attention_mask"]).squeeze()
        ).item()
    probs.append(prob)

check(all(0.0 <= p <= 1.0 for p in probs), "All 20 predictions are valid probabilities")
check(len(set(round(p, 4) for p in probs)) > 1,
      f"Model outputs vary across inputs: min={min(probs):.3f}, max={max(probs):.3f}",
      "Model outputs identical probabilities for all inputs!")

# Check model predicts higher prob for rage quits than non-rage quits on average
rq_probs = [probs[i] for i in range(20) if test1[i]["label"] == 1]
nq_probs = [probs[i] for i in range(20) if test1[i]["label"] == 0]
if rq_probs and nq_probs:
    check(sum(rq_probs)/len(rq_probs) > sum(nq_probs)/len(nq_probs),
          f"Rage quit avg prob ({sum(rq_probs)/len(rq_probs):.3f}) > normal avg prob ({sum(nq_probs)/len(nq_probs):.3f})",
          "Model gives higher prob to normal games than rage quits!")

# ── 4. Figures ───────────────────────────────────────────────────────
section("4. FIGURES")

figures = ["roc_pr_curves.png", "confusion_matrix.png", "model_comparison.png",
           "event_importance.png", "summary_card.png"]

for fig in figures:
    path = Path(f"results/figures/{fig}")
    exists = path.exists()
    size_ok = path.stat().st_size > 10000 if exists else False
    check(exists and size_ok, f"{fig} exists and non-empty ({path.stat().st_size//1024}KB)" if exists else fig,
          f"{fig} missing or too small!")

metrics_time = os.path.getmtime("results/metrics/test_metrics.json")
stale = [f for f in figures if os.path.getmtime(f"results/figures/{f}") < metrics_time]
check(len(stale) == 0, "All figures newer than test_metrics.json",
      f"Stale figures (regenerate): {stale}")

# ── 5. app.py ────────────────────────────────────────────────────────
section("5. APP.PY INTEGRITY")

content = open("app.py").read()

old_numbers = ["0.173", "0.197", "0.283", "0.1915", "0.3064",
               "total_positive\": 28", "total_samples\": 4520",
               "test_sequences.pkl"]
for num in old_numbers:
    check(num not in content, f"No hardcoded old value: '{num}'",
          f"Found hardcoded old value in app.py: '{num}'")

check("sequences.pkl" in content, "app.py loads sequences.pkl")
check("split_by_match_id" in content, "app.py uses split_by_match_id")
check("load_baseline_metrics" in content, "load_baseline_metrics function present")
check("load_lstm_metrics" in content, "load_lstm_metrics function present")
check("bl and lstm" in content, "Table condition checks bl and lstm before rendering")

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'═'*55}")
if FAIL == 0:
    print("  🚀 All tests passed — safe to push!")
else:
    print("  ⚠  Fix failures before pushing.")
print()