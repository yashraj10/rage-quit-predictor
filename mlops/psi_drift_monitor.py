"""
PSI (Population Stability Index) Drift Monitor.

Detects distributional shift between training-time feature distributions
and current production distributions. PSI > 0.2 triggers a pipeline alert.

PSI Formula: Σ (Actual% - Expected%) × ln(Actual% / Expected%)

PSI Interpretation:
    < 0.1  : No significant shift    → PASS
    0.1-0.2: Moderate shift          → PASS with warning
    > 0.2  : Significant shift       → FAIL, block model promotion
"""

import json
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES_MONITORED = [
    "avg_gold_diff",
    "death_count",
    "event_count",
    "xp_deficit_rate",
    "segment_rank",
]

PSI_THRESHOLD_WARN = 0.1
PSI_THRESHOLD_FAIL = 0.2
NUM_BINS = 10


def compute_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = NUM_BINS) -> float:
    """
    Compute PSI between expected (training) and actual (production) distributions.

    Args:
        expected: Training distribution values
        actual:   Current production distribution values
        num_bins: Number of bins for discretization

    Returns:
        PSI score
    """
    # Create bins from expected distribution
    bins = np.percentile(expected, np.linspace(0, 100, num_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Compute bin proportions
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    # Convert to proportions, avoid division by zero
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Clip to avoid log(0)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def generate_baseline_distributions(n: int = 10000, seed: int = 42) -> dict:
    """Generate baseline (training-time) feature distributions."""
    rng = np.random.RandomState(seed)
    return {
        "avg_gold_diff": rng.normal(0, 800, n),
        "death_count": rng.poisson(3, n).astype(float),
        "event_count": rng.poisson(25, n).astype(float),
        "xp_deficit_rate": rng.beta(2, 8, n),
        "segment_rank": rng.randint(1, 11, n).astype(float),
    }


def generate_current_distributions(n: int = 2000, drift_level: str = "high", seed: int = 99) -> dict:
    """
    Generate current production distributions with configurable drift.

    drift_level: 'none', 'moderate', 'high'
    """
    rng = np.random.RandomState(seed)

    drift_multipliers = {
        "none":     {"gold": (0, 800),   "death": 3,  "event": 25, "xp": (2, 8),   "rank": (1, 11)},
        "moderate": {"gold": (200, 900), "death": 4,  "event": 22, "xp": (3, 7),   "rank": (1, 11)},
        "high":     {"gold": (800, 1200),"death": 6,  "event": 18, "xp": (5, 5),   "rank": (3, 11)},
    }

    d = drift_multipliers[drift_level]
    return {
        "avg_gold_diff":  rng.normal(*d["gold"], n),
        "death_count":    rng.poisson(d["death"], n).astype(float),
        "event_count":    rng.poisson(d["event"], n).astype(float),
        "xp_deficit_rate":rng.beta(*d["xp"], n),
        "segment_rank":   rng.randint(*d["rank"], n).astype(float),
    }


def run_psi_check(drift_level: str = "high") -> dict:
    """Run PSI check across all monitored features."""
    baseline = generate_baseline_distributions()
    current = generate_current_distributions(drift_level=drift_level)

    results = {}
    overall_pass = True

    print(f"\n{'='*55}")
    print(f" PSI Drift Monitor — drift_level='{drift_level}'")
    print(f"{'='*55}")
    print(f" {'Feature':<22} {'PSI':>6}  {'Status'}")
    print(f" {'-'*22} {'-'*6}  {'-'*20}")

    for feature in FEATURES_MONITORED:
        psi = compute_psi(baseline[feature], current[feature])

        if psi < PSI_THRESHOLD_WARN:
            status = "✅ PASS"
        elif psi < PSI_THRESHOLD_FAIL:
            status = "⚠️  WARN"
        else:
            status = "❌ FAIL"
            overall_pass = False

        print(f" {feature:<22} {psi:>6.4f}  {status}")
        results[feature] = {"psi": round(psi, 4), "status": status}

    print(f"{'='*55}")
    print(f" Overall: {'✅ PASS — model promotion allowed' if overall_pass else '❌ FAIL — block promotion, alert triggered'}")
    print(f"{'='*55}\n")

    results["overall_pass"] = overall_pass
    results["drift_level"] = drift_level
    return results


if __name__ == "__main__":
    # Run with high drift to demonstrate detection
    results = run_psi_check(drift_level="high")

    # Save results
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    with open("results/metrics/psi_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ PSI results saved to results/metrics/psi_results.json")