#!/usr/bin/env python3
"""Cohort/data-quality audit and per-feature statistics for the Feature Statistics page.

Companion to run_analysis.py, which this script does not modify or duplicate the
pipeline of. It reuses the already-validated top-50 stability-ranked feature list
per endpoint (from run_analysis.py's feature_rankings_*.csv outputs) rather than
recomputing feature selection, and adds descriptive, group-comparison, and
univariate-discrimination statistics that the main pipeline does not produce.

Depends on numpy/pandas only for classification statistics (no scipy/scikit-learn
needed there), since those aren't guaranteed to be installed everywhere this script
runs. Univariate ROC AUC is computed from rank sums (the Mann-Whitney U identity),
which is algebraically the same value sklearn.metrics.roc_auc_score would return
for the binary/one-vs-rest case. The survival endpoint's per-feature hazard-ratio
confidence interval does need `lifelines` (the same dependency run_analysis.py now
uses for its Cox model) — a real Cox partial-likelihood fit, not something to
hand-roll — but its concordance index is computed via lifelines' closed-form
utility function only, no model fitting, so that part stays fast.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

SEED = 20260807
N_BOOTSTRAP = 1000
HIST_BINS = 20
PERCENTILE_STEPS = list(range(0, 101, 5))  # 21 points for the calculator curve


# ---------------------------------------------------------------------------
# Shared helpers copied from run_analysis.py (that script is left untouched;
# these are small, pure functions duplicated here to keep this script standalone).
# ---------------------------------------------------------------------------


def read_duke_sheet(path: Path, sheet: str) -> pd.DataFrame:
    """Read the Duke workbook's fifth-row source headers without changing it."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    headers: list[str] = []
    seen: Counter[str] = Counter()
    for i, value in enumerate(raw.iloc[4].tolist()):
        header = str(value).strip() if pd.notna(value) else f"unnamed_{i}"
        if seen[header]:
            header = f"{header}_{seen[header] + 1}"
        seen[str(value).strip() if pd.notna(value) else f"unnamed_{i}"] += 1
        headers.append(header)
    frame = raw.iloc[5:].copy()
    frame.columns = headers
    frame = frame[frame["Patient ID"].notna()].copy()
    frame["Patient ID"] = frame["Patient ID"].astype(str).str.strip()
    return frame


def parse_feature(name: str) -> dict[str, str]:
    region = "Vessel" if name.endswith("_vessel") else "Tumor"
    core = name[:-7] if region == "Vessel" else name
    parts = core.split("_")
    transform = parts[0]
    family = parts[1] if len(parts) > 1 else "other"
    measure = "_".join(parts[2:]) if len(parts) > 2 else core
    transform_label = (
        "Original"
        if transform == "original"
        else transform.replace("log-sigma-", "LoG ").replace("-mm-3d", " mm").replace("wavelet-", "Wavelet ").upper()
    )
    family_label = {
        "firstorder": "First order",
        "glcm": "GLCM",
        "glrlm": "GLRLM",
        "glszm": "GLSZM",
        "gldm": "GLDM",
    }.get(family, family.upper())
    measure_label = re.sub(r"(?<!^)(?=[A-Z])", " ", measure).replace("_", " ")
    measure_label = measure_label.replace("graylevel", "gray-level").replace("nonuniformity", "non-uniformity")
    measure_label = measure_label.replace("joint", "joint ").replace("dependence", "dependence ")
    measure_label = " ".join(measure_label.split()).title()
    return {
        "region": region,
        "transform": transform_label,
        "family": family_label,
        "measure": measure_label,
        "display": f"{transform_label} · {family_label} · {measure_label} · {region}",
    }


# ---------------------------------------------------------------------------
# New statistics helpers (numpy/pandas only)
# ---------------------------------------------------------------------------


def tukey_bounds(x: np.ndarray) -> tuple[float, float]:
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    return float(q25 - 1.5 * iqr), float(q75 + 1.5 * iqr)


def outlier_rate(x: np.ndarray) -> float:
    lo, hi = tukey_bounds(x)
    if lo == hi:
        return 0.0
    return float(np.mean((x < lo) | (x > hi)))


def is_near_constant(x: np.ndarray) -> bool:
    q25, q75 = np.percentile(x, [25, 75])
    if q75 - q25 == 0:
        return True
    _, counts = np.unique(x, return_counts=True)
    return bool(counts.max() / len(x) >= 0.99)


def quality_flag(missing_rate: float, near_constant: bool, outlier: float) -> str:
    if missing_rate > 0.05 or near_constant or outlier > 0.15:
        return "poor"
    if missing_rate > 0 or outlier > 0.05:
        return "caution"
    return "good"


def describe_distribution(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    q5, q25, median, q75, q95 = np.percentile(x, [5, 25, 50, 75, 95])
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    iqr = float(q75 - q25)
    m2 = float(np.mean((x - mean) ** 2))
    m3 = float(np.mean((x - mean) ** 3))
    skew = float(m3 / (m2**1.5)) if m2 > 0 else 0.0
    lo, hi = tukey_bounds(x)
    extreme_mask = (x < lo) | (x > hi)
    return {
        "n": int(len(x)),
        "min": float(np.min(x)),
        "p5": float(q5),
        "p25": float(q25),
        "median": float(median),
        "p75": float(q75),
        "p95": float(q95),
        "max": float(np.max(x)),
        "mean": mean,
        "sd": sd,
        "iqr": iqr,
        "skewness": skew,
        "extremeCount": int(extreme_mask.sum()),
        "extremePct": round(float(extreme_mask.mean() * 100), 2),
    }


def percentile_curve(x: np.ndarray) -> list[dict]:
    values = np.percentile(x, PERCENTILE_STEPS)
    return [{"percentile": p, "value": round(float(v), 6)} for p, v in zip(PERCENTILE_STEPS, values)]


def shared_histogram(x: np.ndarray, bins: int = HIST_BINS) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(x, bins=bins)
    return counts, edges


def histogram_on_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(x, bins=edges)
    return counts


def average_rank(x: np.ndarray) -> np.ndarray:
    """Rank, ascending. Radiomic feature values are continuous floats extracted
    from images, so exact ties are not expected; plain (untied) ranks are used
    for speed. `np.unique`-based tie correction was tried and measured far too
    slow across the ~350k bootstrap evaluations this script needs."""
    order = np.argsort(x, kind="quicksort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks


def rank_auc(x: np.ndarray, positive: np.ndarray) -> float:
    n_pos = int(positive.sum())
    n_neg = len(x) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_rank(x)
    r_pos = ranks[positive].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def batch_ranks(xb: np.ndarray) -> np.ndarray:
    """Ranks for each row of a (n_boot, n) matrix, vectorized across rows."""
    n_boot, n = xb.shape
    order = np.argsort(xb, axis=1, kind="quicksort")
    ranks = np.empty((n_boot, n), dtype=float)
    row_idx = np.arange(n_boot)[:, None]
    ranks[row_idx, order] = np.arange(1, n + 1, dtype=float)
    return ranks


def batch_auc_from_ranks(ranks: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """AUC per row, given precomputed ranks and a (n_boot, n) positive-class mask."""
    n = ranks.shape[1]
    n_pos = positive.sum(axis=1)
    n_neg = n - n_pos
    r_pos = np.sum(ranks * positive, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def batch_cohens_d(xb: np.ndarray, positive: np.ndarray) -> np.ndarray:
    negative = ~positive
    n1 = positive.sum(axis=1)
    n0 = negative.sum(axis=1)
    mean1 = np.sum(np.where(positive, xb, 0.0), axis=1) / n1
    mean0 = np.sum(np.where(negative, xb, 0.0), axis=1) / n0
    var1 = np.sum(np.where(positive, (xb - mean1[:, None]) ** 2, 0.0), axis=1) / np.clip(n1 - 1, 1, None)
    var0 = np.sum(np.where(negative, (xb - mean0[:, None]) ** 2, 0.0), axis=1) / np.clip(n0 - 1, 1, None)
    pooled = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / np.clip(n0 + n1 - 2, 1, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (mean1 - mean0) / pooled
    correction = 1 - 3 / np.clip(4 * (n0 + n1) - 9, 1, None)
    return d * correction


def batch_eta_squared(xb: np.ndarray, labelsb: np.ndarray, classes: list) -> np.ndarray:
    n_boot = xb.shape[0]
    overall = xb.mean(axis=1)
    ss_total = np.sum((xb - overall[:, None]) ** 2, axis=1)
    ss_between = np.zeros(n_boot)
    for cls in classes:
        mask = labelsb == cls
        count = mask.sum(axis=1)
        group_sum = np.sum(np.where(mask, xb, 0.0), axis=1)
        group_mean = np.divide(group_sum, count, out=np.zeros(n_boot), where=count > 0)
        ss_between += count * (group_mean - overall) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.divide(ss_between, ss_total, out=np.zeros(n_boot), where=ss_total > 0)


def youden_threshold(x: np.ndarray, positive: np.ndarray, higher_is_positive: bool) -> dict:
    """Sensitivity/specificity/PPV/NPV at the Youden's-J-optimal threshold.

    `higher_is_positive` orients the rule so the reported threshold matches the
    feature's already-established direction of association, rather than an
    arbitrary sign.
    """
    score = x if higher_is_positive else -x
    n_pos = int(positive.sum())
    n_neg = len(x) - n_pos
    best = None
    for t in np.unique(score):
        pred_pos = score >= t
        tp = int(np.sum(pred_pos & positive))
        fp = int(np.sum(pred_pos & ~positive))
        fn = int(np.sum(~pred_pos & positive))
        tn = int(np.sum(~pred_pos & ~positive))
        sens = tp / n_pos if n_pos else 0.0
        spec = tn / n_neg if n_neg else 0.0
        j = sens + spec - 1
        if best is None or j > best["j"]:
            best = {"j": j, "t": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "sensitivity": sens, "specificity": spec}
    threshold = float(best["t"]) if higher_is_positive else float(-best["t"])
    ppv = best["tp"] / (best["tp"] + best["fp"]) if (best["tp"] + best["fp"]) else float("nan")
    npv = best["tn"] / (best["tn"] + best["fn"]) if (best["tn"] + best["fn"]) else float("nan")
    rule = f"x ≥ {threshold:.4g} predicts the positive class" if higher_is_positive else f"x ≤ {threshold:.4g} predicts the positive class"
    return {
        "threshold": threshold,
        "thresholdRule": rule,
        "sensitivity": float(best["sensitivity"]),
        "specificity": float(best["specificity"]),
        "ppv": float(ppv),
        "npv": float(npv),
    }


def make_bootstrap_indices(labels: np.ndarray, classes: list, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified bootstrap resample indices, shape (n_boot, n) — one row per draw."""
    strata = [np.flatnonzero(labels == c) for c in classes]
    n = len(labels)
    idx_matrix = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        sample = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in strata])
        rng.shuffle(sample)
        idx_matrix[b] = sample
    return idx_matrix


def ci_from_draws(draws: np.ndarray) -> tuple[float, float]:
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def survival_concordance(durations: np.ndarray, risk: np.ndarray, events: np.ndarray) -> float:
    """Same sign-convention wrapper as run_analysis.py: lifelines' concordance_index
    expects scores that behave like survival time (higher = longer survival), the
    opposite of a risk score, so the risk score is negated here."""
    try:
        return float(concordance_index(durations, -risk, events))
    except ZeroDivisionError:
        return float("nan")


# ---------------------------------------------------------------------------
# Data loading (mirrors run_analysis.py's merge logic; that script is untouched)
# ---------------------------------------------------------------------------


def load_cohort(radiomics_path: Path, clinical_path: Path):
    radiomics = pd.read_csv(radiomics_path, low_memory=False)
    demography = read_duke_sheet(clinical_path, "Demography_tumortypes")
    clinical = read_duke_sheet(clinical_path, "Cancer_specific_clinical_data")
    drugs = read_duke_sheet(clinical_path, "drugs")

    clinical_feature_names = ["menopause", "er", "pr", "her2"]
    id_columns = ["Unnamed: 0", "patient"]
    feature_names = [c for c in radiomics.columns if c not in id_columns + clinical_feature_names]
    x_all = radiomics[feature_names].to_numpy(dtype=float)

    merged = radiomics[["patient"]].merge(
        demography[["Patient ID", "Mol Subtype", "OS_STATUS", "OS_MONTHS"]],
        left_on="patient",
        right_on="Patient ID",
        how="left",
        validate="one_to_one",
    )
    merged["OS_STATUS"] = pd.to_numeric(merged["OS_STATUS"], errors="coerce")
    merged["OS_MONTHS"] = pd.to_numeric(merged["OS_MONTHS"], errors="coerce")
    merged = merged.merge(
        drugs[["Patient ID", "Overall Near-complete Response:  Stricter Definition"]],
        left_on="patient",
        right_on="Patient ID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_drug"),
    )
    merged["strict_response"] = (
        merged["Overall Near-complete Response:  Stricter Definition"].astype("string").str.strip().str.lower()
    )
    return radiomics, feature_names, x_all, merged


SUBTYPE_LABELS = {"BRCA_Basal": "Basal", "BRCA_Her2": "HER2-enriched", "BRCA_LumA": "Luminal A", "BRCA_LumB": "Luminal B"}
KNOWN_RESPONSE = ["complete", "near-complete", "not-complete-or-near-complete"]


# ---------------------------------------------------------------------------
# Section 1: cohort & data-quality summary (all features)
# ---------------------------------------------------------------------------


def build_data_quality(feature_names: list[str], x_all: np.ndarray) -> dict:
    rows = []
    for idx, name in enumerate(feature_names):
        col = x_all[:, idx]
        finite = np.isfinite(col)
        missing_rate = float(1 - finite.mean())
        clean = col[finite]
        near_const = is_near_constant(clean) if len(clean) > 1 else True
        out_rate = outlier_rate(clean) if len(clean) > 1 else 0.0
        flag = quality_flag(missing_rate, near_const, out_rate)
        parsed = parse_feature(name)
        rows.append(
            {
                "feature": name,
                "display": parsed["display"],
                "region": parsed["region"],
                "family": parsed["family"],
                "transform": parsed["transform"],
                "missingRate": round(missing_rate, 4),
                "outlierRate": round(out_rate, 4),
                "nearConstant": near_const,
                "quality": flag,
            }
        )

    def rollup(items: list[dict]) -> dict:
        counts = Counter(r["quality"] for r in items)
        return {"good": counts.get("good", 0), "caution": counts.get("caution", 0), "poor": counts.get("poor", 0)}

    by_region: dict[str, dict] = {}
    by_family: dict[str, dict] = {}
    for region in sorted({r["region"] for r in rows}):
        by_region[region] = rollup([r for r in rows if r["region"] == region])
    for family in sorted({r["family"] for r in rows}):
        by_family[family] = rollup([r for r in rows if r["family"] == family])

    return {
        "rollup": {"overall": rollup(rows), "byRegion": by_region, "byFamily": by_family},
        "features": rows,
    }


# ---------------------------------------------------------------------------
# Sections 2, 4, 5: per-feature deep dive for the top-50 ranked features
# ---------------------------------------------------------------------------


def build_endpoint_features(
    ranking_csv: Path,
    feature_names: list[str],
    x_all: np.ndarray,
    endpoint_mask: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    binary: bool,
    rng: np.random.Generator,
) -> list[dict]:
    ranking = pd.read_csv(ranking_csv).sort_values("rank").head(50)
    feature_index = {name: i for i, name in enumerate(feature_names)}
    x_endpoint = x_all[endpoint_mask]

    boot_idx = make_bootstrap_indices(labels, classes, N_BOOTSTRAP, rng)  # (n_boot, n)
    labels_boot = labels[boot_idx]  # (n_boot, n), reused across every feature

    out = []
    for row_number, (_, row) in enumerate(ranking.iterrows(), start=1):
        name = row["feature"]
        idx = feature_index[name]
        x = x_endpoint[:, idx]

        dist = describe_distribution(x)
        counts, edges = shared_histogram(x)

        groups = []
        for cls in classes:
            mask = labels == cls
            xc = x[mask]
            gcounts = histogram_on_edges(xc, edges)
            groups.append(
                {
                    "label": cls,
                    "n": int(mask.sum()),
                    "median": float(np.median(xc)),
                    "iqr": float(np.percentile(xc, 75) - np.percentile(xc, 25)),
                    "histogram": {"counts": gcounts.tolist()},
                }
            )

        x_boot = x[boot_idx]  # (n_boot, n) — this feature's values under every resample
        ranks_boot = batch_ranks(x_boot)

        if binary:
            positive = labels == classes[1]
            positive_boot = labels_boot == classes[1]
            effect_draws = batch_cohens_d(x_boot, positive_boot)
            auc_point = rank_auc(x, positive)
            auc_lo, auc_hi = ci_from_draws(batch_auc_from_ranks(ranks_boot, positive_boot))
            higher_is_positive = str(row["direction"]).startswith("Higher")
            threshold_stats = youden_threshold(x, positive, higher_is_positive)
            discrimination = {
                "type": "binary",
                "auc": round(float(auc_point), 3),
                "ciLow": round(auc_lo, 3),
                "ciHigh": round(auc_hi, 3),
                **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in threshold_stats.items()},
            }
        else:
            effect_draws = batch_eta_squared(x_boot, labels_boot, classes)
            by_class = []
            aucs = []
            for cls in classes:
                positive = labels == cls
                point = rank_auc(x, positive)
                lo, hi = ci_from_draws(batch_auc_from_ranks(ranks_boot, labels_boot == cls))
                aucs.append((cls, point))
                by_class.append({"label": cls, "auc": round(float(point), 3), "ciLow": round(lo, 3), "ciHigh": round(hi, 3)})
            aucs_sorted = sorted(aucs, key=lambda item: item[1], reverse=True)
            if len(aucs_sorted) >= 2 and aucs_sorted[0][1] - aucs_sorted[1][1] >= 0.05:
                summary = f"Primarily distinguishes {aucs_sorted[0][0]}"
            else:
                summary = "Distinguishes multiple subtypes"
            discrimination = {"type": "oneVsRest", "byClass": by_class, "summary": summary}

        effect_lo, effect_hi = ci_from_draws(effect_draws)
        print(f"  [{row_number}/{len(ranking)}] {name}", flush=True)

        out.append(
            {
                "feature": name,
                "display": row["display"],
                "region": row["region"],
                "family": row["family"],
                "transform": row["transform"],
                "measure": row["measure"],
                "rank": int(row["rank"]),
                "stability": round(float(row["selectionFrequency"]), 3),
                "univariateP": float(row["univariateP"]),
                "fdrQ": float(row["fdrQ"]),
                "effect": float(row["effect"]),
                "effectCiLow": round(effect_lo, 4),
                "effectCiHigh": round(effect_hi, 4),
                "direction": row["direction"],
                "finalModel": bool(row["finalModel"]),
                "distribution": dist,
                "histogram": {"binEdges": [round(float(e), 6) for e in edges], "counts": counts.tolist()},
                "percentileCurve": percentile_curve(x),
                "groups": groups,
                "discrimination": discrimination,
            }
        )
    return out


def build_survival_features(
    ranking_csv: Path,
    feature_names: list[str],
    x_all: np.ndarray,
    survival_mask: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    ranking = pd.read_csv(ranking_csv).sort_values("rank").head(50)
    feature_index = {name: i for i, name in enumerate(feature_names)}
    x_endpoint = x_all[survival_mask]
    classes = [0, 1]
    class_labels = {0: "Censored", 1: "Event (death)"}

    boot_idx = make_bootstrap_indices(events, classes, N_BOOTSTRAP, rng)  # (n_boot, n)
    durations_boot = durations[boot_idx]
    events_boot = events[boot_idx]

    out = []
    for row_number, (_, row) in enumerate(ranking.iterrows(), start=1):
        name = row["feature"]
        idx = feature_index[name]
        x = x_endpoint[:, idx]

        dist = describe_distribution(x)
        counts, edges = shared_histogram(x)

        groups = []
        for cls in classes:
            mask = events == cls
            xc = x[mask]
            gcounts = histogram_on_edges(xc, edges)
            groups.append(
                {
                    "label": class_labels[cls],
                    "n": int(mask.sum()),
                    "median": float(np.median(xc)),
                    "iqr": float(np.percentile(xc, 75) - np.percentile(xc, 25)),
                    "histogram": {"counts": gcounts.tolist()},
                }
            )

        # Concordance index: how well this single feature's value orders survival
        # time on its own. Bootstrapped directly via lifelines (benchmarked at
        # ~1.6ms/call, so 1,000 draws x 50 features stays well under two minutes —
        # no need for the hand-rolled vectorization the AUC path uses).
        c_point = survival_concordance(durations, x, events)
        c_draws = np.array(
            [survival_concordance(durations_boot[b], x[boot_idx[b]], events_boot[b]) for b in range(N_BOOTSTRAP)]
        )
        c_lo, c_hi = ci_from_draws(c_draws)

        # Hazard-ratio CI: one univariate Cox fit (fast — a single covariate),
        # native Wald CI from the fit rather than 1,000 bootstrap refits, matching
        # run_analysis.py's rationale for its own per-feature hazard ratios. The
        # point estimate itself is reused from the ranking CSV (same value
        # run_analysis.py already computed) so this page never disagrees with the
        # main dashboard's Feature registry.
        sd = x.std(ddof=1)
        z = (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)
        df_uni = pd.DataFrame({"x": z, "duration": durations, "event": events})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph = CoxPHFitter(penalizer=0.0)
                cph.fit(df_uni, duration_col="duration", event_col="event")
            effect_lo = float(cph.summary["coef lower 95%"].iloc[0])
            effect_hi = float(cph.summary["coef upper 95%"].iloc[0])
        except Exception:
            effect_lo, effect_hi = float("nan"), float("nan")

        discrimination = {
            "type": "concordance",
            "concordance": round(float(c_point), 3),
            "ciLow": round(c_lo, 3),
            "ciHigh": round(c_hi, 3),
        }

        print(f"  [{row_number}/{len(ranking)}] {name}", flush=True)

        out.append(
            {
                "feature": name,
                "display": row["display"],
                "region": row["region"],
                "family": row["family"],
                "transform": row["transform"],
                "measure": row["measure"],
                "rank": int(row["rank"]),
                "stability": round(float(row["selectionFrequency"]), 3),
                "univariateP": float(row["univariateP"]),
                "fdrQ": float(row["fdrQ"]),
                "effect": float(row["effect"]),
                "effectCiLow": round(effect_lo, 4) if np.isfinite(effect_lo) else None,
                "effectCiHigh": round(effect_hi, 4) if np.isfinite(effect_hi) else None,
                "direction": row["direction"],
                "finalModel": bool(row["finalModel"]),
                "distribution": dist,
                "histogram": {"binEdges": [round(float(e), 6) for e in edges], "counts": counts.tolist()},
                "percentileCurve": percentile_curve(x),
                "groups": groups,
                "discrimination": discrimination,
            }
        )
    return out


FAMILY_GROUPS = [
    {
        "family": "Shape",
        "interpretation": "Tumor size, compactness, irregularity, elongation, and surface complexity.",
        "available": False,
        "note": "No shape-family features are present in this radiomics extraction — only first-order, GLCM, GLRLM, GLSZM, and GLDM features were provided.",
    },
    {
        "family": "First-order intensity",
        "interpretation": "Overall signal intensity and intensity variability within the region.",
        "available": True,
    },
    {
        "family": "Texture (GLCM / GLRLM / GLSZM / GLDM)",
        "interpretation": "Intratumoral heterogeneity, uniformity, coarseness, or local contrast.",
        "available": True,
    },
    {
        "family": "Wavelet / filtered texture",
        "interpretation": "The same first-order and texture measures recomputed on wavelet- or Laplacian-of-Gaussian-filtered images, capturing imaging patterns at different spatial scales.",
        "available": True,
    },
    {
        "family": "Vessel features",
        "interpretation": "Peritumoral vascular burden, geometry, tortuosity, and heterogeneity, computed on the vessel region independently of the tumor region.",
        "available": True,
    },
    {
        "family": "Tumor–vessel relationships",
        "interpretation": "Spatial organization of tumor and surrounding vasculature (e.g. joint or relational measures between the two regions).",
        "available": False,
        "note": "Not present in this feature set. Tumor and vessel features were extracted independently per region; no joint tumor–vessel spatial-relationship features were computed.",
    },
]

UNAVAILABLE = [
    {
        "topic": "Scanner, acquisition, and reconstruction metadata",
        "note": "Not available for this cohort. The radiomics extraction was delivered as a flat per-patient feature table with no accompanying scanner model, field strength, sequence, or reconstruction parameters.",
    },
    {
        "topic": "Segmentation method distribution",
        "note": "Not available for this cohort. No record of segmentation method (manual, semi-automated, or automated) or of which reader/tool produced each contour was provided alongside the features.",
    },
    {
        "topic": "Feature stability across scanners, protocols, or segmentation methods",
        "note": "Cannot be assessed — this would require repeat scans, repeat segmentations, or protocol/site labels, none of which exist in the source files. The fold-stability percentages shown elsewhere in this app describe robustness to cross-validation resampling, a different property from robustness to acquisition variation.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiomics", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rankings-dir", type=Path, default=None, help="Defaults to --output-dir")
    args = parser.parse_args()
    rankings_dir = args.rankings_dir or args.output_dir

    radiomics, feature_names, x_all, merged = load_cohort(args.radiomics, args.clinical)
    rng = np.random.default_rng(SEED)

    survival_mask = (merged["OS_STATUS"].notna() & merged["OS_MONTHS"].notna() & (merged["OS_MONTHS"] > 0)).to_numpy()
    survival_durations = merged.loc[survival_mask, "OS_MONTHS"].to_numpy(dtype=float)
    survival_events = merged.loc[survival_mask, "OS_STATUS"].to_numpy(dtype=int)

    subtype_mask = merged["Mol Subtype"].notna().to_numpy()
    subtype_labels = merged.loc[subtype_mask, "Mol Subtype"].map(SUBTYPE_LABELS).to_numpy()
    subtype_classes = ["Basal", "HER2-enriched", "Luminal A", "Luminal B"]

    pcr_mask = merged["strict_response"].isin(KNOWN_RESPONSE).to_numpy()
    pcr_labels = np.where(merged.loc[pcr_mask, "strict_response"].eq("complete").to_numpy(), "pCR", "No pCR")
    pcr_classes = ["No pCR", "pCR"]

    excluded_reasons_raw = merged.loc[~pcr_mask, "strict_response"].fillna("missing / not recorded")
    exclusion_reasons = [
        {"reason": reason, "count": int(count)}
        for reason, count in excluded_reasons_raw.value_counts().items()
    ]

    print("Building cohort/data-quality summary (all features)...")
    data_quality = build_data_quality(feature_names, x_all)

    print("Building survival endpoint feature statistics...")
    survival_features = build_survival_features(
        rankings_dir / "feature_rankings_survival.csv",
        feature_names,
        x_all,
        survival_mask,
        survival_durations,
        survival_events,
        rng=rng,
    )

    print("Building subtype endpoint feature statistics...")
    subtype_features = build_endpoint_features(
        rankings_dir / "feature_rankings_subtype.csv",
        feature_names,
        x_all,
        subtype_mask,
        subtype_labels,
        subtype_classes,
        binary=False,
        rng=rng,
    )

    print("Building pCR endpoint feature statistics...")
    pcr_features = build_endpoint_features(
        rankings_dir / "feature_rankings_pcr.csv",
        feature_names,
        x_all,
        pcr_mask,
        pcr_labels,
        pcr_classes,
        binary=True,
        rng=rng,
    )

    output = {
        "meta": {
            "patientCount": int(len(radiomics)),
            "radiomicFeatureCount": int(len(feature_names)),
            "tumorFeatureCount": int(sum(not c.endswith("_vessel") for c in feature_names)),
            "vesselFeatureCount": int(sum(c.endswith("_vessel") for c in feature_names)),
            "endpoints": {
                "survival": {
                    "sampleCount": int(survival_mask.sum()),
                    "excludedCount": int(len(merged) - survival_mask.sum()),
                    "exclusionReasons": [{"reason": "missing status, missing follow-up time, or non-positive follow-up", "count": int(len(merged) - survival_mask.sum())}],
                },
                "subtype": {
                    "sampleCount": int(subtype_mask.sum()),
                    "excludedCount": int(len(merged) - subtype_mask.sum()),
                    "exclusionReasons": [],
                },
                "pcr": {
                    "sampleCount": int(pcr_mask.sum()),
                    "excludedCount": int(len(merged) - pcr_mask.sum()),
                    "exclusionReasons": exclusion_reasons,
                },
            },
            "qualityThresholds": {
                "outlierCautionRate": 0.05,
                "outlierPoorRate": 0.15,
                "nearConstantTopValueShare": 0.99,
                "missingCautionRate": 0.0,
                "missingPoorRate": 0.05,
            },
            "discriminationThreshold": "Youden's J statistic (maximizes sensitivity + specificity − 1), computed once on the evaluable cohort for that feature. The survival endpoint's concordance index needs no threshold — it directly measures ranking agreement between the feature and observed survival time.",
            "bootstrap": N_BOOTSTRAP,
            "unavailable": UNAVAILABLE,
        },
        "dataQuality": data_quality,
        "familyGroups": FAMILY_GROUPS,
        "endpoints": {
            "survival": {"sampleCount": int(survival_mask.sum()), "classes": ["Censored", "Event (death)"], "features": survival_features},
            "subtype": {"sampleCount": int(subtype_mask.sum()), "classes": subtype_classes, "features": subtype_features},
            "pcr": {"sampleCount": int(pcr_mask.sum()), "classes": pcr_classes, "features": pcr_features},
        },
    }

    out_path = args.output_dir / "feature-explorer.json"
    out_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "dataQualityFeatures": len(data_quality["features"]), "survivalFeatures": len(survival_features), "subtypeFeatures": len(subtype_features), "pcrFeatures": len(pcr_features)}, indent=2))


if __name__ == "__main__":
    main()
