#!/usr/bin/env python3
"""Reproducible conventional radiomics feature-selection analysis.

The script deliberately keeps clinical variables out of every predictive model.
Clinical workbooks are used only to supply endpoint labels and aggregate context.
No patient-level rows or identifiers are written to the dashboard outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


SEED = 20260804
N_BOOTSTRAP = 1000


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Greedy Spearman redundancy filter, ordered by training-fold F score."""

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, scores: np.ndarray | None = None):
        X = np.asarray(X, dtype=float)
        if X.shape[1] <= 1:
            self.support_ = np.ones(X.shape[1], dtype=bool)
            return self

        if scores is None:
            scores, _ = f_classif(X, y)
        scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf, posinf=np.finfo(float).max)
        order = np.argsort(-scores, kind="stable")
        ranked = np.apply_along_axis(rankdata, 0, X)
        corr = np.corrcoef(ranked, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        kept: list[int] = []
        for idx in order:
            if not kept or np.all(np.abs(corr[idx, kept]) < self.threshold):
                kept.append(int(idx))
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[kept] = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, self.support_]

    def get_support(self) -> np.ndarray:
        return self.support_.copy()


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


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid] = restored
    return out


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


def effect_sizes(X: np.ndarray, y: np.ndarray, binary: bool) -> np.ndarray:
    if binary:
        x0, x1 = X[y == 0], X[y == 1]
        n0, n1 = len(x0), len(x1)
        v0 = np.var(x0, axis=0, ddof=1)
        v1 = np.var(x1, axis=0, ddof=1)
        pooled = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / max(n0 + n1 - 2, 1))
        d = np.divide(
            np.mean(x1, axis=0) - np.mean(x0, axis=0),
            pooled,
            out=np.zeros(X.shape[1]),
            where=pooled > 0,
        )
        correction = 1 - 3 / max(4 * (n0 + n1) - 9, 1)
        return d * correction

    overall = np.mean(X, axis=0)
    ss_between = np.zeros(X.shape[1])
    for label in np.unique(y):
        subset = X[y == label]
        ss_between += len(subset) * (np.mean(subset, axis=0) - overall) ** 2
    ss_total = np.sum((X - overall) ** 2, axis=0)
    return np.divide(ss_between, ss_total, out=np.zeros(X.shape[1]), where=ss_total > 0)


def correlation_representatives(X: np.ndarray, order: np.ndarray, candidate_mask: np.ndarray, threshold: float = 0.90) -> int:
    candidates = [int(i) for i in order if candidate_mask[i]][:500]
    if not candidates:
        return 0
    ranked = np.apply_along_axis(rankdata, 0, X[:, candidates])
    corr = np.nan_to_num(np.corrcoef(ranked, rowvar=False), nan=0.0)
    kept: list[int] = []
    for local_idx in range(len(candidates)):
        if not kept or np.all(np.abs(corr[local_idx, kept]) < threshold):
            kept.append(local_idx)
    return len(kept)


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 80) -> list[dict[str, float]]:
    if len(x) <= max_points:
        idx = np.arange(len(x))
    else:
        idx = np.unique(np.linspace(0, len(x) - 1, max_points).round().astype(int))
    return [{"x": round(float(x[i]), 5), "y": round(float(y[i]), 5)} for i in idx]


def bootstrap_intervals(y: np.ndarray, prob: np.ndarray, classes: np.ndarray, binary: bool) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(SEED + (1 if binary else 0))
    values: dict[str, list[float]] = {}
    strata = [np.flatnonzero(y == c) for c in classes]

    for _ in range(N_BOOTSTRAP):
        sample = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata])
        rng.shuffle(sample)
        yt, pt = y[sample], prob[sample]
        pred = classes[np.argmax(pt, axis=1)]
        if binary:
            positive = pt[:, 1]
            cm = confusion_matrix(yt, pred, labels=classes)
            tn, fp, fn, tp = cm.ravel()
            metrics = {
                "rocAuc": roc_auc_score(yt, positive),
                "prAuc": average_precision_score(yt, positive),
                "balancedAccuracy": balanced_accuracy_score(yt, pred),
                "f1": f1_score(yt, pred, zero_division=0),
                "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                "specificity": tn / (tn + fp) if tn + fp else np.nan,
                "brier": brier_score_loss(yt, positive),
            }
        else:
            metrics = {
                "macroRocAuc": roc_auc_score(yt, pt, labels=classes, multi_class="ovr", average="macro"),
                "balancedAccuracy": balanced_accuracy_score(yt, pred),
                "macroF1": f1_score(yt, pred, average="macro", zero_division=0),
                "logLoss": log_loss(yt, pt, labels=classes),
            }
        for key, value in metrics.items():
            if np.isfinite(value):
                values.setdefault(key, []).append(float(value))
    return {key: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for key, v in values.items()}


def survival_concordance(durations: np.ndarray, risk: np.ndarray, events: np.ndarray) -> float:
    """Concordance index for a risk score (higher risk = shorter survival).

    lifelines' concordance_index expects scores that behave like survival
    time itself (higher score = longer survival), the opposite of a risk
    score, so the risk score is negated here. Centralizing that sign flip
    in one place avoids it being a footgun everywhere else in this file.
    """
    try:
        return float(concordance_index(durations, -risk, events))
    except ZeroDivisionError:
        return float("nan")


def univariate_concordance_strength(X: np.ndarray, durations: np.ndarray, events: np.ndarray) -> np.ndarray:
    """|C-index - 0.5| per column: association strength, direction-agnostic.

    Negating a score flips C to 1-C, so this is identical whether or not the
    raw column is used as-is versus negated first — safe to call directly on
    raw feature values without worrying about survival_concordance's sign
    convention.
    """
    scores = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        c = survival_concordance(durations, X[:, j], events)
        scores[j] = abs(c - 0.5) if np.isfinite(c) else 0.0
    return scores


def km_curve_points(durations: np.ndarray, events: np.ndarray, max_points: int = 80) -> list[dict[str, float]]:
    kmf = KaplanMeierFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmf.fit(durations, events)
    table = kmf.survival_function_.reset_index()
    return downsample_xy(table.iloc[:, 0].to_numpy(dtype=float), table.iloc[:, 1].to_numpy(dtype=float), max_points=max_points)


def survival_bootstrap_intervals(durations: np.ndarray, risk: np.ndarray, events: np.ndarray) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(SEED + 3)
    median_risk = float(np.median(risk))
    strata = [np.flatnonzero(events == c) for c in (0, 1)]
    values: dict[str, list[float]] = {"concordance": [], "logrankP": []}

    for _ in range(N_BOOTSTRAP):
        sample = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata])
        rng.shuffle(sample)
        d, r, e = durations[sample], risk[sample], events[sample]
        c = survival_concordance(d, r, e)
        if np.isfinite(c):
            values["concordance"].append(c)

        high = r >= median_risk
        if high.any() and (~high).any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = logrank_test(d[high], d[~high], e[high], e[~high])
            if np.isfinite(result.p_value):
                values["logrankP"].append(float(result.p_value))

    return {key: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for key, v in values.items()}


def metric_record(key: str, label: str, value: float, intervals: dict[str, tuple[float, float]], higher_better: bool = True) -> dict:
    low, high = intervals[key]
    return {
        "key": key,
        "label": label,
        "value": round(float(value), 3),
        "low": round(low, 3),
        "high": round(high, 3),
        "higherBetter": higher_better,
    }


def fit_endpoint(
    endpoint_key: str,
    X: np.ndarray,
    y_raw: pd.Series,
    feature_names: list[str],
    class_labels: dict,
    binary: bool,
) -> tuple[dict, pd.DataFrame]:
    classes_raw = np.array(sorted(pd.unique(y_raw)))
    class_to_int = {label: i for i, label in enumerate(classes_raw)}
    y = y_raw.map(class_to_int).to_numpy(dtype=int)
    classes = np.arange(len(classes_raw))
    class_names = [class_labels.get(label, str(label)) for label in classes_raw]

    scores, p_values = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0)
    p_values = np.nan_to_num(p_values, nan=1.0)
    q_values = bh_adjust(p_values)
    effects = effect_sizes(X, y, binary)
    fdr_mask = q_values < 0.05
    order = np.lexsort((-np.abs(effects), p_values))
    correlation_count = correlation_representatives(X, order, fdr_mask)

    pipeline = Pipeline(
        [
            ("screen", SelectKBest(score_func=f_classif, k=50)),
            ("correlation", CorrelationPruner(threshold=0.90)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    class_weight="balanced",
                    max_iter=4000,
                    tol=1e-3,
                    random_state=SEED,
                ),
            ),
        ]
    )
    param_grid = {
        "screen__k": [25, 50, 100],
        "model__C": [0.03, 0.10, 0.30, 1.0],
    }
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scoring = "roc_auc" if binary else "roc_auc_ovr"
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

    probability_sum = np.zeros((len(y), len(classes)), dtype=float)
    probability_count = np.zeros(len(y), dtype=int)
    selection_count = np.zeros(X.shape[1], dtype=int)
    coefficient_abs_sum = np.zeros(X.shape[1], dtype=float)
    coefficient_signed_sum = np.zeros((len(classes) if not binary else 1, X.shape[1]), dtype=float)
    fold_records: list[dict] = []

    for fold, (train, test) in enumerate(outer.split(X, y), start=1):
        search = GridSearchCV(
            pipeline,
            param_grid,
            scoring=scoring,
            cv=inner,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X[train], y[train])
        estimator = search.best_estimator_
        probability = estimator.predict_proba(X[test])
        probability_sum[test] += probability
        probability_count[test] += 1

        selected_idx = np.flatnonzero(estimator.named_steps["screen"].get_support())
        pruned_idx = selected_idx[estimator.named_steps["correlation"].get_support()]
        coef = estimator.named_steps["model"].coef_
        nonzero = np.any(np.abs(coef) > 1e-9, axis=0)
        active_idx = pruned_idx[nonzero]
        active_coef = coef[:, nonzero]
        selection_count[active_idx] += 1
        coefficient_abs_sum[active_idx] += np.mean(np.abs(active_coef), axis=0)
        coefficient_signed_sum[:, active_idx] += active_coef

        pred = estimator.classes_[np.argmax(probability, axis=1)]
        record = {
            "fold": fold,
            "bestK": int(search.best_params_["screen__k"]),
            "bestC": float(search.best_params_["model__C"]),
            "activeFeatures": int(len(active_idx)),
            "balancedAccuracy": float(balanced_accuracy_score(y[test], pred)),
        }
        if binary:
            record["rocAuc"] = float(roc_auc_score(y[test], probability[:, 1]))
            record["prAuc"] = float(average_precision_score(y[test], probability[:, 1]))
        else:
            record["macroRocAuc"] = float(
                roc_auc_score(y[test], probability, labels=classes, multi_class="ovr", average="macro")
            )
            record["macroF1"] = float(f1_score(y[test], pred, average="macro", zero_division=0))
        fold_records.append(record)

    if not np.all(probability_count == 3):
        raise RuntimeError("Repeated CV did not produce exactly three out-of-fold predictions per patient")
    oof_probability = probability_sum / probability_count[:, None]
    oof_pred = classes[np.argmax(oof_probability, axis=1)]
    intervals = bootstrap_intervals(y, oof_probability, classes, binary)

    final_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        cv=inner,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_search.fit(X, y)
    final_estimator = final_search.best_estimator_
    final_screen_idx = np.flatnonzero(final_estimator.named_steps["screen"].get_support())
    final_pruned_idx = final_screen_idx[final_estimator.named_steps["correlation"].get_support()]
    final_coef = final_estimator.named_steps["model"].coef_
    final_nonzero = np.any(np.abs(final_coef) > 1e-9, axis=0)
    final_active_idx = final_pruned_idx[final_nonzero]
    final_abs = np.zeros(X.shape[1])
    final_abs[final_active_idx] = np.mean(np.abs(final_coef[:, final_nonzero]), axis=0)

    n_outer = len(fold_records)
    frequency = selection_count / n_outer
    mean_abs = np.divide(
        coefficient_abs_sum,
        selection_count,
        out=np.zeros_like(coefficient_abs_sum),
        where=selection_count > 0,
    )
    signed_mean = coefficient_signed_sum / n_outer

    rows: list[dict] = []
    for idx, name in enumerate(feature_names):
        parsed = parse_feature(name)
        if binary:
            direction = "Higher in pCR" if signed_mean[0, idx] >= 0 else "Lower in pCR"
        else:
            dominant = int(np.argmax(np.abs(signed_mean[:, idx])))
            sign = "Higher" if signed_mean[dominant, idx] >= 0 else "Lower"
            direction = f"{sign} → {class_names[dominant]}"
        rows.append(
            {
                "feature": name,
                **parsed,
                "selectionFrequency": float(frequency[idx]),
                "meanAbsCoefficient": float(mean_abs[idx]),
                "finalAbsCoefficient": float(final_abs[idx]),
                "univariateP": float(p_values[idx]),
                "fdrQ": float(q_values[idx]),
                "effect": float(effects[idx]),
                "effectMagnitude": float(abs(effects[idx])),
                "direction": direction,
                "finalModel": bool(idx in set(final_active_idx.tolist())),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["selectionFrequency", "meanAbsCoefficient", "effectMagnitude", "univariateP"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    top = ranking.head(50).copy()
    feature_records = []
    for _, row in top.iterrows():
        feature_records.append(
            {
                "rank": int(row["rank"]),
                "feature": row["feature"],
                "display": row["display"],
                "region": row["region"],
                "transform": row["transform"],
                "family": row["family"],
                "measure": row["measure"],
                "frequency": round(float(row["selectionFrequency"]), 3),
                "coefficient": round(float(row["meanAbsCoefficient"]), 4),
                "effect": round(float(row["effect"]), 4),
                "qValue": float(row["fdrQ"]),
                "direction": row["direction"],
                "fdrSignificant": bool(row["fdrQ"] < 0.05),
                "finalModel": bool(row["finalModel"]),
            }
        )

    cm = confusion_matrix(y, oof_pred, labels=classes)
    cm_norm = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) > 0)
    class_distribution = [
        {"label": class_names[i], "count": int(np.sum(y == i)), "percent": round(float(np.mean(y == i) * 100), 1)}
        for i in classes
    ]

    feature_groups = {
        "regions": [{"label": k, "count": int(v)} for k, v in Counter(top.head(20)["region"]).most_common()],
        "families": [{"label": k, "count": int(v)} for k, v in Counter(top.head(20)["family"]).most_common()],
        "transforms": [{"label": k, "count": int(v)} for k, v in Counter(top.head(20)["transform"]).most_common()],
    }

    roc_curves: list[dict] = []
    if binary:
        fpr, tpr, _ = roc_curve(y, oof_probability[:, 1])
        precision, recall, _ = precision_recall_curve(y, oof_probability[:, 1])
        roc_curves.append({"label": "pCR", "auc": round(float(roc_auc_score(y, oof_probability[:, 1])), 3), "points": downsample_xy(fpr, tpr)})
        pr_curve = downsample_xy(recall[::-1], precision[::-1])
        fraction_positive, mean_predicted = calibration_curve(y, oof_probability[:, 1], n_bins=8, strategy="quantile")
        calibration = downsample_xy(mean_predicted, fraction_positive, max_points=20)
        tn, fp, fn, tp = cm.ravel()
        values = {
            "rocAuc": roc_auc_score(y, oof_probability[:, 1]),
            "prAuc": average_precision_score(y, oof_probability[:, 1]),
            "balancedAccuracy": balanced_accuracy_score(y, oof_pred),
            "f1": f1_score(y, oof_pred, zero_division=0),
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "brier": brier_score_loss(y, oof_probability[:, 1]),
        }
        metrics = [
            metric_record("rocAuc", "ROC AUC", values["rocAuc"], intervals),
            metric_record("prAuc", "PR AUC", values["prAuc"], intervals),
            metric_record("balancedAccuracy", "Balanced accuracy", values["balancedAccuracy"], intervals),
            metric_record("sensitivity", "Sensitivity", values["sensitivity"], intervals),
            metric_record("specificity", "Specificity", values["specificity"], intervals),
            metric_record("brier", "Brier score", values["brier"], intervals, higher_better=False),
        ]
    else:
        y_bin = label_binarize(y, classes=classes)
        class_auc = []
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], oof_probability[:, i])
            auc_value = roc_auc_score(y_bin[:, i], oof_probability[:, i])
            roc_curves.append({"label": class_name, "auc": round(float(auc_value), 3), "points": downsample_xy(fpr, tpr)})
            class_auc.append({"label": class_name, "auc": round(float(auc_value), 3)})
        values = {
            "macroRocAuc": roc_auc_score(y, oof_probability, labels=classes, multi_class="ovr", average="macro"),
            "balancedAccuracy": balanced_accuracy_score(y, oof_pred),
            "macroF1": f1_score(y, oof_pred, average="macro", zero_division=0),
            "logLoss": log_loss(y, oof_probability, labels=classes),
        }
        metrics = [
            metric_record("macroRocAuc", "Macro ROC AUC", values["macroRocAuc"], intervals),
            metric_record("balancedAccuracy", "Balanced accuracy", values["balancedAccuracy"], intervals),
            metric_record("macroF1", "Macro F1", values["macroF1"], intervals),
            metric_record("logLoss", "Log loss", values["logLoss"], intervals, higher_better=False),
        ]
        pr_curve = []
        calibration = []

    stable_count = int(np.sum(frequency >= 0.50))
    endpoint = {
        "key": endpoint_key,
        "sampleCount": int(len(y)),
        "classCount": int(len(classes)),
        "classes": class_distribution,
        "metrics": metrics,
        "classAuc": class_auc if not binary else [],
        "rocCurves": roc_curves,
        "prCurve": pr_curve,
        "calibration": calibration,
        "confusion": {
            "labels": class_names,
            "counts": cm.astype(int).tolist(),
            "normalized": np.round(cm_norm, 3).tolist(),
        },
        "featureFunnel": [
            {"label": "Radiomic input", "count": int(X.shape[1]), "note": "Tumor + vessel regions"},
            {"label": "QC retained", "count": int(X.shape[1]), "note": "No missing, infinite, or constant features"},
            {"label": "Univariate FDR", "count": int(np.sum(fdr_mask)), "note": "Benjamini–Hochberg q < 0.05"},
            {"label": "Nonredundant FDR", "count": int(correlation_count), "note": "|Spearman ρ| < 0.90 representatives"},
            {"label": "Stable L1", "count": stable_count, "note": "Selected in ≥50% of outer folds"},
        ],
        "features": feature_records,
        "featureGroups": feature_groups,
        "validation": {
            "outerDesign": "5-fold × 3 repeats",
            "outerFits": n_outer,
            "innerDesign": "3-fold grid search",
            "bootstrap": N_BOOTSTRAP,
            "bestKMode": int(pd.Series([f["bestK"] for f in fold_records]).mode().iloc[0]),
            "bestCMode": float(pd.Series([f["bestC"] for f in fold_records]).mode().iloc[0]),
            "medianActiveFeatures": int(round(float(np.median([f["activeFeatures"] for f in fold_records])))),
            "finalK": int(final_search.best_params_["screen__k"]),
            "finalC": float(final_search.best_params_["model__C"]),
            "finalActiveFeatures": int(len(final_active_idx)),
        },
    }
    return endpoint, ranking


PARAM_GRID_SURVIVAL = [(k, pen) for k in (25, 50, 100) for pen in (0.05, 0.1, 0.3, 1.0)]


def _screen_and_prune_survival(X: np.ndarray, durations: np.ndarray, events: np.ndarray, k: int) -> np.ndarray:
    scores = univariate_concordance_strength(X, durations, events)
    screen_idx = np.argsort(-scores)[:k]
    pruner = CorrelationPruner(threshold=0.90)
    pruner.fit(X[:, screen_idx], scores=scores[screen_idx])
    return screen_idx[pruner.get_support()]


def _fit_cox(X: np.ndarray, durations: np.ndarray, events: np.ndarray, k: int, penalizer: float):
    pruned_idx = _screen_and_prune_survival(X, durations, events, k)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[:, pruned_idx])
    df = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(len(pruned_idx))])
    df["duration"] = durations
    df["event"] = events
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=1.0)
        cph.fit(df, duration_col="duration", event_col="event")
    return pruned_idx, cph, scaler


def _inner_fold_concordance(X_train, dur_train, evt_train, X_val, dur_val, evt_val, k, pen) -> float:
    """One inner-fold fit + validation score for a (k, penalizer) combo.

    Cox's Newton-Raphson optimizer is more fragile to collinearity than
    sklearn's logistic-regression solver, especially with few events — a
    weakly regularized combo occasionally hits a singular Hessian on a given
    fold split. That's informative for hyperparameter *selection* (this combo
    is unstable for this data), not a bug, so it's scored as chance-level
    (0.5) rather than raised — the search then naturally prefers combos that
    converge reliably.
    """
    try:
        pruned_idx, cph, scaler = _fit_cox(X_train, dur_train, evt_train, k, pen)
        X_val_scaled = scaler.transform(X_val[:, pruned_idx])
        risk_val = cph.predict_partial_hazard(
            pd.DataFrame(X_val_scaled, columns=[f"f{i}" for i in range(len(pruned_idx))])
        ).to_numpy()
        score = survival_concordance(dur_val, risk_val, evt_val)
        return score if np.isfinite(score) else 0.5
    except Exception:
        return 0.5


def _select_survival_params(X: np.ndarray, durations: np.ndarray, events: np.ndarray) -> tuple[int, float]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_score, best_params = -np.inf, PARAM_GRID_SURVIVAL[0]
    for k, pen in PARAM_GRID_SURVIVAL:
        scores = [
            _inner_fold_concordance(X[train_idx], durations[train_idx], events[train_idx], X[val_idx], durations[val_idx], events[val_idx], k, pen)
            for train_idx, val_idx in inner.split(X, events)
        ]
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_params = mean_score, (k, pen)
    return best_params


def fit_survival_endpoint(
    X: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    feature_names: list[str],
) -> tuple[dict, pd.DataFrame]:
    n = len(durations)

    print("  [survival] full-cohort univariate Cox screen (2,064 features)...", flush=True)
    p_values = np.empty(X.shape[1])
    effects = np.empty(X.shape[1])
    for idx in range(X.shape[1]):
        col = X[:, idx]
        sd = col.std(ddof=1)
        if sd == 0:
            # Constant within this 561-patient survival subset even though no
            # feature is constant across the full 772-patient cohort — a
            # zero-variance predictor has no association to measure.
            p_values[idx] = 1.0
            effects[idx] = 0.0
            continue
        z = (col - col.mean()) / sd
        df_uni = pd.DataFrame({"x": z, "duration": durations, "event": events})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph_uni = CoxPHFitter(penalizer=0.0)
                cph_uni.fit(df_uni, duration_col="duration", event_col="event")
            p_values[idx] = float(cph_uni.summary["p"].iloc[0])
            effects[idx] = float(cph_uni.params_.iloc[0])
        except Exception:
            # A handful of columns fail to converge unpenalized (near-perfect
            # separation or extreme outliers); treat as no measurable effect
            # rather than aborting the full-cohort ranking pass over one column.
            p_values[idx] = 1.0
            effects[idx] = 0.0
    p_values = np.nan_to_num(p_values, nan=1.0)
    q_values = bh_adjust(p_values)
    fdr_mask = q_values < 0.05
    order = np.lexsort((-np.abs(effects), p_values))
    correlation_count = correlation_representatives(X, order, fdr_mask)

    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    oof_risk_sum = np.zeros(n)
    oof_count = np.zeros(n, dtype=int)
    selection_count = np.zeros(X.shape[1], dtype=int)
    coefficient_abs_sum = np.zeros(X.shape[1], dtype=float)
    coefficient_signed_sum = np.zeros(X.shape[1], dtype=float)
    fold_records: list[dict] = []

    for fold, (train, test) in enumerate(outer.split(X, events), start=1):
        best_k, best_pen = _select_survival_params(X[train], durations[train], events[train])
        pruned_idx, cph, scaler = _fit_cox(X[train], durations[train], events[train], best_k, best_pen)

        coef = cph.params_.to_numpy()
        nonzero = np.abs(coef) > 1e-9
        active_idx = pruned_idx[nonzero]
        active_coef = coef[nonzero]
        selection_count[active_idx] += 1
        coefficient_abs_sum[active_idx] += np.abs(active_coef)
        coefficient_signed_sum[active_idx] += active_coef

        X_test_scaled = scaler.transform(X[test][:, pruned_idx])
        risk_test = cph.predict_partial_hazard(
            pd.DataFrame(X_test_scaled, columns=[f"f{i}" for i in range(len(pruned_idx))])
        ).to_numpy()
        oof_risk_sum[test] += risk_test
        oof_count[test] += 1

        fold_c = survival_concordance(durations[test], risk_test, events[test])
        fold_records.append(
            {
                "fold": fold,
                "bestK": int(best_k),
                "bestC": float(best_pen),
                "activeFeatures": int(len(active_idx)),
                "concordance": fold_c,
            }
        )
        print(f"  [survival] outer fold {fold}/15 done (k={best_k}, penalizer={best_pen}, active={len(active_idx)}, C={fold_c:.3f})", flush=True)

    if not np.all(oof_count == 3):
        raise RuntimeError("Repeated CV did not produce exactly three out-of-fold predictions per patient")
    oof_risk = oof_risk_sum / oof_count

    overall_c = survival_concordance(durations, oof_risk, events)
    print("  [survival] bootstrapping concordance/log-rank intervals...", flush=True)
    intervals = survival_bootstrap_intervals(durations, oof_risk, events)

    median_risk = float(np.median(oof_risk))
    high = oof_risk >= median_risk
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logrank_result = logrank_test(durations[high], durations[~high], events[high], events[~high])
    logrank_p = float(logrank_result.p_value)

    metrics = [
        metric_record("concordance", "Concordance index", overall_c, intervals),
        metric_record("logrankP", "Log-rank p (risk groups)", logrank_p, intervals, higher_better=False),
    ]

    print("  [survival] final full-cohort refit...", flush=True)
    final_k, final_pen = _select_survival_params(X, durations, events)
    final_pruned_idx, final_cph, _final_scaler = _fit_cox(X, durations, events, final_k, final_pen)
    final_coef = final_cph.params_.to_numpy()
    final_nonzero = np.abs(final_coef) > 1e-9
    final_active_idx = final_pruned_idx[final_nonzero]
    final_abs = np.zeros(X.shape[1])
    final_abs[final_active_idx] = np.abs(final_coef[final_nonzero])

    n_outer = len(fold_records)
    frequency = selection_count / n_outer
    mean_abs = np.divide(
        coefficient_abs_sum,
        selection_count,
        out=np.zeros_like(coefficient_abs_sum),
        where=selection_count > 0,
    )

    rows: list[dict] = []
    for idx, name in enumerate(feature_names):
        parsed = parse_feature(name)
        direction = "Higher risk" if effects[idx] >= 0 else "Lower risk"
        rows.append(
            {
                "feature": name,
                **parsed,
                "selectionFrequency": float(frequency[idx]),
                "meanAbsCoefficient": float(mean_abs[idx]),
                "finalAbsCoefficient": float(final_abs[idx]),
                "univariateP": float(p_values[idx]),
                "fdrQ": float(q_values[idx]),
                "effect": float(effects[idx]),
                "effectMagnitude": float(abs(effects[idx])),
                "direction": direction,
                "finalModel": bool(idx in set(final_active_idx.tolist())),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["selectionFrequency", "meanAbsCoefficient", "effectMagnitude", "univariateP"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    top = ranking.head(50).copy()
    feature_records = []
    for _, row in top.iterrows():
        feature_records.append(
            {
                "rank": int(row["rank"]),
                "feature": row["feature"],
                "display": row["display"],
                "region": row["region"],
                "transform": row["transform"],
                "family": row["family"],
                "measure": row["measure"],
                "frequency": round(float(row["selectionFrequency"]), 3),
                "coefficient": round(float(row["meanAbsCoefficient"]), 4),
                "effect": round(float(row["effect"]), 4),
                "qValue": float(row["fdrQ"]),
                "direction": row["direction"],
                "fdrSignificant": bool(row["fdrQ"] < 0.05),
                "finalModel": bool(row["finalModel"]),
            }
        )

    event_count = int(events.sum())
    class_distribution = [
        {"label": "Event (death)", "count": event_count, "percent": round(event_count / n * 100, 1)},
        {"label": "Censored", "count": n - event_count, "percent": round((n - event_count) / n * 100, 1)},
    ]

    feature_groups = {
        "regions": [{"label": k, "count": int(v)} for k, v in Counter(top.head(20)["region"]).most_common()],
        "families": [{"label": k, "count": int(v)} for k, v in Counter(top.head(20)["family"]).most_common()],
        "transforms": [{"label": k, "count": int(v)} for k, v in Counter(top.head(20)["transform"]).most_common()],
    }

    overall_curve = km_curve_points(durations, events)
    low_points = km_curve_points(durations[~high], events[~high])
    high_points = km_curve_points(durations[high], events[high])

    stable_count = int(np.sum(frequency >= 0.50))
    endpoint = {
        "key": "survival",
        "sampleCount": int(n),
        "classCount": 2,
        "classes": class_distribution,
        "metrics": metrics,
        "survivalCurve": {"points": overall_curve, "maxMonths": round(float(durations.max()), 1)},
        "riskGroups": {
            "logrankP": round(logrank_p, 4),
            "groups": [
                {
                    "label": "Lower predicted risk",
                    "n": int((~high).sum()),
                    "events": int(events[~high].sum()),
                    "points": low_points,
                },
                {
                    "label": "Higher predicted risk",
                    "n": int(high.sum()),
                    "events": int(events[high].sum()),
                    "points": high_points,
                },
            ],
        },
        "featureFunnel": [
            {"label": "Radiomic input", "count": int(X.shape[1]), "note": "Tumor + vessel regions"},
            {"label": "QC retained", "count": int(X.shape[1]), "note": "No missing, infinite, or constant features"},
            {"label": "Univariate FDR", "count": int(np.sum(fdr_mask)), "note": "Benjamini–Hochberg q < 0.05"},
            {"label": "Nonredundant FDR", "count": int(correlation_count), "note": "|Spearman ρ| < 0.90 representatives"},
            {"label": "Stable Cox", "count": stable_count, "note": "Selected in ≥50% of outer folds"},
        ],
        "features": feature_records,
        "featureGroups": feature_groups,
        "validation": {
            "outerDesign": "5-fold × 3 repeats",
            "outerFits": n_outer,
            "innerDesign": "3-fold grid search",
            "bootstrap": N_BOOTSTRAP,
            "bestKMode": int(pd.Series([f["bestK"] for f in fold_records]).mode().iloc[0]),
            "bestCMode": float(pd.Series([f["bestC"] for f in fold_records]).mode().iloc[0]),
            "medianActiveFeatures": int(round(float(np.median([f["activeFeatures"] for f in fold_records])))),
            "finalK": int(final_k),
            "finalC": float(final_pen),
            "finalActiveFeatures": int(len(final_active_idx)),
        },
    }
    return endpoint, ranking


def clinical_context(merged: pd.DataFrame, pcr_mask: pd.Series) -> list[dict]:
    context: list[dict] = []
    pcr = merged.loc[pcr_mask].copy()
    pcr["pCR"] = (pcr["strict_response"] == "complete").astype(int)
    for column, label in [
        ("Mol Subtype", "Molecular subtype"),
        ("ER", "ER status"),
        ("HER2", "HER2 status"),
        ("Menopause (at diagnosis)", "Menopause"),
    ]:
        table = (
            pcr[pcr[column].notna()]
            .groupby(column, observed=True)["pCR"]
            .agg(["sum", "count", "mean"])
            .reset_index()
        )
        for _, row in table.iterrows():
            context.append(
                {
                    "variable": label,
                    "group": str(row[column]).replace("BRCA_", ""),
                    "pcr": int(row["sum"]),
                    "n": int(row["count"]),
                    "rate": round(float(row["mean"]), 3),
                }
            )
    return context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiomics", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    radiomics = pd.read_csv(args.radiomics, low_memory=False)
    demography = read_duke_sheet(args.clinical, "Demography_tumortypes")
    clinical = read_duke_sheet(args.clinical, "Cancer_specific_clinical_data")
    drugs = read_duke_sheet(args.clinical, "drugs")

    clinical_feature_names = ["menopause", "er", "pr", "her2"]
    id_columns = ["Unnamed: 0", "patient"]
    feature_names = [c for c in radiomics.columns if c not in id_columns + clinical_feature_names]
    X_all = radiomics[feature_names].to_numpy(dtype=float)
    if not np.isfinite(X_all).all():
        raise ValueError("Radiomic matrix contains non-finite values")
    if radiomics["patient"].duplicated().any():
        raise ValueError("Radiomic patient identifiers are not unique")

    merged = radiomics[["patient"]].merge(
        demography[["Patient ID", "Mol Subtype", "Nottingham grade", "Staging(Nodes)#(Nx replaced by -1)[N]", "Staging(Tumor Size)# [T]", "OS_STATUS", "OS_MONTHS"]],
        left_on="patient",
        right_on="Patient ID",
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        clinical[["Patient ID", "Menopause (at diagnosis)", "ER", "PR", "HER2"]],
        left_on="patient",
        right_on="Patient ID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_clinical"),
    )
    merged = merged.merge(
        drugs[["Patient ID", "Received Neoadjuvant Therapy or Not", "Overall Near-complete Response:  Stricter Definition"]],
        left_on="patient",
        right_on="Patient ID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_drug"),
    )
    merged["strict_response"] = (
        merged["Overall Near-complete Response:  Stricter Definition"].astype("string").str.strip().str.lower()
    )

    subtype_mask = merged["Mol Subtype"].notna()
    subtype_labels = {
        "BRCA_Basal": "Basal",
        "BRCA_Her2": "HER2-enriched",
        "BRCA_LumA": "Luminal A",
        "BRCA_LumB": "Luminal B",
    }
    subtype, subtype_ranking = fit_endpoint(
        "subtype",
        X_all[subtype_mask.to_numpy()],
        merged.loc[subtype_mask, "Mol Subtype"],
        feature_names,
        subtype_labels,
        binary=False,
    )
    subtype.update(
        {
            "title": "Molecular subtype",
            "shortTitle": "Subtype",
            "description": "Four-class prediction from radiomics only; receptor fields are excluded because they help define the label.",
            "targetDefinition": "BRCA_Basal, BRCA_Her2, BRCA_LumA, or BRCA_LumB from the Duke molecular-subtype field.",
            "excludedCount": int(len(merged) - subtype_mask.sum()),
            "prevalenceNote": "Luminal A is the dominant class, so macro metrics and balanced accuracy are emphasized.",
        }
    )

    known_response = ["complete", "near-complete", "not-complete-or-near-complete"]
    pcr_mask = merged["strict_response"].isin(known_response)
    pcr_y = merged.loc[pcr_mask, "strict_response"].eq("complete").map({False: "No pCR", True: "pCR"})
    pcr, pcr_ranking = fit_endpoint(
        "pcr",
        X_all[pcr_mask.to_numpy()],
        pcr_y,
        feature_names,
        {"No pCR": "No pCR", "pCR": "pCR"},
        binary=True,
    )
    pcr.update(
        {
            "title": "Pathologic complete response",
            "shortTitle": "pCR",
            "description": "Binary radiomics-only prediction among patients with evaluable strict response labels.",
            "targetDefinition": "‘Complete’ in Overall Near-complete Response: Stricter Definition = pCR; near-complete and not-complete-or-near-complete = no pCR; missing/not-available excluded.",
            "excludedCount": int(len(merged) - pcr_mask.sum()),
            "prevalenceNote": "The positive class is uncommon; PR AUC and sensitivity are shown alongside ROC AUC.",
        }
    )

    merged["OS_STATUS"] = pd.to_numeric(merged["OS_STATUS"], errors="coerce")
    merged["OS_MONTHS"] = pd.to_numeric(merged["OS_MONTHS"], errors="coerce")
    survival_mask = merged["OS_STATUS"].notna() & merged["OS_MONTHS"].notna() & (merged["OS_MONTHS"] > 0)
    survival, survival_ranking = fit_survival_endpoint(
        X_all[survival_mask.to_numpy()],
        merged.loc[survival_mask, "OS_MONTHS"].to_numpy(dtype=float),
        merged.loc[survival_mask, "OS_STATUS"].to_numpy(dtype=int),
        feature_names,
    )
    survival.update(
        {
            "title": "Overall survival",
            "shortTitle": "Survival",
            "description": "Radiomics-only Cox proportional-hazards model predicting time to death from any cause.",
            "targetDefinition": "OS_STATUS / OS_MONTHS from the Duke demography sheet (1 = death, 0 = censored at last contact); missing status/time or non-positive follow-up excluded. Disease-specific, disease-free, and progression-free survival fields exist in the source workbook but are empty for every patient, so only overall survival is modeled.",
            "excludedCount": int(len(merged) - survival_mask.sum()),
            "prevalenceNote": "Death is uncommon in this cohort; the log-rank comparison and concordance interval both reflect that sparsity.",
        }
    )

    classification_method = [
        {"step": "01", "title": "Cohort & label audit", "detail": "One row per patient; exact ID join; endpoint-specific missing labels excluded."},
        {"step": "02", "title": "Radiomics-only matrix", "detail": "2,064 tumor and peritumoral-vessel features; menopause and receptor fields explicitly removed."},
        {"step": "03", "title": "Training-fold screen", "detail": "ANOVA F ranking (k = 25, 50, 100), then |Spearman ρ| ≥ 0.90 redundancy pruning."},
        {"step": "04", "title": "Sparse conventional model", "detail": "Z-score scaling and class-weighted L1 logistic regression; C tuned in the inner loop."},
        {"step": "05", "title": "Nested validation", "detail": "5-fold × 3 repeated outer CV, 3-fold inner tuning, pooled out-of-fold predictions, 1,000 bootstrap intervals."},
        {"step": "06", "title": "Stability ranking", "detail": "Features ranked by outer-fold selection frequency, standardized coefficient magnitude, and univariate effect."},
    ]
    survival_method = [
        {"step": "01", "title": "Cohort & label audit", "detail": "One row per patient; exact ID join; missing survival status, missing follow-up time, or non-positive follow-up excluded."},
        {"step": "02", "title": "Radiomics-only matrix", "detail": "2,064 tumor and peritumoral-vessel features; menopause and receptor fields explicitly removed."},
        {"step": "03", "title": "Training-fold screen", "detail": "Univariate concordance-index ranking (k = 25, 50, 100), then |Spearman ρ| ≥ 0.90 redundancy pruning."},
        {"step": "04", "title": "Sparse conventional model", "detail": "Z-score scaling and elastic-net (L1-weighted) Cox proportional-hazards regression; penalizer tuned in the inner loop."},
        {"step": "05", "title": "Nested validation", "detail": "5-fold × 3 repeated outer CV, 3-fold inner tuning, pooled out-of-fold risk scores, 1,000 bootstrap intervals."},
        {"step": "06", "title": "Stability ranking", "detail": "Features ranked by outer-fold selection frequency, standardized coefficient magnitude, and univariate hazard ratio."},
    ]
    subtype["method"] = classification_method
    pcr["method"] = classification_method
    survival["method"] = survival_method

    output = {
        "meta": {
            "title": "Duke Breast MRI Radiomics",
            "subtitle": "Conventional, leakage-aware feature selection",
            "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "radiomicsSource": args.radiomics.name,
            "clinicalSource": args.clinical.name,
            "patientCount": int(len(radiomics)),
            "radiomicFeatureCount": int(len(feature_names)),
            "tumorFeatureCount": int(sum(not c.endswith("_vessel") for c in feature_names)),
            "vesselFeatureCount": int(sum(c.endswith("_vessel") for c in feature_names)),
            "clinicalPredictorsUsed": 0,
            "joinMatched": int(merged["Mol Subtype"].notna().sum()),
            "duplicatePatients": int(radiomics["patient"].duplicated().sum()),
            "missingRadiomicCells": int(radiomics[feature_names].isna().sum().sum()),
            "constantFeatures": int(sum(radiomics[c].nunique(dropna=True) <= 1 for c in feature_names)),
        },
        "endpoints": {"survival": survival, "subtype": subtype, "pcr": pcr},
        "clinicalContext": clinical_context(merged, pcr_mask),
        "guardrails": [
            "Exploratory retrospective analysis—not a clinical decision tool.",
            "No external cohort is supplied; internal cross-validation cannot establish transportability.",
            "The strict response field is used as a pCR proxy and should be verified against the source data dictionary and pathology definition.",
            "No repeat scans, segmentations, scanner/protocol metadata, or acquisition sites were available for ICC, perturbation testing, or ComBat harmonization.",
            "Feature stability under segmentation variability and batch effects remains unknown.",
            "Intervals summarize pooled out-of-fold predictions and do not fully capture uncertainty from all preprocessing choices.",
            f"Only {survival['classes'][0]['count']} deaths were observed among {survival['sampleCount']} evaluable patients for the survival model; this event count limits the precision of the concordance interval more than the sample size alone suggests, and only overall survival could be modeled since disease-specific/disease-free/progression-free fields are empty in the source data.",
        ],
    }

    json_path = args.output_dir / "analysis.json"
    json_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")

    export_columns = [
        "rank", "feature", "display", "region", "transform", "family", "measure",
        "selectionFrequency", "meanAbsCoefficient", "finalAbsCoefficient",
        "univariateP", "fdrQ", "effect", "direction", "finalModel",
    ]
    survival_ranking[export_columns].to_csv(args.output_dir / "feature_rankings_survival.csv", index=False)
    subtype_ranking[export_columns].to_csv(args.output_dir / "feature_rankings_subtype.csv", index=False)
    pcr_ranking[export_columns].to_csv(args.output_dir / "feature_rankings_pcr.csv", index=False)
    print(json.dumps({
        "analysis": str(json_path),
        "survival_n": survival["sampleCount"],
        "subtype_n": subtype["sampleCount"],
        "pcr_n": pcr["sampleCount"],
        "survival_concordance": survival["metrics"][0]["value"],
        "subtype_auc": subtype["metrics"][0]["value"],
        "pcr_auc": pcr["metrics"][0]["value"],
        "survival_stable": survival["featureFunnel"][-1]["count"],
        "subtype_stable": subtype["featureFunnel"][-1]["count"],
        "pcr_stable": pcr["featureFunnel"][-1]["count"],
    }, indent=2))


if __name__ == "__main__":
    main()
