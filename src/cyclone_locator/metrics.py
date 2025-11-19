import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _clean_scores(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    return y_true[mask], y_score[mask]


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    order = np.argsort(x[mask])
    xs = x[mask][order]
    ys = y[mask][order]
    return float(np.trapz(ys, xs))


def pr_curve(y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 201) -> List[Dict[str, float]]:
    y_true, y_score = _clean_scores(y_true, y_score)
    if len(y_true) == 0:
        return [{"threshold": float("nan"), "precision": float("nan"), "recall": float("nan"), "pr_auc": float("nan")}]
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    total_pos = y_true.sum()
    precision = []
    recall = []
    for tau in thresholds:
        pred = y_score >= tau
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        fn = float(((pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / total_pos if total_pos > 0 else float("nan")
        precision.append(prec)
        recall.append(rec)
    recall_arr = np.asarray(recall, dtype=float)
    precision_arr = np.asarray(precision, dtype=float)
    pr_auc = _auc(recall_arr, precision_arr) if total_pos > 0 else float("nan")
    rows = []
    for tau, prec, rec in zip(thresholds, precision, recall):
        rows.append({
            "threshold": float(tau),
            "precision": float(prec) if math.isfinite(prec) else float("nan"),
            "recall": float(rec) if math.isfinite(rec) else float("nan"),
            "pr_auc": float(pr_auc) if math.isfinite(pr_auc) else float("nan"),
        })
    return rows


def roc_curve(y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 201) -> List[Dict[str, float]]:
    y_true, y_score = _clean_scores(y_true, y_score)
    if len(y_true) == 0:
        return [{"threshold": float("nan"), "tpr": float("nan"), "fpr": float("nan"), "roc_auc": float("nan")}]
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    tpr_vals = []
    fpr_vals = []
    for tau in thresholds:
        pred = y_score >= tau
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        fn = float(((pred == 0) & (y_true == 1)).sum())
        tn = float(((pred == 0) & (y_true == 0)).sum())
        tpr = tp / pos if pos > 0 else float("nan")
        fpr = fp / neg if neg > 0 else float("nan")
        tpr_vals.append(tpr)
        fpr_vals.append(fpr)
    tpr_arr = np.asarray(tpr_vals, dtype=float)
    fpr_arr = np.asarray(fpr_vals, dtype=float)
    roc_auc = _auc(fpr_arr, tpr_arr) if (pos > 0 and neg > 0) else float("nan")
    rows = []
    for tau, tpr, fpr in zip(thresholds, tpr_vals, fpr_vals):
        rows.append({
            "threshold": float(tau),
            "tpr": float(tpr) if math.isfinite(tpr) else float("nan"),
            "fpr": float(fpr) if math.isfinite(fpr) else float("nan"),
            "roc_auc": float(roc_auc) if math.isfinite(roc_auc) else float("nan"),
        })
    return rows


def confusion_at_thresh(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, int]:
    y_true, y_score = _clean_scores(y_true, y_score)
    if len(y_true) == 0:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    pred = y_score >= threshold
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def presence_aggregate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: Optional[float],
    pr_curve_data: List[Dict[str, float]],
    roc_curve_data: List[Dict[str, float]],
) -> Dict[str, float]:
    y_true, y_score = _clean_scores(y_true, y_score)
    if threshold is None:
        threshold = 0.5
    if pr_curve_data:
        auprc = pr_curve_data[0]["pr_auc"]
    else:
        auprc = float("nan")
    if roc_curve_data:
        roc_auc = roc_curve_data[0]["roc_auc"]
    else:
        roc_auc = float("nan")
    cm = confusion_at_thresh(y_true, y_score, threshold)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "auprc": float(auprc),
        "roc_auc": float(roc_auc),
        "threshold": float(threshold),
        "precision_at_tau": float(precision),
        "recall_at_tau": float(recall),
        "f1_at_tau": float(f1),
        "confusion_matrix": cm,
    }


def center_errors_px(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    pred_x = np.asarray(pred_x, dtype=float)
    pred_y = np.asarray(pred_y, dtype=float)
    gt_x = np.asarray(gt_x, dtype=float)
    gt_y = np.asarray(gt_y, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    errors: List[float] = []
    for px, py, gx, gy, keep in zip(pred_x, pred_y, gt_x, gt_y, mask):
        if not keep:
            continue
        if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(gx) and math.isfinite(gy)):
            continue
        dx = px - gx
        dy = py - gy
        errors.append(float(math.hypot(dx, dy)))
    return np.asarray(errors, dtype=float)


def percent_within_thresholds(errors: np.ndarray, thresholds: Optional[Sequence[float]]) -> Dict[str, float]:
    if not thresholds:
        return {}
    errors = np.asarray(errors, dtype=float)
    total = len(errors)
    out: Dict[str, float] = {}
    for thr in thresholds:
        if total == 0:
            out[str(thr)] = float("nan")
            continue
        pct = 100.0 * float((errors <= thr).sum()) / total
        out[str(thr)] = float(pct)
    return out


def center_metrics_summary(
    errors: np.ndarray,
    thresholds: Optional[Sequence[float]],
    unit: str,
    policy: str,
) -> Dict[str, object]:
    errors = np.asarray(errors, dtype=float)
    n = len(errors)
    mae = float(np.mean(errors)) if n else float("nan")
    medae = float(np.median(errors)) if n else float("nan")
    summary = {
        "policy": policy,
        "unit": unit,
        "count": int(n),
        f"mae_{unit}": mae,
        f"medae_{unit}": medae,
        f"pct_within_{unit}": percent_within_thresholds(errors, thresholds),
    }
    return summary
