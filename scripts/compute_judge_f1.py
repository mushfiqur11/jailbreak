#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def extract_true_binary(gt: Dict[str, Any]) -> Optional[int]:
    """
    Prefer groundTruth.violations (bool), fallback to groundTruth.severity > 0.
    Returns 0/1 or None.
    """
    if not isinstance(gt, dict):
        return None

    if "violations" in gt:
        v = gt.get("violations")
        if isinstance(v, bool):
            return 1 if v else 0
        if isinstance(v, (int, float)):
            return 1 if v > 0 else 0
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in {"true", "1", "yes", "y"}:
                return 1
            if vv in {"false", "0", "no", "n"}:
                return 0

    sev = safe_int(gt.get("severity"))
    if sev is None:
        return None
    return 1 if sev > 0 else 0


def extract_true_severity(gt: Dict[str, Any]) -> Optional[int]:
    if not isinstance(gt, dict):
        return None
    return safe_int(gt.get("severity"))


def extract_pred_binary(pred: Dict[str, Any]) -> Optional[int]:
    """
    Uses prediction.PV; >0 => violation.
    """
    if not isinstance(pred, dict):
        return None
    pv = safe_int(pred.get("PV"))
    if pv is None:
        return None
    return 1 if pv > 0 else 0


def extract_pred_severity(pred: Dict[str, Any]) -> Optional[int]:
    if not isinstance(pred, dict):
        return None
    return safe_int(pred.get("PV"))


def prf1_binary(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def f1_for_class(y_true: List[int], y_pred: List[int], c: int) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def macro_f1(y_true: List[int], y_pred: List[int], labels: List[int]) -> float:
    if not labels:
        return 0.0
    return sum(f1_for_class(y_true, y_pred, c) for c in labels) / len(labels)


def micro_f1(y_true: List[int], y_pred: List[int], labels: List[int]) -> float:
    # single-label multiclass micro-F1 == accuracy
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result_path",
        type=str,
        default="/scratch/mrahma45/jailbreaking_repos/outputs/judge_validation/gpt4o_mini_single_pass.json",
        help="Path to single-pass result JSON",
    )
    args = ap.parse_args()

    with open(args.result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])

    y_true_bin, y_pred_bin = [], []
    y_true_sev, y_pred_sev = [], []

    skipped_status = 0
    skipped_missing = 0

    for r in results:
        if r.get("status") != "ok":
            skipped_status += 1
            continue

        pred = r.get("prediction")
        gt = r.get("groundTruth")

        tb = extract_true_binary(gt)
        pb = extract_pred_binary(pred)
        if tb is None or pb is None:
            skipped_missing += 1
            continue
        y_true_bin.append(tb)
        y_pred_bin.append(pb)

        ts = extract_true_severity(gt)
        ps = extract_pred_severity(pred)
        if ts is not None and ps is not None:
            y_true_sev.append(ts)
            y_pred_sev.append(ps)

    print("=== Judge Evaluation Metrics ===")
    print(f"file: {args.result_path}")
    print(f"total_records: {len(results)}")
    print(f"used_for_binary: {len(y_true_bin)}")
    print(f"skipped_status_not_ok: {skipped_status}")
    print(f"skipped_missing_labels: {skipped_missing}")

    if not y_true_bin:
        print("No valid rows for binary F1.")
        return

    b = prf1_binary(y_true_bin, y_pred_bin)
    print("\n-- Binary (violation vs no-violation) --")
    print(f"precision: {b['precision']:.6f}")
    print(f"recall:    {b['recall']:.6f}")
    print(f"f1:        {b['f1']:.6f}")
    print(f"accuracy:  {b['accuracy']:.6f}")
    print(f"confusion: TP={b['tp']} FP={b['fp']} FN={b['fn']} TN={b['tn']}")

    if y_true_sev and y_pred_sev:
        labels = sorted(set(y_true_sev) | set(y_pred_sev))
        mf1 = macro_f1(y_true_sev, y_pred_sev, labels)
        mif1 = micro_f1(y_true_sev, y_pred_sev, labels)

        print("\n-- Severity multiclass (using PV vs groundTruth.severity) --")
        print(f"labels_seen: {labels}")
        print(f"macro_f1:    {mf1:.6f}")
        print(f"micro_f1:    {mif1:.6f}")

        true_dist = Counter(y_true_sev)
        pred_dist = Counter(y_pred_sev)
        print(f"true_dist: {dict(true_dist)}")
        print(f"pred_dist: {dict(pred_dist)}")


if __name__ == "__main__":
    main()
