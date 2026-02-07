from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class QualityResult:
    ok: bool
    issues: List[str]
    summary: Dict[str, float]


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proto3-hybrid学習データCSVの品質をチェックします")
    parser.add_argument("--csv", type=Path, default=Path("temp/proto3-hybrid_samples.csv"))
    parser.add_argument("--safe-col", type=str, default="safe")
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--min-safe-ratio", type=float, default=0.05)
    parser.add_argument("--max-duplicate-ratio", type=float, default=0.2)
    parser.add_argument("--min-feature-std", type=float, default=1e-6)
    parser.add_argument("--max-outlier-ratio", type=float, default=0.05)
    parser.add_argument("--max-corr", type=float, default=0.999)
    parser.add_argument("--split-ratios", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--max-cross-split-duplicate-ratio", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args(argv)


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _quantile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return sorted_values[low]
    weight = pos - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _pearson_corr(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    den = math.sqrt(den_x * den_y)
    if den == 0:
        return 0.0
    return num / den


def _load_csv(path: Path, safe_col: str) -> Tuple[List[str], List[List[float]], List[int]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV header is missing")
        if safe_col not in reader.fieldnames:
            raise ValueError(f"safe column '{safe_col}' is missing")
        feature_names = [name for name in reader.fieldnames if name != safe_col]
        if not feature_names:
            raise ValueError("no feature columns found")
        features: List[List[float]] = []
        labels: List[int] = []
        for row_index, row in enumerate(reader, start=2):
            feature_row: List[float] = []
            for name in feature_names:
                raw = (row.get(name) or "").strip()
                if raw == "":
                    raise ValueError(f"missing value in '{name}' at line {row_index}")
                value = float(raw)
                if not _is_finite(value):
                    raise ValueError(f"non-finite value in '{name}' at line {row_index}")
                feature_row.append(value)
            raw_label = (row.get(safe_col) or "").strip()
            if raw_label == "":
                raise ValueError(f"missing label in '{safe_col}' at line {row_index}")
            label = int(float(raw_label))
            if label not in (0, 1):
                raise ValueError(f"label must be 0/1 in '{safe_col}' at line {row_index}")
            features.append(feature_row)
            labels.append(label)
    return feature_names, features, labels


def _analyze(
    feature_names: List[str],
    features: List[List[float]],
    labels: List[int],
    min_rows: int,
    min_safe_ratio: float,
    max_duplicate_ratio: float,
    min_feature_std: float,
    max_outlier_ratio: float,
    max_corr: float,
    split_ratios: Tuple[float, float, float],
    max_cross_split_duplicate_ratio: float,
    split_seed: int,
) -> QualityResult:
    issues: List[str] = []
    row_count = len(features)
    if row_count < min_rows:
        issues.append(f"rows {row_count} < min_rows {min_rows}")

    label_counts = Counter(labels)
    safe_ratio = label_counts.get(1, 0) / max(row_count, 1)
    if safe_ratio < min_safe_ratio:
        issues.append(f"safe ratio {safe_ratio:.3f} < min_safe_ratio {min_safe_ratio:.3f}")

    tuples = [tuple(row) for row in features]
    unique_count = len(set(tuples))
    duplicate_ratio = 1.0 - unique_count / max(row_count, 1)
    if duplicate_ratio > max_duplicate_ratio:
        issues.append(
            f"duplicate ratio {duplicate_ratio:.3f} > max_duplicate_ratio {max_duplicate_ratio:.3f}"
        )

    for idx, name in enumerate(feature_names):
        col_values = [row[idx] for row in features]
        col_std = _std(col_values)
        if col_std < min_feature_std:
            issues.append(
                f"feature '{name}' std {col_std:.6g} < min_feature_std {min_feature_std:.6g}"
            )

    outlier_rows = 0
    if row_count > 0:
        lower_bounds: List[float] = []
        upper_bounds: List[float] = []
        for col_idx in range(len(feature_names)):
            col = sorted(row[col_idx] for row in features)
            q1 = _quantile(col, 0.25)
            q3 = _quantile(col, 0.75)
            iqr = q3 - q1
            lower_bounds.append(q1 - 1.5 * iqr)
            upper_bounds.append(q3 + 1.5 * iqr)
        for row in features:
            if any(v < lo or v > hi for v, lo, hi in zip(row, lower_bounds, upper_bounds)):
                outlier_rows += 1
    outlier_ratio = outlier_rows / max(row_count, 1)
    if outlier_ratio > max_outlier_ratio:
        issues.append(f"outlier ratio {outlier_ratio:.3f} > max_outlier_ratio {max_outlier_ratio:.3f}")

    columns: List[List[float]] = [[row[i] for row in features] for i in range(len(feature_names))]
    max_abs_corr = 0.0
    high_corr_pairs = 0
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            abs_corr = abs(_pearson_corr(columns[i], columns[j]))
            if abs_corr > max_abs_corr:
                max_abs_corr = abs_corr
            if abs_corr > max_corr:
                high_corr_pairs += 1
    if high_corr_pairs > 0:
        issues.append(f"high-correlation pairs: {high_corr_pairs} (>|{max_corr:.3f}|)")

    train_ratio, val_ratio, test_ratio = split_ratios
    total = train_ratio + val_ratio + test_ratio
    cross_split_duplicate_ratio = 0.0
    if min(train_ratio, val_ratio, test_ratio) < 0:
        issues.append("split ratios must be non-negative")
    elif total <= 0:
        issues.append("split ratios total must be > 0")
    elif row_count > 0:
        indices = list(range(row_count))
        rnd = random.Random(split_seed)
        rnd.shuffle(indices)
        train_end = int(row_count * (train_ratio / total))
        val_end = train_end + int(row_count * (val_ratio / total))
        train_set = {tuple(features[i]) for i in indices[:train_end]}
        val_set = {tuple(features[i]) for i in indices[train_end:val_end]}
        test_set = {tuple(features[i]) for i in indices[val_end:]}
        leaked = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
        cross_split_duplicate_ratio = len(leaked) / max(unique_count, 1)
        if cross_split_duplicate_ratio > max_cross_split_duplicate_ratio:
            issues.append(
                f"cross-split duplicate ratio {cross_split_duplicate_ratio:.3f} > "
                f"max_cross_split_duplicate_ratio {max_cross_split_duplicate_ratio:.3f}"
            )

    summary = {
        "rows": float(row_count),
        "safe_ratio": safe_ratio,
        "duplicate_ratio": duplicate_ratio,
        "features": float(len(feature_names)),
        "outlier_ratio": outlier_ratio,
        "max_abs_corr": max_abs_corr,
        "high_corr_pairs": float(high_corr_pairs),
        "cross_split_duplicate_ratio": cross_split_duplicate_ratio,
    }
    return QualityResult(ok=not issues, issues=issues, summary=summary)


def main() -> None:
    args = _parse_args(None)
    split_parts = [p.strip() for p in str(args.split_ratios).split(",") if p.strip()]
    if len(split_parts) != 3:
        raise ValueError("--split-ratios must have exactly 3 comma-separated values")
    split_ratios = tuple(float(p) for p in split_parts)

    feature_names, features, labels = _load_csv(args.csv, args.safe_col)
    result = _analyze(
        feature_names,
        features,
        labels,
        args.min_rows,
        args.min_safe_ratio,
        args.max_duplicate_ratio,
        args.min_feature_std,
        args.max_outlier_ratio,
        args.max_corr,
        split_ratios,
        args.max_cross_split_duplicate_ratio,
        args.split_seed,
    )

    print("Proto3-hybrid data quality report")
    print(f"Rows: {int(result.summary['rows'])}")
    print(f"Features: {int(result.summary['features'])}")
    print(f"Safe ratio: {result.summary['safe_ratio']:.3f}")
    print(f"Duplicate ratio: {result.summary['duplicate_ratio']:.3f}")
    print(f"Outlier ratio: {result.summary['outlier_ratio']:.3f}")
    print(f"Max abs corr: {result.summary['max_abs_corr']:.3f}")
    print(f"High corr pairs: {int(result.summary['high_corr_pairs'])}")
    print(f"Cross-split duplicate ratio: {result.summary['cross_split_duplicate_ratio']:.3f}")

    if result.ok:
        print("Result: OK")
    else:
        print("Result: NG")
        for issue in result.issues:
            print(f"- {issue}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
