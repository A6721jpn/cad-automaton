from __future__ import annotations

"""
Proto3-hybrid が出力する学習データCSVの品質チェック用スクリプト。
FreeCAD など依存ライブラリは conda 環境 (fcad / fcad-codex / b123d) に
インストール済みであることを前提とします。
"""

import argparse
import csv
import math
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
    parser = argparse.ArgumentParser(
        description=(
            "Proto3-hybrid が生成した学習データの品質を軽量チェックします。"
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("temp/proto3-hybrid_samples.csv"),
        help="proto3-hybrid が出力する学習データCSV (デフォルト: temp/proto3-hybrid_samples.csv)",
    )
    parser.add_argument("--safe-col", type=str, default="safe", help="安全ラベル列名")
    parser.add_argument("--min-rows", type=int, default=50, help="最低行数")
    parser.add_argument(
        "--min-safe-ratio",
        type=float,
        default=0.05,
        help="安全ラベルの最低比率 (0-1)",
    )
    parser.add_argument(
        "--max-duplicate-ratio",
        type=float,
        default=0.2,
        help="許容する重複率 (0-1)",
    )
    parser.add_argument(
        "--min-feature-std",
        type=float,
        default=1e-6,
        help="各特徴量の最低標準偏差 (小さすぎると変動が不足)",
    )
    return parser.parse_args(argv)


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


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

    summary = {
        "rows": float(row_count),
        "safe_ratio": safe_ratio,
        "duplicate_ratio": duplicate_ratio,
        "features": float(len(feature_names)),
    }
    return QualityResult(ok=not issues, issues=issues, summary=summary)


def main() -> None:
    args = _parse_args(None)
    feature_names, features, labels = _load_csv(args.csv, args.safe_col)
    result = _analyze(
        feature_names,
        features,
        labels,
        args.min_rows,
        args.min_safe_ratio,
        args.max_duplicate_ratio,
        args.min_feature_std,
    )

    print("Proto3-hybrid data quality report")
    print(f"Rows: {int(result.summary['rows'])}")
    print(f"Features: {int(result.summary['features'])}")
    print(f"Safe ratio: {result.summary['safe_ratio']:.3f}")
    print(f"Duplicate ratio: {result.summary['duplicate_ratio']:.3f}")
    if result.ok:
        print("Result: OK")
    else:
        print("Result: NG")
        for issue in result.issues:
            print(f"- {issue}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
