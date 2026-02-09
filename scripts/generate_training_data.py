"""
FreeCAD用教師データ生成スクリプト

使用方法（FreeCADのfreecadcmd.exeから実行）:
  freecadcmd.exe generate_training_data.py --samples 5000 --output path/to/output.csv

このスクリプトはproto3-hybridのhybrid_solver.pyをラップし、
コマンドライン引数で出力先やサンプル数を指定できるようにしたものです。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# リポジトリルートを取得
REPO_ROOT = Path(__file__).resolve().parents[1]

# デフォルトパス
DEFAULT_FCSTD = REPO_ROOT / "ref" / "TH1-ref.FCStd"
DEFAULT_DIMS_CSV = REPO_ROOT / "input" / "dims_deg.csv"
DEFAULT_TEMPLATE = REPO_ROOT / "temp" / "constraints.json"
DEFAULT_OUTPUT = REPO_ROOT / "src" / "ai-v0" / "proto3-hybrid_samples_new.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training data for ai-v0 using proto3-hybrid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples", type=int, default=5000,
        help="Total number of samples to generate"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output CSV path (absolute path recommended)"
    )
    parser.add_argument(
        "--fcstd", type=Path, default=DEFAULT_FCSTD,
        help="FreeCAD file path"
    )
    parser.add_argument(
        "--dims", type=Path, default=DEFAULT_DIMS_CSV,
        help="Dimensions CSV file path"
    )
    parser.add_argument(
        "--template", type=Path, default=DEFAULT_TEMPLATE,
        help="Constraint template JSON path"
    )
    parser.add_argument(
        "--ratio-min", type=float, default=0.78,
        help="Minimum ratio for sampling"
    )
    parser.add_argument(
        "--ratio-max", type=float, default=1.22,
        help="Maximum ratio for sampling"
    )
    parser.add_argument(
        "--init-samples", type=int, default=1000,
        help="Initial LHS samples"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="Batch size per iteration"
    )
    parser.add_argument(
        "--iters", type=int, default=10,
        help="Active learning iterations"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 出力パスを絶対パスに変換
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # hybrid_solver.pyへの引数を構築
    hybrid_args = [
        str(args.fcstd.resolve()),
        str(args.dims.resolve()),
        "--template", str(args.template.resolve()),
        "--samples", str(args.samples),
        "--ratio-min", str(args.ratio_min),
        "--ratio-max", str(args.ratio_max),
        "--init-samples", str(args.init_samples),
        "--batch-size", str(args.batch_size),
        "--iters", str(args.iters),
        "--seed", str(args.seed),
        "--out-csv", str(output_path),
        "--simple-frac", "0.3",  # 30% pure random before active learning
        "--explore-frac", "0.4",
        "--narrow-ratio", "0.05",
    ]
    
    # sys.argvを設定してhybrid_solver.pyを実行
    hybrid_script = REPO_ROOT / "src" / "proto3-hybrid" / "hybrid_solver.py"
    sys.argv = [str(hybrid_script)] + hybrid_args
    
    print(f"[generate_training_data] Starting...")
    print(f"  Output: {output_path}")
    print(f"  Samples: {args.samples}")
    print(f"  Ratio range: [{args.ratio_min}, {args.ratio_max}]")
    
    # hybrid_solver.pyを実行
    code = hybrid_script.read_text(encoding="utf-8")
    exec(compile(code, str(hybrid_script), "exec"), {
        "__name__": "__main__",
        "__file__": str(hybrid_script),
    })
    
    print(f"[generate_training_data] Complete. Output: {output_path}")


if __name__ == "__main__":
    main()
