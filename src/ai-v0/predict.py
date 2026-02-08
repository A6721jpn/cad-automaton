import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_ARTIFACTS_DIR = Path(__file__).with_name("artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict safe probability with v0 model.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV containing feature rows.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory containing model artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Output CSV path for predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.artifacts / "mlp_model.joblib")
    scaler = joblib.load(args.artifacts / "scaler.joblib")

    with (args.artifacts / "feature_columns.json").open("r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    data = pd.read_csv(args.input)
    features = data[feature_columns]
    features_scaled = scaler.transform(features)

    probas = model.predict_proba(features_scaled)[:, 1]
    preds = (probas >= 0.5).astype(int)

    output = data.copy()
    output["safe_proba"] = probas
    output["safe_pred"] = preds
    output.to_csv(args.output, index=False)

    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
