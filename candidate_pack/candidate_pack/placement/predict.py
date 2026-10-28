"""Score registered students with the fitted placement pipeline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib
import pandas as pd


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "placement_model.joblib"
CONFIG_PATH = HERE / "model_config.json"
REQUIRED_COLUMNS = [
    "student_id", "department", "city_tier", "cgpa", "backlogs",
    "internship_months", "coding_score", "projects", "workshops", "status_code",
]


def predict(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different; refusing to overwrite the input CSV.")
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    repeated = sorted({name for name in header if header.count(name) > 1})
    if repeated:
        raise ValueError(f"Input CSV contains duplicate column names: {', '.join(repeated)}")
    frame = pd.read_csv(input_path, dtype={"student_id": "string"})
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Input CSV contains duplicate column names: {', '.join(map(str, duplicate_columns))}")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
    if "placed" in frame.columns:
        raise ValueError("Input must not contain the training target column 'placed'.")
    if frame.empty:
        raise ValueError("Input CSV contains no student rows.")
    if frame["student_id"].isna().any() or frame["student_id"].str.strip().eq("").any():
        raise ValueError("Input contains an empty student_id.")

    config = __import__("json").loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)
    features = config["features"]
    probabilities = model.predict_proba(frame[features])[:, 1]
    predictions = (probabilities >= float(config["threshold"])).astype("int64")
    result = pd.DataFrame({"student_id": frame["student_id"], "placed": predictions})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input student CSV")
    parser.add_argument("--output", required=True, type=Path, help="Output predictions CSV")
    args = parser.parse_args()
    try:
        predict(args.input, args.output)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"predict.py: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
