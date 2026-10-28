"""Placement input/output contract and edge-case tests."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import predict


HEADER = [
    "student_id", "department", "city_tier", "cgpa", "backlogs", "internship_months",
    "coding_score", "projects", "workshops", "status_code",
]


class PredictionContractTests(unittest.TestCase):
    def write_csv(self, path: Path, header: list[str], rows: list[list[str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def valid_rows(self) -> list[list[str]]:
        return [
            ["SMOKE-A", "CS", "1", "8.1", "0", "2", "70", "3", "2", "0"],
            ["SMOKE-B", "EE", "2", "", "", "", "", "1", "0", "1"],
        ]

    def test_prediction_schema_order_and_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input.csv", root / "output.csv"
            self.write_csv(source, HEADER, self.valid_rows())
            predict.predict(source, output)
            with output.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["student_id", "placed"])
                records = list(reader)
            self.assertEqual([row["student_id"] for row in records], ["SMOKE-A", "SMOKE-B"])
            self.assertTrue(all(row["placed"] in {"0", "1"} for row in records))

    def test_rejects_missing_columns_and_training_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            self.write_csv(source, HEADER[:-1], [["A", "CS", "1", "8", "0", "1", "50", "1", "0"]])
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                predict.predict(source, root / "out.csv")
            self.write_csv(source, HEADER + ["placed"], [self.valid_rows()[0] + ["1"]])
            with self.assertRaisesRegex(ValueError, "must not contain.*placed"):
                predict.predict(source, root / "out.csv")

    def test_rejects_duplicate_headers_empty_ids_and_same_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            duplicate = HEADER + ["cgpa"]
            self.write_csv(source, duplicate, [self.valid_rows()[0] + ["8"]])
            with self.assertRaisesRegex(ValueError, "duplicate column"):
                predict.predict(source, root / "out.csv")
            self.write_csv(source, HEADER, [["  "] + self.valid_rows()[0][1:]])
            with self.assertRaisesRegex(ValueError, "empty student_id"):
                predict.predict(source, root / "out.csv")
            with self.assertRaisesRegex(ValueError, "must be different"):
                predict.predict(source, source)

    def test_rejects_missing_file_and_empty_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                predict.predict(root / "missing.csv", root / "out.csv")
            empty = root / "empty.csv"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                predict.predict(empty, root / "out.csv")


if __name__ == "__main__":
    unittest.main()
