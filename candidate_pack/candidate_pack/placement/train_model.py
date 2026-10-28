"""Reproduce the holdout comparison and save the final placement pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = Path(__file__).resolve().parent
SEED = 42
FEATURES = ["department", "city_tier", "cgpa", "backlogs", "internship_months", "coding_score", "projects", "workshops"]


def make_pipeline(estimator: object) -> Pipeline:
    numeric = ["city_tier", "cgpa", "backlogs", "internship_months", "coding_score", "projects", "workshops"]
    categorical = ["department"]
    prep = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    return Pipeline([("preprocess", prep), ("classifier", estimator)])


def main() -> None:
    data = pd.read_csv(HERE / "data" / "train.csv")
    y = data["placed"].astype(int)
    X = data[FEATURES]
    train_idx, valid_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    candidates = {
        "logistic_regression_c0_3": LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced", random_state=SEED),
        "logistic_regression_c1": LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample", random_state=SEED, n_jobs=1),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }
    fitted: dict[str, Pipeline] = {}
    scores: dict[str, dict[str, float]] = {}
    thresholds = np.round(np.arange(0.10, 0.901, 0.01), 2)
    for name, estimator in candidates.items():
        model = make_pipeline(estimator)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_valid)[:, 1]
        best_threshold = max(thresholds, key=lambda threshold: f1_score(y_valid, probabilities >= threshold, zero_division=0))
        scores[name] = {
            "f1_at_0_5": float(f1_score(y_valid, probabilities >= 0.5, zero_division=0)),
            "accuracy_at_0_5": float(accuracy_score(y_valid, probabilities >= 0.5)),
            "best_threshold": float(best_threshold),
            "best_f1": float(f1_score(y_valid, probabilities >= best_threshold, zero_division=0)),
            "accuracy_at_best_f1_threshold": float(accuracy_score(y_valid, probabilities >= best_threshold)),
            "precision_at_best": float(precision_score(y_valid, probabilities >= best_threshold, zero_division=0)),
            "recall_at_best": float(recall_score(y_valid, probabilities >= best_threshold, zero_division=0)),
        }
        fitted[name] = model
    # Check whether the slight holdout improvement survives nested threshold tuning.
    # The outer validation folds never participate in fitting or choosing a threshold.
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    cross_validation: dict[str, dict[str, object]] = {}
    for name in ("logistic_regression_c0_3", "logistic_regression_c1"):
        tuned = TunedThresholdClassifierCV(
            estimator=make_pipeline(candidates[name]),
            scoring="f1",
            cv=inner_cv,
            thresholds=101,
        )
        cv_result = cross_validate(
            tuned,
            X,
            y,
            cv=outer_cv,
            scoring={"f1": "f1", "accuracy": "accuracy"},
            return_estimator=True,
            n_jobs=1,
        )
        cross_validation[name] = {
            "method": "nested 5-fold evaluation; threshold tuned in inner 3-fold CV",
            "fold_f1": [float(value) for value in cv_result["test_f1"]],
            "mean_f1": float(np.mean(cv_result["test_f1"])),
            "std_f1": float(np.std(cv_result["test_f1"])),
            "fold_accuracy_at_f1_tuned_threshold": [float(value) for value in cv_result["test_accuracy"]],
            "mean_accuracy_at_f1_tuned_threshold": float(np.mean(cv_result["test_accuracy"])),
            "std_accuracy_at_f1_tuned_threshold": float(np.std(cv_result["test_accuracy"])),
            "inner_selected_thresholds": [float(model.best_threshold_) for model in cv_result["estimator"]],
        }
    # Estimate the accuracy/F1 tradeoff when threshold selection itself targets accuracy.
    # This uses the same nested protocol so the outer fold remains untouched.
    accuracy_tuned = TunedThresholdClassifierCV(
        estimator=make_pipeline(candidates["logistic_regression_c0_3"]),
        scoring="accuracy",
        cv=inner_cv,
        thresholds=101,
    )
    accuracy_cv = cross_validate(
        accuracy_tuned,
        X,
        y,
        cv=outer_cv,
        scoring={"f1": "f1", "accuracy": "accuracy"},
        return_estimator=True,
        n_jobs=1,
    )
    # Choose the strongest fixed-holdout candidate and retain nested-CV evidence
    # so the small gain is not presented as a guarantee of future-cohort lift.
    selected = max(scores, key=lambda name: scores[name]["best_f1"])
    threshold = scores[selected]["best_threshold"]
    final_model = make_pipeline(candidates[selected]).fit(X, y)
    joblib.dump(final_model, HERE / "placement_model.joblib")
    config = {
        "features": FEATURES,
        "threshold": threshold,
        "random_seed": SEED,
        "validation": {
            "method": "stratified 80/20 holdout; decision threshold selected on this validation split",
            "test_size": 0.2,
            "seed": SEED,
            "majority_class_accuracy_baseline": float(max(y.mean(), 1 - y.mean())),
        },
        "selected_model": selected,
        "model_parameters": {
            "logistic_regression_c0_3": {"C": 0.3, "class_weight": "balanced"},
            "logistic_regression_c1": {"C": 1.0, "class_weight": "balanced"},
        },
        "validation_results": scores,
        "nested_cross_validation": cross_validation,
        "nested_accuracy_threshold_comparison": {
            "model": "logistic_regression_c0_3",
            "method": "nested 5-fold evaluation; accuracy threshold tuned in inner 3-fold CV",
            "fold_accuracy": [float(value) for value in accuracy_cv["test_accuracy"]],
            "mean_accuracy": float(np.mean(accuracy_cv["test_accuracy"])),
            "std_accuracy": float(np.std(accuracy_cv["test_accuracy"])),
            "fold_f1": [float(value) for value in accuracy_cv["test_f1"]],
            "mean_f1": float(np.mean(accuracy_cv["test_f1"])),
            "std_f1": float(np.std(accuracy_cv["test_f1"])),
            "inner_selected_thresholds": [float(model.best_threshold_) for model in accuracy_cv["estimator"]],
        },
        "training_rows": int(len(data)),
        "excluded_features": {
            "student_id": "Identifier with no defensible predictive meaning; excluded to avoid memorization.",
            "status_code": "Undocumented portal field; status=1 and status=0 placement rates are 92.1% and 2.0%, suggesting potential outcome leakage.",
        },
    }
    (HERE / "model_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
