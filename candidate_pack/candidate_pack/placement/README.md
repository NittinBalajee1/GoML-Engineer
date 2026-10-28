# Placement prediction

The fitted sklearn pipeline is `placement_model.joblib`; `model_config.json` stores its feature list, threshold, seed, and validation results. `train_model.py` reproduces the fixed validation comparison and saves the final refit. `analysis.ipynb` contains the EDA and modelling record.

## Setup and run

From this directory in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python predict.py --input some_students.csv --output predictions.csv
python -m notebook analysis.ipynb
```

`predict.py` loads the saved pipeline without training. Input must contain all ten documented feature columns and no `placed` column. Output is exactly `student_id,placed`, in input row order.

## Validation and feature decisions

The supplied file has 2,500 rows, 11 columns, 702 placed (28.08%), no duplicate records, and no duplicate student IDs. Missing counts: `cgpa` 118, `backlogs` 168, `coding_score` 525, `status_code` 113. A stratified 80/20 holdout with seed 42 compared logistic regression, random forest, extra trees, and gradient boosting. Numeric medians, scaling, and categorical encoding are fitted within each candidate pipeline.

On the fixed holdout, the selected logistic-regression pipeline (C=0.3, balanced class weights) scores positive-class F1 0.5786 at threshold 0.59, with precision and recall both 0.5786. Its accuracy on that same thresholded holdout is 0.764, versus a 0.719 majority-class baseline. This holdout threshold was selected using the holdout labels, so the holdout scores are model-selection estimates and may be optimistic. Nested 5-fold CV with threshold tuning inside each training fold estimates mean F1 0.5667 and mean accuracy 0.6812 for the F1-focused decision rule. Tuning the threshold for accuracy instead estimates mean accuracy 0.7516 but lowers mean F1 to 0.4000. This is a material precision/recall/accuracy tradeoff, not a free improvement. Logistic regression with C=1.0 scores holdout F1 0.5765; random forest 0.5538, extra trees 0.5547, and gradient boosting 0.5383. The complete results are recorded in `model_config.json`. The selected pipeline is refit on all 2,500 labeled rows. Adding `student_id` changes an earlier holdout result to about 0.5806, a small gain that is not a sound reason to use an arbitrary identifier. These estimates may vary on the separate evaluation set.

`status_code` is documented only as a portal field, without its definition or availability time. In this dataset the placement rate is about 92.1% when it is 1 and 2.0% when it is 0; including it produces approximately 0.92 F1 and 0.96 accuracy in five-fold out-of-fold experiments. That gain is suspicious and consistent with possible outcome leakage. It is excluded unless its meaning and pre-outcome availability are established. `student_id` is an identifier without documented predictive meaning and is excluded to avoid memorizing it.

## Limitations

The data is small and contains no evaluation labels. No conclusion can be made about performance on future cohorts. The score intentionally avoids the undocumented high-signal field; confirm `status_code` timing with the data owner before considering it for use.
