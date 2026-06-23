# ==========================================================
# PHASE 5
# SELF LEARNING & FEEDBACK ENGINE
# FINAL VERSION
# ==========================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("Loading Dataset...")

df = pd.read_csv(
    "phase1_processed_events.csv"
)

df.columns = df.columns.str.lower()

print("Dataset Loaded")
print(df.shape)

# ==========================================================
# RECREATE MISSING FEATURES
# ==========================================================

if "is_weekend" not in df.columns:

    weekend_days = [
        "Saturday",
        "Sunday"
    ]

    df["is_weekend"] = (
        df["day_name"]
        .astype(str)
        .str.strip()
        .isin(weekend_days)
    ).astype(int)

    print("Created is_weekend")

if "is_peak_hour" not in df.columns:

    peak_hours = [
        8, 9, 10,
        17, 18, 19, 20
    ]

    df["is_peak_hour"] = (
        df["hour"]
        .isin(peak_hours)
    ).astype(int)

    print("Created is_peak_hour")

# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading Model...")

model = joblib.load(
    "event_impact_xgb_model.pkl"
)

print("Model Loaded")

# ==========================================================
# TARGET COLUMN
# ==========================================================

TARGET = "requires_road_closure"

# ==========================================================
# EXACT FEATURES USED IN PHASE 2
# ==========================================================

FEATURES = [

    "event_type",
    "event_cause",
    "zone",
    "junction",
    "priority",
    "hour",
    "day_name",
    "month",
    "latitude",
    "longitude",
    "is_weekend",
    "is_peak_hour"

]

# ==========================================================
# VERIFY FEATURES
# ==========================================================

missing_features = [

    col

    for col in FEATURES

    if col not in df.columns

]

if len(missing_features) > 0:

    print("\nMissing Features:")

    print(missing_features)

    raise ValueError(
        "Dataset is missing required features."
    )

# ==========================================================
# BUILD X AND Y
# ==========================================================

X = df[FEATURES]

y = df[TARGET]

# ==========================================================
# PREDICTIONS
# ==========================================================

print("\nGenerating Predictions...")

pred_probs = model.predict_proba(X)[:, 1]

pred_labels = (
    pred_probs >= 0.5
).astype(int)

# ==========================================================
# PERFORMANCE
# ==========================================================

accuracy = accuracy_score(
    y,
    pred_labels
)

roc_auc = roc_auc_score(
    y,
    pred_probs
)

print("\n")
print("=" * 60)
print("LEARNING REPORT")
print("=" * 60)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y,
        pred_labels
    )
)

print("\nClassification Report")

print(
    classification_report(
        y,
        pred_labels
    )
)

# ==========================================================
# FEEDBACK DATAFRAME
# ==========================================================

feedback_df = df.copy()

feedback_df[
    "predicted_probability"
] = pred_probs

feedback_df[
    "predicted_label"
] = pred_labels

feedback_df[
    "prediction_correct"
] = (
    feedback_df[
        "predicted_label"
    ]
    ==
    feedback_df[
        TARGET
    ]
)

feedback_df[
    "prediction_error"
] = abs(
    feedback_df[
        TARGET
    ]
    -
    feedback_df[
        "predicted_probability"
    ]
)

# ==========================================================
# ZONE LEARNING REPORT
# ==========================================================

print(
    "\nBuilding Zone Learning Report..."
)

zone_report = (

    feedback_df

    .groupby("zone")

    .agg({

        TARGET: "mean",

        "predicted_probability": "mean",

        "prediction_correct": "mean",

        "prediction_error": "mean"

    })

)

zone_report.columns = [

    "actual_closure_rate",

    "avg_predicted_probability",

    "prediction_accuracy",

    "avg_prediction_error"

]

zone_report = zone_report.sort_values(

    by="actual_closure_rate",

    ascending=False

)

zone_report.to_csv(

    "zone_learning_report.csv"

)

print(
    "Saved: zone_learning_report.csv"
)

# ==========================================================
# JUNCTION LEARNING REPORT
# ==========================================================

print(
    "Building Junction Learning Report..."
)

junction_report = (

    feedback_df

    .groupby("junction")

    .agg({

        TARGET: "mean",

        "prediction_correct": "mean",

        "prediction_error": "mean"

    })

)

junction_report.columns = [

    "closure_rate",

    "prediction_accuracy",

    "avg_prediction_error"

]

junction_report = junction_report.sort_values(

    by="closure_rate",

    ascending=False

)

junction_report.to_csv(

    "junction_learning_report.csv"

)

print(
    "Saved: junction_learning_report.csv"
)

# ==========================================================
# UPDATED ZONE RISK SCORES
# ==========================================================

print(
    "Updating Zone Risk Scores..."
)

zone_risk = (

    feedback_df

    .groupby("zone")

    [TARGET]

    .mean()

)

zone_risk = (

    zone_risk * 100

).round(2)

zone_risk = zone_risk.reset_index()

zone_risk.columns = [

    "zone",

    "risk_score"

]

zone_risk.to_csv(

    "updated_zone_risk_scores.csv",

    index=False

)

print(
    "Saved: updated_zone_risk_scores.csv"
)

# ==========================================================
# HIGH RISK JUNCTIONS
# ==========================================================

high_risk = (

    feedback_df

    .groupby("junction")

    [TARGET]

    .mean()

)

high_risk = (

    high_risk * 100

).round(2)

high_risk = high_risk.sort_values(

    ascending=False

)

high_risk.to_csv(

    "high_risk_junctions.csv"

)

print(
    "Saved: high_risk_junctions.csv"
)

# ==========================================================
# RETRAINING DATASET
# ==========================================================

feedback_df.to_csv(

    "model_retraining_dataset.csv",

    index=False

)

print(
    "Saved: model_retraining_dataset.csv"
)

# ==========================================================
# TOP 10 HIGH RISK ZONES
# ==========================================================

print("\n")
print("=" * 60)
print("TOP 10 HIGH RISK ZONES")
print("=" * 60)

print(

    zone_risk

    .sort_values(

        by="risk_score",

        ascending=False

    )

    .head(10)

)

# ==========================================================
# TOP 10 HIGH RISK JUNCTIONS
# ==========================================================

print("\n")
print("=" * 60)
print("TOP 10 HIGH RISK JUNCTIONS")
print("=" * 60)

print(
    high_risk.head(10)
)

print("\n")
print("=" * 60)
print("PHASE 5 COMPLETE")
print("=" * 60)
