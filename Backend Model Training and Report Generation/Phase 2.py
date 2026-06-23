# =====================================================
# PHASE 2.1
# EVENT IMPACT PREDICTION (ROAD CLOSURE FORECASTING)
# XGBOOST + SMOTE
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("phase1_processed_events.csv")

print("="*60)
print("DATASET SHAPE")
print(df.shape)
print("="*60)

# =====================================================
# COLUMN CLEANING
# =====================================================

df.columns = df.columns.str.strip().str.lower()

# =====================================================
# TARGET COLUMN
# =====================================================

target = "requires_road_closure"

if target not in df.columns:
    raise ValueError(
        f"{target} column not found.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )

# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Weekend Feature

if "day_name" in df.columns:

    df["is_weekend"] = df["day_name"].isin(
        ["Saturday", "Sunday"]
    ).astype(int)

# Peak Hour Feature

if "hour" in df.columns:

    df["is_peak_hour"] = np.where(
        (
            ((df["hour"] >= 7) & (df["hour"] <= 10))
            |
            ((df["hour"] >= 17) & (df["hour"] <= 20))
        ),
        1,
        0
    )

# =====================================================
# SELECT FEATURES
# =====================================================

candidate_features = [

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

features = [
    col for col in candidate_features
    if col in df.columns
]

print("\nFeatures Used:")
print(features)

# =====================================================
# DATASET
# =====================================================

X = df[features]
y = df[target]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# FEATURE TYPES
# =====================================================

categorical_features = [
    col for col in features
    if X[col].dtype == "object"
]

numeric_features = [
    col for col in features
    if col not in categorical_features
]

# =====================================================
# PREPROCESSING
# =====================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            categorical_transformer,
            categorical_features
        ),
        (
            "num",
            numeric_transformer,
            numeric_features
        )
    ]
)

# =====================================================
# CLASS IMBALANCE RATIO
# =====================================================

negative = sum(y_train == 0)
positive = sum(y_train == 1)

scale_pos_weight = negative / positive

print("\nNegative:", negative)
print("Positive:", positive)
print("Scale Pos Weight:", scale_pos_weight)

# =====================================================
# XGBOOST MODEL
# =====================================================

xgb_model = XGBClassifier(

    n_estimators=500,

    max_depth=8,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1
)

# =====================================================
# PIPELINE
# =====================================================

pipeline = ImbPipeline(
    steps=[

        ("preprocessor", preprocessor),

        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),

        ("classifier", xgb_model)
    ]
)

# =====================================================
# TRAIN MODEL
# =====================================================

print("\nTraining XGBoost...")

pipeline.fit(X_train, y_train)

print("Training Complete!")

# =====================================================
# PREDICTIONS
# =====================================================

predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)[:, 1]

# =====================================================
# EVALUATION
# =====================================================

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

accuracy = accuracy_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print("\nAccuracy:")
print(round(accuracy, 4))

print("\nROC-AUC:")
print(round(roc_auc, 4))

print("\nConfusion Matrix:")
cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

# =====================================================
# CONFUSION MATRIX PLOT
# =====================================================

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =====================================================
# ROC CURVE
# =====================================================

RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("ROC Curve")
plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

print("\nGenerating Feature Importance...")

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = pipeline.named_steps[
    "classifier"
].feature_importances_

feature_importance = pd.DataFrame({

    "Feature": feature_names,
    "Importance": importance

})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 20 Important Features")

print(
    feature_importance.head(20)
)

plt.figure(figsize=(10,8))

sns.barplot(

    data=feature_importance.head(20),

    x="Importance",

    y="Feature"
)

plt.title("Top 20 Important Features")

plt.tight_layout()

plt.show()

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    pipeline,
    "event_impact_xgb_model.pkl"
)

print("\nModel Saved Successfully")
print("File: event_impact_xgb_model.pkl")

# =====================================================
# SAVE FEATURE IMPORTANCE
# =====================================================

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nFeature Importance Saved")
print("File: feature_importance.csv")
