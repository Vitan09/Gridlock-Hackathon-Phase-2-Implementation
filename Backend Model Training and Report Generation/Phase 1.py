# ==========================================
# PHASE 1: EVENT ANALYSIS & IMPACT ASSESSMENT
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# Load Dataset
# ------------------------------------------

file_path = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows")
print(df.head())

# ------------------------------------------
# Data Cleaning
# ------------------------------------------

df.columns = df.columns.str.strip().str.lower()

# Remove duplicate rows
df = df.drop_duplicates()

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# ------------------------------------------
# Detect Important Columns
# ------------------------------------------

event_col = None
zone_col = None
junction_col = None
road_col = None
time_col = None

for col in df.columns:

    if 'cause' in col or 'event' in col:
        event_col = col

    if 'zone' in col:
        zone_col = col

    if 'junction' in col:
        junction_col = col

    if 'closure' in col or 'road' in col:
        road_col = col

    if 'time' in col or 'date' in col:
        time_col = col

print("\nDetected Columns")
print("Event:", event_col)
print("Zone:", zone_col)
print("Junction:", junction_col)
print("Road:", road_col)
print("Time:", time_col)

# ------------------------------------------
# Datetime Features
# ------------------------------------------

if time_col:

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    df['hour'] = df[time_col].dt.hour
    df['day_name'] = df[time_col].dt.day_name()
    df['month'] = df[time_col].dt.month_name()

# ------------------------------------------
# Event Type Analysis
# ------------------------------------------

if event_col:

    plt.figure(figsize=(12,6))

    event_counts = df[event_col].value_counts().head(15)

    sns.barplot(
        x=event_counts.values,
        y=event_counts.index
    )

    plt.title("Top Event Types")
    plt.xlabel("Count")
    plt.ylabel("Event Type")
    plt.show()

# ------------------------------------------
# Zone Analysis
# ------------------------------------------

if zone_col:

    plt.figure(figsize=(12,6))

    zone_counts = df[zone_col].value_counts().head(15)

    sns.barplot(
        x=zone_counts.values,
        y=zone_counts.index
    )

    plt.title("Events by Zone")
    plt.xlabel("Count")
    plt.ylabel("Zone")
    plt.show()

# ------------------------------------------
# Junction Analysis
# ------------------------------------------

if junction_col:

    plt.figure(figsize=(12,6))

    junction_counts = df[junction_col].value_counts().head(15)

    sns.barplot(
        x=junction_counts.values,
        y=junction_counts.index
    )

    plt.title("Top Affected Junctions")
    plt.xlabel("Count")
    plt.ylabel("Junction")
    plt.show()

# ------------------------------------------
# Hourly Event Distribution
# ------------------------------------------

if 'hour' in df.columns:

    plt.figure(figsize=(12,6))

    sns.countplot(
        data=df,
        x='hour'
    )

    plt.title("Events by Hour")
    plt.show()

# ------------------------------------------
# Day-wise Analysis
# ------------------------------------------

if 'day_name' in df.columns:

    order = [
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday',
        'Sunday'
    ]

    plt.figure(figsize=(10,5))

    sns.countplot(
        data=df,
        x='day_name',
        order=order
    )

    plt.title("Events by Day")
    plt.xticks(rotation=30)
    plt.show()

# ------------------------------------------
# Road Closure Analysis
# ------------------------------------------

if road_col:

    plt.figure(figsize=(8,5))

    sns.countplot(
        data=df,
        x=road_col
    )

    plt.title("Road Closure Distribution")
    plt.xticks(rotation=20)
    plt.show()

# ------------------------------------------
# Event vs Zone Heatmap
# ------------------------------------------

if event_col and zone_col:

    top_events = df[event_col].value_counts().head(10).index

    heat_data = pd.crosstab(
        df[df[event_col].isin(top_events)][event_col],
        df[df[event_col].isin(top_events)][zone_col]
    )

    plt.figure(figsize=(14,8))

    sns.heatmap(
        heat_data,
        cmap='YlOrRd'
    )

    plt.title("Event Type vs Zone Heatmap")
    plt.show()

# ------------------------------------------
# Congestion Risk Score (Simple Rule-Based)
# ------------------------------------------

risk_score = []

for _, row in df.iterrows():

    score = 0

    if event_col:

        event_text = str(row[event_col]).lower()

        if 'rally' in event_text:
            score += 5

        elif 'festival' in event_text:
            score += 4

        elif 'vip' in event_text:
            score += 3

        elif 'construction' in event_text:
            score += 2

    if road_col:

        road_text = str(row[road_col]).lower()

        if 'yes' in road_text:
            score += 5

    risk_score.append(score)

df['impact_score'] = risk_score

# ------------------------------------------
# Risk Category
# ------------------------------------------

conditions = [
    df['impact_score'] <= 2,
    (df['impact_score'] > 2) & (df['impact_score'] <= 5),
    (df['impact_score'] > 5) & (df['impact_score'] <= 8),
    (df['impact_score'] > 8)
]

labels = [
    'Low',
    'Moderate',
    'High',
    'Critical'
]

df['impact_category'] = np.select(
    conditions,
    labels,
    default='Low'
)

print("\nImpact Category Distribution")
print(df['impact_category'].value_counts())

# ------------------------------------------
# Save Processed Dataset
# ------------------------------------------

df.to_csv(
    "phase1_processed_events.csv",
    index=False
)

print("\nSaved:")
print("phase1_processed_events.csv")
