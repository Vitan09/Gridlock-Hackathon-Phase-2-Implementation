# ==========================================================
# PHASE 3
# SMART RESOURCE RECOMMENDATION ENGINE
# FINAL VERSION
# ==========================================================

import pandas as pd
import numpy as np
import joblib

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

model = joblib.load(
    "event_impact_xgb_model.pkl"
)

print("Model Loaded Successfully")

# ==========================================================
# LOAD HISTORICAL DATA
# ==========================================================

history = pd.read_csv(
    "phase1_processed_events.csv"
)

history.columns = history.columns.str.lower()

# ==========================================================
# BUILD ZONE MULTIPLIERS
# ==========================================================

ZONE_MULTIPLIERS = {}

if (
    "zone" in history.columns
    and
    "requires_road_closure" in history.columns
):

    zone_stats = (

        history
        .groupby("zone")
        ["requires_road_closure"]
        .mean()

    )

    for zone, closure_rate in zone_stats.items():

        multiplier = round(
            1 + closure_rate,
            2
        )

        ZONE_MULTIPLIERS[
            zone
        ] = multiplier

print("\nZone Multipliers")

for zone, value in ZONE_MULTIPLIERS.items():

    print(
        f"{zone}: {value}"
    )

# ==========================================================
# IMPACT LEVEL
# ==========================================================

def get_impact_level(prob):

    if prob < 0.30:
        return "LOW"

    elif prob < 0.60:
        return "MODERATE"

    elif prob < 0.80:
        return "HIGH"

    else:
        return "CRITICAL"

# ==========================================================
# ALERT LEVEL
# ==========================================================

def get_alert_level(prob):

    if prob < 0.30:
        return "GREEN"

    elif prob < 0.60:
        return "YELLOW"

    elif prob < 0.80:
        return "ORANGE"

    else:
        return "RED"

# ==========================================================
# DIVERSION STRATEGY
# ==========================================================

def get_diversion(prob):

    if prob < 0.30:

        return "NO DIVERSION REQUIRED"

    elif prob < 0.60:

        return "OPTIONAL DIVERSION"

    elif prob < 0.80:

        return "RECOMMENDED DIVERSION"

    else:

        return "MANDATORY DIVERSION"

# ==========================================================
# ADVANCED RESOURCE MULTIPLIER
# ==========================================================

def calculate_resource_multiplier(event):

    multiplier = 1.0

    # Priority

    priority = str(
        event.get(
            "priority",
            ""
        )
    ).lower()

    if priority == "high":

        multiplier *= 1.5

    elif priority == "medium":

        multiplier *= 1.2

    # Peak Hour

    if event.get(
        "is_peak_hour",
        0
    ) == 1:

        multiplier *= 1.2

    # Weekend

    if event.get(
        "is_weekend",
        0
    ) == 1:

        multiplier *= 1.1

    # Planned Events

    event_type = str(
        event.get(
            "event_type",
            ""
        )
    ).lower()

    if event_type == "planned":

        multiplier *= 1.3

    return round(
        multiplier,
        2
    )

# ==========================================================
# DYNAMIC RESOURCE ALLOCATION
# ==========================================================

def dynamic_resources(
        probability,
        resource_multiplier):

    base_police = int(
        round(
            2 + (
                probability * 25
            )
        )
    )

    base_barricades = int(
        round(
            1 + (
                probability * 12
            )
        )
    )

    police = int(
        round(
            base_police *
            resource_multiplier
        )
    )

    barricades = int(
        round(
            base_barricades *
            resource_multiplier
        )
    )

    return (

        police,
        barricades

    )

# ==========================================================
# ZONE SCALING
# ==========================================================

def apply_zone_scaling(
        police,
        barricades,
        zone):

    multiplier = (

        ZONE_MULTIPLIERS
        .get(
            zone,
            1.0
        )

    )

    police = int(
        round(
            police *
            multiplier
        )
    )

    barricades = int(
        round(
            barricades *
            multiplier
        )
    )

    return (

        police,
        barricades,
        multiplier

    )

# ==========================================================
# RECOMMENDATION ENGINE
# ==========================================================

def generate_recommendation(
        probability,
        zone,
        event):

    impact = get_impact_level(
        probability
    )

    alert = get_alert_level(
        probability
    )

    diversion = get_diversion(
        probability
    )

    resource_multiplier = (

        calculate_resource_multiplier(
            event
        )

    )

    police, barricades = (

        dynamic_resources(

            probability,

            resource_multiplier

        )

    )

    police, barricades, zone_multiplier = (

        apply_zone_scaling(

            police,

            barricades,

            zone

        )

    )

    return {

        "closure_probability":

            round(
                probability * 100,
                2
            ),

        "impact_level":
            impact,

        "alert_level":
            alert,

        "resource_multiplier":
            resource_multiplier,

        "zone_multiplier":
            zone_multiplier,

        "police_required":
            police,

        "barricades_required":
            barricades,

        "diversion_plan":
            diversion
    }

# ==========================================================
# SINGLE EVENT PREDICTION
# ==========================================================

def predict_event(event):

    event_df = pd.DataFrame(
        [event]
    )

    probability = (

        model
        .predict_proba(
            event_df
        )[0][1]

    )

    recommendation = (

        generate_recommendation(

            probability,

            event["zone"],

            event

        )

    )

    return recommendation

# ==========================================================
# SAMPLE EVENT
# ==========================================================

sample_event = {

    "event_type":
        "planned",

    "event_cause":
        "public_event",

    "zone":
        "Central Zone 2",

    "junction":
        "MekhriCircle",

    "priority":
        "High",

    "hour":
        18,

    "day_name":
        "Saturday",

    "month":
        "October",

    "latitude":
        12.998,

    "longitude":
        77.592,

    "is_weekend":
        1,

    "is_peak_hour":
        1
}

# ==========================================================
# TEST
# ==========================================================

result = predict_event(
    sample_event
)

print("\n")
print("=" * 60)
print("SMART RESOURCE RECOMMENDATION")
print("=" * 60)

for k, v in result.items():

    print(
        f"{k}: {v}"
    )

# ==========================================================
# BATCH PREDICTION
# ==========================================================

def batch_predict(
        input_csv,
        output_csv):

    df = pd.read_csv(
        input_csv
    )

    probabilities = (

        model
        .predict_proba(
            df
        )[:, 1]

    )

    recommendations = []

    for idx, prob in enumerate(
            probabilities):

        event = (
            df
            .iloc[idx]
            .to_dict()
        )

        zone = event["zone"]

        recommendation = (

            generate_recommendation(

                prob,

                zone,

                event

            )

        )

        recommendations.append(
            recommendation
        )

    rec_df = pd.DataFrame(
        recommendations
    )

    final_df = pd.concat(

        [

            df.reset_index(
                drop=True
            ),

            rec_df

        ],

        axis=1

    )

    final_df.to_csv(

        output_csv,

        index=False

    )

    print(
        f"\nSaved: {output_csv}"
    )

# ==========================================================
# EXAMPLE
# ==========================================================

"""
batch_predict(

    input_csv=
    "future_events.csv",

    output_csv=
    "resource_recommendations.csv"

)
"""
