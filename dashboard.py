import streamlit as st
import pandas as pd
import joblib
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Gridlock AI",
    page_icon="🚦",
    layout="wide"
)

st.markdown("""
<div style='text-align:center'>
<b>Gridlock Hackathon 2026</b><br>
Smart Event Congestion Forecasting & Resource Planning
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD FILES
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "phase1_processed_events.csv"
    )

    return df

@st.cache_resource
def load_model():

    return joblib.load(
        "event_impact_xgb_model.pkl"
    )

df = load_data()
model = load_model()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🚦 Gridlock AI")

page = st.sidebar.radio(

    "Navigation",

    [

        "Command Center",
        "Impact Prediction",
        "Resource Planning",
        "Diversion Routing",
        "Learning Dashboard"

    ]

)

# =====================================================
# COMMAND CENTER
# =====================================================

if page == "Command Center":

    st.title(
        "🚦 Smart Traffic Command Center"
    )

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "Total Events",
        len(df)
    )

    col2.metric(
        "Junctions",
        df["junction"].nunique()
    )

    col3.metric(
        "Zones",
        df["zone"].nunique()
    )

    col4.metric(
        "Road Closures",
        int(
            df[
                "requires_road_closure"
            ].sum()
        )
    )

    st.markdown("---")

    zone_counts = (
        df["zone"]
        .value_counts()
        .reset_index()
    )

    zone_counts.columns = [
        "Zone",
        "Events"
    ]

    fig = px.bar(

        zone_counts,

        x="Zone",

        y="Events",

        title="Events by Zone"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# IMPACT PREDICTION
# =====================================================

elif page == "Impact Prediction":

    st.title(
        "🤖 Event Impact Prediction"
    )

    event_type = st.selectbox(

        "Event Type",

        sorted(
            df["event_type"]
            .dropna()
            .unique()
        )

    )

    event_cause = st.selectbox(

        "Cause",

        sorted(
            df["event_cause"]
            .dropna()
            .unique()
        )

    )

    zone = st.selectbox(

        "Zone",

        sorted(
            df["zone"]
            .dropna()
            .unique()
        )

    )

    junction = st.selectbox(

        "Junction",

        sorted(
            df["junction"]
            .dropna()
            .unique()
        )

    )

    priority = st.selectbox(

        "Priority",

        sorted(
            df["priority"]
            .dropna()
            .unique()
        )

    )

    hour = st.slider(
        "Hour",
        0,
        23,
        18
    )

    day_name = st.selectbox(

        "Day",

        sorted(
            df["day_name"]
            .dropna()
            .unique()
        )

    )

    month = st.selectbox(

        "Month",

        sorted(
            df["month"]
            .dropna()
            .unique()
        )

    )

    lat = st.number_input(
        "Latitude",
        value=float(
            df["latitude"].mean()
        )
    )

    lon = st.number_input(
        "Longitude",
        value=float(
            df["longitude"].mean()
        )
    )

    if st.button("Predict Impact"):

        is_weekend = int(
            day_name in [
                "Saturday",
                "Sunday"
            ]
        )

        is_peak_hour = int(
            hour in [
                8,9,10,
                17,18,19,20
            ]
        )

        input_df = pd.DataFrame({

            "event_type":[event_type],
            "event_cause":[event_cause],
            "zone":[zone],
            "junction":[junction],
            "priority":[priority],
            "hour":[hour],
            "day_name":[day_name],
            "month":[month],
            "latitude":[lat],
            "longitude":[lon],
            "is_weekend":[is_weekend],
            "is_peak_hour":[is_peak_hour]

        })

        prob = model.predict_proba(
            input_df
        )[0][1]

        st.metric(

            "Closure Probability",

            f"{prob*100:.2f}%"

        )

        if prob >= 0.75:

            st.error(
                "CRITICAL IMPACT"
            )

        elif prob >= 0.50:

            st.warning(
                "HIGH IMPACT"
            )

        else:

            st.success(
                "LOW IMPACT"
            )

# =====================================================
# RESOURCE PLANNING
# =====================================================

elif page == "Resource Planning":

    st.title(
        "👮 Resource Recommendation"
    )

    probability = st.slider(

        "Closure Probability",

        0.0,
        1.0,
        0.80

    )

    police = int(
        20 + probability * 50
    )

    barricades = int(
        police / 2
    )

    st.metric(
        "Police Required",
        police
    )

    st.metric(
        "Barricades Required",
        barricades
    )

# =====================================================
# DIVERSION ROUTING
# =====================================================

elif page == "Diversion Routing":

    st.title("🗺 Diversion Route Optimizer")

    try:

        G = joblib.load(
            "traffic_network_graph.pkl"
        )

        st.success(
            f"Graph Loaded Successfully | "
            f"Nodes: {len(G.nodes())} | "
            f"Edges: {len(G.edges())}"
        )

        junctions = sorted(
            list(G.nodes())
        )

        blocked_junction = st.selectbox(

            "Blocked Junction",

            junctions

        )

        destination = st.selectbox(

            "Destination Junction",

            junctions,

            index=min(
                1,
                len(junctions)-1
            )

        )

        if st.button(
            "Find Best Diversion Route"
        ):

            try:

                route = nx.shortest_path(

                    G,

                    source=blocked_junction,

                    target=destination,

                    weight="weight"

                )

                cost = nx.shortest_path_length(

                    G,

                    source=blocked_junction,

                    target=destination,

                    weight="weight"

                )

                st.subheader(
                    "Recommended Route"
                )

                route_text = " → ".join(
                    route
                )

                st.success(
                    route_text
                )

                st.metric(
                    "Route Cost",
                    round(cost,2)
                )

                st.metric(
                    "Junctions Crossed",
                    len(route)
                )

                delay_reduction = min(

                    50,

                    round(
                        cost * 2,
                        2
                    )

                )

                st.metric(

                    "Estimated Delay Reduction",

                    f"{delay_reduction}%"

                )

            except Exception as e:

                st.error(
                    f"Routing Error: {e}"
                )

    except Exception as e:

        st.error(
            f"Graph Loading Error: {e}"
        )

# =====================================================
# LEARNING DASHBOARD
# =====================================================

elif page == "Learning Dashboard":

    st.title(
        "📈 Learning Dashboard"
    )

    risk = pd.read_csv(
        "updated_zone_risk_scores.csv"
    )

    fig = px.bar(

        risk,

        x="zone",

        y="risk_score",

        title="Zone Risk Scores"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    junctions = pd.read_csv(
        "high_risk_junctions.csv"
    )

    st.dataframe(
        junctions.head(20)
    )
