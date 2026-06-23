# ==========================================================
# PHASE 4
# DIVERSION ROUTE OPTIMIZATION
# FAST DIJKSTRA VERSION
# ==========================================================

import pandas as pd
import networkx as nx
from geopy.distance import geodesic
import pickle

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
# BUILD UNIQUE JUNCTION TABLE
# ==========================================================

junction_df = (

    df[
        [
            "junction",
            "latitude",
            "longitude",
            "requires_road_closure"
        ]
    ]

    .dropna()

    .groupby("junction")

    .agg({

        "latitude": "mean",

        "longitude": "mean",

        "requires_road_closure": "mean"

    })

    .reset_index()

)

print(
    f"\nUnique Junctions: {len(junction_df)}"
)

# ==========================================================
# CREATE GRAPH
# ==========================================================

print("\nBuilding Traffic Network...")

G = nx.Graph()

for _, row in junction_df.iterrows():

    G.add_node(

        row["junction"],

        latitude=row["latitude"],

        longitude=row["longitude"],

        risk=row["requires_road_closure"]

    )

# ==========================================================
# CONNECT NEAREST JUNCTIONS
# ==========================================================

K = 10

for i, row1 in junction_df.iterrows():

    nearby = []

    coord1 = (
        row1["latitude"],
        row1["longitude"]
    )

    for j, row2 in junction_df.iterrows():

        if i == j:
            continue

        coord2 = (
            row2["latitude"],
            row2["longitude"]
        )

        distance = geodesic(
            coord1,
            coord2
        ).km

        nearby.append(

            (

                row2["junction"],

                distance,

                row2[
                    "requires_road_closure"
                ]

            )

        )

    nearby.sort(
        key=lambda x: x[1]
    )

    for target, distance, risk in nearby[:K]:

        weight = (
            distance * 0.7
        ) + (
            risk * 0.3
        )

        G.add_edge(

            row1["junction"],

            target,

            weight=weight,

            distance=distance,

            risk=risk

        )

print(
    f"Nodes: {G.number_of_nodes()}"
)

print(
    f"Edges: {G.number_of_edges()}"
)

# ==========================================================
# SAVE GRAPH
# ==========================================================

with open(
        "traffic_network_graph.pkl",
        "wb"
) as f:

    pickle.dump(
        G,
        f
    )

print(
    "\nGraph Saved Successfully"
)

# ==========================================================
# ROUTE COST
# ==========================================================

def calculate_route_cost(path):

    total_cost = 0

    total_distance = 0

    for i in range(
            len(path)-1
    ):

        edge = G[
            path[i]
        ][
            path[i+1]
        ]

        total_cost += edge[
            "weight"
        ]

        total_distance += edge[
            "distance"
        ]

    return (

        round(
            total_cost,
            2
        ),

        round(
            total_distance,
            2
        )

    )

# ==========================================================
# SHORTEST DIVERSION ROUTE
# ==========================================================

def get_best_route(

        source,

        destination

):

    try:

        path = nx.shortest_path(

            G,

            source=source,

            target=destination,

            weight="weight"

        )

        cost, distance = \
            calculate_route_cost(
                path
            )

        return {

            "route": path,

            "cost": cost,

            "distance_km":
                distance

        }

    except Exception as e:

        print(
            "Route Error:",
            e
        )

        return None

# ==========================================================
# DELAY REDUCTION
# ==========================================================

def estimate_delay_reduction(

        probability,

        distance

):

    score = (

        probability * 100

    )

    reduction = (

        score * 0.4

    ) - (

        distance * 0.5

    )

    reduction = max(
        reduction,
        5
    )

    return round(
        reduction,
        2
    )

# ==========================================================
# DIVERSION PLAN
# ==========================================================

def generate_diversion_plan(

        blocked_junction,

        destination,

        closure_probability

):

    route = get_best_route(

        blocked_junction,

        destination

    )

    if route is None:

        return None

    delay_reduction = \
        estimate_delay_reduction(

            closure_probability,

            route[
                "distance_km"
            ]

        )

    return {

        "blocked_junction":
            blocked_junction,

        "destination":
            destination,

        "route":

            " -> ".join(
                route["route"]
            ),

        "distance_km":
            route[
                "distance_km"
            ],

        "route_cost":
            route[
                "cost"
            ],

        "estimated_delay_reduction":

            f"{delay_reduction}%"

    }

# ==========================================================
# TEST
# ==========================================================

blocked_junction = \
    "MekhriCircle"

destination = \
    "TrinityCircle"

closure_probability = \
    0.8037

result = generate_diversion_plan(

    blocked_junction,

    destination,

    closure_probability

)

print("\n")
print("=" * 70)
print("DIVERSION ROUTE PLAN")
print("=" * 70)

if result:

    for k, v in result.items():

        print(
            f"{k}: {v}"
        )
