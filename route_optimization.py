import pandas as pd
import networkx as nx
import itertools
import math
import json

CSV_FILE = "iot_bins.csv"
ROUTE_FILE = "optimized_route.json"

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/lon points in km."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def build_graph(df):
    """Create a complete graph of bins using haversine distance."""
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row["Bin ID"], pos=(row["Latitude"], row["Longitude"]))
    for (i, r1), (j, r2) in itertools.combinations(df.iterrows(), 2):
        dist = haversine(r1["Latitude"], r1["Longitude"], r2["Latitude"], r2["Longitude"])
        G.add_edge(r1["Bin ID"], r2["Bin ID"], weight=dist)
    return G

def optimize_route():
    """Find shortest route visiting all bins >70% filled (TSP approximation)."""
    df = pd.read_csv(CSV_FILE)
    full_bins = df[df["Fill Level (%)"] > 70]
    
    if len(full_bins) == 0:
        print("No bins require collection.")
        return []

    G = build_graph(full_bins)
    
    # Approximate TSP (Christofides not in base nx, so use greedy)
    route = list(nx.approximation.traveling_salesman_problem(G, weight="weight", cycle=False))
    
    # Save optimized route
    result = full_bins.set_index("Bin ID").loc[route][["Latitude", "Longitude"]].reset_index()
    result.to_json(ROUTE_FILE, orient="records")
    print("Optimized route saved to", ROUTE_FILE)
    return result

if __name__ == "__main__":
    optimize_route()
