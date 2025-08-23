# app.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from math import radians, cos, sin, asin, sqrt
from streamlit_autorefresh import st_autorefresh

# --- Page Config ---
st.set_page_config(page_title="EcoRoute AI", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("EcoRoute AI Dashboard")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Bin Detection", "IoT Simulation", "Route Optimization", "Driver Dashboard", "Agent Reports"]
)

# --- Helper Functions ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in km between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def nearest_neighbor_route(start, bins):
    """Nearest neighbor route for a list of bins [(lat, lon, id), ...]."""
    route = [start]
    unvisited = bins.copy()
    while unvisited:
        last = route[-1]
        next_bin = min(unvisited, key=lambda x: haversine(last[0], last[1], x[0], x[1]))
        route.append(next_bin)
        unvisited.remove(next_bin)
    return route

# --- Initialize bins in session_state ---
if "bins_df" not in st.session_state:
    num_bins = 20
    np.random.seed(42)
    st.session_state['bins_df'] = pd.DataFrame({
        "Bin ID": list(range(1, num_bins + 1)),
        "Latitude": np.random.uniform(17.68, 17.80, num_bins),
        "Longitude": np.random.uniform(83.20, 83.31, num_bins),
        "Fill Level (%)": np.random.randint(0, 101, num_bins)
    })

df = st.session_state['bins_df']

# --- HOME PAGE ---
if page == "Home":
    st.title("🌱 EcoRoute AI")
    st.markdown("""
    EcoRoute AI is an AI-powered smart waste management system.
    Upload bin images, track fill levels, simulate IoT updates,
    and calculate optimized collection routes for cleaner, greener cities.
    """)

    # Auto-refresh every 5 seconds
    st_autorefresh(interval=5000, key="home_autorefresh")

    # Simulate bins filling over time
    df["Fill Level (%)"] = np.clip(df["Fill Level (%)"] + np.random.randint(0, 5, size=len(df)), 0, 100)
    st.session_state['bins_df'] = df

    # Dashboard metrics
    total_bins = len(df)
    bins_filled = len(df[df["Fill Level (%)"] >= 70])
    bins_pending = total_bins - bins_filled
    avg_fill = df["Fill Level (%)"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Bins", total_bins)
    col2.metric("Bins ≥ 70%", bins_filled)
    col3.metric("Average Fill %", f"{avg_fill:.1f}%")

    # Bar chart for bin fill levels
    st.bar_chart(df["Fill Level (%)"])

# --- BIN DETECTION PAGE ---
elif page == "Bin Detection":
    st.header("📷 Upload Bin Image for AI Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Bin Image", use_container_width=True)
        fill_level = np.random.choice(["Empty", "Half-Full", "Full"])
        st.success(f"Predicted Fill Level: {fill_level}")

# --- IoT SIMULATION PAGE ---
elif page == "IoT Simulation":
    st.header("📊 Simulated Real-Time Bin Status Across Vizag")
    st.dataframe(df)

    # Folium map
    m = folium.Map(location=[17.74, 83.255], zoom_start=12)
    for _, row in df.iterrows():
        if row["Fill Level (%)"] < 50:
            color = "#2ca02c"
        elif row["Fill Level (%)"] < 70:
            color = "#ff7f0e"
        else:
            color = "#d62728"
        radius = 5 + row["Fill Level (%)"] / 10
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"Bin {row['Bin ID']} - {row['Fill Level (%)']}%",
            tooltip=f"Fill: {row['Fill Level (%)']}%"
        ).add_to(m)

    # Legend
    legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 160px; height: 100px; 
            background-color: #2b2b2b; z-index:9999; font-size:14px;
            border:2px solid grey; border-radius:5px; padding: 10px; color: white;">
<b>Fill Level Legend</b><br>
<span style="color:#2ca02c;">●</span> Fill < 50%<br>
<span style="color:#ff7f0e;">●</span> Fill 50-70%<br>
<span style="color:#d62728;">●</span> Fill > 70%
</div>
'''
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=700)

# --- ROUTE OPTIMIZATION PAGE ---
elif page == "Route Optimization":
    st.header("🗺️ Optimized Collection Route Across Vizag")
    st.markdown("Showing bins with fill level > 70%")

    high_fill_bins = df[df["Fill Level (%)"] > 70].copy()
    bins_list = high_fill_bins.to_dict('records')
    route = []

    if bins_list:
        route.append(bins_list.pop(0))
        while bins_list:
            last = route[-1]
            next_bin = min(bins_list, key=lambda x: haversine(last['Latitude'], last['Longitude'], x['Latitude'], x['Longitude']))
            route.append(next_bin)
            bins_list.remove(next_bin)

    # Draw map
    m = folium.Map(location=[17.74, 83.255], zoom_start=12)
    prev_bin = None
    for bin in route:
        folium.Marker(
            [bin["Latitude"], bin["Longitude"]],
            popup=f"Bin {bin['Bin ID']} - {bin['Fill Level (%)']}%",
            icon=folium.Icon(color="red", icon="trash")
        ).add_to(m)
        if prev_bin:
            folium.PolyLine(
                locations=[[prev_bin["Latitude"], prev_bin["Longitude"]],
                           [bin["Latitude"], bin["Longitude"]]],
                color="blue", weight=3
            ).add_to(m)
        prev_bin = bin
    st_folium(m, width=700)

# --- DRIVER DASHBOARD PAGE (Person 2) ---
elif page == "Driver Dashboard":
    st.header("🚛 Driver Dashboard")
    depot = (17.74, 83.255)

    target_bins = df[df["Fill Level (%)"] >= 70][["Latitude", "Longitude", "Bin ID"]].values.tolist()
    if "route" not in st.session_state:
        st.session_state.route = []
    if "collected" not in st.session_state:
        st.session_state.collected = []

    if st.button("🔄 Generate Route"):
        if target_bins:
            st.session_state.route = nearest_neighbor_route(depot, [(lat, lon, bid) for lat, lon, bid in target_bins])
            st.session_state.collected = []
            st.success("✅ Route generated!")
        else:
            st.warning("No bins above threshold to collect.")

    if st.session_state.route:
        m = folium.Map(location=depot, zoom_start=13)
        folium.Marker(depot, popup="Depot", icon=folium.Icon(color="green")).add_to(m)
        for lat, lon, bid in st.session_state.route[1:]:
            color = "blue" if bid not in st.session_state.collected else "gray"
            folium.Marker([lat, lon], popup=f"Bin {bid}", icon=folium.Icon(color=color)).add_to(m)
        folium.PolyLine([(p[0], p[1]) for p in st.session_state.route], color="blue", weight=2.5).add_to(m)
        st_folium(m, width=700, height=500)

        st.write("### Route Progress")
        for lat, lon, bid in st.session_state.route[1:]:
            if bid not in st.session_state.collected:
                if st.button(f"✅ Collect Bin {bid}"):
                    st.session_state.collected.append(bid)
                    st.experimental_rerun()

        st.info(f"Bins Collected: {len(st.session_state.collected)} / {len(st.session_state.route)-1}")

# --- AGENT REPORTS PAGE ---
elif page == "Agent Reports":
    st.header("📝 AI Agent Reports")
    st.markdown(f"""
    **Summary for Today:**  
    - Bins collected: {len(df[df["Fill Level (%)"] > 70])}  
    - Pending bins: {len(df[df["Fill Level (%)"] <= 70])}  
    - Estimated fuel saved: 12%  

    *(This can be dynamically updated by integrating AgentOps outputs)*
    """)
