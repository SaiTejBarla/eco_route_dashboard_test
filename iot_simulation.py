import pandas as pd
import numpy as np
import time
import random

# File where simulated data will be stored
CSV_FILE = "iot_bins.csv"

def generate_initial_bins(num_bins=20, seed=42):
    """Generate initial random bin data for Vizag city area."""
    np.random.seed(seed)
    df = pd.DataFrame({
        "Bin ID": list(range(1, num_bins + 1)),
        "Latitude": np.random.uniform(17.68, 17.80, num_bins),
        "Longitude": np.random.uniform(83.20, 83.31, num_bins),
        "Fill Level (%)": np.random.randint(0, 101, num_bins)
    })
    df.to_csv(CSV_FILE, index=False)
    return df

def update_bins():
    """Simulate IoT bin updates over time."""
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        df = generate_initial_bins()

    # Each cycle, bins fill up slightly
    df["Fill Level (%)"] = np.clip(
        df["Fill Level (%)"] + np.random.randint(0, 5, size=len(df)), 
        0, 
        100
    )
    df.to_csv(CSV_FILE, index=False)
    return df

if __name__ == "__main__":
    print("Starting IoT Bin Simulation... Press Ctrl+C to stop.")
    generate_initial_bins()
    while True:
        df = update_bins()
        print(df.head(5))  # show first 5 bins as preview
        time.sleep(5)  # simulate update every 5 sec
