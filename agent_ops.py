import pandas as pd
import numpy as np
from datetime import datetime

def generate_agent_report(bins_df):
    """
    Generate a summary report for AI agents based on bin fill levels.
    
    Args:
        bins_df (pd.DataFrame): DataFrame with columns ['Bin ID', 'Latitude', 'Longitude', 'Fill Level (%)']
    
    Returns:
        dict: Report containing bins collected, pending bins, and estimated fuel saved
    """
    bins_collected = len(bins_df[bins_df["Fill Level (%)"] > 70])
    bins_pending = len(bins_df[bins_df["Fill Level (%)"] <= 70])
    
    # Simple fuel saving estimation based on number of bins collected
    estimated_fuel_saved = round(bins_collected * 5, 2)  # e.g., 5% per collected bin
    
    report = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "bins_collected": bins_collected,
        "bins_pending": bins_pending,
        "estimated_fuel_saved": estimated_fuel_saved
    }
    
    return report

if __name__ == "__main__":
    # Example usage
    data = {
        "Bin ID": [1,2,3,4],
        "Latitude": [17.70, 17.71, 17.72, 17.73],
        "Longitude": [83.21, 83.22, 83.23, 83.24],
        "Fill Level (%)": [80, 45, 90, 75]
    }
    df = pd.DataFrame(data)
    report = generate_agent_report(df)
    print("Agent Report for Today:")
    print(report)
