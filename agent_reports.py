import pandas as pd
from datetime import datetime

def generate_report(bin_data, route_data=None):
    """
    Generate a simple daily agent report from bin data & route optimization.
    
    Args:
        bin_data (pd.DataFrame): DataFrame with columns ['Bin ID', 'Fill Level (%)'].
        route_data (list of dict): Optimized route with bin details (optional).
    
    Returns:
        dict: Summary report with stats.
    """
    total_bins = len(bin_data)
    bins_filled = len(bin_data[bin_data["Fill Level (%)"] > 70])
    bins_pending = total_bins - bins_filled
    avg_fill = bin_data["Fill Level (%)"].mean()

    fuel_saved = 0
    if route_data:
        # Assume ~5% fuel saved for every 10 bins optimized
        fuel_saved = (len(route_data) // 10) * 5

    report = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_bins": total_bins,
        "bins_collected": bins_filled,
        "bins_pending": bins_pending,
        "average_fill": round(avg_fill, 2),
        "estimated_fuel_saved_percent": fuel_saved
    }
    return report

def save_report(report, filename="agent_report.csv"):
    """Append the report to a CSV file (persistent log)."""
    df = pd.DataFrame([report])
    try:
        old = pd.read_csv(filename)
        df = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        "Bin ID": [1, 2, 3, 4],
        "Fill Level (%)": [20, 80, 95, 40]
    })
    sample_route = [
        {"Bin ID": 2, "Latitude": 17.74, "Longitude": 83.25, "Fill Level (%)": 80},
        {"Bin ID": 3, "Latitude": 17.75, "Longitude": 83.26, "Fill Level (%)": 95}
    ]
    report = generate_report(sample_data, sample_route)
    print(report)
    save_report(report)
