"""
Flight Data Generator
Generates realistic simulated flight telemetry data with controlled anomalies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def generate_flight_data(
    n_samples: int = 10000,
    anomaly_rate: float = 0.05,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate simulated flight telemetry data with injected anomalies.
    
    Parameters
    ----------
    n_samples : int, default=10000
        Number of data points to generate
    anomaly_rate : float, default=0.05
        Proportion of anomalies to inject (0.0 to 1.0)
    seed : int or None, default=42
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing flight parameters with timestamps
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Validate inputs
    if not 0 <= anomaly_rate <= 1:
        raise ValueError("anomaly_rate must be between 0 and 1")
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100")
    
    # Generate timestamps (10-second intervals)
    start_date = datetime(2024, 1, 1, 10, 0, 0)
    timestamps = [start_date + timedelta(seconds=i*10) for i in range(n_samples)]
    
    # Generate normal flight parameters
    altitude = 10000 + np.random.normal(0, 100, n_samples)
    speed = 250 + np.random.normal(0, 10, n_samples)
    fuel = 5000 - np.linspace(0, 2000, n_samples) + np.random.normal(0, 50, n_samples)
    engine_temp = 200 + np.random.normal(0, 15, n_samples)
    
    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['altitude', 'speed', 'temp', 'combined'])
        
        if anomaly_type == 'altitude':
            altitude[idx] += np.random.choice([-500, 500])
        elif anomaly_type == 'speed':
            speed[idx] += np.random.choice([-50, 50])
        elif anomaly_type == 'temp':
            engine_temp[idx] += np.random.choice([-40, 40])
        else:
            altitude[idx] += np.random.choice([-300, 300])
            speed[idx] += np.random.choice([-30, 30])
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'altitude_ft': altitude,
        'speed_knots': speed,
        'fuel_lbs': fuel,
        'engine_temp_c': engine_temp
    })
    
    return df


if __name__ == "__main__":
    print("Generating flight data...")
    flight_data = generate_flight_data(n_samples=10000, anomaly_rate=0.05)
    
    print(f"\nDataset shape: {flight_data.shape}")
    print(f"\nFirst few rows:")
    print(flight_data.head())
    
    print(f"\nStatistical summary:")
    print(flight_data.describe())
    
    # Save to CSV
    flight_data.to_csv('data/raw/flight_data.csv', index=False)
    print(f"\nData saved to 'data/raw/flight_data.csv'")
