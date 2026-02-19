"""
Test script for visualizer
"""

from src.data_generator import generate_flight_data
from src.detector import AnomalyDetector
from src.visualizer import (
    plot_flight_analysis,
    plot_anomaly_distribution,
    plot_correlation_matrix
)
import matplotlib.pyplot as plt

print("=" * 60)
print("FLIGHT DATA VISUALIZATION - TEST")
print("=" * 60)

# Generate data
print("\n1. Generating flight data...")
df = generate_flight_data(n_samples=10000, anomaly_rate=0.05)
print(f"   ✓ Generated {len(df)} data points")

# Detect anomalies
print("\n2. Detecting anomalies...")
detector = AnomalyDetector(method='zscore', threshold=3)
results = detector.fit_detect(df, columns=['altitude_ft', 'speed_knots', 'engine_temp_c'])
print(f"   ✓ Detected {results['anomaly'].sum()} anomalies")

# Create visualizations
print("\n3. Creating visualizations...")

print("   - Flight analysis plot...")
plot_flight_analysis(results, save_path='outputs/figures/flight_analysis.png')

print("   - Anomaly distribution plot...")
plot_anomaly_distribution(results, save_path='outputs/figures/anomaly_distribution.png')

print("   - Correlation matrix...")
plot_correlation_matrix(results, save_path='outputs/figures/correlation_matrix.png')

print("\n4. All visualizations saved to outputs/figures/")
print("\n" + "=" * 60)
print("VISUALIZATION TEST COMPLETED!")
print("=" * 60)

# Show plots
print("\nDisplaying plots (close windows to continue)...")
plt.show()
