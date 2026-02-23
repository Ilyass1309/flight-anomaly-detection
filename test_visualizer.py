"""
Test script for visualizer
"""

from src.data_generator import generate_flight_data
from src.detector import AnomalyDetector
from src.ml_detector import IsolationForestDetector
from src.visualizer import (
    plot_flight_analysis,
    plot_anomaly_distribution,
    plot_correlation_matrix,
    plot_ml_comparison
)
import matplotlib.pyplot as plt

print("=" * 60)
print("FLIGHT DATA VISUALIZATION - TEST")
print("=" * 60)

# Generate data
print("\n1. Generating flight data...")
df = generate_flight_data(n_samples=10000, anomaly_rate=0.05)
print(f"   ✓ Generated {len(df)} data points")

# Detect anomalies with Statistical method
print("\n2. Detecting anomalies with Z-Score...")
stat_detector = AnomalyDetector(method='zscore', threshold=3)
stat_results = stat_detector.fit_detect(df, columns=['altitude_ft', 'speed_knots', 'engine_temp_c'])
print(f"   ✓ Z-Score detected {stat_results['anomaly'].sum()} anomalies")

# Detect anomalies with ML method
print("\n3. Detecting anomalies with Isolation Forest...")
ml_detector = IsolationForestDetector(contamination=0.05)
ml_results = ml_detector.fit_detect(df, columns=['altitude_ft', 'speed_knots', 'engine_temp_c'])
print(f"   ✓ Isolation Forest detected {ml_results['anomaly_ml'].sum()} anomalies")

# Create visualizations
print("\n4. Creating visualizations...")

print("   - Flight analysis plot (Z-Score)...")
plot_flight_analysis(stat_results, save_path='outputs/figures/flight_analysis.png')

print("   - Anomaly distribution plot (Z-Score)...")
plot_anomaly_distribution(stat_results, save_path='outputs/figures/anomaly_distribution.png')

print("   - Correlation matrix...")
plot_correlation_matrix(stat_results, save_path='outputs/figures/correlation_matrix.png')

print("   - ML vs Statistical comparison...")
plot_ml_comparison(stat_results, ml_results, save_path='outputs/figures/ml_vs_statistical.png')

print("\n5. All visualizations saved to outputs/figures/")
print("\n" + "=" * 60)
print("VISUALIZATION TEST COMPLETED!")
print("=" * 60)

# Show plots
print("\nDisplaying plots (close windows to continue)...")
plt.show()
