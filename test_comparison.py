"""
Compare Statistical vs ML Detection
"""

from src.data_generator import generate_flight_data
from src.detector import AnomalyDetector
from src.ml_detector import IsolationForestDetector

print("=" * 60)
print("STATISTICAL vs MACHINE LEARNING COMPARISON")
print("=" * 60)

# Generate data
df = generate_flight_data(n_samples=1000, anomaly_rate=0.05)

# Statistical detection
print("\n1. Z-Score Detection...")
stat_detector = AnomalyDetector(method='zscore', threshold=3)
stat_results = stat_detector.fit_detect(df, columns=['altitude_ft', 'speed_knots'])

# ML detection
print("2. Isolation Forest Detection...")
ml_detector = IsolationForestDetector(contamination=0.05)
ml_results = ml_detector.fit_detect(df, columns=['altitude_ft', 'speed_knots'])

# Compare
print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)

stat_anomalies = stat_results['anomaly'].sum()
ml_anomalies = ml_results['anomaly_ml'].sum()

print(f"\nZ-Score detected:         {stat_anomalies} anomalies")
print(f"Isolation Forest detected: {ml_anomalies} anomalies")

# Agreement
both = (stat_results['anomaly']) & (ml_results['anomaly_ml'])
only_stat = (stat_results['anomaly']) & (~ml_results['anomaly_ml'])
only_ml = (~stat_results['anomaly']) & (ml_results['anomaly_ml'])

print(f"\nDetected by BOTH methods:  {both.sum()}")
print(f"Only by Z-Score:           {only_stat.sum()}")
print(f"Only by Isolation Forest:  {only_ml.sum()}")

print(f"\nAgreement rate: {(both.sum() / max(stat_anomalies, ml_anomalies)) * 100:.1f}%")
