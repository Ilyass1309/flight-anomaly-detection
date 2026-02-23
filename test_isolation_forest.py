"""
Test Isolation Forest detector
"""

from src.data_generator import generate_flight_data
from src.ml_detector import IsolationForestDetector

print("=" * 60)
print("ISOLATION FOREST - TEST")
print("=" * 60)

# 1. Generate data
print("\n1. Generating flight data...")
df = generate_flight_data(n_samples=1000, anomaly_rate=0.05)
print(f"   ✓ Generated {len(df)} samples")
print(f"   Expected anomalies: ~{int(1000 * 0.05)} (5%)")

# 2. Create detector
print("\n2. Creating Isolation Forest detector...")
detector = IsolationForestDetector(
    contamination=0.05,
    n_estimators=100,
    random_state=42
)
print(f"   ✓ Contamination: {detector.contamination}")
print(f"   ✓ Number of trees: {detector.n_estimators}")

# 3. Detect anomalies
print("\n3. Detecting anomalies...")
results = detector.fit_detect(
    df, 
    columns=['altitude_ft', 'speed_knots', 'engine_temp_c']
)
print(f"   ✓ Detection completed")

# 4. Display results
print("\n4. Results:")
print("-" * 60)
n_anomalies = results['anomaly_ml'].sum()
percentage = (n_anomalies / len(results)) * 100

print(f"Total anomalies detected: {n_anomalies}")
print(f"Percentage: {percentage:.2f}%")
print(f"Expected: ~5%")
print(f"Difference: {abs(percentage - 5):.2f}%")

# 5. Show some examples
print("\n5. Example anomalies (first 5):")
print("-" * 60)
anomalies = results[results['anomaly_ml']].head()
if len(anomalies) > 0:
    print(anomalies[['timestamp', 'altitude_ft', 'speed_knots', 
                     'engine_temp_c', 'anomaly_score']].to_string(index=False))
else:
    print("No anomalies detected!")

# 6. Score statistics
print("\n6. Anomaly Score Statistics:")
print("-" * 60)
print(f"Min score (most anomalous):  {results['anomaly_score'].min():.4f}")
print(f"Max score (most normal):     {results['anomaly_score'].max():.4f}")
print(f"Mean score:                  {results['anomaly_score'].mean():.4f}")
print(f"Median score:                {results['anomaly_score'].median():.4f}")

# 7. Distribution of scores
print("\n7. Score Distribution:")
print("-" * 60)
normal_scores = results[~results['anomaly_ml']]['anomaly_score']
anomaly_scores = results[results['anomaly_ml']]['anomaly_score']

print(f"Normal points - Mean score:   {normal_scores.mean():.4f}")
print(f"Anomaly points - Mean score:  {anomaly_scores.mean():.4f}")
print(f"Difference:                   {abs(normal_scores.mean() - anomaly_scores.mean()):.4f}")

print("\n" + "=" * 60)
print("TEST COMPLETED SUCCESSFULLY!")
print("=" * 60)
