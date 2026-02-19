"""
Anomaly Detector
Statistical methods for detecting anomalies in flight data.
"""

import pandas as pd
import numpy as np
from typing import List, Literal


class AnomalyDetector:
    """
    Anomaly detection using statistical methods.
    
    Supports multiple detection methods:
    - Z-score: Standard deviation based
    - IQR: Interquartile range based
    - Moving average: Deviation from trend
    
    Parameters
    ----------
    method : {'zscore', 'iqr', 'moving_avg'}, default='zscore'
        Detection method to use
    threshold : float, default=3.0
        Threshold for anomaly detection
    window_size : int, default=30
        Window size for moving average method
    """
    
    def __init__(
        self,
        method: Literal['zscore', 'iqr', 'moving_avg'] = 'zscore',
        threshold: float = 3.0,
        window_size: int = 30
    ):
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        self.stats_ = {}
        
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'AnomalyDetector':
        """
        Compute statistics needed for anomaly detection.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        columns : list of str
            Columns to analyze
            
        Returns
        -------
        self : AnomalyDetector
            Fitted detector
        """
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            
            # Calculate statistics for this column
            self.stats_[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'q1': df[col].quantile(0.25),
                'q3': df[col].quantile(0.75),
                'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
            }
        
        return self
    
    def detect(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Detect anomalies in the data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to analyze
        columns : list of str
            Columns to check for anomalies
            
        Returns
        -------
        pd.DataFrame
            Original DataFrame with added anomaly columns
        """
        if not self.stats_:
            raise ValueError("Detector must be fitted before detection. Call fit() first.")
        
        df_result = df.copy()
        
        # Apply the chosen method to each column
        for col in columns:
            if self.method == 'zscore':
                df_result = self._detect_zscore(df_result, col)
            elif self.method == 'iqr':
                df_result = self._detect_iqr(df_result, col)
            elif self.method == 'moving_avg':
                df_result = self._detect_moving_avg(df_result, col)
        
        # Create combined anomaly flag (True if ANY column has anomaly)
        anomaly_cols = [f'{col}_anomaly' for col in columns]
        df_result['anomaly'] = df_result[anomaly_cols].any(axis=1)
        
        return df_result
    
    def fit_detect(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Fit and detect in one step (convenience method).
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to analyze
        columns : list of str
            Columns to check for anomalies
            
        Returns
        -------
        pd.DataFrame
            DataFrame with anomaly detection results
        """
        self.fit(df, columns)
        return self.detect(df, columns)
    
    def _detect_zscore(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Z-score based detection (internal method)."""
        mean = self.stats_[col]['mean']
        std = self.stats_[col]['std']
        
        # Calculate Z-score: (value - mean) / std
        df[f'{col}_zscore'] = (df[col] - mean) / std
        
        # Mark as anomaly if |Z| > threshold
        df[f'{col}_anomaly'] = np.abs(df[f'{col}_zscore']) > self.threshold
        
        return df
    
    def _detect_iqr(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """IQR based detection (internal method)."""
        q1 = self.stats_[col]['q1']
        q3 = self.stats_[col]['q3']
        iqr = self.stats_[col]['iqr']
        
        # Calculate bounds: Q1 - threshold*IQR, Q3 + threshold*IQR
        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr
        
        # Mark as anomaly if outside bounds
        df[f'{col}_anomaly'] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        return df
    
    def _detect_moving_avg(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Moving average based detection (internal method)."""
        # Calculate moving average
        df[f'{col}_ma'] = df[col].rolling(window=self.window_size, center=True).mean()
        
        # Calculate deviation from moving average
        df[f'{col}_deviation'] = np.abs(df[col] - df[f'{col}_ma'])
        
        # Use threshold as multiple of standard deviation of deviations
        threshold_value = df[f'{col}_deviation'].std() * self.threshold
        df[f'{col}_anomaly'] = df[f'{col}_deviation'] > threshold_value
        
        return df
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics of detected anomalies.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with anomaly detection results
            
        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        summary = []
        
        for col in self.stats_.keys():
            anomaly_col = f'{col}_anomaly'
            if anomaly_col in df.columns:
                n_anomalies = df[anomaly_col].sum()
                pct_anomalies = (n_anomalies / len(df)) * 100
                
                summary.append({
                    'parameter': col,
                    'total_points': len(df),
                    'anomalies': n_anomalies,
                    'percentage': f'{pct_anomalies:.2f}%'
                })
        
        return pd.DataFrame(summary)


if __name__ == "__main__":
    # Example usage when running this file directly
    from .data_generator import generate_flight_data
    
    print("Generating test data...")
    df = generate_flight_data(n_samples=1000, anomaly_rate=0.05)
    
    print("\nDetecting anomalies with Z-score method...")
    detector = AnomalyDetector(method='zscore', threshold=3)
    results = detector.fit_detect(df, columns=['altitude_ft', 'speed_knots'])
    
    print("\nAnomaly Summary:")
    print(detector.get_anomaly_summary(results))
    
    print(f"\nTotal anomalies detected: {results['anomaly'].sum()}")
    print(f"Percentage of anomalies: {(results['anomaly'].sum() / len(results)) * 100:.2f}%")
