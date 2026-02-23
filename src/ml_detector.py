from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np


class IsolationForestDetector:
    """
    Anomaly detector using Isolation Forest algorithm.
    
    Isolation Forest isolates anomalies by randomly selecting a feature
    and then randomly selecting a split value. Anomalies are easier to
    isolate and require fewer splits.
    
    Parameters
    ----------
    contamination : float, default=0.05
        Expected proportion of outliers in the dataset (e.g., 0.05 = 5%)
    n_estimators : int, default=100
        Number of trees in the forest
    random_state : int, default=42
        Random seed for reproducibility
        
    """


    def __init__(
            self,
            ## Number of trees in the forest (The more there are, the greater the accuracy will be.)
            ## I guess, I'll have to experiment what is the best value for this parameter.
            contamination: float = 0.05,
            n_estimators: int = 100, 
            random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            contamination=self.contamination
        )
    
    def fit_detect(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Fit the model and detect anomalies.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data containing flight parameters
        columns : list of str
            Column names to use for anomaly detection
            
        Returns
        -------
        pd.DataFrame
            Original DataFrame with added columns:
            - 'anomaly_ml': Boolean, True if anomaly detected
            - 'anomaly_score': Float, anomaly score (lower = more anomalous)
            
        Examples
        --------
        >>> detector = IsolationForestDetector(contamination=0.05)
        >>> results = detector.fit_detect(df, columns=['altitude_ft', 'speed_knots'])
        >>> print(f"Anomalies: {results['anomaly_ml'].sum()}")
        """
        X = df[columns].values
        
        predictions = self.model.fit_predict(X)
        scores = self.model.score_samples(X)
        
        df_result = df.copy()  # Copy of original
        df_result['anomaly_ml'] = predictions == -1
        df_result['anomaly_score'] = scores
    
        return df_result
    
if __name__ == "__main__":
    isolation = IsolationForestDetector(contamination=0.05, n_estimators=100, random_state=42)
    print("Model parameters:", isolation.model.get_params())

    fit_detect = isolation.fit_detect
    print("fit_detect method:", fit_detect)