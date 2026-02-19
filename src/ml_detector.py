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

if __name__ == "__main__":
    isolation = IsolationForestDetector(contamination=0.05, n_estimators=100, random_state=42)
    print("Model parameters:", isolation.model.get_params())