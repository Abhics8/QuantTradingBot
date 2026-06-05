import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from quantbot.strategies.base import Strategy
from quantbot.features.pipeline import build_feature_matrix

class XGBoostSignalModel:
    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.05, 
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
        
    def feature_importance(self) -> pd.Series:
        if self.feature_names is None:
            return pd.Series()
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)

class XGBoostStrategy(Strategy):
    """
    Strategy wrapper that uses a trained XGBoost model to generate signals.
    """
    def __init__(self, model: XGBoostSignalModel, feature_columns: list[str]):
        self.model = model
        self.feature_columns = feature_columns
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        print(f"🤖 Generating signals using {self.name}...")
        
        # Build features for inference
        feature_df = build_feature_matrix(df, include_target=False)
        
        # Initialize signals as 0.0
        signals = pd.Series(0.0, index=df.index, name='Signal')
        
        # Only predict on rows where we have valid features
        X = feature_df[self.feature_columns]
        predictions = self.model.predict(X)
        
        # Assign predictions back to the original index
        signals.loc[feature_df.index] = predictions
        
        return signals

    @property
    def name(self) -> str:
        return "XGBoost ML Signal"
