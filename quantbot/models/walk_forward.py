import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class WalkForwardValidator:
    """
    Anchored walk-forward cross-validation engine.
    """
    def __init__(self, train_period_days=504, test_period_days=63, min_train_size=252):
        """
        504 days approx 2 years training
        63 days approx 3 months testing
        """
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.min_train_size = min_train_size
        
    def split(self, df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate (train_df, test_df) splits using anchored walk-forward.
        Train window grows each fold, test window slides forward.
        """
        splits = []
        n_samples = len(df)
        
        if n_samples < self.min_train_size + self.test_period_days:
            # Not enough data for even one fold, return a single split
            train_size = int(n_samples * 0.8)
            return [(df.iloc[:train_size], df.iloc[train_size:])]
            
        # Start at min_train_size or train_period_days
        current_train_end = max(self.min_train_size, self.train_period_days)
        
        while current_train_end + self.test_period_days <= n_samples:
            train_df = df.iloc[:current_train_end]
            test_df = df.iloc[current_train_end:current_train_end + self.test_period_days]
            
            splits.append((train_df, test_df))
            current_train_end += self.test_period_days
            
        # Handle the remainder if there's a gap
        if current_train_end < n_samples:
            train_df = df.iloc[:current_train_end]
            test_df = df.iloc[current_train_end:]
            splits.append((train_df, test_df))
            
        return splits
        
    def validate(self, model_class, model_kwargs: dict, 
                 feature_df: pd.DataFrame, feature_cols: list, 
                 target_col: str = 'target') -> dict:
        """
        Run walk-forward validation.
        """
        splits = self.split(feature_df)
        
        all_predictions = []
        all_actuals = []
        fold_metrics = []
        
        for i, (train_df, test_df) in enumerate(splits):
            print(f"   🔄 Fold {i+1}/{len(splits)}: Train={len(train_df)} Test={len(test_df)}")
            
            # Instantiate fresh model
            model = model_class(**model_kwargs)
            
            # Fit
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            model.fit(X_train, y_train)
            
            # Predict
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]
            preds = model.predict(X_test)
            
            # Since LSTM pads predictions with 0s at the start, we might need to truncate
            if len(preds) > len(y_test):
                 preds = preds[-len(y_test):]
            elif len(preds) < len(y_test):
                 # This shouldn't happen with proper seq_length handling, but just in case
                 y_test = y_test[-len(preds):]

            all_predictions.extend(preds)
            all_actuals.extend(y_test.values)
            
            # Metrics for fold
            acc = accuracy_score(y_test, preds)
            fold_metrics.append(acc)
            
        # Aggregate
        y_true = np.array(all_actuals)
        y_pred = np.array(all_predictions)
        
        # Avoid division by zero warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            aggregate_metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
            }
        
        return {
            'all_predictions': np.array(all_predictions),
            'all_actuals': np.array(all_actuals),
            'fold_metrics': fold_metrics,
            'aggregate_metrics': aggregate_metrics
        }
