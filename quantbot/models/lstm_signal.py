import pandas as pd
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("⚠️ PyTorch not installed. LSTM module will not function.")

from quantbot.strategies.base import Strategy
from quantbot.features.pipeline import build_feature_matrix

if HAS_PYTORCH:
    class LSTMNet(nn.Module):
        """2-layer LSTM with dropout"""
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            # Take the output of the last time step
            last_out = lstm_out[:, -1, :]
            out = self.fc(last_out)
            return self.sigmoid(out).squeeze(-1)

class LSTMSignalModel:
    def __init__(self, seq_length=20, hidden_size=64, num_layers=2, 
                 dropout=0.3, epochs=50, lr=0.001, batch_size=32):
        if not HAS_PYTORCH:
            raise ImportError("PyTorch is required for LSTMSignalModel")
            
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_sequences(self, data: np.ndarray, labels: np.ndarray = None):
        X_seq, y_seq = [], []
        for i in range(len(data) - self.seq_length + 1):
            X_seq.append(data[i:(i + self.seq_length)])
            if labels is not None:
                # Target is the label corresponding to the last day in the sequence
                y_seq.append(labels[i + self.seq_length - 1])
                
        if labels is not None:
            return np.array(X_seq), np.array(y_seq)
        return np.array(X_seq)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X.values)
        y_vals = y.values
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_vals)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = LSTMNet(
            input_size=X.shape[1], 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            dropout=self.dropout
        ).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        history = {'loss': []}
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            history['loss'].append(epoch_loss / len(loader))
            
        return history
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        
        X_scaled = self.scaler.transform(X.values)
        X_seq = self._create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            return np.array([])
            
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy()
            
        # Pad the beginning to match original DataFrame length
        padding = np.zeros(self.seq_length - 1)
        return np.concatenate([padding, outputs])
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(float)


class LSTMStrategy(Strategy):
    """
    Strategy wrapper using trained LSTM model.
    """
    def __init__(self, model: LSTMSignalModel, feature_columns: list[str]):
        self.model = model
        self.feature_columns = feature_columns
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        print(f"🧠 Generating signals using {self.name}...")
        
        # Build features for inference
        feature_df = build_feature_matrix(df, include_target=False)
        
        # Initialize signals as 0.0
        signals = pd.Series(0.0, index=df.index, name='Signal')
        
        if len(feature_df) < self.model.seq_length:
            return signals
            
        X = feature_df[self.feature_columns]
        predictions = self.model.predict(X)
        
        signals.loc[feature_df.index] = predictions
        
        return signals

    @property
    def name(self) -> str:
        return "LSTM Deep Learning"
