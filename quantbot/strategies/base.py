from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the input data.
        
        Args:
            df: DataFrame containing market data (and features if needed).
            
        Returns:
            pd.Series containing 1.0 (buy/long) or 0.0 (sell/flat).
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the strategy."""
        pass
    
    @property
    def params(self) -> dict:
        """Parameters of the strategy."""
        return {}
