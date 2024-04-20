from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class Strategy(ABC):
    def __init__(self):
        self.N = 0

    @abstractmethod 
    def checkIfCanEnterPosition(self, df: pd.DataFrame, i: int, capital: float) -> tuple[bool, float, float, float, str]:
        pass

    @abstractmethod 
    def checkIfCanEnterPosition_2(self, df: np.ndarray, capital: float) -> tuple[bool, float, float, float]:
        pass

    @abstractmethod 
    def setParams(self, **kwargs):
        pass

    @abstractmethod 
    def updateSl(self, currentPrice: float, entryPrice:float, tpInPips:float) -> float:
        pass

