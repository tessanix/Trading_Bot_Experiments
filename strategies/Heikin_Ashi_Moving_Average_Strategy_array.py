from strategies.Strategy import Strategy
import utils.utility as utility
import numpy as np

class HeikinAshiMovingAverage(Strategy):

    def __init__(self, N:int=3,  percentZoneFromMA:float=2):
        self.maxRisk = 0.02 # 2%
        self.N = N
        self.percentZoneFromMA = percentZoneFromMA


    def checkIfCanEnterPosition_2(self, data:np.ndarray, capital:float) -> tuple[bool, float, float, float]:
        inPosition, slInPips, tpInPips, entryPrice = False, 0, 0, 0

        close_idx = 0
        # open_idx = 1
        # high_idx = 2
        # low_idx = 3
        shortTermMa_idx = 4
        HA_close_idx = 5
        HA_open_idx = 6
        # HA_high_idx = 7
        # HA_low_idx = 8

        shortTermMAZoneMin = data[-1][shortTermMa_idx]-(data[-1][close_idx]/100)*self.percentZoneFromMA # => MA - 3% du prix
        shortTermMAZoneMax = data[-1][shortTermMa_idx]+(data[-1][close_idx]/100)*self.percentZoneFromMA # => MA + 3% du prix
    
        isLastNCandlesInshortTermMAZone = False
        for j in range(self.N):
            if utility.between(data[-1-j][close_idx], shortTermMAZoneMin, shortTermMAZoneMax):
                isLastNCandlesInshortTermMAZone = True
                break
        isHACandleBullish = True if data[-1][HA_close_idx] < data[-1][HA_open_idx] else False

        if data[-1][shortTermMa_idx] < data[-1][HA_open_idx] and isHACandleBullish and isLastNCandlesInshortTermMAZone:
            entryPrice = data[-1][close_idx]

            slInPips = -utility.getSlInPipsForTrade(
                invested = capital*self.maxRisk,
                pipValue = 50, # valeur du pip pour le SP500 pour un lot standard = 50
                lotSize = 0.01 # micro lot
            )
            inPosition = True
            tpInPips = -slInPips

        return inPosition, slInPips, tpInPips, entryPrice
    

    def checkIfCanEnterPosition(self, df, i, capital):
        pass
        
    def updateSl(self, currentPrice, entryPrice, tpInPips) :
        pass

    def setParams(self, **kwargs):
        pass
        # self.N = kwargs["N"]
        # self.keyLevels = kwargs["keyLevels"]
        # self.useSR = kwargs["useSR"]
        # self.useUpdateSl = kwargs["useUpdateSl"]
        # self.uselongTermMA = kwargs["uselongTermMA"]
        # self.longTermMAPeriod = kwargs["longTermMAPeriod"]
        # self.percentZoneFromMA = kwargs["percentZoneFromMA"]
