from strategies.Strategy import Strategy
import utility
import pandas as pd
from actor_critic.trading_agent.agent import Agent

class HeikinAshiMovingAverage(Strategy):

    def __init__(self, agent:Agent, N:int=3, levels: list[float]=[], useSR:str="simpleSR", useUpdateSl:bool=True, longTermMAPeriod=200, uselongTermMA:bool=True, percentZoneFromMA:float=2):
        self.maxRisk = 0.02 # 2%
        self.N = N
        self.keyLevels = levels
        self.useSR = useSR # values = NoSR, "simpleSR", "SRbyRL"
        self.useUpdateSl = useUpdateSl
        self.uselongTermMA = uselongTermMA
        self.longTermMAPeriod = longTermMAPeriod
        self.percentZoneFromMA = percentZoneFromMA

        self.agent = agent

    def setParams(self, **kwargs):
        self.N = kwargs["N"]
        self.keyLevels = kwargs["keyLevels"]
        self.useSR = kwargs["useSR"]
        self.useUpdateSl = kwargs["useUpdateSl"]
        self.uselongTermMA = kwargs["uselongTermMA"]
        self.longTermMAPeriod = kwargs["longTermMAPeriod"]
        self.percentZoneFromMA = kwargs["percentZoneFromMA"]
    
    def determineSlAndTp(self, capital:float, price: float, keyLevels: list[float], last2DaysVariation:float) -> tuple[bool, float, float]:
        _slInPips = -utility.getSlInPipsForTrade(
                            invested = capital*self.maxRisk,
                            pipValue = 50, # valeur du pip pour le SP500 pour un lot standard = 50
                            lotSize = 0.01 # micro lot
                        )
        _tpInPips = -_slInPips
        # print(f"slInPips:{_slInPips}, tpInPips: {_tpInPips}")
        resistance = support = 0
        # done = False

        for i in range(len(keyLevels)):
            sl = price+_slInPips
            # keyLevels[i==0] is the greatest level
            if i==0 and sl < keyLevels[i] and keyLevels[i] < price: 
                # slInPips = keyLevels[i]-price # négatif
                resistance = price + _tpInPips
                support = keyLevels[i]
                # done = True
                break
            elif i!=0 and keyLevels[i] < price and price < keyLevels[i-1]:
                resistance = keyLevels[i-1]
                support = keyLevels[i]
                break

        # if done: 
        #     return False, slInPips, tpInPips
        
        middleSR = (resistance+support)/2

        _isBelowMiddleSR = price < middleSR

        if _isBelowMiddleSR:
            sl = price+_slInPips
            if sl < support and price*0.01 <= price-support: 
                _slInPips = support-price # négatif
            if resistance-price < _tpInPips: 
                _tpInPips = resistance-price
            if 0 < last2DaysVariation and last2DaysVariation < _tpInPips: 
                _tpInPips = last2DaysVariation
            # print(f'resistance : {tpInPips}, slInPips : {slInPips}')

        return _isBelowMiddleSR, _slInPips, _tpInPips

    def updateSl(self, currentPrice: float, entryPrice:float, tpInPips:float) -> float:
        newSlInPips = 0
        if self.useUpdateSl and tpInPips/2 < currentPrice-entryPrice:
            newSlInPips = -tpInPips/4 
        return newSlInPips
    
    def updateSlAndTpWithRL(self, capital, df):
        _maxSlInPips = -utility.getSlInPipsForTrade(
                            invested = capital*self.maxRisk,
                            pipValue = 50, # valeur du pip pour le SP500 pour un lot standard = 50
                            lotSize = 0.01 # micro lot
                        )
        _maxTpInPips = -_maxSlInPips
        
        observation = df.to_numpy()
        sl, tp = self.agent.choose_action(observation)
        
        if 0 < tp and tp <= 1:
            tp = tp*_maxTpInPips
        else: tp = _maxTpInPips

        if 0 < sl and sl <= 1:
            sl = sl*_maxSlInPips
        else: sl = _maxSlInPips

        return sl, tp
    
    def determineSlAndTpWithRL(self, capital:float, df:pd.DataFrame):
        _maxSlInPips = -utility.getSlInPipsForTrade(
                            invested = capital*self.maxRisk,
                            pipValue = 50, # valeur du pip pour le SP500 pour un lot standard = 50
                            lotSize = 0.01 # micro lot
                        )
        _maxTpInPips = -_maxSlInPips
        
        observation = df[["open", "close","low", "high"]].to_numpy()
        sl, tp = self.agent.choose_action(observation)

        if 0 < tp and tp <= 1:
            tp = tp*_maxTpInPips
        else: tp = _maxTpInPips

        if 0 < sl and sl <= 1:
            sl = sl*_maxSlInPips
        else: sl = _maxSlInPips

        return sl, tp


    def checkIfCanEnterPosition(self, df: pd.DataFrame, i: int, capital: float) -> tuple[bool, float, float, float, str]:
        inPosition, slInPips, tpInPips, entryPrice, entryDate = False, 0, 0, 0, ""
        
        allowedToTrade = True
        
        if self.uselongTermMA:
           allowedToTrade = True if df["longTermMA"].loc[i] < df["HA open"].loc[i] else False

        if allowedToTrade:
            shortTermMAZoneMin = df["shortTermMA"].loc[i]-(df["close"].loc[i]/100)*self.percentZoneFromMA # => MA - 3% du prix
            shortTermMAZoneMax = df["shortTermMA"].loc[i]+(df["close"].loc[i]/100)*self.percentZoneFromMA # => MA + 3% du prix
        
            isLastNCandlesInshortTermMAZone = False
            for j in range(i-self.N, i):
                if utility.between(df["HA close"].loc[j], shortTermMAZoneMin, shortTermMAZoneMax):
                    isLastNCandlesInshortTermMAZone = True
                    break
            
            if df["shortTermMA"].loc[i] < df["HA open"].loc[i] and df["HA color"].loc[i] == "green" and isLastNCandlesInshortTermMAZone:
                entryDate = df["datetime"].loc[i]
                entryPrice =  df["close"].loc[i]
                if self.useSR=="simpleSR":
                    isBelowMiddleSR, slInPips, tpInPips = self.determineSlAndTp(capital, entryPrice, self.keyLevels, df["last2DaysVariation"].loc[i])
                    inPosition = isBelowMiddleSR

                elif self.useSR=="SRbyRL":
                    slInPips, tpInPips = self.determineSlAndTpWithRL(capital, df)
                    inPosition = True

                elif self.useSR=="NoSR":
                    slInPips = -utility.getSlInPipsForTrade(
                                invested = capital*self.maxRisk,
                                pipValue = 50, # valeur du pip pour le SP500 pour un lot standard = 50
                                lotSize = 0.01 # micro lot
                            )
                    inPosition = True
                    tpInPips = -slInPips

        return inPosition, slInPips, tpInPips, entryPrice, entryDate 
        
