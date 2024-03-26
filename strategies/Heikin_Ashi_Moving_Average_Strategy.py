from strategies.Strategy import Strategy
import utility
import pandas as pd



class HeikinAshiMovingAverage(Strategy):

    def __init__(self, levels: list[float]=[], useSR: bool=True, useUpdateSl:bool=True, longTermMAPeriod=200, uselongTermMA:bool=True):
        self.maxRisk = 0.02 # 2%
        self.N = 3
        self.keyLevels = levels
        self.useSR = useSR
        self.useUpdateSl = useUpdateSl
        self.uselongTermMA = uselongTermMA
        self.longTermMAPeriod = longTermMAPeriod

    def setParams(self, **kwargs):
        self.keyLevels = kwargs["keyLevels"]
        self.useSR = kwargs["useSR"]
        self.useUpdateSl = kwargs["useUpdateSl"]
        self.uselongTermMA = kwargs["uselongTermMA"]
        self.longTermMAPeriod = kwargs["longTermMAPeriod"]
    
    def determineSlAndTp(self, capital:float, price: float, keyLevels: list[float]) -> tuple[bool, float, float]:
        slInPips = -utility.getSlInPipsForTrade(
                            invested = capital*self.maxRisk,
                            pipValue = 50, # valeur du pip pour le SP500 pour un lot standard = 50
                            lotSize = 0.01 # micro lot
                        )
        tpInPips = -slInPips
        resistance = support = 0
        done = False

        for i in range(len(keyLevels)):
            sl = price+slInPips
            # keyLevels[i==0] is the greatest level
            if i==0 and sl < keyLevels[i] and keyLevels[i] < price: 
                slInPips = keyLevels[i]-price # négatif
                done = True
                break
            elif i!=0 and keyLevels[i] < price and price < keyLevels[i-1]:
                resistance = keyLevels[i-1]
                support = keyLevels[i]
                break

        if done: 
            return False, slInPips, tpInPips
        
        middleSR = (resistance+support)/2

        isBelowMiddleSR = price < middleSR

        if isBelowMiddleSR:
            sl = price+slInPips
            tp = price+tpInPips
            if sl < support and price*0.02 <= price-support: 
                slInPips = support-price # négatif
            if tp < resistance: 
                tpInPips = resistance-price

        return isBelowMiddleSR, slInPips, tpInPips

    def updateSl(self, currentPrice: float, entryPrice:float, tpInPips:float) -> float:
        newSlInPips = 0
        if self.useUpdateSl and tpInPips/2 < currentPrice-entryPrice:
            newSlInPips = tpInPips/4 #positif
        return newSlInPips


    def checkIfCanEnterPosition(self, df: pd.DataFrame, i: int, capital: float) -> tuple[bool, float, float, float, str]:
        inPosition, slInPips, tpInPips, entryPrice, entryDate = False, 0, 0, 0, ""
        
        allowedToTrade = True
        
        if self.uselongTermMA:
           allowedToTrade = True if df["longTermMA"][i] < df["HA open"][i] else False

        if allowedToTrade:
            shortTermMAZoneMin = df["shortTermMA"][i]-(df["close"][i]/100)*3 # => MA - 2% du prix
            shortTermMAZoneMax = df["shortTermMA"][i]+(df["close"][i]/100)*3 # => MA + 2% du prix
        
            isLastNCandlesInshortTermMAZone = False
            for j in range(i-self.N, i):
                if utility.between(df["HA close"][j], shortTermMAZoneMin, shortTermMAZoneMax):
                    isLastNCandlesInshortTermMAZone = True
                    break
            
            if df["shortTermMA"][i] < df["HA open"][i] and df["HA color"][i] == "green" and isLastNCandlesInshortTermMAZone:
                entryDate = df["datetime"][i]
                entryPrice =  df["close"][i]
                if self.useSR:
                    isBelowMiddleSR, slInPips, tpInPips = self.determineSlAndTp(capital, entryPrice, self.keyLevels)
                    inPosition = isBelowMiddleSR
                    
                else:
                    slInPips = -utility.getSlInPipsForTrade(
                                invested = capital*self.maxRisk,
                                pipValue = 50, # valeur du pip pour le SP500 pour un lot standard = 50
                                lotSize = 0.01 # micro lot
                            )
                    inPosition = True
                    tpInPips = -slInPips

        return inPosition, slInPips, tpInPips, entryPrice, entryDate    
        
