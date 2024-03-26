from strategies.Strategy import Strategy
import utility
import pandas as pd



class MovingAverage(Strategy):

    def __init__(self, levels: list[float], useSR: bool, useUpdateSl:bool):
        self.maxRisk = 0.02 # 2%
        self.N = 3
        self.keyLevels = levels
        self.useSR = useSR
        self.useUpdateSl = useUpdateSl
    
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

        ma50ZoneMin = df["MA50"][i]-(df["close"][i]/100)*3 # => MA - 2% du prix
        ma50ZoneMax = df["MA50"][i]+(df["close"][i]/100)*3 # => MA + 2% du prix

        isLastNCandlesInMA50Zone = False
        for j in range(i-self.N, i):
            if utility.between(df["close"][j], ma50ZoneMin, ma50ZoneMax):
                isLastNCandlesInMA50Zone = True
                break
        
        if df["MA50"][i] < df["open"][i] and df["open"][i] < df["close"][i] and isLastNCandlesInMA50Zone:
            entryDate = df["datetime"][i]
            entryPrice =  df["close"][i]
            if self.useSR:
                # print("entry condition == True ")
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
        
