import pandas as pd
from strategies.Strategy import Strategy


def strategyLoop(df: pd.DataFrame, strategy: Strategy, longTermMAPeriod:int=200, pipValue:float=50.0) -> pd.DataFrame:
    
    capital = 4000 #$
    inPosition = False
    entryPrice, sl, tp = 0, 0, 0
    slInPips, tpInPips = 0, 0
    pipValue = pipValue
    lot_size = 0.01
    entryDate = df["datetime"].iloc[0]
    tradesData = []

    for i in df.index[longTermMAPeriod+strategy.N:]:

        currentPrice = df["close"].iloc[i]

        if not inPosition:
            inPosition, slInPips, tpInPips, entryPrice, entryDate = strategy.checkIfCanEnterPosition(df, i, capital)
        else:
            newSlInPips = strategy.updateSl(currentPrice, entryPrice, tpInPips)
            if newSlInPips != 0: slInPips = newSlInPips
            sl, tp = entryPrice+slInPips, entryPrice+tpInPips
            lose = currentPrice <= sl
            win = tp <= currentPrice
            if lose or win:
                profit = tpInPips*pipValue*lot_size if win else slInPips*pipValue*lot_size
                capital += profit 
                tradesData.append({
                    "entry_date":entryDate, 
                    "exit_date":df["datetime"].iloc[i], 
                    "entry_price":entryPrice, 
                    "stop_loss":sl, 
                    "take_profit":tp, 
                    "profit":profit, 
                    "capital_after_trade":capital
                })
                inPosition = False

    return pd.DataFrame(tradesData)
    
