import pandas as pd
from strategies.Strategy import Strategy


def strategyLoop(df: pd.DataFrame, strategy: Strategy, longTermMAPeriod:int=200) -> pd.DataFrame:

    CAPITAL = 1000 #$
    inPosition = False
    entryPrice, sl, tp = 0, 0, 0
    slInPips, tpInPips = 0, 0
    pipValue = 50
    lot_size = 0.01
    entryDate = df["datetime"][0]

    trades = pd.DataFrame(columns=["entry_date","exit_date", "entry_price", "stop_loss", "take_profit", "profit", "capital_after_trade"])

    for i in range(longTermMAPeriod+strategy.N, len(df)):

        currentPrice = df["close"][i]

        if not inPosition:
            inPosition, slInPips, tpInPips, entryPrice, entryDate = strategy.checkIfCanEnterPosition(df, i, CAPITAL)
        else:
            newSlInPips = strategy.updateSl(currentPrice, entryPrice, tpInPips)
            if newSlInPips != 0: slInPips = newSlInPips
            sl, tp = entryPrice+slInPips, entryPrice+tpInPips
            lose = currentPrice <= sl
            win = tp <= currentPrice
            if lose or win:
                profit = tpInPips*pipValue*lot_size if win else slInPips*pipValue*lot_size
                CAPITAL += profit 
                new_trade = {"entry_date":entryDate, "exit_date":df["datetime"][i], "entry_price":entryPrice, "stop_loss":sl, "take_profit":tp, "profit":profit, "capital_after_trade":CAPITAL} 
                trades.loc[len(trades)] = new_trade
                inPosition = False

    return trades
    
