import pandas as pd

def processListOfCandlesFromXtb(candlesList):
    open = close = low = high = []
    for candle in candlesList:
        open.append(candle['open'])
        close.append(candle['close']+candle['open'])
        high.append(candle['high']+candle['open'])
        low.append(candle['low']+candle['open'])
    return pd.DataFrame({'open':open, 'close':close, 'high':high, 'low':low})
