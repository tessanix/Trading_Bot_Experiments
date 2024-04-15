import pandas as pd

def processListOfCandlesFromXtb(candlesList, maPeriod=0):
    open = close = low = high = []
    for candle in candlesList:
        open.append(candle['open'])
        close.append(candle['close']+candle['open'])
        high.append(candle['high']+candle['open'])
        low.append(candle['low']+candle['open'])

    df = pd.DataFrame({'open':open, 'close':close, 'high':high, 'low':low})
    if 0 < maPeriod:
        df["shortTermMA"] = df["close"].rolling(window=maPeriod).mean() # add moving average 50
    return df
