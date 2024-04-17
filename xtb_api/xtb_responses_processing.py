import pandas as pd
from utils import utility

def processListOfCandlesFromXtb(candlesList, digits, maPeriod=0):
    open = close = low = high = []
    divider = 10**digits
    for candle in candlesList:
        open.append(candle['open']/divider)
        close.append((candle['close']+candle['open'])/divider)
        high.append((candle['high']+candle['open'])/divider)
        low.append((candle['low']+candle['open'])/divider)

    df = pd.DataFrame({'open':open, 'close':close, 'high':high, 'low':low})
    if 0 < maPeriod:
        df["shortTermMA"] = df["close"].rolling(window=maPeriod).mean() # add moving average 50

    df = utility.heikinashi(df)
    df = utility.addHACandleColor(df)
    return df
