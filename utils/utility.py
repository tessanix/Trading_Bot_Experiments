from operator import index
import pandas as pd

def between(a, min, max): 
    return min <= a and a <= max

def getSlInPipsForTrade(invested:float, pipValue:float, lotSize:float) -> float:
    sl_in_pips = invested/lotSize/pipValue
    return sl_in_pips


def heikinashi(df: pd.DataFrame) -> pd.DataFrame:
    df['HA close']=(df['open']+ df['high']+ df['low']+df['close'])/4
    HA_open = []
    for i in df.index:
        if i == df.index[0]:
            HA_open.append( (df['open'][i] + df['close'][i] )/ 2)
        else:
            HA_open.append( (df['open'][i-1] + df['close'][i-1] )/ 2)
    df['HA open'] = HA_open
    df['HA high']=df[['open','close','high']].max(axis=1)
    df['HA low']=df[['open','close','low']].min(axis=1)
    return df


def addHACandleColor(df: pd.DataFrame) -> pd.DataFrame:
    colors = []
    for i in df.index:
        color = "green" if df["HA open"][i] < df["HA close"][i] else "red"
        colors.append(color)
    df["HA color"] = colors
    return  df

def getTotalProfitInPercent(trades: pd.DataFrame, display: bool=False):
    startCapital = trades["capital_after_trade"][trades.index[0]] - trades["profit"][trades.index[0]]
    endCapital = trades["capital_after_trade"][trades.index[-1]]
    profitInPercent = ((endCapital-startCapital)/startCapital)*100
    if display: print("profit in percentage: ", profitInPercent, "%")
    return profitInPercent


def countNbLossesAndWins(trades: pd.DataFrame, display: bool=False) -> tuple[int, int, int]:
    nbLosses = nbWins = nbNull = 0
    nbLosses = trades.loc[trades['profit']<0, 'profit'].count()
    nbWins = trades.loc[trades['profit']>0, 'profit'].count()
    nbNull = trades.loc[trades['profit']==0, 'profit'].count()
    if display:  print("nbLosses: ", nbLosses, "nbWins: ",nbWins, "nbNull: ", nbNull)
    return nbLosses, nbWins, nbNull

def computeAvgLossesAndWins(trades: pd.DataFrame, display: bool=False) -> tuple[float, float]:
    avgLosses = trades.loc[trades['profit']<0, 'profit'].mean()
    avgWins = trades.loc[trades['profit']>0, 'profit'].mean()
    if display:  print("avgLosses: ", avgLosses, "avgWins: ", avgWins)
    return avgLosses, avgWins


import plotly.express as px
import plotly.graph_objects as go

def displayCapitalOverTime(trades: pd.DataFrame):
    fig = px.line(trades, x="entry_date", y="capital_after_trade")
    fig.show()



def displayChartPrice(df:pd.DataFrame, keylevels:list[float]):
    fig = go.Figure(
        data=[go.Candlestick(
                x=df["date"],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                increasing_line_color= 'green', 
                decreasing_line_color= 'red')
            ]
    )
    for k in keylevels:
        fig.add_hline(y=k)

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(paper_bgcolor='black', plot_bgcolor='black')
    fig.show()

def getProfitEachMonths(trades: pd.DataFrame):
    pass

