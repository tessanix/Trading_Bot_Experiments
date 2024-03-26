import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift

def pivotid(df, l, n1, n2): #n1 n2 before and after candle l
    if l-n1 < 0 or l+n2 >= len(df):
        return 0
    
    pividlow=1
    pividhigh=1
    for i in range(l-n1, l+n2+1):
        if(df.low[l]>df.low[i]):
            pividlow=0
        if(df.high[l]<df.high[i]):
            pividhigh=0
    if pividlow and pividhigh:
        return 3
    elif pividlow:
        return 1
    elif pividhigh:
        return 2
    else:
        return 0


def pointpos(df, eps=20):
    if df['pivot']==1:
        return df['low']-eps
    elif df['pivot']==2:
        return df['high']+eps
    else:
        return np.nan


def getKeyZonesMinMaxFlattened(df: pd.DataFrame, unique_labels:list[float]) -> list[float]:
    keyZonesMinMax = []
    # keyZonesMean = []
    keyZonesMinMaxFlattened = []
    for label in unique_labels:
        min = df[df['cluster_labels'] == label]['pointpos'].min()
        max = df[df['cluster_labels'] == label]['pointpos'].max()
        # mean = df[df['cluster_labels'] == label]['pointpos'].mean()
        keyZonesMinMax.append((min, max))
        # keyZonesMean.append(mean)
        keyZonesMinMaxFlattened.append(min)
        keyZonesMinMaxFlattened.append(max)

    keyZonesMinMaxFlattened.sort(reverse=True)
    return keyZonesMinMaxFlattened

def filterKeyZonesTooClose(df: pd.DataFrame, keyZones: list[float], percent:int=4) -> list[float]:
    i = 0
    lastIndexKeyZone = 0
    keysToRemove = []
    lenght = len(keyZones)

    while True:
        if i == lenght-1: break

        percentOfCurrentPrice = (df["close"][i]/100)*percent

        if abs(keyZones[lastIndexKeyZone] - keyZones[i+1]) < percentOfCurrentPrice \
        and keyZones[lastIndexKeyZone] not in keysToRemove: # niveaux trop proches
            keysToRemove.append(keyZones[i+1]) 
            i = i+1 
        else:
            lastIndexKeyZone = i+1
            i = i+1 

    for key in keysToRemove: keyZones.remove(key)

    return keyZones


def getKeyLevels(df: pd.DataFrame, pivotN1N2:tuple[int,int]=(20,10), useFilter:bool=True, filterPercent:int=4, clusterBW:int=80) -> list[float]:

    df['pivot'] = df.apply(lambda x: pivotid(df, x.name,*pivotN1N2), axis=1)

    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)

    X = df[['pointpos']].dropna()

    clustering = MeanShift(bandwidth=clusterBW, cluster_all=True).fit(X)

    df['cluster_labels'] = np.NaN

    for index, label in zip(X.index, clustering.labels_):
        df.loc[index, 'cluster_labels'] = label

    unique_labels = df['cluster_labels'].value_counts()
    unique_labels = list(unique_labels.index)

    keyZonesMinMaxFlattened = getKeyZonesMinMaxFlattened(df, unique_labels)
    if useFilter:
        keyZonesMinMaxFlattened = filterKeyZonesTooClose(df, keyZonesMinMaxFlattened, filterPercent)

    return keyZonesMinMaxFlattened
