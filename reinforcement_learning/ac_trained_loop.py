import pandas as pd
from strategies.Strategy import Strategy
from reinforcement_learning.trading_agent.actor_critic.agent import Agent
from tqdm import tqdm
import tensorflow as tf

def actorCritictrainedLoop(df: pd.DataFrame, strategy: Strategy, agent:Agent, longTermMAPeriod:int=200, pipValue:float=50.0, capital:float=4000.0):
    candlesWindow = 100
    capital = capital #$
    pipValue = pipValue
    lot_size = 0.01
    i = df.index[0] + longTermMAPeriod+strategy.N+candlesWindow
    tradesData = []

    with tqdm(total=len(df)-i) as pbar:
        while i < len(df):
            ### starting values ###
            entryDate = ''
            inPosition = False
            slInPips, tpInPips, maxSlInPips, maxTpInPips = 0, 0, 0, 0
            entryPrice = 0.0
            done = False
            observation = df[["open", "close","low", "high"]].loc[i-candlesWindow:i]

            while not done:

                if not inPosition:
                    inPosition, maxSlInPips, maxTpInPips, entryPrice, entryDate = strategy.checkIfCanEnterPosition(df.loc[i-candlesWindow:i], i, capital)
                    if inPosition: 
                        slInPips, tpInPips = agent.updateSlAndTp(observation, maxSlInPips, maxTpInPips) # choose action
                    else: 
                        done = True
                else:
                    observation_ = df[["open", "close","low", "high"]].loc[i-candlesWindow:i]

                    sl, tp = entryPrice+slInPips, entryPrice+tpInPips
                    currentPrice = df["close"].loc[i]
                    lose = currentPrice <= sl
                    win = tp <= currentPrice
                    if lose or win:
                        profit = tpInPips*pipValue*lot_size if win else slInPips*pipValue*lot_size
                        capital += profit 
                        done = True
                        tradesData.append({
                            "entry_date":entryDate, 
                            "exit_date":df["datetime"].iloc[i], 
                            "entry_price":entryPrice, 
                            "stop_loss":tf.get_static_value(sl), 
                            "take_profit":tf.get_static_value(tp), 
                            "profit":tf.get_static_value(profit), 
                            "capital_after_trade":tf.get_static_value(capital)
                        })
                    if i == len(df) and not done: # si l'épisode n'est pas fini mais qu'on a plus de données pour continuer
                        done = True
                    else:
                        slInPips, tpInPips = agent.updateSlAndTp(observation_, maxSlInPips, maxTpInPips)

                        observation = observation_
                i+=1
                pbar.update(1)
                if i==len(df):break

    return pd.DataFrame(tradesData)
    
    