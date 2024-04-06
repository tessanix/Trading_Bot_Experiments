import pandas as pd
from strategies.Strategy import Strategy
from random import choice
import numpy as np
from reinforcement_learning.trading_agent.actor_critic.agent import Agent


def actorCritictrainingLoop(df: pd.DataFrame, strategy: Strategy, agent:Agent, longTermMAPeriod:int=200, pipValue:float=50.0, n_games:int=1000, capital:float=4000.0):
    
    n_games = n_games
    score_history = []
    best_score = 0
    candlesWindow = 100

    for episode in range(n_games):

        ### starting values ###
        capital = capital #$
        pipValue = pipValue
        inPosition = False
        sl, tp, slInPips, tpInPips, maxSlInPips, maxTpInPips = 0, 0, 0, 0, 0, 0
        entryPrice = 0.0
        lot_size = 0.01
        done = False
        startIndex = choice(df.index[longTermMAPeriod+strategy.N:])
        i = 0
        reward = 0.0
        observation = df[["open", "close","low", "high"]].loc[startIndex+i-candlesWindow:startIndex+i]

        while not done:

            if not inPosition:
                inPosition, maxSlInPips, maxTpInPips, entryPrice, _ = strategy.checkIfCanEnterPosition(df.loc[startIndex+i-candlesWindow:startIndex+i], startIndex+i, capital)
                if inPosition: 
                    slInPips, tpInPips = agent.updateSlAndTp(observation, maxSlInPips, maxTpInPips) # choose action
                    i+=1
                    reward -= 0.5
                else: 
                    done = True
            else:
                observation_ = df[["open", "close","low", "high"]].loc[startIndex-candlesWindow+i:startIndex+i]
                
                sl, tp = entryPrice+slInPips, entryPrice+tpInPips
                currentPrice = df["close"].loc[startIndex+i]
                lose = currentPrice <= sl
                win = tp <= currentPrice
                if lose or win:
                    profit = tpInPips*pipValue*lot_size if win else slInPips*pipValue*lot_size
                    capital += profit 
                    reward += profit
                    inPosition = False
                    i+=1
                    done = True
                else:
                    if startIndex+i == len(df) and not done: # si l'épisode n'est pas fini mais qu'on a plus de données pour continuer
                        done = True
                    else:
                        if i<=1:
                            reward += (observation_["close"].iloc[-1] - entryPrice)/entryPrice # reward basé sur la variation du prix depuis entry_price
                            agent.learn(observation.to_numpy(), reward, observation_.to_numpy(), True)

                        else:
                            agent.learn(observation.to_numpy(), reward, observation_.to_numpy(), True)
                            reward += (observation_["close"].iloc[-1] - entryPrice)/entryPrice # reward basé sur la variation du prix depuis entry_price

                        slInPips, tpInPips = agent.updateSlAndTp(observation_, maxSlInPips, maxTpInPips)

                        observation = observation_
                        i+=1

        score_history.append(reward)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
    
        print(f'episode: {episode}, score: {reward}, avg_score: {avg_score}, nb iter: {i}')
        
                    

"""
LE REWARD DESIGN:

option1:
    - l'agent commence avec N points et N est décrémenté de x à chaque t periode 
        => plus le trade dure longtemps et plus l'agent perd de point
        => l'agent pourrait stoper le trade dès le début pour gagner tout le temps N points 
    - en touchant le take profit, l'agent gagne TP*M points
        => incite l'agent à toucher un TP éloigné

option 2:
    - en touchant le take profit, l'agent gagne TP*M points
        => incite l'agent à toucher un TP éloigné


"""


    
