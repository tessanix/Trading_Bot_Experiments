from strategies.Strategy import Strategy
from random import randrange
import numpy as np
from reinforcement_learning.trading_agent.actor_critic.agent import Agent
import tensorflow as tf

def actorCritictrainingLoop(data: np.ndarray, strategy: Strategy, agent:Agent, pipValue:float=50.0, n_games:int=1000, capital:float=4000.0, load_checkPt=False, save_model=True):
    load_checkPoint = load_checkPt
    n_games = n_games
    score_history = []
    best_score = 0
    candlesWindow = 100
    capital = capital #$
    pipValue = pipValue
    lot_size = 0.01
    dataLength = len(data)
    episodeTrainedOn = 0

    for episode in range(n_games):

        ### starting values ###
        inPosition, done = False, False
        sl, tp, slInPips, tpInPips, maxSlInPips, maxTpInPips = 0, 0, 0, 0, 0, 0
        entryPrice, reward = 0.0, 0.0
        i = 0
        startIndex = randrange(candlesWindow+strategy.N, dataLength) # FIXME: if startIndex == dataLength => error index
        observation = data[startIndex+i-candlesWindow:startIndex+i, :4] # => "close", "open","high", "low"
        slList, tpList = [], []

        while not done:

            if not inPosition:
                inPosition, maxSlInPips, maxTpInPips, entryPrice = strategy.checkIfCanEnterPosition_2(data[startIndex+i-candlesWindow:startIndex+i], capital)
                if inPosition: 
                    slInPips, tpInPips = agent.updateSlAndTp(observation, maxSlInPips, maxTpInPips, entryPrice, entryPrice) # choose action
                    slList.append(tf.get_static_value(slInPips))
                    tpList.append(tf.get_static_value(tpInPips))

                    i+=1
                    reward -= 0.5
                else: 
                    done = True
            else:
                observation_ = data[startIndex+i-candlesWindow:startIndex+i, :4]
                
                sl, tp = entryPrice+slInPips, entryPrice+tpInPips
                currentPrice = observation_[-1, 0]
                lose = currentPrice <= sl
                win = tp <= currentPrice
                if lose or win:
                    profit = tpInPips*pipValue*lot_size if win else slInPips*pipValue*lot_size
                    capital += profit 
                    reward += profit
                    done = True

                if (startIndex+i == dataLength or i>300) and not done: # si l'épisode n'est pas fini mais qu'on a plus de données pour continuer
                    done = True
                else:
                    if i==1:
                        reward += (currentPrice - entryPrice)/entryPrice # reward basé sur la variation du prix depuis entry_price
                        if not load_checkPoint:
                            agent.learn(observation, reward, observation_, maxSlInPips, maxTpInPips, done, entryPrice)

                    else:
                        if not load_checkPoint:
                            agent.learn(observation, reward, observation_, maxSlInPips, maxTpInPips, done, entryPrice)
                        reward += (currentPrice - entryPrice)/entryPrice # reward basé sur la variation du prix depuis entry_price

                    slInPips, tpInPips = agent.updateSlAndTp(observation_, maxSlInPips, maxTpInPips, currentPrice, entryPrice)
                    slList.append(tf.get_static_value(slInPips))
                    tpList.append(tf.get_static_value(tpInPips))

                    observation = observation_
                i+=1

        if inPosition:
            episodeTrainedOn+=1
            score_history.append(reward)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score: 
                best_score = avg_score
                if save_model:
                    agent.save_models()

            slList = [ round(elem,2) for elem in slList ][-10:]
            tpList = [ round(elem,2) for elem in tpList ][-10:]
            print(f"episode:{episode}, capital:{capital:.2f}, ep. trained on:{episodeTrainedOn}, slInPips:{slList}, tpInPips:{tpList}, score:{reward:.3f}, avg_score:{avg_score:.3f}, nb iter:{i}")


    return score_history
    
        
                    

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


    
