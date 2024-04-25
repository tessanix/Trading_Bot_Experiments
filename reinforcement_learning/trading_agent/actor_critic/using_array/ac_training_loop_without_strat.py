from random import randrange
import numpy as np
from reinforcement_learning.trading_agent.actor_critic.agent_no_strat import Agent
# import sys; sys.path.insert(1, '../../../../')
from utils import utility
import tensorflow as tf

# 0 = enter BUY position
# 1 = exit BUY position (SELL)
# 2 = do nothing

def actorCritictrainingLoop(data: np.ndarray, agent:Agent, pipValue:float=50.0, n_games:int=1000, capital:float=4000.0, save_model=True):
    # load_checkPoint = load_checkPt
    score_history = []
    best_score = 0.0
    candlesWindow = 100
    lot_size = 0.01
    maxRisk = 0.02 # 2%
    dataLength = len(data)
    # capital = _capital
    # previousBestCap = capital

    for episode in range(n_games):

        ### starting values ###
        inPosition, done = False, False
        entryPrice, reward, maxSlInPips = 0.0, 0.0, 0.0
        i = 0
        startIndex = randrange(candlesWindow, dataLength) # FIXME: if startIndex == dataLength => error index
        observation = data[startIndex+i-candlesWindow:startIndex+i, :4] # => "close", "open","high", "low"
        observation_:np.ndarray
        actionList = []
        score_this_ep = 0

        while not done:

            if not inPosition:
                entryPrice = observation[-1, 0]
                maxSlInPips = -utility.getSlInPipsForTrade(invested=capital*maxRisk, pipValue=pipValue, lotSize=0.01)

                action = agent.choose_action(observation, maxSlInPips, entryPrice) # choose action

                actionList.append(tf.get_static_value(action))
                i+=1
                observation_ = data[startIndex+i-candlesWindow:startIndex+i, :4]
                currentPrice = observation_[-1, 0]

                if action == 0: # enter market
                    inPosition = True
                    reward = -1.0

                    lose = currentPrice <= entryPrice+maxSlInPips
                    if lose:
                        profit = maxSlInPips*pipValue*lot_size
                        capital += profit 
                        reward += profit
                        done = True
                    else:
                        reward += (currentPrice - entryPrice) # reward basé sur la variation du prix depuis entry_price

                elif action == 1: # exit market
                    reward = -10.0
                    done = True

                else: # do nothing
                    # reward += 0.0
                    done = True
                
                
                agent.learn(observation, reward, observation_, maxSlInPips, done, entryPrice)
                observation = observation_
                score_this_ep += reward

            else:
                # if i!=1:
                #     currentPrice = observation_[-1, 0]
                i+=1
                observation_ = data[startIndex+i-candlesWindow:startIndex+i, :4]
                currentPrice = observation_[-1, 0]

                action = agent.choose_action(observation, maxSlInPips, entryPrice)
                actionList.append(tf.get_static_value(action))
            

                if action == 0: # enter market
                    reward = -10.0
                    done = True

                elif action == 1: # exit market
                    lose = currentPrice <= entryPrice+maxSlInPips
                    profit = maxSlInPips*pipValue*lot_size if lose else (currentPrice - entryPrice)
                    capital += profit
                    reward = profit # reward basé sur la variation du prix depuis entry_price
                    done = True

                # else: # do nothing
                #     pass
                    # reward += 0.0
                    # done = True
                
                agent.learn(observation, reward, observation_, maxSlInPips, done, entryPrice)
                observation = observation_
                score_this_ep += reward

        # END OF WHILE LOOP
        score_history.append(score_this_ep)
        avg_score = np.mean(score_history[-80:])

        if avg_score > best_score: #and len(score_history) > 10: 
            best_score = avg_score
            if save_model:
                agent.save_models()
        # elif episodeTrainedOn%20==0 and previousBestCap < capital:
        #     previousBestCap = capital
        #     if save_model:
        #         agent.save_models()

        print(f"episode:{episode}, capital:{capital:.2f}, actions:{actionList[-10:]}, score:{score_this_ep:.3f}, avg_score:{avg_score:.3f}, nb iter:{i}")


    return score_history