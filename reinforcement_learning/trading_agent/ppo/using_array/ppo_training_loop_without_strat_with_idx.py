import math
import numpy as np
import tensorflow as tf
from utils import utility
from tqdm.notebook import tqdm
from reinforcement_learning.trading_agent.ppo.agent_no_strat import Agent

# 0 = enter BUY position
# 1 = exit BUY position (SELL)
# 2 = do nothing

def PPOtrainingLoop(data: np.ndarray, train_indices:list, agent:Agent, pipValue:float=50.0, n_games:int=1000, capital:float=4000.0, save_model=True, d_model=4, candlesWindow=100):
    # load_checkPoint = load_checkPt
    score_history = []
    capitals = []
    
    best_score = 0.0
    lot_size = 0.01
    maxRisk = 0.02 # 2%
    # dataLength = len(data)
    # previousBestCap = capital
    N = 20
    n_steps = 0
    learn_iters = 0

    for episode in tqdm(range(n_games), leave=False):

        ### starting values ###
        inPosition, done = False, False
        entryPrice, reward, maxSlInPips = 0.0, 0.0, 0.0
        i = 0
        startIndex = train_indices[episode%len(train_indices)] # FIXME: if startIndex == dataLength => error index
        observation = data[startIndex+i-candlesWindow:startIndex+i, :d_model] # => "close", "open","high", "low"
        observation_:np.ndarray
        actionList = []
        rewards_this_ep = []

        while not done:

            if not inPosition:
                entryPrice = observation[-1, 0]
                maxSlInPips = -utility.getSlInPipsForTrade(invested=capital*maxRisk, pipValue=pipValue, lotSize=0.01)
                action_mask = np.array([[0.0, -math.inf, 0.0]])
                action, prob, val = agent.choose_action(observation, maxSlInPips, entryPrice, action_mask, training=True) # choose action
                n_steps += 1
                actionList.append(tf.get_static_value(action))
                i+=1
                observation_ = data[startIndex+i-candlesWindow:startIndex+i, :d_model]
                currentPrice = observation_[-1, 0]

                if action == 0: # enter market
                    inPosition = True
                    lose = currentPrice <= entryPrice+maxSlInPips
                    if lose:
                        profit = maxSlInPips*pipValue*lot_size
                        capital += profit 
                        reward = profit - 1.0 # - spread
                        done = True
                    else:
                        reward = (currentPrice - entryPrice) - 1.0 # reward basé sur la variation du prix depuis entry_price - spread

                elif action == 1: # exit market
                    reward = -50.0
                    print("INVALID ACTION : AGENT CAN'T EXIT MARKET, HE IS NOT IN POSITION!")
                    done = True

                else: # action == 2 => do nothing
                    reward = 0.0
                    done = True
                
                agent.store_transition(observation, maxSlInPips, entryPrice, action, prob, val, reward, done)
                if n_steps % N == 0:
                    # agent.learn()
                    agent.learn()
                    learn_iters += 1

                observation = observation_
                rewards_this_ep.append(reward)

            else:
                i+=1
                observation_ = data[startIndex+i-candlesWindow:startIndex+i, :d_model]
                currentPrice = observation_[-1, 0]
                action_mask = np.array([[-math.inf, 0.0, 0.0]])
                action, prob, val = agent.choose_action(observation, maxSlInPips, entryPrice, action_mask, training=True)
                n_steps += 1
                actionList.append(tf.get_static_value(action))
            
                if action == 0: # enter market
                    reward = -50.0
                    print("INVALID ACTION : AGENT CAN'T ENTER MARKET AGAIN, HE IS ALREADY IN!")
                    done = True

                elif action == 1: # exit market
                    lose = currentPrice <= entryPrice+maxSlInPips
                    profit = maxSlInPips*pipValue*lot_size if lose else (currentPrice - entryPrice)
                    capital += profit
                    reward = profit # reward basé sur la variation du prix depuis entry_price
                    done = True

                else: # action == 2 => do nothing
                    reward = 0.0
                    
                agent.store_transition(observation, maxSlInPips, entryPrice, action, prob, val, reward, done)
                if n_steps % N == 0:
                    # agent.learn()
                    agent.learn()
                    learn_iters += 1

                observation = observation_
                rewards_this_ep.append(reward)

        # END OF WHILE LOOP
        capitals.append(capital)
        score_history.append(sum(rewards_this_ep))
        avg_score = np.mean(score_history[-80:])

        if avg_score > best_score and len(score_history) > 10: 
            best_score = avg_score
            if save_model:
                agent.save_models()
        # elif episode%500==0 and previousBestCap < capital:
        #     previousBestCap = capital
        #     if save_model:
        #         agent.save_models()
        rewards_this_ep = [ round(elem,1) for elem in rewards_this_ep ][-10:]
        print(f"episode:{episode}, capital:{capital:.2f}, actions:{actionList[-10:]}, rewards:{rewards_this_ep}, avg_score:{avg_score:.3f}, nb iter:{i}")


    return score_history, capitals