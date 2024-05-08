import math
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import utility
from tqdm.notebook import tqdm
from reinforcement_learning.trading_agent.ppo.agent_no_strat import Agent

# 0 = enter BUY position
# 1 = exit BUY position (SELL)
# 2 = do nothing

def PPOtrainedLoop(data: np.ndarray, test_indices:list, agent:Agent, pipValue:float=50.0, n_games:int=1000, capital:float=4000.0, d_model=4, candlesWindow=100):
    # load_checkPoint = load_checkPt
    capitals = []
    profits  = []
    trades_start_index = []
    nb_period_for_trade = []

    lot_size = 0.01
    maxRisk = 0.02 # 2%
    # dataLength = len(data)
    # previousBestCap = capital

    for episode in tqdm(range(n_games), leave=False):

        ### starting values ###
        inPosition, done = False, False
        entryPrice, maxSlInPips = 0.0, 0.0
        i = 0
        startIndex = test_indices[episode%len(test_indices)] # FIXME: if startIndex == dataLength => error index
        observation = data[startIndex+i-candlesWindow:startIndex+i, :d_model] # => "close", "open","high", "low"
        observation_:np.ndarray
        actionList = []
        profit = 0

        while not done:

            if not inPosition:
                entryPrice = observation[-1, 0]
                maxSlInPips = -utility.getSlInPipsForTrade(invested=capital*maxRisk, pipValue=pipValue, lotSize=0.01)
                action_mask = np.array([[0.0, -math.inf, 0.0]])
                action, _, _ = agent.choose_action(observation, maxSlInPips, entryPrice, action_mask) # choose action
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
                        done = True
                    else:
                        pass
                elif action == 1: # exit market
                    print("INVALID ACTION : AGENT CAN'T EXIT MARKET, HE IS NOT IN POSITION!")
                    done = True

                else: # action == 2 => do nothing
                    done = True

                observation = observation_

            else:
                i+=1
                observation_ = data[startIndex+i-candlesWindow:startIndex+i, :d_model]
                currentPrice = observation_[-1, 0]
                action_mask = np.array([[-math.inf, 0.0, 0.0]])
                action, _, _ = agent.choose_action(observation, maxSlInPips, entryPrice, action_mask)
                actionList.append(tf.get_static_value(action))
            
                if action == 0: # enter market
                    print("INVALID ACTION : AGENT CAN'T ENTER MARKET AGAIN, HE IS ALREADY IN!")
                    done = True

                elif action == 1: # exit market
                    lose = currentPrice <= entryPrice+maxSlInPips
                    profit = maxSlInPips*pipValue*lot_size if lose else (currentPrice - entryPrice)
                    capital += profit
                    done = True

                else: # action == 2 => do nothing
                    pass                    
                observation = observation_

        # END OF WHILE LOOP
        capitals.append(capital)
        profits.append(profits)
        trades_start_index.append(startIndex)
        nb_period_for_trade.append(i)

        print(f"episode:{episode}, capital:{capital:.2f}, actions:{actionList[-10:]}, nb iter:{i}")

    return pd.DataFrame({'capital':capitals, 
                         'profit':profits, 
                         'start_index':trades_start_index, 
                         'nb_period_for_trade': nb_period_for_trade
                        })