import pandas as pd
from strategies.Strategy import Strategy
from random import randrange

def actorCritictrainingLoop(df: pd.DataFrame, strategy: Strategy, longTermMAPeriod:int=200, pipValue:float=50.0) -> pd.DataFrame:
    
    
    n_games = 1000

    for _ in range(n_games):
        capital = 4000 #$
        inPosition = False
        entryPrice, sl, tp = 0, 0, 0
        slInPips, tpInPips = 0, 0
        pipValue = pipValue
        lot_size = 0.01
        entryDate = df["datetime"].iloc[0]
        tradesData = []
        
        # starting values:
        # done = False
        # score = 0.0
        startIndex = randrange(start=longTermMAPeriod+strategy.N, stop=len(df))
        i = 0
        reward = 30.0 
        observation = df[["open", "close","low", "high"]].iloc[startIndex-100:startIndex]

        while True:
            currentPrice = df["close"].iloc[startIndex+i]

            if not inPosition:
                inPosition, slInPips, tpInPips, entryPrice, entryDate = strategy.checkIfCanEnterPosition(df.iloc[startIndex-100+i+1:startIndex+i+1], startIndex+i-1, capital)
                i+=1
                reward -= 0.5
                if inPosition == False: break 
            else:
                i+=1
                reward -= 0.5
                observation_ = df[["open", "close","low", "high"]].iloc[startIndex-100+i+1:startIndex+i+1]
                
                
                newSlInPips = strategy.updateSl(currentPrice, entryPrice, tpInPips)
                if newSlInPips != 0: slInPips = newSlInPips
                sl, tp = entryPrice+slInPips, entryPrice+tpInPips
                lose = currentPrice <= sl
                win = tp <= currentPrice

                strategy.agent.learn(observation.to_numpy(), reward, observation_.to_numpy(), True)
                observation = observation_
                
                if lose or win:
                    profit = tpInPips*pipValue*lot_size if win else slInPips*pipValue*lot_size
                    capital += profit 
                    tradesData.append({
                        "entry_date":entryDate, 
                        "exit_date":df["datetime"].iloc[i], 
                        "entry_price":entryPrice, 
                        "stop_loss":sl, 
                        "take_profit":tp, 
                        "profit":profit, 
                        "capital_after_trade":capital
                    })
                    inPosition = False
                    break

    return pd.DataFrame(tradesData)


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


    
