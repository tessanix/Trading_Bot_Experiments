import pytz
from datetime import datetime 
from threading import Lock, Thread

import sys; sys.path.insert(1, '..')
from xtb_api.xtb_requests import XTBRequests
from reinforcement_learning.trading_agent.actor_critic.agent import Agent
from strategies.Heikin_Ashi_Moving_Average_Strategy import HeikinAshiMovingAverage
# from datetime import datetime, timedelta


class TradingLoop:
    def __init__(self, symbol='US500'):
        ### starting values ###
        self.running = False
        self.runningLock = Lock()
        self.loopThread = Thread(target=self.loop)

        self.xtbRequest = XTBRequests(symbol)
        self.strategy = HeikinAshiMovingAverage(useSR=False, useUpdateSl=False, uselongTermMA=False)
        self.agent = Agent()

        self._CetCestTimezone = pytz.timezone("Europe/Paris") # it is a CET/CEST timezone

    def get_running(self) -> bool:
        with self.runningLock:
            return self.running
        
    def set_running(self, _running:bool): 
        with self.runningLock:
            self.running = _running

    def runLoop(self):
        self.set_running(True)
        self.loopThread.start()

    def stopLoop(self):
        self.set_running(False)
        self.loopThread.join()
        self.loopThread = Thread(target=self.loop)

    def loop(self):

        last_hour = datetime.now(self._CetCestTimezone).now().hour
        slInPips, tpInPips, maxSlInPips, maxTpInPips = 0, 0, 0, 0
        entryPrice = 0.0
        entryDate = ''
        inPosition = False
        priceAlreadySeen = False

        _ = self.xtbRequest.login()
        observation = self.xtbRequest.getLastNCandlesH4(nCandles=100)

        while self.get_running():

            if not priceAlreadySeen:
                if not inPosition:
                    capital = self.xtbRequest.getBalance() # TODO: change "balance" to "equity" in the function
                    inPosition, maxSlInPips, maxTpInPips, entryPrice, entryDate = self.strategy.checkIfCanEnterPosition(df=observation, i=-1, capital=capital)
                    if inPosition: 
                        slInPips, tpInPips = self.agent.updateSlAndTp(observation, maxSlInPips, maxTpInPips) # choose action
                        sl, tp = entryPrice+slInPips, entryPrice+tpInPips
                        _ = self.xtbRequest.openBuyPosition(entryPrice, sl=sl, tp=tp, vol=0.01)
                else:
                    self.currentPrice = observation["close"].iloc[-1]
                    status = self.xtbRequest.checkPositionStatus()
                    closed, profit = status
                    if closed: 
                        inPosition = False
                        self.xtbRequest.positionId = 0
                    else:
                        slInPips, tpInPips = self.agent.updateSlAndTp(observation, maxSlInPips, maxTpInPips)
                        _ = self.xtbRequest.modifyPosition(sl, tp, 0.01)

                priceAlreadySeen = True

            actual_hour = datetime.now(self._CetCestTimezone).hour        
            if actual_hour % 4 == 0 and actual_hour != last_hour:
                # time.sleep(5) #sleep 5 seconds to let the server refresh his data ???
                last_hour = actual_hour
                observation = self.xtbRequest.getLastNCandlesH4(nCandles=100)
                priceAlreadySeen = False

        # END OF WHILE LOOP
        self.xtbRequest.closeSocket()

 
    