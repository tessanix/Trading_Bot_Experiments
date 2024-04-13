import ssl
import pytz
import socket
from datetime import datetime 
from threading import Lock, Thread

import sys; sys.path.insert(1, '..')
from xtb_api import xtb_requests
from reinforcement_learning.trading_agent.actor_critic.agent import Agent
from strategies.Heikin_Ashi_Moving_Average_Strategy import HeikinAshiMovingAverage
# from datetime import datetime, timedelta


class TradingLoop:
    def __init__(self):
        ### starting values ###
        self.running = False
        self.runningLock = Lock()
        self.loopThread = Thread(target=self.loop)

        self.host = 'xapi.xtb.com'
        self.port = 5124 # port for DEMO account
        self.PERIOD_H4 = 240
        self.symbol = 'US500'

        self.strategy = HeikinAshiMovingAverage(useSR=False, useUpdateSl=False, uselongTermMA=False)
        self.agent = Agent()

        self.context = ssl.create_default_context()
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

    def loop(self,):

        positionId = 0
        last_hour = datetime.now(self._CetCestTimezone).now().hour
        slInPips, self.tpInPips, self.maxSlInPips, self.maxTpInPips = 0, 0, 0, 0
        entryPrice = 0.0
        entryDate = ''
        inPosition = False
        priceAlreadySeen = False

        with socket.create_connection((self.host, self.port)) as sock:
            with self.context.wrap_socket(sock, server_hostname=self.host) as ssock:

                _ = xtb_requests.login(ssock)
                observation = xtb_requests.getLastNCandlesH4(ssock, 100, 'US500')

                while self.get_running():

                    if not priceAlreadySeen:
                        if not inPosition:
                            capital:float = xtb_requests.getBalance(ssock) # TODO: change "balance" to "equity" in the function
                            inPosition, maxSlInPips, maxTpInPips, entryPrice, entryDate = self.strategy.checkIfCanEnterPosition(df=observation, i=-1, capital=capital)
                            if inPosition: 
                                slInPips, tpInPips = self.agent.updateSlAndTp(observation, maxSlInPips, maxTpInPips) # choose action
                                sl, tp = entryPrice+slInPips, entryPrice+tpInPips
                                positionId = xtb_requests.openBuyPosition(ssock, entryPrice, sl=sl, tp=tp, vol=0.01, symbol='US500')
                        else:
                            self.currentPrice = observation["close"].iloc[-1]
                            status = xtb_requests.checkPositionStatus(ssock, positionId)
                            closed, profit = status
                            if closed: 
                                inPosition = False
                            else:
                                slInPips, tpInPips = self.agent.updateSlAndTp(observation, maxSlInPips, maxTpInPips)
                                xtb_requests.modifyPosition(ssock, sl, tp, 0.01, positionId, symbol='US500')

                        priceAlreadySeen = True

                    actual_hour = datetime.now(self._CetCestTimezone).hour        
                    if actual_hour % 4 == 0 and actual_hour != last_hour:
                        # time.sleep(5) #sleep 5 seconds to let the server refresh his data ???
                        last_hour = actual_hour
                        observation = xtb_requests.getLastNCandlesH4(ssock, 100, 'US500')
                        priceAlreadySeen = False
                # END OF WHILE LOOP
 
    