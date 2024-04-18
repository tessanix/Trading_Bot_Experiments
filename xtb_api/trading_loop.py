import pytz
from datetime import datetime 
import time
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, Future
import sys; sys.path.insert(1, '..')
from xtb_api.xtb_requests import XTBRequests
from reinforcement_learning.trading_agent.actor_critic.agent import Agent
from strategies.Heikin_Ashi_Moving_Average_Strategy import HeikinAshiMovingAverage


class TradingLoop:
    def __init__(self, clientSocket, symbol='US500'):
        ### starting values ###
        self.running = False
        self.runningLock = Lock()
        # self.tradingLoopThread = Thread(target=self.tradingLoop)
        self.clientSocket = clientSocket
        self.isLogged = False
        self.isLoggedLock = Lock()

        self.pingExecution = Future()
        self.tradingExecution = Future()
        self.lastTimePinged = None
        # self.pingloopThread = Thread(target=self.pingLoop)
        self.pool = ThreadPoolExecutor(max_workers=4)

        self.xtbRequest = XTBRequests(symbol)
        self.strategy = HeikinAshiMovingAverage(useSR=False, useUpdateSl=False, uselongTermMA=False)
        self.agent = Agent()

        self._CetCestTimezone = pytz.timezone("Europe/Paris") # it is a CET/CEST timezone

    def login(self):
        print("TradingLoop::login func")
        response = self.xtbRequest.login()
        if response["status"] == True:
            self.set_isLogged(True)
            self.pingExecution = self.pool.submit(self.pingLoop)
            # self.pingloopThread.start()
        else:
            print(f"command sent: 'login'")
            print(f'Error code: {response["errorCode"]}')
            print(f'Error description: {response["errorDescr"]}')
            # self.stopPingLoop()
            self.set_isLogged(False)
            self.xtbRequest.closeSocket()
            self.pingExecution.result(timeout=2)
    
    def logout(self):
        self.xtbRequest.logout()
        self.set_isLogged(False)
        print('logout done!')


    def pingLoop(self):

        while self.get_isLogged(): #TODO: dev here
            # actual_time = datetime.now(self._CetCestTimezone)
            # if(actual_time.minute%2==0 and actual_time!=self.lastTimePinged):
            response = self.xtbRequest.ping()
            self.lastTimePinged = datetime.now(self._CetCestTimezone)
            print(f'ping status: {response["status"]}')
            if response["status"] == True:
                self.clientSocket.emit('ping', str(self.lastTimePinged))
                time.sleep(60*9) # sleep 9 minites
                print('ping time elapsed')
            else:
                self.set_isLogged(False)



    def set_isLogged(self, value:bool):
        with self.isLoggedLock:
            self.isLogged = value

    def get_isLogged(self):
        with self.isLoggedLock:
            return self.isLogged
    
    def get_running(self) -> bool:
        with self.runningLock:
            return self.running
        
    def set_running(self, _running:bool): 
        with self.runningLock:
            self.running = _running

    def runTradingLoop(self):
        print("TradingLoop::runTradingLoop func")
        self.set_running(True)
        self.tradingExecution = self.pool.submit(self.tradingLoop)

    def stopTradingLoop(self):
        self.set_running(False)
        self.tradingExecution.result()
     

    def tradingLoop(self):

        last_hour = datetime.now(self._CetCestTimezone).now().hour
        slInPips, tpInPips, maxSlInPips, maxTpInPips = 0, 0, 0, 0
        entryPrice = 0.0
        inPosition = False
        priceAlreadySeen = False
        print('entering trading loop')
        if self.get_isLogged():
            
            observation = self.xtbRequest.getLastNCandlesH4(nCandles=100, maPeriod=50)

            while self.get_running():

                if not priceAlreadySeen:
                    print('checking new price')
                    if not inPosition:
                        capital = self.xtbRequest.getBalance() # TODO: change "balance" to "equity" in the function
                        inPosition, maxSlInPips, maxTpInPips, entryPrice, _ = self.strategy.checkIfCanEnterPosition(df=observation, i=-1, capital=capital)
                        if inPosition: 
                            print('has entered position')
                            slInPips, tpInPips = self.agent.updateSlAndTp(observation, maxSlInPips, maxTpInPips) # choose action
                            sl, tp = entryPrice+slInPips, entryPrice+tpInPips
                            _ = self.xtbRequest.openBuyPosition(entryPrice, sl=sl, tp=tp, vol=0.01)
                    else:
                        print('checking position status')
                        self.currentPrice = observation["close"].iloc[-1]
                        status = self.xtbRequest.checkPositionStatus()
                        self.clientSocket.emit('trade_status', status)
                        if status['closed']: 
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

 
    