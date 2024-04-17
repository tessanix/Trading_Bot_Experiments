import ssl
import json
import socket 
from datetime import datetime, timedelta
from xtb_api.xtb_responses_processing import processListOfCandlesFromXtb


class XTBRequests():
    def __init__(self, symbol='US500'):
        
        self.orderCommands = {"BUY":0, "SELL":1}
        self.orderTypes = {"OPEN":0, "PENDING":1, "CLOSE":2, "MODIFY":3, "DELETE":4}
        self.positionId = 0
        self.PERIOD_H4 = 240
        self.symbol = symbol

        self.host = 'xapi.xtb.com'
        self.port = 5124 # port for DEMO account

        self.sock:socket.socket
        self.ssock:ssl.SSLSocket
      
    def closeSocket(self):
        self.ssock.shutdown(2)
        self.ssock.close()
        self.sock.close()

    def readConfig(self):
        config = {}
        with open("../xtb_api/config.ini") as configFile:
            for line in configFile:
                name, _, var = line.partition("=")
                config[name] = var.strip('\n')
        return config
    
    def sendCommand(self, command:dict, returnAll=False):
        request = json.dumps(command).encode("UTF-8")
        self.ssock.send(request)
        END = b'\n\n'
        response_buffer = b''
        while True:
            chunk = self.ssock.recv(4096)
            response_buffer += chunk
            if END in chunk: break

        jsonObject:dict = json.loads(response_buffer.decode())
        return jsonObject

    def ping(self):
        command = {"command": "ping"}
        jsonObject = self.sendCommand(command)
        return jsonObject

    def login(self):
        self.sock = socket.create_connection((self.host, self.port))
        self.ssock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=self.host)

        config = self.readConfig()
        command = {
            "command": "login",
            "arguments": {
                "userId":   config['DEMO_ID'],
                "password": config['PWD']
            }
        }
        jsonObject = self.sendCommand(command)
        return jsonObject
    
    def logout(self):
        command = {"command": "logout"}
        self.sendCommand(command)
        self.closeSocket()
    

    def getServerTime(self):
        command = {"command": "getServerTime"}
        jsonObject = self.sendCommand(command)
        timestampInMs = jsonObject["returnData"]["time"]
        datetime_object = datetime.fromtimestamp(timestampInMs/1000.0)
        return datetime_object
        

    def getLastNCandlesH4(self, nCandles=100, maPeriod=50):
        currentDatetime = datetime.now()
        delta = timedelta(days=50) 
        new_datetime = currentDatetime - delta # 2 mois en arrière
        timestampInMs = new_datetime.timestamp()*1000
        command = {
            "command": "getChartLastRequest",
            "arguments": {
                "info": {
                    "period": self.PERIOD_H4,
                    "start": timestampInMs,
                    "symbol": self.symbol
                }
            }
        }
        jsonObject = self.sendCommand(command)
        listOfCandles = jsonObject["returnData"]["rateInfos"]
        digits = jsonObject["returnData"]["digits"]
        if nCandles+maPeriod <= len(listOfCandles):
            data = processListOfCandlesFromXtb(listOfCandles, digits, maPeriod)
            return data.iloc[-nCandles:]
        else:
            return processListOfCandlesFromXtb(listOfCandles[-nCandles:], digits)
       
        
    def openBuyPosition(self, price:float, sl:float, tp:float, vol:float):
        command = {
            "command": "tradeTransaction",
            "arguments": {
                "tradeTransInfo": {
                    "cmd": self.orderCommands["BUY"],
                    "customComment": "",
                    "expiration": 0,
                    "order": self.positionId,
                    "price":price,
                    "sl": sl,
                    "tp": tp,
                    "symbol": self.symbol,
                    "type": self.orderTypes["OPEN"],
                    "volume": vol
                }
            }
        }
        jsonObject = self.sendCommand(command)
        self.positionId = jsonObject["returnData"]["order"]
        return self.positionId


    def modifyPosition(self, sl:float, tp:float, vol:float):
        command = {
            "command": "tradeTransaction",
            "arguments": {
                "tradeTransInfo": {
                    "order": self.positionId,
                    "price":1,
                    "sl": sl,
                    "tp": tp,
                    "symbol": self.symbol,
                    "type": self.orderTypes["MODIFY"],
                    "volume": vol
                }
            }
        }

        jsonObject = self.sendCommand(command)
        positionId = jsonObject["returnData"]["order"]
        return positionId


    def checkPositionStatus(self):
        command = {
            "command": "getTradeRecords",
            "arguments": {
                "orders": [self.positionId]
            }
        }
        jsonObject = self.sendCommand(command)
        return jsonObject["returnData"][0]
       

    def getBalance(self):
        command =  {"command": "getMarginLevel"}
        jsonObject = self.sendCommand(command)
        balance:float = jsonObject["returnData"]["balance"]
        return balance
        
