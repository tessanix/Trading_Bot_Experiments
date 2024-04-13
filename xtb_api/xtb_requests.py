import json
from ssl import SSLSocket
from datetime import datetime, timedelta
from xtb_api.xtb_responses_processing import processListOfCandlesFromXtb

PERIOD_H4 = 240 # == 240 minutes
END = b'\n\n'

orderCommands = {"BUY":0, "SELL":1}
orderTypes = {"OPEN":0, "PENDING":1, "CLOSE":2, "MODIFY":3, "DELETE":4}
NEW_ORDER = 0


def login(socket:SSLSocket):
    config = {}
    with open("config.ini") as configFile:
        for line in configFile:
            name, _ , var = line.partition("=")
            config[name] = var.strip('\n')

    request = json.dumps({
        "command": "login",
        "arguments": {
            "userId":   config['DEMO_ID'],     # account number
            "password": config['PWD']
        }
    }).encode("UTF-8")
    socket.send(request)

    response = socket.recv(8192)
    jsonObject = json.loads(response.decode())

    if jsonObject["status"] == True:
        return jsonObject["streamSessionId"]
    else:
        print(f'Error code: {jsonObject["errorCode"]}')
        print(f'Error description: {jsonObject["errorDescr"]}')
        return None
    

def getServerTime(socket:SSLSocket):
    request = json.dumps({
        "command": "getServerTime"
    }).encode("UTF-8")

    socket.send(request)
    response = socket.recv(8192)
    jsonObject = json.loads(response.decode())

    if jsonObject["status"] == True:
        timestampInMs = jsonObject["returnData"]["time"]
        datetime_object = datetime.fromtimestamp(timestampInMs/1000.0)
        return datetime_object
    else:
        print(f'Error code: {jsonObject["errorCode"]}')
        print(f'Error description: {jsonObject["errorDescr"]}')
        return None
    

def getLastNCandlesH4(socket:SSLSocket, nCandles=100, symbol="US500"):

    currentDatetime = datetime.now()
    delta = timedelta(days=60) # environ 2 mois
    new_datetime = currentDatetime - delta # 2 mois en arrière
    timestampInMs = new_datetime.timestamp()*1000

    request = json.dumps({
        "command": "getChartLastRequest",
        "arguments": {
            "info": {
                "period": PERIOD_H4,
                "start": timestampInMs,
                "symbol": symbol
            }
        }
    }).encode("UTF-8")

    socket.send(request)
        
    response_buffer = b''
    while True:
        chunk = socket.recv(4096)
        response_buffer += chunk
        if END in chunk: break

    jsonObject = json.loads(response_buffer.decode())

    if jsonObject["status"] == True:
        listOfCandles = jsonObject["returnData"]["rateInfos"]
        listOfCandles = listOfCandles[-nCandles:]
        return processListOfCandlesFromXtb(listOfCandles)
    else:
        print(f'Error code: {jsonObject["errorCode"]}')
        print(f'Error description: {jsonObject["errorDescr"]}')
        return None
    

def openBuyPosition(socket:SSLSocket, price:float, sl:float, tp:float, vol:float, symbol='US500'):
    request = json.dumps( {
        "command": "tradeTransaction",
        "arguments": {
            "tradeTransInfo": {
                "cmd": orderCommands["BUY"],
                "customComment": "",
                "expiration": 0,
                "order": NEW_ORDER,
                "price":price,
                "sl": sl,
                "tp": tp,
                "symbol": symbol,
                "type": orderTypes["OPEN"],
                "volume": vol
            }
        }
    }).encode("UTF-8")

    socket.send(request)
        
    response_buffer = b''
    while True:
        chunk = socket.recv(4096)
        response_buffer += chunk
        if END in chunk: break

    jsonObject = json.loads(response_buffer.decode())

    if jsonObject["status"] == True:
        positionId = jsonObject["returnData"]["order"]
        return positionId
    else:
        print(f'Error code: {jsonObject["errorCode"]}')
        print(f'Error description: {jsonObject["errorDescr"]}')
        return None


def modifyPosition(socket:SSLSocket, sl:float, tp:float, vol:float, positionId:float, symbol='US500'):
    request = json.dumps( {
        "command": "tradeTransaction",
        "arguments": {
            "tradeTransInfo": {
                "order": positionId,
                "price":1,
                "sl": sl,
                "tp": tp,
                "symbol": symbol,
                "type": orderTypes["MODIFY"],
                "volume": vol
            }
        }
    }).encode("UTF-8")

    socket.send(request)
        
    response_buffer = b''
    while True:
        chunk = socket.recv(4096)
        response_buffer += chunk
        if END in chunk: break

    jsonObject = json.loads(response_buffer.decode())

    if jsonObject["status"] == True:
        positionId = jsonObject["returnData"]["order"]
        return positionId
    else:
        print(f'Error code: {jsonObject["errorCode"]}')
        print(f'Error description: {jsonObject["errorDescr"]}')
        return None


def checkPositionStatus(socket:SSLSocket, positionId:float):
    request = json.dumps({
        "command": "getTradeRecords",
        "arguments": {
            "orders": [positionId]
        }
    }).encode("UTF-8")

    socket.send(request)
        
    response_buffer = b''
    while True:
        chunk = socket.recv(4096)
        response_buffer += chunk
        if END in chunk: break

    jsonObject = json.loads(response_buffer.decode())

    if jsonObject["status"] == True:
        closed = jsonObject["returnData"][0]["closed"]
        profit = jsonObject["returnData"][0]["profit"]
        return (closed, profit)
    else:
        print(f'Error code: {jsonObject["errorCode"]}')
        print(f'Error description: {jsonObject["errorDescr"]}')
        return None


def getBalance(socket:SSLSocket):
    request = json.dumps({
        "command": "getMarginLevel"
    }).encode("UTF-8")

    socket.send(request)
    response = socket.recv(8192)
    jsonObject = json.loads(response.decode())

    if jsonObject["status"] == True:
        return jsonObject["returnData"]["balance"]
    else:
        print(f'Error code: {jsonObject["errorCode"]}')
        print(f'Error description: {jsonObject["errorDescr"]}')
        return None

    
   
   
