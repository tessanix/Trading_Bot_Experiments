from flask import Flask, render_template, redirect, jsonify, session, request
import sys; sys.path.insert(1, '..')
from xtb_api.trading_loop import TradingLoop

app = Flask(__name__)
tradingLoop = TradingLoop()

app.secret_key = "your_secret_key_here"
username = "123"
password = "123"


@app.route("/", methods=["GET"])
def home():
    return redirect("/login")

@app.route("/login", methods=["GET", "POST"])
def login():
    print(f'session from login:{session}')

    if request.method == "POST":
        form_username = request.form["username"]
        form_password = request.form["password"]
        if form_username == username and form_password == password:
                session["logged_in"] = True
                return redirect("/run")
        else:
            return ">:("
    return render_template("login.html")


@app.route("/run", methods=["GET", "POST"])
def runTradingBot():
    global tradingLoop
    print(f'session from run:{session}')

    if not session.get("logged_in"):
        return redirect("/")
    
    if request.method == "POST":
        if request.json != None:
            buttonValue = request.json['value'] # button on HTML page
            print("button value: ", request.json['value'])
            botRunning = tradingLoop.get_running() # value allowing loop to run
            if (buttonValue == "STOP") and botRunning:
                # tradingLoop.stopLoop()
                return jsonify({
                    'button_value' : 'RUN',
                    'text_result': 'Bot stoped!'
                })
            
            elif (buttonValue == "RUN") and not botRunning:
                # tradingLoop.runLoop()
                return jsonify({
                    'button_value' : 'STOP',
                    'text_result': 'Bot launched!'
                })
            
            elif (buttonValue == "STOP") and not botRunning:
                return jsonify({
                    'button_value' : 'STOP',
                    'text_result': "Bot can't be stoped, he is already not running!"
                })
            
            elif (buttonValue == "RUN") and botRunning:
                return jsonify({
                    'button_value':'RUN',
                    'text_result': "Bot can't be launched, he is already running!"
                })
            
            else:
                return jsonify({
                    'button_value': buttonValue,
                    'text_result': "IDK"
                })
        
    return render_template("index.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=5555, debug=True)

