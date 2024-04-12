from flask import Flask, render_template, redirect, session, request
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
        botRunning = tradingLoop.get_running()
        if (request.form["runBot"] == False) and botRunning:
            tradingLoop.stopLoop()
        elif (request.form["runBot"] == True) and not botRunning:
            tradingLoop.runLoop()
        elif (request.form["runBot"] == False) and not botRunning:
            return "The bot is not running you dumb dumb"
        elif (request.form["runBot"] == True) and botRunning:
            return "The bot is already running you super dumb"
        else:
            return "IDK"
        
    return render_template("index.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__": 
    app.run(debug=True)

