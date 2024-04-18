from flask import Flask, render_template, redirect, jsonify, session, request
import sys; sys.path.insert(1, '..')
from xtb_api.trading_loop import TradingLoop
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

tradingLoop = TradingLoop(socketio)

app.secret_key = "your_secret_key_here"
username = "123"
password = "123"


@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

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
                return redirect("/index")
        else:
            return ">:("
    return render_template("login.html")


@app.route("/index", methods=["GET", "POST"])
def index_page():
    global tradingLoop
    print(f'session from run:{session}')

    if not session.get("logged_in"):
        return redirect("/")
    
    isLogged = tradingLoop.get_isLogged()
    botRunning = tradingLoop.get_running() # value allowing loop to run

    login_btn = {
        'value': isLogged,
        'text': 'Logout' if isLogged else 'Login', 
        'status': 'You are logged.' if isLogged else 'You are not logged.'
    }       
    on_off_btn = {
        'value': botRunning,
        'text': 'Stop bot' if botRunning else 'Run bot', 
        'status': 'Bot is running.' if botRunning else 'Bot is not running.'
    }  
    ping_status = f'Last time pinged: {tradingLoop.lastTimePinged}'
    return render_template("index.html", login_btn=login_btn, on_off_btn=on_off_btn, ping_status=ping_status)


@app.route("/run_trading_bot", methods=["POST"])
def run_trading_bot():
    global tradingLoop

    isLogged = tradingLoop.get_isLogged()
    botRunning = tradingLoop.get_running() 

    response = {}

    if isLogged:
        if botRunning: 
            print('stoping bot ...')
            tradingLoop.stopTradingLoop()
            response = {
                'on_off_button_text'  : 'Run it!',
                'on_off_button_status': 'Bot stoped.'
            }
        
        else: #bot is not running
            print('starting bot ...')
            tradingLoop.runTradingLoop()
            response = {
                'on_off_button_text'  : 'Stop it!',
                'on_off_button_status': 'Bot launched.'
            } 
    else: # Not logged
        response = {
            'on_off_button_text'  : 'Run it!',
            'on_off_button_status': 'You can\'t run the bot if you are not logged.'
        }
        
    return jsonify(response)

@app.route("/login_xtb_api", methods=["POST"])
def login_xtb():
    global tradingLoop
    print("calling login")
    tradingLoop.login()
    isLogged = tradingLoop.get_isLogged()
    return jsonify({
        'login_button_text': 'Logout' if isLogged else 'Login', 
        'login_button_status': 'You are logged.' if isLogged else 'You are not logged.'
    })

@app.route("/logout_xtb_api", methods=["POST"])
def logout_xtb():
    print("calling logout")
    global tradingLoop
    tradingLoop.logout()
    return jsonify({
        'login_button_value' : False,
        'login_button_text'  : 'Login', 
        'login_button_status': 'You are not logged.'
    })


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=5555, debug=True)

