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
    return render_template("index.html", login_btn=login_btn, on_off_btn=on_off_btn)


@app.route("/run_trading_bot", methods=["POST"])
def run_trading_bot():
    global tradingLoop

    isLogged = tradingLoop.get_isLogged()
    botRunning = tradingLoop.get_running() 

    response = {}

    if isLogged:
        if request.json != None:
            buttonValue = bool(request.json['value']) # button on HTML page

            print("button value: ", request.json['value'])

            if (buttonValue == False) and (botRunning == True): # buttonValue == False == STOP bot
                # tradingLoop.stopLoop()
                response = {
                    'on_off_button_value' : True,
                    'on_off_button_text'  : 'Run it!',
                    'on_off_button_status': 'Bot stoped.'
                }
            
            elif (buttonValue == True) and (botRunning == False): # buttonValue == True == RUN bot
                # tradingLoop.runLoop()
                response = {
                    'on_off_button_value' : False,
                    'on_off_button_text'  : 'Stop it!',
                    'on_off_button_status': 'Bot launched.'
                }
            
            elif (buttonValue == False) and (botRunning == False): # TODO: continue devs
                response = {
                    'on_off_button_value' : True,
                    'on_off_button_text'  : 'Run it!',
                    'on_off_button_status': 'Bot can\'t be stoped, he is already not running.'
                }
            
            elif (buttonValue == True) and (botRunning == True):
                response = {
                    'on_off_button_value' : False,
                    'on_off_button_text'  : 'Stop it!',
                    'on_off_button_status': 'Bot can\'t be launched, he is already running.'
                }
            
            else:
                response['on_off_button_status'] = 'IDK'
            
        else: # Not logged
             response = {
                'on_off_button_value' : True,
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
        'login_button_value': isLogged,
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

