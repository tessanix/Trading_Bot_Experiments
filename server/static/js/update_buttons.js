const onOffButton       = document.getElementById('onOffButton');
const tradingLoopStatus = document.getElementById('trading-loop-status');

onOffButton.addEventListener('click', () => {
    // When the button is clicked, send a POST request to the server
    fetch('/run_trading_bot', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        console.log("data from response of onOffButton click:")
        console.log(data)
        // Update the paragraph element with the response from the server
        if (data.hasOwnProperty('on_off_button_status')){
            tradingLoopStatus.innerText = 'Status: ' + data.on_off_button_status;
        }
        if(data.hasOwnProperty('on_off_button_text')){
            onOffButton.innerText = data.on_off_button_text;
        }
    })
    .catch(error => console.error('Error:', error));
});

const loginButton = document.getElementById('loginButton');
const loginStatus = document.getElementById('login-status');

loginButton.addEventListener('click', () => {
    strBtnValue = loginButton.innerText.toLowerCase()
    console.log(strBtnValue)
    logUrl = (strBtnValue === 'login') ? '/login_xtb_api' : '/logout_xtb_api';

    console.log(logUrl)
    // When the button is clicked, send a POST request to the server
    fetch(logUrl, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        console.log("data from response of loginButton click:")
        console.log(data)

        if (data.hasOwnProperty('login_button_status')){
            loginStatus.innerText = 'Status: ' + data.login_button_status;
        }
        if(data.hasOwnProperty('login_button_text')){
            loginButton.innerText = data.login_button_text;
        }
    })
    .catch(error => console.error('Error:', error));
});