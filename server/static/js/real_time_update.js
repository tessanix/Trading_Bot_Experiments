
const pingStatus = document.getElementById('ping-monitor');
var socket = io.connect('http://' + document.domain + ':' + location.port);
console.log('http://' + document.domain + ':' + location.port);
socket.on('connect', function() {
    console.log('Connected');
});
socket.on('disconnect', function() {
    console.log('Disconnected');
});
socket.on('ping', function(data) {
    console.log('Data received:', data);
    pingStatus.innerText = 'Last time pinged: ' + data;
});
socket.on('trade_status', function(data) {
    console.log('Data received:', data);
    document.getElementById('order').innerText = data.order;
    document.getElementById('digits').innerText = data.digits;
    document.getElementById('symbol').innerText = data.symbol;
    document.getElementById('margin_rate').innerText = data.margin_rate;
    document.getElementById('close_price').innerText = data.close_price;
    document.getElementById('open_price').innerText = data.open_price;
    document.getElementById('profit').innerText = data.profit;
    document.getElementById('volume').innerText = data.volume;
    document.getElementById('sl').innerText = data.sl;
    document.getElementById('tp').innerText = data.tp;
    document.getElementById('closed').innerText = data.closed;
    document.getElementById('timestamp').innerText = new Date(data.timestamp).toLocaleString();
    document.getElementById('spread').innerText = data.spread;
    document.getElementById('taxes').innerText = data.taxes;
    document.getElementById('open_time').innerText = new Date(data.open_time).toLocaleString();
    document.getElementById('close_time').innerText = data.close_time ? new Date(data.close_time).toLocaleString() : 'Not closed yet';
});

