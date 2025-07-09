from flask import Flask, render_template, request, redirect, url_for
import time

app = Flask(__name__)

# In-memory storage (reset on each restart)
logs_store = [
    {"timestamp": "2025-07-09 13:14", "event": "Startup complete"},
    {"timestamp": "2025-07-09 13:20", "event": "No camera detected"},
]

events_store = [
    {"timestamp": "2025-07-09 13:45", "description": "Bird detected: Cardinal"},
    {"timestamp": "2025-07-09 13:46", "description": "Motion detected in frame"},
]

@app.route('/')
def dashboard():
    return render_template('dashboard.html', logs=reversed(logs_store), events=reversed(events_store))

@app.route('/control', methods=['POST'])
def control():
    direction = request.form.get('direction')
    logs_store.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": f"PTZ button pressed: {direction}"
    })
    return redirect(url_for('dashboard'))

@app.route('/logs')
def logs():
    return render_template('logs.html', logs=reversed(logs_store))

@app.route('/events')
def events():
    return render_template('events.html', events=reversed(events_store))

@app.route('/snapshots')
def snapshots():
    # Placeholder for future snapshot page
    return "<h2>Snapshots page coming soon</h2>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
