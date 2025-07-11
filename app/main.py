from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from onvif import ONVIFCamera
from config import RTSP_USERNAME, RTSP_PASSWORD, RTSP_IP, RTSP_PORT, RTSP_STREAM_PATH_MAIN, RTSP_STREAM_PATH_SUB
from flask_socketio import SocketIO, emit
from collections import deque
import cv2
import urllib.parse
import time
import psutil

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

logs_store = []
events_store = []

def log_event(message):
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": message
    }
    logs_store.append(log_entry)
    socketio.emit('new_log', log_entry)

@app.route('/')
def dashboard():
    return render_template('dashboard.html', logs=reversed(logs_store), events=reversed(events_store))

camera = ONVIFCamera(RTSP_IP, 8000, RTSP_USERNAME, RTSP_PASSWORD)
media_service = camera.create_media_service()
ptz_service = camera.create_ptz_service()

@app.route('/control', methods=['POST'])
def control():
    data = request.get_json()
    velocity = 1.0
    pan = float(data.get('pan', 0.0))
    tilt = float(data.get('tilt', 0.0))
    duration = float(data.get('seconds', 0.5))
    command_type = str(data.get('command_type', "unknown"))

    log_event(f"Received PTZ control {command_type} request → pan: {pan}, tilt: {tilt}, duration: {duration}s")

    try:
        log_event("Using existing ONVIFCamera connection for control")
        token = media_service.GetProfiles()[0].token

        move = ptz_service.create_type('ContinuousMove')
        move.ProfileToken = token
        move.Velocity = {}

        if pan != 0 or tilt != 0:
            move.Velocity['PanTilt'] = {'x': pan * velocity, 'y': tilt * velocity}
            ptz_service.ContinuousMove(move)
            log_event(f"PTZ move → pan velocity: {pan * velocity}, tilt velocity: {tilt * velocity}, duration: {duration}s")
            time.sleep(duration)
            ptz_service.Stop({'ProfileToken': token})
            log_event("PTZ movement stopped")

        return jsonify({'success': 'PTZ control executed successfully'}), 200

    except Exception as e:
        log_event(f"PTZ control error: {str(e)}")
        return jsonify({'error': 'PTZ control failed'}), 500

@app.route('/live_stream')
def live_stream():
    return render_template('live_stream.html')

@app.route('/processed_stream')
def processed_stream():
    return Response(gen_processed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status', methods=['GET'])
def get_ptz_status():
    try:
        camera = ONVIFCamera(RTSP_IP, 8000, RTSP_USERNAME, RTSP_PASSWORD)
        log_event("Connected to ONVIFCamera for status")
        media_service = camera.create_media_service()
        ptz_service = camera.create_ptz_service()
        token = media_service.GetProfiles()[0].token
        status = ptz_service.GetStatus({'ProfileToken': token})
        log_event(f"PTZ status raw: {status}")
    except Exception as e:
        log_event(f"PTZ status error: {str(e)}")
        return jsonify({'error': 'Unable to fetch PTZ status'}), 500

@app.route('/camera_diagnostics')
def camera_diagnostics():
    diagnostics = {}
    try:
        camera = ONVIFCamera(RTSP_IP, 8000, RTSP_USERNAME, RTSP_PASSWORD)
        log_event("Connected to ONVIFCamera for diagnostics")
        media_service = camera.create_media_service()
        ptz_service = camera.create_ptz_service()
        profiles = media_service.GetProfiles()
        diagnostics['profiles'] = [str(profile) for profile in profiles]
        log_event(f"Found {len(profiles)} media profiles")
        profile_info = []
        for p in profiles:
            info = {
                "name": p.Name,
                "token": p.token,
                "videoSourceConfiguration": str(p.VideoSourceConfiguration),
                "videoEncoderConfiguration": str(p.VideoEncoderConfiguration),
                "ptzConfiguration": str(p.PTZConfiguration),
            }
            profile_info.append(info)
        diagnostics['profile_details'] = profile_info
        ptz_capabilities = ptz_service.GetServiceCapabilities()
        diagnostics['ptz_capabilities'] = str(ptz_capabilities)
        log_event(f"PTZ Capabilities: {ptz_capabilities}")
        ptz_config_token = profiles[0].PTZConfiguration.token if profiles[0].PTZConfiguration else None
        if ptz_config_token:
            ptz_options = ptz_service.GetConfigurationOptions({'ConfigurationToken': ptz_config_token})
            diagnostics['ptz_options'] = str(ptz_options)
            log_event(f"PTZ Options: {ptz_options}")
        else:
            diagnostics['ptz_options'] = 'No PTZConfiguration token available'
        has_home = hasattr(ptz_service, 'GotoHomePosition') and hasattr(ptz_service, 'SetHomePosition')
        diagnostics['has_home_support'] = has_home
        log_event(f"Home Position Supported: {has_home}")
        ptz_methods = dir(ptz_service)
        diagnostics['ptz_methods'] = ptz_methods
        return jsonify(diagnostics)
    except Exception as e:
        log_event(f"Diagnostics error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/logs')
def logs():
    return render_template('logs.html', logs=reversed(logs_store))

@app.route('/events')
def events():
    return render_template('events.html', events=reversed(events_store))

@app.route('/snapshots')
def snapshots():
    return "<h2>Snapshots page coming soon</h2>"

def gen_frames():
    username = urllib.parse.quote(RTSP_USERNAME)
    password = urllib.parse.quote(RTSP_PASSWORD)
    rtsp_url = f'rtsp://{username}:{password}@{RTSP_IP}:{RTSP_PORT}/{RTSP_STREAM_PATH_SUB}'
    camera = cv2.VideoCapture(rtsp_url)
    if not camera.isOpened():
        log_event("Camera not accessible (failed to open RTSP stream)")
        return
    log_event("Camera accessed successfully for stream")
    while True:
        success, frame = camera.read()
        if not success:
            log_event("Camera read failed during streaming")
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_processed_frames():
    username = urllib.parse.quote(RTSP_USERNAME)
    password = urllib.parse.quote(RTSP_PASSWORD)
    rtsp_url = f'rtsp://{username}:{password}@{RTSP_IP}:{RTSP_PORT}/{RTSP_STREAM_PATH_MAIN}'

    def open_camera():
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            log_event("Camera not accessible (failed to open RTSP stream for processed view)")
            return None
        log_event("Camera accessed successfully for processed stream")
        return cap

    camera = open_camera()
    last_reset = time.time()
    read_failures = 0

    while True:
        if camera is None:
            time.sleep(1)
            camera = open_camera()
            continue

        grabbed = camera.grab()
        if not grabbed:
            read_failures += 1
            log_event(f"[Grab Fail] Attempt #{read_failures} | Elapsed: {time.time() - last_reset:.2f}s")
            if read_failures >= 5 or time.time() - last_reset > 5:
                log_event("[Stream Reset] Too many failures or stalled. Reinitializing camera.")
                log_event(f"[System Stats] CPU: {psutil.cpu_percent()}%, RAM: {psutil.virtual_memory().percent}%")
                camera.release()
                camera = open_camera()
                last_reset = time.time()
                read_failures = 0
            continue

        success, frame = camera.retrieve()
        if not success:
            read_failures += 1
            log_event(f"[Retrieve Fail] Attempt #{read_failures}")
            continue

        read_failures = 0
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            log_event("[Encoding Fail] Could not encode frame.")
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(2)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)