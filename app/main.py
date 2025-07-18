from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_file
from onvif import ONVIFCamera
from flask_socketio import SocketIO, emit
from bird_classifier.bird_classifier import classify_species_chriamue
import requests
from datetime import datetime
import cv2
import urllib.parse
import time
import psutil
from config import *
import os
from ultralytics import YOLO
import sqlite3

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

camera = ONVIFCamera(CAMERA_01_IP, CAMERA_01_ONVIF_PORT, CAMERA_01_USERNAME, CAMERA_01_PASSWORD)
media_service = camera.create_media_service()
ptz_service = camera.create_ptz_service()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "static", "snapshots")
CROPS_DIR = os.path.join(BASE_DIR, "static", "crops")
model = YOLO(MODEL_PATH_01)

print("BASE_DIR =", BASE_DIR)
print("CROPS_DIR =", CROPS_DIR)
print("SNAPSHOT_DIR =", SNAPSHOT_DIR)

logs_store = []
events_store = []

DB_PATH = os.path.join(BASE_DIR, "data", "birdwatch.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        label TEXT,
        confidence REAL,
        bbox TEXT,
        original_path TEXT,
        crop_path TEXT,
        species TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_detection(label, confidence, bbox, original_path, crop_path, species=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO detections (timestamp, label, confidence, bbox, original_path, crop_path, species)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        label,
        confidence,
        str(bbox),
        original_path,
        crop_path,
        species
    ))
    conn.commit()
    conn.close()

def get_recent_detections(limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    SELECT timestamp, label, confidence, crop_path, species FROM detections
    ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def detect_and_crop(image_path):
    image = cv2.imread(image_path)
    results = model(image)

    detections = []
    for i, r in enumerate(results):
        boxes = r.boxes
        for j, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]
            crop_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_crop_{label}_{j}.jpg"
            crop_path = os.path.join(CROPS_DIR, crop_filename)
            cv2.imwrite(crop_path, crop)

            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "crop_path": crop_path
            })

    return detections

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

@app.route('/status', methods=['GET'])
def get_ptz_status():
    try:
        camera = ONVIFCamera(CAMERA_01_IP, CAMERA_01_ONVIF_PORT, CAMERA_01_USERNAME, CAMERA_01_PASSWORD)
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
        camera = ONVIFCamera(CAMERA_01_IP, CAMERA_01_ONVIF_PORT, CAMERA_01_USERNAME, CAMERA_01_PASSWORD)
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
    detections = get_recent_detections(100)
    return render_template('snapshots.html', detections=detections)

@app.route('/snapshot', methods=['POST'])
def snapshot():
    url = f"http://{CAMERA_01_IP}/cgi-bin/api.cgi"
    params = {
        "cmd": "Snap",
        "channel": 0,
        "rs": "abc123",
        "user": CAMERA_01_USERNAME,
        "password": CAMERA_01_PASSWORD
    }

    try:
        response = requests.get(url, params=params, stream=True, timeout=10)
        if response.status_code == 200:
            filename = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.jpg")
            filepath = os.path.join(SNAPSHOT_DIR, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            log_event(f"Snapshot taken and saved to {filepath}")
            detections = detect_and_crop(filepath)
            log_event(f"Detected {len(detections)} objects in {filepath}")
            for det in detections:
                species = None
                #if det['label'] == 'bird':
                if 1 == 1:
                    log_event(f"Triggering species classifier for {det['crop_path']}")
                    label, conf = classify_species_chriamue(det['crop_path'])
                    log_event(f"Species: {label} ({conf})")
                else:
                    species = None
                insert_detection(
                    label=det['label'],
                    confidence=det['confidence'],
                    bbox=det['bbox'],
                    original_path=filepath,
                    crop_path=det['crop_path'],
                    species=species
                )
            return jsonify({'success': f'Snapshot saved as {filename}'}), 200
        else:
            log_event(f"Snapshot failed with status code: {response.status_code}")
            return jsonify({'error': f'Snapshot failed: {response.status_code}'}), 500
    except Exception as e:
        log_event(f"Snapshot error: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

def gen_frames_sub():
    username = urllib.parse.quote(CAMERA_01_USERNAME)
    password = urllib.parse.quote(CAMERA_01_PASSWORD)
    rtsp_url = f'rtsp://{username}:{password}@{CAMERA_01_IP}:{CAMERA_01_RTSP_PORT}/{CAMERA_01_RTSP_STREAM_PATH_SUB}'
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

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames_sub(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    init_db()
