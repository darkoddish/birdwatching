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
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import queue
import shutil
import traceback
from logger_config import main_logger, video_logger, debug_logger

processing_queue = queue.Queue()

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

camera = ONVIFCamera(CAMERA_01_IP, CAMERA_01_ONVIF_PORT, CAMERA_01_USERNAME, CAMERA_01_PASSWORD)
media_service = camera.create_media_service()
ptz_service = camera.create_ptz_service()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE_DIR, exist_ok=True)
SNAPSHOT_DIR = os.path.join(BASE_DIR, "static", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
CROPS_DIR = os.path.join(BASE_DIR, "static", "crops")
os.makedirs(CROPS_DIR, exist_ok=True)
FTP_VIDEO_DIR = os.path.join(BASE_DIR, "static", "ftp_video")
os.makedirs(FTP_VIDEO_DIR, exist_ok=True)
GENERAL_CLASSIFIER_PATH = os.path.join(BASE_DIR, "general_classifier", "yolov8n.pt")
#os.makedirs(GENERAL_CLASSIFIER_PATH, exist_ok=True)
PROCESSED_MARKER_DIR = os.path.join(BASE_DIR, "static", "ftp_video_processed")
os.makedirs(PROCESSED_MARKER_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, "data", "birdwatch.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

#touch /home/thjones29/Documents/birdwatching/app/static/ftp_video_processed/test_marker_1.processed

model = YOLO(GENERAL_CLASSIFIER_PATH)

print("BASE_DIR =", BASE_DIR)
print("CROPS_DIR =", CROPS_DIR)
print("SNAPSHOT_DIR =", SNAPSHOT_DIR)

logs_store = []
events_store = []



def run_periodic_snapshots(interval_seconds=300):
    def snapshot_loop():
        while True:
            try:
                with app.app_context():
                    requests.post("http://localhost:5000/snapshot")
            except Exception as e:
                log_event(f"Periodic snapshot error: {e}")
            time.sleep(interval_seconds)

    thread = threading.Thread(target=snapshot_loop, daemon=True)
    thread.start()

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Create snapshots table
        c.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            path TEXT
        )
        """)
        debug_logger.info("Created/verified snapshots table")
        
        # Create detections table with snapshot_id
        c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER,
            timestamp TEXT,
            label TEXT,
            confidence REAL,
            bbox TEXT,
            original_path TEXT,
            crop_path TEXT,
            species TEXT,
            feedback TEXT,
            feedback_notes TEXT,
            FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
        )
        """)
        debug_logger.info("Created/verified detections table")
        
        # Create manual_labels table
        c.execute("""
        CREATE TABLE IF NOT EXISTS manual_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER,
            x INTEGER,
            y INTEGER,
            width INTEGER,
            height INTEGER,
            label TEXT,
            timestamp TEXT,
            FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
        )
        """)
        debug_logger.info("Created/verified manual_labels table")
        
        # Create feedback_missed table
        c.execute("""
        CREATE TABLE IF NOT EXISTS feedback_missed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_path TEXT,
            label TEXT,
            timestamp TEXT
        )
        """)
        debug_logger.info("Created/verified feedback_missed table")
        
        conn.commit()
        conn.close()
        main_logger.info("Database initialized successfully")
        
    except Exception as e:
        main_logger.error(f"Database initialization error: {e}")
        debug_logger.debug(traceback.format_exc())
        raise

def get_manual_labels_for_snapshots():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT snapshot_id, x, y, width, height, label
        FROM manual_labels
    """)
    rows = cur.fetchall()
    conn.close()

    # organize by snapshot_id
    label_map = {}
    for row in rows:
        sid = row[0]
        label_data = {
            'x': row[1],
            'y': row[2],
            'width': row[3],
            'height': row[4],
            'label': row[5]
        }
        label_map.setdefault(sid, []).append(label_data)
    return label_map

def get_recent_detections(limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(f"""
        SELECT s.timestamp, d.label, d.confidence, d.crop_path, d.species,
               d.feedback, d.feedback_notes,
               EXISTS (
                   SELECT 1 FROM manual_labels ml WHERE ml.snapshot_id = d.snapshot_id
               ) as has_manual
        FROM detections d
        JOIN snapshots s ON d.snapshot_id = s.id
        ORDER BY s.timestamp DESC
        LIMIT ?
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

def log_event(message, level="info"):
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": message
    }
    logs_store.append(log_entry)
    socketio.emit('new_log', log_entry)
    
    # Also write to file logger
    if level == "error":
        main_logger.error(message)
    elif level == "warning":
        main_logger.warning(message)
    elif level == "debug":
        main_logger.debug(message)
    else:
        main_logger.info(message)

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

@app.route('/feedback_missed', methods=['POST'])
def feedback_missed():
    snapshot_path = request.form['snapshot_path']
    missed_label = request.form['missed_label']
    timestamp = extract_timestamp_from_filename(snapshot_path)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO feedback_missed (snapshot_path, label, timestamp)
        VALUES (?, ?, ?)
    """, (snapshot_path, missed_label, timestamp))
    conn.commit()
    conn.close()

    return redirect(request.referrer or url_for('snapshots'))

@app.route('/feedback_wrong', methods=['POST'])
def feedback_wrong():
    detection_id_raw = request.form.get('detection_id', '').strip()
    if not detection_id_raw.isdigit():
        log_event("Invalid detection_id in feedback_wrong")
        return redirect(request.referrer or url_for('detections'))

    detection_id = int(detection_id_raw)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        UPDATE detections
        SET feedback = 'wrong'
        WHERE id = ?
    """, (detection_id,))
    conn.commit()
    conn.close()
    log_event(f"Marked detection {detection_id} as wrong")
    return redirect(request.referrer or url_for('detections'))

@app.route('/feedback_ignore', methods=['POST'])
def feedback_ignore():
    detection_id_raw = request.form.get('detection_id', '').strip()
    if not detection_id_raw.isdigit():
        log_event("Invalid detection_id in feedback_ignore")
        return redirect(request.referrer or url_for('detections'))

    detection_id = int(detection_id_raw)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        UPDATE detections
        SET feedback = 'ignore'
        WHERE id = ?
    """, (detection_id,))
    conn.commit()
    conn.close()
    log_event(f"Marked detection {detection_id} as ignore")
    return redirect(request.referrer or url_for('detections'))

def get_snapshots_list():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, timestamp, filename, path
        FROM snapshots
        ORDER BY timestamp DESC
        LIMIT 100
    """)
    rows = cur.fetchall()
    conn.close()

    # Convert to list of dicts for template use
    return [
        {
            "id": row[0],
            "timestamp": row[1],
            "filename": row[2],
            "path": row[3]
        }
        for row in rows
    ]
    
def insert_snapshot(filename, path):
    snapshot_id = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        # First check if snapshots table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='snapshots'")
        if not cur.fetchone():
            debug_logger.error("snapshots table does not exist!")
            # Create the table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    filename TEXT,
                    path TEXT
                )
            """)
            conn.commit()
            debug_logger.info("Created snapshots table")
        
        cur.execute("""
            INSERT INTO snapshots (timestamp, filename, path)
            VALUES (?, ?, ?)
        """, (timestamp, filename, path))
        conn.commit()
        snapshot_id = cur.lastrowid
        debug_logger.debug(f"Inserted snapshot {filename} with ID {snapshot_id}")
        conn.close()
    except Exception as e:
        debug_logger.error(f"Insert snapshot error: {str(e)}")
        debug_logger.debug(traceback.format_exc())
        log_event(f"Insert snapshot error: {str(e)}", "error")
    return snapshot_id

def insert_detection(snapshot_id, label, confidence, bbox, crop_path, species=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Check if detections table has snapshot_id column
        c.execute("PRAGMA table_info(detections)")
        columns = [col[1] for col in c.fetchall()]
        
        if 'snapshot_id' not in columns:
            debug_logger.warning("detections table missing snapshot_id column, adding it")
            c.execute("ALTER TABLE detections ADD COLUMN snapshot_id INTEGER")
            conn.commit()
        
        c.execute("""
        INSERT INTO detections (snapshot_id, label, confidence, bbox, crop_path, species)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            snapshot_id,
            label,
            confidence,
            str(bbox),
            crop_path,
            species
        ))
        conn.commit()
        debug_logger.debug(f"Inserted detection: {label} (conf: {confidence}) for snapshot {snapshot_id}")
        conn.close()
    except Exception as e:
        debug_logger.error(f"Insert detection error: {str(e)}")
        debug_logger.debug(traceback.format_exc())
        log_event(f"Insert detection error: {str(e)}", "error")

def insert_manual_label(snapshot_id, x, y, width, height, label):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO manual_labels (snapshot_id, x, y, width, height, label, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (snapshot_id, x, y, width, height, label, datetime.now().isoformat()))
    conn.commit()

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

            snapshot_id = insert_snapshot(filename, filepath)
            detections = detect_and_crop(filepath)
            log_event(f"Detected {len(detections)} objects in {filepath}")
            for det in detections:
                species = None
                if det['label'] == 'bird':
                    log_event(f"Triggering species classifier for {det['crop_path']}")
                    label, conf = classify_species_chriamue(det['crop_path'])
                    log_event(f"Species: {label} ({conf})")
                    species = label
                insert_detection(
                    snapshot_id=snapshot_id,
                    label=det['label'],
                    confidence=det['confidence'],
                    bbox=det['bbox'],
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

@app.route('/feedback_draw', methods=['POST'])
def feedback_draw():
    snapshot_id = int(request.form['snapshot_id'])
    x = int(request.form['x'])
    y = int(request.form['y'])
    width = int(request.form['width'])
    height = int(request.form['height'])
    label = request.form['label']

    insert_manual_label(snapshot_id, x, y, width, height, label)
    return redirect(request.referrer or url_for('snapshots'))

@app.route('/detections')
def detections():
    species = request.args.get('species')
    label = request.args.get('label')
    min_conf = request.args.get('min_conf', type=float)
    max_conf = request.args.get('max_conf', type=float)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    query = """
        SELECT s.timestamp, d.label, d.confidence, d.crop_path, d.species,d.feedback, d.feedback_notes,
               EXISTS (SELECT 1 FROM manual_labels ml WHERE ml.snapshot_id = d.snapshot_id) as has_manual
        FROM detections d
        JOIN snapshots s ON d.snapshot_id = s.id
        WHERE 1=1
    """
    params = []

    if species:
        query += " AND d.species = ?"
        params.append(species)
    if label:
        query += " AND d.label = ?"
        params.append(label)
    if min_conf is not None:
        query += " AND d.confidence >= ?"
        params.append(min_conf)
    if max_conf is not None:
        query += " AND d.confidence <= ?"
        params.append(max_conf)
    if start_date:
        query += " AND s.timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND s.timestamp <= ?"
        params.append(end_date + "T23:59:59")

    query += " ORDER BY s.timestamp DESC LIMIT 100"

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()

    return render_template('detections.html', detections=rows)

@app.route('/snapshots')
def snapshots():
    snapshots = get_snapshots_list()
    labels_by_snapshot = get_manual_labels_for_snapshots()
    return render_template('snapshots.html', snapshots=snapshots, labels_by_snapshot=labels_by_snapshot)

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

def process_i_frames_from_video(video_path):
    import subprocess
    
    video_logger.info(f"=== STARTING VIDEO PROCESSING: {video_path} ===")
    video_logger.debug(f"Video file size: {os.path.getsize(video_path) / 1024 / 1024:.2f} MB")
    
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    frame_output_dir = os.path.join(BASE_DIR, "temp_frames")
    os.makedirs(frame_output_dir, exist_ok=True)
    video_logger.debug(f"Frame output directory: {frame_output_dir}")

    output_pattern = os.path.join(frame_output_dir, f"{base_filename}_%04d.jpg")
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "select='eq(pict_type\\,I)'",
        "-vsync", "vfr",
        output_pattern
    ]
    
    video_logger.debug(f"FFmpeg command: {' '.join(command)}")
    
    frames_processed = 0
    detections_found = 0
    
    try:
        video_logger.info(f"Running FFmpeg to extract I-frames...")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        video_logger.debug(f"FFmpeg stdout: {result.stdout}")
        if result.stderr:
            video_logger.debug(f"FFmpeg stderr: {result.stderr}")
        
        # Count extracted frames
        extracted_frames = sorted([f for f in os.listdir(frame_output_dir) if f.endswith('.jpg')])
        video_logger.info(f"Extracted {len(extracted_frames)} I-frames from video")
        
        for frame_index, frame_file in enumerate(extracted_frames):
            frame_path = os.path.join(frame_output_dir, frame_file)
            video_logger.debug(f"Processing frame {frame_index + 1}/{len(extracted_frames)}: {frame_file}")
            
            try:
                detections = detect_and_crop(frame_path)
                video_logger.info(f"Frame {frame_file}: Found {len(detections)} objects")
                detections_found += len(detections)
                
                snapshot_id = insert_snapshot(frame_file, frame_path)
                video_logger.debug(f"Created snapshot with ID: {snapshot_id}")
                
                for det_index, det in enumerate(detections):
                    video_logger.debug(f"Processing detection {det_index + 1}/{len(detections)}: {det['label']} (conf: {det['confidence']})")
                    species = None
                    
                    if det['label'] == 'bird':
                        video_logger.info(f"Running species classifier on: {det['crop_path']}")
                        try:
                            label, conf = classify_species_chriamue(det['crop_path'])
                            video_logger.info(f"Species identified: {label} (confidence: {conf})")
                            species = label
                        except Exception as e:
                            video_logger.error(f"Species classification failed: {e}")
                            video_logger.debug(traceback.format_exc())
                    
                    try:
                        insert_detection(
                            snapshot_id=snapshot_id,
                            label=det['label'],
                            confidence=det['confidence'],
                            bbox=det['bbox'],
                            crop_path=det['crop_path'],
                            species=species
                        )
                        video_logger.debug(f"Detection saved to database")
                    except Exception as e:
                        video_logger.error(f"Failed to save detection: {e}")
                        video_logger.debug(traceback.format_exc())
                
                frames_processed += 1
                
            except Exception as e:
                video_logger.error(f"Error processing frame {frame_file}: {e}")
                video_logger.debug(traceback.format_exc())
        
        # Create processed marker file
        video_logger.info(f"All frames processed. Creating marker file...")
        try:
            base = os.path.basename(video_path)
            marker_path = os.path.join(PROCESSED_MARKER_DIR, base + ".processed")
            
            # Ensure directory exists
            os.makedirs(PROCESSED_MARKER_DIR, exist_ok=True)
            
            video_logger.debug(f"Marker file path: {marker_path}")
            video_logger.debug(f"Marker directory exists: {os.path.exists(PROCESSED_MARKER_DIR)}")
            video_logger.debug(f"Marker directory writable: {os.access(PROCESSED_MARKER_DIR, os.W_OK)}")
            
            with open(marker_path, "w") as f:
                timestamp = datetime.now().isoformat()
                f.write(timestamp)
                f.flush()
                os.fsync(f.fileno())
            
            # Verify file was created
            if os.path.exists(marker_path):
                video_logger.info(f"Marker file created successfully: {marker_path}")
                video_logger.debug(f"Marker file size: {os.path.getsize(marker_path)} bytes")
            else:
                video_logger.error(f"Marker file was not created at: {marker_path}")
                
        except Exception as e:
            video_logger.error(f"CRITICAL: Failed to create marker file for {video_path}: {e}")
            video_logger.debug(traceback.format_exc())
            log_event(f"ERROR writing marker for {video_path}: {e}", "error")

        video_logger.info(f"Video processing completed. Frames: {frames_processed}, Detections: {detections_found}")
        log_event(f"Processed video {base_filename}: {frames_processed} frames, {detections_found} detections")

    except subprocess.CalledProcessError as e:
        video_logger.error(f"FFmpeg failed for {video_path}: {e}")
        video_logger.debug(f"FFmpeg return code: {e.returncode}")
        if e.stderr:
            video_logger.debug(f"FFmpeg error output: {e.stderr}")
        log_event(f"Error during video processing {video_path}: FFmpeg failed", "error")
        
    except Exception as e:
        video_logger.error(f"Unexpected error processing video {video_path}: {e}")
        video_logger.debug(traceback.format_exc())
        log_event(f"Error during processing video {video_path}: {e}", "error")
        
    finally:
        video_logger.info("Cleaning up temporary frames...")
        try:
            if os.path.exists(frame_output_dir):
                shutil.rmtree(frame_output_dir, ignore_errors=True)
                video_logger.debug(f"Cleaned up {frame_output_dir}")
        except Exception as e:
            video_logger.error(f"Error cleaning up temp frames: {e}")
        
        video_logger.info(f"=== FINISHED VIDEO PROCESSING: {video_path} ===")

class FTPVideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.mp4'):
            video_logger.info(f"New MP4 file detected: {event.src_path}")
            base = os.path.basename(event.src_path)
            marker_path = os.path.join(PROCESSED_MARKER_DIR, base + ".processed")
            
            video_logger.debug(f"Checking for existing marker: {marker_path}")
            video_logger.debug(f"Marker exists: {os.path.exists(marker_path)}")

            # Wait for the file to stabilize (not grow anymore)
            prev_size = -1
            wait_count = 0
            video_logger.info(f"Waiting for file to stabilize: {event.src_path}")
            
            while True:
                try:
                    curr_size = os.path.getsize(event.src_path)
                    video_logger.debug(f"File size check #{wait_count}: {curr_size / 1024 / 1024:.2f} MB")
                    
                    if curr_size == prev_size:
                        video_logger.info(f"File stabilized at {curr_size / 1024 / 1024:.2f} MB after {wait_count} checks")
                        break
                    prev_size = curr_size
                    wait_count += 1
                    time.sleep(1)
                except Exception as e:
                    video_logger.error(f"Error waiting for file to stabilize: {e}")
                    log_event(f"Error waiting for file to stabilize: {e}", "error")
                    return

            # Now queue it
            if not os.path.exists(marker_path):
                video_logger.info(f"Queueing new file for processing: {event.src_path}")
                log_event(f"Queueing new file for processing: {event.src_path}")
                processing_queue.put(event.src_path)
                video_logger.debug(f"Current queue size: {processing_queue.qsize()}")
            else:
                video_logger.info(f"File already processed (marker exists): {event.src_path}")
                log_event(f"Skipping already processed file: {base}")


def start_ftp_video_watcher():
    observer = Observer()
    event_handler = FTPVideoHandler()
    observer.schedule(event_handler, path=FTP_VIDEO_DIR, recursive=False)
    observer.start()
    log_event(f"Watching FTP video directory: {FTP_VIDEO_DIR}")


def queue_existing_unprocessed_videos():
    video_logger.info(f"=== SCANNING FOR UNPROCESSED VIDEOS ===")
    video_logger.info(f"Video directory: {FTP_VIDEO_DIR}")
    video_logger.info(f"Marker directory: {PROCESSED_MARKER_DIR}")
    
    log_event(f"Checking for unprocessed files in: {FTP_VIDEO_DIR}")
    
    total_files = 0
    mp4_files = 0
    queued_files = 0
    already_processed = 0
    
    try:
        files = sorted(os.listdir(FTP_VIDEO_DIR))
        video_logger.info(f"Found {len(files)} total files in video directory")
        
        for f in files:
            total_files += 1
            full_path = os.path.join(FTP_VIDEO_DIR, f)
            marker_path = os.path.join(PROCESSED_MARKER_DIR, f + ".processed")
            
            video_logger.debug(f"Checking file: {f}")
            video_logger.debug(f"  Full path: {full_path}")
            video_logger.debug(f"  Marker path: {marker_path}")
            video_logger.debug(f"  Is MP4: {f.endswith('.mp4')}")
            video_logger.debug(f"  Marker exists: {os.path.exists(marker_path)}")
            
            if f.endswith(".mp4"):
                mp4_files += 1
                if not os.path.exists(marker_path):
                    video_logger.info(f"Queueing unprocessed file: {f} ({os.path.getsize(full_path) / 1024 / 1024:.2f} MB)")
                    log_event(f"Queueing unprocessed file on startup: {f}")
                    processing_queue.put(full_path)
                    queued_files += 1
                else:
                    video_logger.debug(f"File already processed: {f}")
                    already_processed += 1
            else:
                video_logger.debug(f"Skipping non-MP4 file: {f}")
        
        video_logger.info(f"Scan complete: {total_files} total files, {mp4_files} MP4s, {queued_files} queued, {already_processed} already processed")
        log_event(f"Startup scan: {queued_files} videos queued for processing")
        
    except Exception as e:
        video_logger.error(f"Error scanning video directory: {e}")
        video_logger.debug(traceback.format_exc())
        log_event(f"Error scanning video directory: {e}", "error")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames_sub(), mimetype='multipart/x-mixed-replace; boundary=frame')

def processing_worker():
    video_logger.info("Video processing worker started")
    processed_count = 0
    error_count = 0
    
    while True:
        try:
            video_logger.debug(f"Worker waiting for next video. Queue size: {processing_queue.qsize()}")
            video_path = processing_queue.get()
            
            if not video_path:
                video_logger.warning("Received None video path")
                processing_queue.task_done()
                continue
                
            if not os.path.isfile(video_path):
                video_logger.error(f"Video file not found: {video_path}")
                log_event(f"Skipping invalid video path: {video_path}", "error")
                processing_queue.task_done()
                error_count += 1
                continue
            
            try:
                video_logger.info(f"Worker starting processing #{processed_count + 1}: {video_path}")
                log_event(f"Worker processing: {video_path}")
                
                start_time = time.time()
                process_i_frames_from_video(video_path)
                elapsed_time = time.time() - start_time
                
                processed_count += 1
                video_logger.info(f"Successfully processed video in {elapsed_time:.2f} seconds")
                log_event(f"Worker completed: {os.path.basename(video_path)} in {elapsed_time:.2f}s")
                
            except Exception as e:
                error_count += 1
                video_logger.error(f"Worker error processing {video_path}: {e}")
                video_logger.debug(traceback.format_exc())
                log_event(f"Worker error on {video_path}: {e}", "error")
                
        except Exception as e:
            video_logger.error(f"Critical error in processing worker: {e}")
            video_logger.debug(traceback.format_exc())
            
        finally:
            processing_queue.task_done()
            video_logger.info(f"Worker stats - Processed: {processed_count}, Errors: {error_count}, Queue remaining: {processing_queue.qsize()}")



if __name__ == '__main__':
    # Initialize logging
    main_logger.info("=== BIRDWATCH APPLICATION STARTING ===")
    main_logger.info(f"Base directory: {BASE_DIR}")
    main_logger.info(f"Video directory: {FTP_VIDEO_DIR}")
    main_logger.info(f"Processed markers directory: {PROCESSED_MARKER_DIR}")
    
    # Initialize database
    main_logger.info("Initializing database...")
    init_db()
    
    # Start FTP watcher
    main_logger.info("Starting FTP video watcher...")
    start_ftp_video_watcher()
    
    # Queue existing videos
    main_logger.info("Scanning for existing unprocessed videos...")
    queue_existing_unprocessed_videos()
    
    # Start processing worker
    main_logger.info("Starting video processing worker thread...")
    worker_thread = threading.Thread(target=processing_worker, daemon=True, name="VideoProcessor")
    worker_thread.start()
    
    # Start Flask app
    main_logger.info("Starting Flask application on 0.0.0.0:5000...")
    log_event("Birdwatch application started")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        main_logger.error(f"Application crashed: {e}")
        main_logger.debug(traceback.format_exc())
        raise