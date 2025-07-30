#!/usr/bin/env python3
"""
Debug script to check the status of video processing
"""
import os
import sqlite3
from datetime import datetime
from config import *

def check_directories():
    print("\n=== DIRECTORY STATUS ===")
    dirs = {
        "BASE_DIR": BASE_DIR,
        "FTP_VIDEO_DIR": os.path.join(BASE_DIR, "static", "ftp_video"),
        "PROCESSED_MARKER_DIR": os.path.join(BASE_DIR, "static", "ftp_video_processed"),
        "SNAPSHOT_DIR": os.path.join(BASE_DIR, "static", "snapshots"),
        "CROPS_DIR": os.path.join(BASE_DIR, "static", "crops"),
        "LOGS_DIR": os.path.join(BASE_DIR, "logs"),
    }
    
    for name, path in dirs.items():
        exists = os.path.exists(path)
        writable = os.access(path, os.W_OK) if exists else False
        print(f"{name}:")
        print(f"  Path: {path}")
        print(f"  Exists: {exists}")
        print(f"  Writable: {writable}")
        
        if name == "FTP_VIDEO_DIR" and exists:
            mp4_files = [f for f in os.listdir(path) if f.endswith('.mp4')]
            print(f"  MP4 files: {len(mp4_files)}")
            for f in mp4_files[:5]:  # Show first 5
                size = os.path.getsize(os.path.join(path, f)) / 1024 / 1024
                print(f"    - {f} ({size:.2f} MB)")
                
        elif name == "PROCESSED_MARKER_DIR" and exists:
            markers = [f for f in os.listdir(path) if f.endswith('.processed')]
            print(f"  Marker files: {len(markers)}")
            for f in markers[:5]:  # Show first 5
                print(f"    - {f}")

def check_database():
    print("\n=== DATABASE STATUS ===")
    db_path = os.path.join(BASE_DIR, "data", "birdwatch.db")
    
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return
        
    print(f"Database path: {db_path}")
    print(f"Database size: {os.path.getsize(db_path) / 1024:.2f} KB")
    
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Check tables
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = c.fetchall()
        print(f"\nTables: {[t[0] for t in tables]}")
        
        # Check snapshots
        try:
            c.execute("SELECT COUNT(*) FROM snapshots")
            snapshot_count = c.fetchone()[0]
            print(f"\nSnapshots: {snapshot_count}")
            
            c.execute("SELECT * FROM snapshots ORDER BY timestamp DESC LIMIT 5")
            recent = c.fetchall()
            if recent:
                print("  Recent snapshots:")
                for row in recent:
                    print(f"    ID={row[0]}, Time={row[1]}, File={row[2]}")
        except sqlite3.OperationalError as e:
            print(f"  Error reading snapshots: {e}")
        
        # Check detections
        try:
            c.execute("SELECT COUNT(*) FROM detections")
            detection_count = c.fetchone()[0]
            print(f"\nDetections: {detection_count}")
            
            c.execute("SELECT label, COUNT(*) FROM detections GROUP BY label")
            label_counts = c.fetchall()
            if label_counts:
                print("  By label:")
                for label, count in label_counts:
                    print(f"    {label}: {count}")
        except sqlite3.OperationalError as e:
            print(f"  Error reading detections: {e}")
            
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

def check_unprocessed_videos():
    print("\n=== UNPROCESSED VIDEOS ===")
    video_dir = os.path.join(BASE_DIR, "static", "ftp_video")
    marker_dir = os.path.join(BASE_DIR, "static", "ftp_video_processed")
    
    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        return
        
    all_videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"Total MP4 files: {len(all_videos)}")
    
    unprocessed = []
    for video in all_videos:
        marker_path = os.path.join(marker_dir, video + ".processed")
        if not os.path.exists(marker_path):
            unprocessed.append(video)
    
    print(f"Unprocessed videos: {len(unprocessed)}")
    for video in unprocessed[:10]:  # Show first 10
        path = os.path.join(video_dir, video)
        size = os.path.getsize(path) / 1024 / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        print(f"  - {video} ({size:.2f} MB, modified: {mtime})")

def check_logs():
    print("\n=== LOG FILES ===")
    log_dir = os.path.join(BASE_DIR, "logs")
    
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return
        
    for log_file in os.listdir(log_dir):
        if log_file.endswith('.log'):
            path = os.path.join(log_dir, log_file)
            size = os.path.getsize(path) / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            print(f"\n{log_file}:")
            print(f"  Size: {size:.2f} KB")
            print(f"  Modified: {mtime}")
            
            # Show last few lines
            try:
                with open(path, 'r') as f:
                    lines = f.readlines()
                    print(f"  Last 3 lines:")
                    for line in lines[-3:]:
                        print(f"    {line.strip()}")
            except Exception as e:
                print(f"  Error reading: {e}")

if __name__ == "__main__":
    print("=== BIRDWATCH DEBUG STATUS ===")
    print(f"Time: {datetime.now()}")
    
    check_directories()
    check_database()
    check_unprocessed_videos()
    check_logs()
    
    print("\n=== END OF DEBUG REPORT ===")