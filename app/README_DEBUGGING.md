# Video Processing Debugging Guide

## Overview
I've enhanced your application with comprehensive logging and debugging tools to help identify why videos are being processed but .processed marker files aren't being created.

## What I've Created

### 1. Enhanced Main Application (`main_enhanced.py`)
- Added detailed file-based logging using Python's logging module
- Three separate log files in the `logs/` directory:
  - `birdwatch.log` - Main application events
  - `video_processing.log` - Detailed video processing logs (DEBUG level)
  - `debug.log` - Database operations and debugging info
- Enhanced error handling and traceback logging
- Added detailed logging for:
  - FFmpeg command execution
  - Frame extraction progress
  - Detection processing
  - Marker file creation (with permission checks)
  - Queue management
  - Worker thread status

### 2. Logger Configuration (`logger_config.py`)
- Rotating file handlers (max 10MB per file, keeps 5 backups)
- Console output for real-time monitoring
- Structured log format with timestamps, module names, and line numbers

### 3. Debug Status Script (`debug_status.py`)
Run this to check system status:
```bash
python debug_status.py
```

Shows:
- Directory status and permissions
- Database contents and statistics
- List of unprocessed videos
- Recent log entries

### 4. Test Script (`test_video_processing.py`)
Run this to diagnose specific issues:
```bash
python test_video_processing.py
```

Tests:
- FFmpeg installation
- Directory permissions
- Marker file creation
- FFmpeg I-frame extraction command variations

## How to Use

### 1. Run the Test Script First
```bash
python test_video_processing.py
```
This will identify any system-level issues.

### 2. Use the Enhanced Main Application
Replace your current main.py with main_enhanced.py:
```bash
cp main_enhanced.py main.py
python main.py
```

### 3. Monitor the Logs
In another terminal, tail the logs:
```bash
# Watch all video processing activity
tail -f logs/video_processing.log

# Watch for errors only
tail -f logs/*.log | grep ERROR

# Watch marker file creation
tail -f logs/video_processing.log | grep -i marker
```

### 4. Check System Status
While the app is running, periodically check status:
```bash
python debug_status.py
```

## Key Things to Look For

### In `video_processing.log`:
1. **FFmpeg Success**: Look for "Extracted X I-frames from video"
2. **Marker Creation**: Look for "Marker file created successfully" or "CRITICAL: Failed to create marker"
3. **Permission Issues**: Look for "Marker directory writable: False"
4. **Processing Stats**: Look for "Video processing completed. Frames: X, Detections: Y"

### Common Issues and Solutions:

1. **FFmpeg Command Failing**
   - The log will show which FFmpeg command variation works
   - Look for "FFmpeg failed" or "FFmpeg return code"

2. **Permission Denied**
   - Check "Marker directory writable" in logs
   - Fix: `chmod -R 755 static/ftp_video_processed`

3. **Database Issues**
   - Check debug.log for "table does not exist"
   - The enhanced code will auto-create missing tables

4. **No Marker File Created**
   - Look for "CRITICAL: Failed to create marker file"
   - Check if exception occurs before marker creation

## Quick Troubleshooting Commands

```bash
# Check if markers directory exists and is writable
ls -la static/ftp_video_processed/

# See how many videos vs markers
ls static/ftp_video/*.mp4 | wc -l
ls static/ftp_video_processed/*.processed | wc -l

# Find recent errors
grep -n ERROR logs/*.log | tail -20

# See marker creation attempts
grep -A5 -B5 "marker" logs/video_processing.log

# Check FFmpeg issues
grep -i ffmpeg logs/video_processing.log | grep -i error
```

## Next Steps

1. Run the application with enhanced logging
2. Try processing a single video file
3. Check the logs for the specific point of failure
4. Share the relevant log sections that show where the process fails

The enhanced logging will show exactly where in the process things go wrong, making it much easier to fix the issue.