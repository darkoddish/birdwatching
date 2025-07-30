#!/usr/bin/env python3
"""
Test script to diagnose video processing issues
"""
import os
import subprocess
import tempfile
from datetime import datetime

# Test configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FTP_VIDEO_DIR = os.path.join(BASE_DIR, "static", "ftp_video")
PROCESSED_MARKER_DIR = os.path.join(BASE_DIR, "static", "ftp_video_processed")

def test_ffmpeg():
    """Test if FFmpeg is installed and working"""
    print("\n=== TESTING FFMPEG ===")
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg is installed")
            print(f"  Version: {result.stdout.split('\\n')[0]}")
        else:
            print("✗ FFmpeg error:", result.stderr)
            return False
    except FileNotFoundError:
        print("✗ FFmpeg not found in PATH")
        return False
    return True

def test_marker_creation():
    """Test creating marker files"""
    print("\n=== TESTING MARKER FILE CREATION ===")
    
    # Ensure directory exists
    os.makedirs(PROCESSED_MARKER_DIR, exist_ok=True)
    print(f"Marker directory: {PROCESSED_MARKER_DIR}")
    print(f"  Exists: {os.path.exists(PROCESSED_MARKER_DIR)}")
    print(f"  Writable: {os.access(PROCESSED_MARKER_DIR, os.W_OK)}")
    
    # Try to create a test marker
    test_marker = os.path.join(PROCESSED_MARKER_DIR, "test_marker.processed")
    try:
        with open(test_marker, 'w') as f:
            f.write(datetime.now().isoformat())
        print(f"✓ Successfully created test marker: {test_marker}")
        
        # Verify it exists
        if os.path.exists(test_marker):
            print("✓ Marker file exists after creation")
            with open(test_marker, 'r') as f:
                content = f.read()
            print(f"  Content: {content}")
            
            # Clean up
            os.remove(test_marker)
            print("✓ Cleaned up test marker")
        else:
            print("✗ Marker file does not exist after creation!")
            
    except Exception as e:
        print(f"✗ Failed to create marker: {e}")
        return False
    
    return True

def test_ffmpeg_command():
    """Test the specific FFmpeg command used for I-frame extraction"""
    print("\n=== TESTING FFMPEG I-FRAME EXTRACTION ===")
    
    # Find a test video
    if not os.path.exists(FTP_VIDEO_DIR):
        print(f"✗ Video directory not found: {FTP_VIDEO_DIR}")
        return False
        
    videos = [f for f in os.listdir(FTP_VIDEO_DIR) if f.endswith('.mp4')]
    if not videos:
        print("✗ No MP4 files found in video directory")
        return False
    
    test_video = os.path.join(FTP_VIDEO_DIR, videos[0])
    print(f"Using test video: {test_video}")
    print(f"  Size: {os.path.getsize(test_video) / 1024 / 1024:.2f} MB")
    
    # Create temp directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        output_pattern = os.path.join(temp_dir, "test_frame_%04d.jpg")
        
        # Test different FFmpeg command variations
        commands = [
            # Original command with escaped backslash
            ["ffmpeg", "-i", test_video, "-vf", "select='eq(pict_type\\\\,I)'", "-vsync", "vfr", output_pattern],
            # Without escape
            ["ffmpeg", "-i", test_video, "-vf", "select='eq(pict_type,I)'", "-vsync", "vfr", output_pattern],
            # Alternative syntax
            ["ffmpeg", "-i", test_video, "-vf", "select=eq(pict_type,I)", "-vsync", "vfr", output_pattern],
        ]
        
        for i, cmd in enumerate(commands):
            print(f"\\nTesting command variation {i+1}:")
            print(f"  Command: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    frames = [f for f in os.listdir(temp_dir) if f.endswith('.jpg')]
                    print(f"  ✓ Success! Extracted {len(frames)} frames")
                    
                    # Clean up frames for next test
                    for f in frames:
                        os.remove(os.path.join(temp_dir, f))
                    
                    return True
                else:
                    print(f"  ✗ Failed with return code {result.returncode}")
                    if result.stderr:
                        print(f"  Error: {result.stderr[:200]}...")
                        
            except subprocess.TimeoutExpired:
                print("  ✗ Command timed out after 30 seconds")
            except Exception as e:
                print(f"  ✗ Exception: {e}")
    
    return False

def check_permissions():
    """Check file system permissions"""
    print("\n=== CHECKING PERMISSIONS ===")
    
    dirs_to_check = [
        ("Base", BASE_DIR),
        ("Video", FTP_VIDEO_DIR),
        ("Markers", PROCESSED_MARKER_DIR),
        ("Temp frames", os.path.join(BASE_DIR, "temp_frames")),
    ]
    
    all_good = True
    for name, path in dirs_to_check:
        if os.path.exists(path):
            readable = os.access(path, os.R_OK)
            writable = os.access(path, os.W_OK)
            executable = os.access(path, os.X_OK)
            
            status = "✓" if (readable and writable and executable) else "✗"
            print(f"{status} {name}: R={readable} W={writable} X={executable}")
            
            if not (readable and writable and executable):
                all_good = False
        else:
            print(f"✗ {name}: Does not exist")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("=== VIDEO PROCESSING DIAGNOSTIC TEST ===")
    print(f"Time: {datetime.now()}")
    print(f"Working directory: {os.getcwd()}")
    
    tests = [
        ("FFmpeg", test_ffmpeg),
        ("Permissions", check_permissions),
        ("Marker Creation", test_marker_creation),
        ("FFmpeg I-frame", test_ffmpeg_command),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    print("\n=== SUMMARY ===")
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    if all(r[1] for r in results):
        print("\\nAll tests passed! The issue might be in the application logic.")
    else:
        print("\\nSome tests failed. Fix the issues above before running the main application.")