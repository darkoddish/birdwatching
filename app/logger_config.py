import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log file paths
MAIN_LOG_FILE = os.path.join(LOG_DIR, "birdwatch.log")
VIDEO_PROCESSING_LOG = os.path.join(LOG_DIR, "video_processing.log")
DEBUG_LOG_FILE = os.path.join(LOG_DIR, "debug.log")

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create loggers
main_logger = setup_logger('main', MAIN_LOG_FILE, logging.INFO)
video_logger = setup_logger('video_processing', VIDEO_PROCESSING_LOG, logging.DEBUG)
debug_logger = setup_logger('debug', DEBUG_LOG_FILE, logging.DEBUG)