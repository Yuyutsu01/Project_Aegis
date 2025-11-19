import logging
import os
import sys
from datetime import datetime

def setup_logger(name: str = "ProjectAegis", log_level: str = "INFO"):
    """Setup application logger with Windows compatibility"""
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Formatter without emojis for Windows compatibility
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler (always works)
    log_file = os.path.join(log_dir, f"trading_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Console handler with Windows compatibility
    console_handler = logging.StreamHandler()
    
    # Check if we're on Windows and adjust encoding
    if sys.platform == "win32":
        # For Windows, use a handler that can handle encoding issues
        try:
            # Try to set up console with UTF-8
            import io
            console_handler.stream = io.TextIOWrapper(
                console_handler.stream.buffer, 
                encoding='utf-8', 
                errors='replace'
            )
        except:
            # Fallback: remove emojis from formatter for console
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
    else:
        console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger