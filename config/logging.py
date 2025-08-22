import logging
import os
from pythonjsonlogger import jsonlogger

def setup_camera_logger(camera_id: int) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/camera_{camera_id}.json"

    logger = logging.getLogger(f"camera_{camera_id}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        formatter = jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger