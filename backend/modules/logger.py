import sys
import logging
from .config import config 

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if config.DEBUG else logging.INFO)
    
    if not logger.handlers:
        logger.addHandler(stream_handler)
    return logger

log = get_logger(__name__)