import logging

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger: logging.Logger = logging.getLogger(name)
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(format=LOG_FORMAT, level=level)
    return logger
