import logging
import os

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    dirpath = os.path.abspath(os.path.dirname(__file__))
    logfileDir = os.path.join(dirpath, 'logs\\')
    if not os.path.exists(logfileDir):
        os.makedirs(logfileDir)
    logfile = os.path.join(dirpath, "logs\\log.log")
    handler = logging.FileHandler(logfile, mode='w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger