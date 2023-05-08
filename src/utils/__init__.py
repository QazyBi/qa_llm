from pathlib import Path


def get_data_path():
    return Path(__file__).parent.parent.parent / 'data'


import logging


def get_logger():
    logging.basicConfig()
    logger = logging.getLogger("question-answering")
    logger.setLevel(logging.INFO)

    return logger
