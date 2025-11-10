import logging
from pathlib import Path


def setup_logger(log_dir: Path) -> logging.Logger:
    log_file = log_dir / "train.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.setLevel(logging.ERROR)
    if logger.handlers:
        return logger

    filehandler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(name)s: %(message)s")
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger
