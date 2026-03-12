import logging
import os
from datetime import datetime
from pathlib import Path

LOG_FILE = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"

# Anchor logs/ to the project root regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
logs_path = PROJECT_ROOT / "logs"
logs_path.mkdir(exist_ok=True)

LOG_FILE_PATH = str(logs_path / LOG_FILE)

_formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

_file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
_file_handler.setFormatter(_formatter)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_file_handler)