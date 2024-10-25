"""This file contains global paths."""

from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEMO_DATA_DIR = DATA_DIR / "demo_data"

CONFIG_DIR = PROJ_ROOT / "config"
MODELS_DIR = PROJ_ROOT / "models"

LOGS_DIR = PROJ_ROOT / "logs"
