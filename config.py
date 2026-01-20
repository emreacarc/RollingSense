"""
Configuration file for RollingSense Predictive Maintenance System.
Centralizes all paths and settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODEL_REPORT_PATH = MODELS_DIR / "model_report.json"

# Source paths
SRC_DIR = PROJECT_ROOT / "src"

# UCI AI4I 2020 dataset URL
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

# Cross-validation settings
CV_FOLDS = 10
CV_SCORING = {
    'accuracy': 'accuracy',
    'f1': 'f1_macro'
}

# Inference speed test settings
INFERENCE_TEST_SIZE = 10000

# Model selection threshold (1% = 0.01)
F1_DIFFERENCE_THRESHOLD = 0.01

# Failure scenarios CSV path
FAILURE_SCENARIOS_CSV = MODELS_DIR / "failurescenarios.csv"
FAILURE_SCENARIOS_TARGET = 2000  # Target number of failure scenarios to collect

# Original failures CSV path (from raw dataset)
ORIGINAL_FAILURES_CSV = MODELS_DIR / "original_failures.csv"

# Failure indicator predictor path
FAILURE_INDICATOR_PREDICTOR_PATH = MODELS_DIR / "failure_indicator_predictor.pkl"

# Create necessary directories
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, SRC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

