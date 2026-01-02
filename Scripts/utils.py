# scripts/utils.py

import os
import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

def load_config(path="config.yaml"):
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    """Create directory if it doesn’t exist."""
    os.makedirs(path, exist_ok=True)

def save_object(obj, path):
    """Save any Python object (model, scaler, etc.)"""
    joblib.dump(obj, path)

def load_data(csv_path):
    """Load dataset as pandas DataFrame."""
    return pd.read_csv(csv_path)

def split_data(df, test_size, val_size, label_col, random_state=42, stratify=True):
    """Split dataset into train/val/test sets."""
    y = df[label_col]
    strat = y if stratify else None

    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=strat)
    val_portion = val_size / (1 - test_size)
    y_temp = temp_df[label_col]
    strat_temp = y_temp if stratify else None
    val_df, test_df = train_test_split(temp_df, test_size=val_portion, random_state=random_state, stratify=strat_temp)

    return train_df, val_df, test_
    
import logging

def ensure_dir(directory_path: str):
    """Create directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Setup a logger with optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers in case of reruns
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

        # Stream handler (prints to console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Optional file handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

