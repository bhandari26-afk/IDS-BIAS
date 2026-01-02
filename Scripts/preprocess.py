# scripts/preprocess.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
from utils import setup_logger, ensure_dir


def load_config():
    """Load config.yaml relative to script location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_data(input_path, output_path, config, chunk_size=100_000):
    """Main preprocessing pipeline."""
    logger = setup_logger("preprocess")
    logger.info(f"Reading data from {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Initial shape: {df.shape}")

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)
    logger.info(f"After dropping duplicates: {df.shape}")

    # Drop columns with a single unique value (no information)
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    logger.info(f"Dropped {len(cols_to_drop)} constant columns")

    # Handle missing values
    missing_threshold = config["preprocessing"]["missing_threshold"]
    df.dropna(thresh=int((1 - missing_threshold) * len(df.columns)), inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    logger.info("Handled missing values")

    # Label encode target column
    target_col = config["data"]["target_col"]
    if target_col in df.columns:
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        logger.info(f"Encoded target column '{target_col}' with {len(le.classes_)} classes")
    else:
        logger.error(f"Target column '{target_col}' not found!")
        return

    # Select numeric features for scaling
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]

    # Convert to float32 to save memory
    df[numeric_cols] = df[numeric_cols].astype(np.float32)

    # Replace inf/-inf and fill NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df[numeric_cols] = df[numeric_cols].clip(lower=-1e10, upper=1e10)

    # Scale numeric columns in batches
    scaler = StandardScaler()
    scaler.fit(df[numeric_cols])
    batch_size = config.get("batch_size", 100000)
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        df.iloc[start:end, df.columns.get_indexer(numeric_cols)] = scaler.transform(
            df[numeric_cols].iloc[start:end]
        )
        if start % (10 * batch_size) == 0:
            logger.info(f"Scaled {start:,} / {len(df):,} rows...")
    logger.info(f"Scaled {len(numeric_cols)} numeric columns")


    # Save preprocessed dataset
    ensure_dir(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved to {output_path}")

    return df


if __name__ == "__main__":
    config = load_config()
    input_path = config["paths"]["combined_data"]
    output_path = config["paths"]["preprocessed_data"]

    preprocess_data(input_path, output_path, config)
