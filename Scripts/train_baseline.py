import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data(config):
    data_path = config["paths"]["processed"]
    target_col = config["data"]["target"]
    sample_size = config["data"].get("sample_size", 100000)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=True)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} cols")

    # ✅ Downsample before conversion to save RAM
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {len(df):,} rows for training.")

    # Convert types to save memory
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

    # Split into features and labels
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    # --- Handle non-numeric columns (drop or encode) ---
    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns
    if len(non_numeric_cols) > 0:
        print(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
        X = X.drop(columns=non_numeric_cols)


    # Ensure y is categorical for classification
    y = y.astype('category').cat.codes

    # Train/Val/Test Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_baseline(X_train, y_train, X_val, y_val):
    print("Training RandomForest baseline model...")
    model = RandomForestClassifier(
        n_estimators=50,  # reduce trees to save time/memory
        max_depth=10,     # shallow trees = less RAM
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Validation performance
    y_pred = model.predict(X_val)
    print("\nValidation Results:")
    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    return model

if __name__ == "__main__":
    config = load_config()
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(config)

    model = train_baseline(X_train, y_train, X_val, y_val)

    # Save model
    model_path = config["paths"]["models"] + "/baseline_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
