import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipeline import SpamDetectionSystem


def ensure_data_directory(base_dir: str = ".") -> Path:
    data_dir = Path(base_dir) / "data"
    if data_dir.exists() and not data_dir.is_dir():
        data_dir.unlink()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def run_training():
    # Load dataset dynamically and save raw data to data/ folder
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    print("Fetching dataset...")
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})

    # Ensure local data directory exists
    data_dir = ensure_data_directory()
    df.to_csv(data_dir / "sms_dataset.csv", index=False)
    print(f"Saved dataset copy to {data_dir / 'sms_dataset.csv'}")

    # Split dataset (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['target'], test_size=0.2, random_state=42, stratify=df['target']
    )

    # Initialize, train, evaluate, and save pipeline
    system = SpamDetectionSystem(model_path="models/spam_pipeline.joblib")
    system.train(X_train, y_train)
    system.evaluate(X_test, y_test)
    system.save_model()

if __name__ == "__main__":
    run_training()