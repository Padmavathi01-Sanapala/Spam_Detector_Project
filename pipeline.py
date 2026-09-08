import os
import re
import logging
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TextCleaner(BaseEstimator, TransformerMixin):
    """Custom Transformer for advanced NLP text preprocessing."""
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
        words = [self.ps.stem(word) for word in text.split() if word not in self.stop_words]
        return ' '.join(words)

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.apply(self._clean_text)
        return [self._clean_text(text) for text in X]

class SpamDetectionSystem:
    """Production ML Manager class for training, evaluating, saving, and serving models."""
    def __init__(self, model_path: str = "models/spam_pipeline.joblib"):
        self.model_path = model_path
        self.pipeline = None

    def build_pipeline(self) -> Pipeline:
        """Constructs an end-to-end Machine Learning Pipeline."""
        return Pipeline([
            ('cleaner', TextCleaner()),
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', LinearSVC(C=1.0, random_state=42))
        ])

    def train(self, X_train, y_train):
        logger.info("Initializing and training the Pipeline...")
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        logger.info("Model training completed successfully.")

    def evaluate(self, X_test, y_test):
        if not self.pipeline:
            raise ValueError("Model has not been trained yet.")
        predictions = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        logger.info(f"Model Accuracy: {acc * 100:.2f}%")
        print("\n--- Classification Performance Report ---")
        print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        logger.info(f"Model artifact saved to {self.model_path}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No artifact found at {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        logger.info(f"Loaded existing model from {self.model_path}")

    def predict(self, text: str) -> dict:
        if not self.pipeline:
            self.load_model()
        
        prediction = self.pipeline.predict([text])[0]
        label = "SPAM" if prediction == 1 else "HAM"
        decision_score = float(self.pipeline.decision_function([text])[0])
        
        return {
            "text": text,
            "prediction": label,
            "decision_score": round(decision_score, 4),
            "is_spam": bool(prediction == 1)
        }