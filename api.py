from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.pipeline import SpamDetectionSystem

app = FastAPI(
    title="Enterprise Spam Detection Microservice",
    description="High-performance NLP REST API service for classifying SMS content.",
    version="2.0.0"
)

# Initialize Model Engine
detector = SpamDetectionSystem(model_path="models/spam_pipeline.joblib")

@app.on_event("startup")
def load_ml_model():
    try:
        detector.load_model()
    except FileNotFoundError:
        print("Warning: Model file not found. Run 'python src/train.py' first.")

class MessageRequest(BaseModel):
    message: str = Field(..., example="WINNER! Claim your $1000 prize now at http://bit.ly/claim")

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    decision_score: float
    is_spam: bool

@app.get("/")
def health_check():
    return {"status": "Online", "service": "Spam Detection Engine API"}

@app.post("/predict", response_model=PredictionResponse)
def classify_message(payload: MessageRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Input message cannot be empty.")
    
    try:
        result = detector.predict(payload.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    