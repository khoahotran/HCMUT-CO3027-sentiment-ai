from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_sentiment

app = FastAPI(title="Vietnamese Sentiment API")

class ReviewRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok", "message": "Sentiment server running"}

@app.post("/sentiment")
def sentiment(req: ReviewRequest):
    result = predict_sentiment(req.text)
    return {
        "text": req.text,
        "sentiment": result
    }
