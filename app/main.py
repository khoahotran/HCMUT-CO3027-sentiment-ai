from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_review, predict_aspect_sentiment

app = FastAPI(title="Vietnamese Sentiment API")


class ReviewRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"status": "ok", "message": "Sentiment server running"}


@app.post("/sentiment")
def sentiment(req: ReviewRequest):
    return {
        "overall_sentiment": predict_review(req.text),
        "aspect_sentiment": predict_aspect_sentiment(req.text),
    }
