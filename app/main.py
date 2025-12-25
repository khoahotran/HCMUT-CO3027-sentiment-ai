from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model import predict_review, predict_aspect_sentiment

app = FastAPI(title="Vietnamese Sentiment API")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
