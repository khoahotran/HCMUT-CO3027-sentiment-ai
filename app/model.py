import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "5CD-AI/Vietnamese-Sentiment-visobert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

LABEL_MAP = {
    0: "negative",
    1: "positive",
    2: "neutral"
}

def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = int(torch.argmax(outputs.logits, dim=1).item())
    return LABEL_MAP[pred_id]
