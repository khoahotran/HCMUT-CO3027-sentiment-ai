import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "5CD-AI/Vietnamese-Sentiment-visobert"
STRONG_POSITIVE = [
    "rất tốt",
    "rất thích",
    "hài lòng",
    "tuyệt vời",
    "xuất sắc",
    "rất nhanh",
]

WEAK_POSITIVE = ["ổn", "tạm ổn", "chấp nhận được", "bình thường"]

NEGATIVE_HINTS = ["kém", "tệ", "xấu", "chậm", "hỏng", "thất vọng", "cao"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

LABEL_MAP = {0: "negative", 1: "positive", 2: "neutral"}


def split_sentences(text: str):
    separators = r"[.!?,]| nhưng | tuy nhiên | nhưng mà "
    sentences = re.split(separators, text, flags=re.IGNORECASE)
    return [s.strip() for s in sentences if s.strip()]


def predict_sentence(text: str):
    text_lower = text.lower()

    # strong positive
    if any(p in text_lower for p in STRONG_POSITIVE):
        if any(n in text_lower for n in NEGATIVE_HINTS):
            return "neutral"
        return "positive"

    # weak positive → neutral
    if any(w in text_lower for w in WEAK_POSITIVE):
        return "neutral"

    # negative
    if any(n in text_lower for n in NEGATIVE_HINTS):
        return "negative"

    # fallback to model
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = int(torch.argmax(outputs.logits, dim=1).item())
    return LABEL_MAP[pred_id]


SENTIMENT_SCORE = {"negative": -1, "neutral": 0, "positive": 1}


def predict_review(text: str):
    sentences = split_sentences(text)
    if not sentences:
        return "neutral"

    scores = {"positive": 0, "neutral": 0, "negative": 0}

    for s in sentences:
        sentiment = predict_sentence(s)
        scores[sentiment] += 1

    # mixed positive + negative → neutral
    if scores["positive"] > 0 and scores["negative"] > 0:
        return "neutral"

    return max(scores, key=lambda x: scores[x])


ASPECT_KEYWORDS = {
    "shipping": ["ship", "giao", "vận chuyển", "gửi hàng"],
    "price": ["giá", "đắt", "rẻ", "tiền"],
    "quality": ["chất lượng", "hỏng", "bền", "kém", "dùng", "sản phẩm"],
    "service": ["shop", "tư vấn", "phục vụ", "CSKH"],
}


def detect_aspects(sentence: str):
    sentence_lower = sentence.lower()
    aspects = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(k in sentence_lower for k in keywords):
            aspects.append(aspect)

    return aspects if aspects else ["other"]


def predict_aspect_sentiment(text: str):
    sentences = split_sentences(text)
    result = {}

    for s in sentences:
        sentiment = predict_sentence(s)
        aspects = detect_aspects(s)
        s_lower = s.lower()

        for aspect in aspects:
            if aspect not in result:
                result[aspect] = {"positive": 0, "neutral": 0, "negative": 0}

            # quality override
            if aspect == "quality" and any(n in s_lower for n in NEGATIVE_HINTS):
                result[aspect]["negative"] += 2
                continue

            # price override
            if aspect == "price":
                if "rẻ" in s_lower or "hợp lý" in s_lower:
                    result[aspect]["positive"] += 2
                    continue
                if "cao" in s_lower:
                    result[aspect]["negative"] += 2
                    continue

            # normal vote
            result[aspect][sentiment] += 1

    # final vote
    final_result = {}
    for aspect, scores in result.items():
        final_result[aspect] = max(scores, key=lambda x: scores[x])

    return final_result
