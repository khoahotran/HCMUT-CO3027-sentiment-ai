import pytest
from app.model import predict_review, predict_aspect_sentiment

# ========= OVERALL SENTIMENT TESTS =========

def test_positive_simple():
    assert predict_review("Sản phẩm rất tốt") == "positive"

def test_negative_simple():
    assert predict_review("Hàng kém chất lượng") == "negative"

def test_neutral_simple():
    assert predict_review("Chất lượng ổn trong tầm giá") == "neutral"

def test_positive_shipping():
    assert predict_review("Ship rất nhanh, đóng gói cẩn thận") == "positive"

def test_negative_experience():
    assert predict_review("Thất vọng thật sự, không giống mô tả") == "negative"

def test_mixed_positive_negative():
    assert predict_review("Ship nhanh nhưng sản phẩm kém") == "neutral"

def test_mixed_negative_positive():
    assert predict_review("Sản phẩm xấu nhưng shop hỗ trợ tốt") == "neutral"

def test_short_neutral():
    assert predict_review("Tạm ổn") == "neutral"

def test_long_review_positive():
    text = "Sản phẩm dùng rất thích, giao hàng nhanh, sẽ ủng hộ lần sau"
    assert predict_review(text) == "positive"

def test_long_review_negative():
    text = "Chất lượng kém, giao hàng chậm, đóng gói sơ sài"
    assert predict_review(text) == "negative"


# ========= ASPECT SENTIMENT TESTS =========

def test_aspect_shipping_positive():
    result = predict_aspect_sentiment("Ship nhanh, rất hài lòng")
    assert result["shipping"] == "positive"

def test_aspect_price_negative():
    result = predict_aspect_sentiment("Giá hơi cao so với chất lượng")
    assert result["price"] == "negative"

def test_aspect_quality_negative():
    result = predict_aspect_sentiment("Chất lượng kém, dùng nhanh hỏng")
    assert result["quality"] == "negative"

def test_aspect_service_positive():
    result = predict_aspect_sentiment("Shop tư vấn rất nhiệt tình")
    assert result["service"] == "positive"

def test_multi_aspect_mixed():
    result = predict_aspect_sentiment(
        "Ship nhanh nhưng giá cao, chất lượng thì ổn"
    )
    assert result["shipping"] == "positive"
    assert result["price"] == "negative"
    assert result["quality"] == "neutral"

def test_multi_sentence_multi_aspect():
    result = predict_aspect_sentiment(
        "Giao hàng chậm. Giá rẻ. Chất lượng kém."
    )
    assert result["shipping"] == "negative"
    assert result["price"] == "positive"
    assert result["quality"] == "negative"

def test_other_aspect():
    result = predict_aspect_sentiment("Màu sắc đẹp, đúng như hình")
    assert "other" in result

def test_aspect_vote_majority():
    result = predict_aspect_sentiment(
        "Ship nhanh. Ship rất nhanh. Ship hơi chậm."
    )
    assert result["shipping"] == "positive"


# ========= EDGE CASES =========

def test_empty_text():
    assert predict_review("") == "neutral"

def test_only_symbols():
    assert predict_review("!!!") == "neutral"
