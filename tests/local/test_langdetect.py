from app.pipeline.stage2_comment_fetcher import is_english


def test_english_returns_true():
    assert is_english("This product is absolutely amazing, I love it!") is True


def test_nonenglish_returns_false():
    assert is_english("Ce produit est absolument incroyable, je l'adore!") is False


def test_empty_string_returns_false():
    assert is_english("") is False


def test_emoji_only_returns_false():
    assert is_english("😂🔥💯🎉👍") is False


def test_very_short_string_returns_false():
    assert is_english("hi") is False
