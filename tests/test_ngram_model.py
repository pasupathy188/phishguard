import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ngram_model import train_ngram_model, predict_ngram_proba


def test_ngram_model_trains_and_predicts():
    urls = [
        "http://paypal-secure-login.com/verify",
        "http://amaz0n-account-suspended.top",
        "http://payp4l-confirm.net",
        "https://google.com",
        "https://github.com/docs",
        "https://python.org/downloads",
    ]
    y = [1, 1, 1, 0, 0, 0]
    vectorizer, clf = train_ngram_model(urls, y)
    probs = predict_ngram_proba(vectorizer, clf, urls)
    assert len(probs) == len(urls)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_ngram_model_handles_unseen_url():
    # A URL with characters/substrings never seen during fit should not
    # raise - unseen n-grams simply contribute zero to the vector.
    urls = ["http://example.com", "http://phishing-test.net"]
    y = [0, 1]
    vectorizer, clf = train_ngram_model(urls, y)
    probs = predict_ngram_proba(vectorizer, clf, ["http://completely-unseen-string-zzzqqq.io"])
    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
