"""
Character n-gram phishing model.

Motivation (see README's false-positive incident): the hand-crafted feature
model (LightGBM on url_length, entropy, TLD flags, etc.) has weak signal on
short/bare domains and relies on a hardcoded brand list to catch lookalikes
like "payp4l" or "arnaz0n". A character n-gram model sees the raw substrings
directly - it can learn that "payp" + "4l" co-occurring is suspicious without
anyone having to enumerate every possible brand/substitution combination in
advance. This generalizes to brands and obfuscation patterns not on any
hardcoded list.

This is intentionally a SEPARATE, simple model (TF-IDF character n-grams +
SGDClassifier) rather than folding n-gram features into the LightGBM
feature set - keeping it separate makes the ensemble's behavior easier to
reason about and lets either half be retrained/disabled independently
(see config.NGRAM_ENABLED).

Memory note: fitting a classifier on many millions of URLs is genuinely
memory-heavy. Two settings control this:
  - config.NGRAM_MAX_FEATURES: caps the vocabulary size (fewer features =
    less memory). Lower this (e.g. to 2000) on memory-constrained machines.
  - SGDClassifier instead of LogisticRegression: SGD trains via stochastic
    gradient descent over mini-batches rather than needing the full matrix
    plus solver internals in memory at once. This is the standard
    scikit-learn recommendation for "logistic-regression-equivalent
    training on data that doesn't comfortably fit in memory" - loss="log_loss"
    makes it mathematically equivalent to logistic regression, just trained
    differently.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from app import config

log = logging.getLogger("ngram_model")


def build_ngram_pipeline() -> TfidfVectorizer:
    """
    Character n-grams (default 3-5 chars) over the raw URL string.
    analyzer="char_wb" respects word boundaries within the string (treats
    it as one token stream), which is appropriate here since URLs aren't
    whitespace-separated natural language - we want raw substring patterns,
    not "words".
    """
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(config.NGRAM_RANGE_MIN, config.NGRAM_RANGE_MAX),
        max_features=config.NGRAM_MAX_FEATURES,
        lowercase=True,
    )


def train_ngram_model(urls: list[str], y) -> tuple[TfidfVectorizer, SGDClassifier]:
    """Fits the vectorizer and classifier together. Returns both - the
    vectorizer must be reused at inference time to produce a consistent
    feature space (this is the n-gram equivalent of train/serve skew
    protection that app/features.py provides for the lexical features)."""
    vectorizer = build_ngram_pipeline()
    log.info("Fitting character n-gram vectorizer (range=%d-%d, max_features=%d)...",
              config.NGRAM_RANGE_MIN, config.NGRAM_RANGE_MAX, config.NGRAM_MAX_FEATURES)
    X_ngram = vectorizer.fit_transform(urls)

    clf = SGDClassifier(
        loss="log_loss",        # mathematically equivalent to logistic regression
        class_weight="balanced",
        max_iter=50,
        n_jobs=-1,
        random_state=42,
    )
    log.info("Training SGD classifier (log_loss) on %d n-gram features...", X_ngram.shape[1])
    clf.fit(X_ngram, y)
    return vectorizer, clf


def predict_ngram_proba(vectorizer: TfidfVectorizer, clf: SGDClassifier, urls: list[str]) -> np.ndarray:
    """Returns phishing-class probability for each URL. Never raises on a
    single bad URL - TfidfVectorizer.transform handles arbitrary strings
    (including ones with characters unseen during fit) without erroring,
    since unseen n-grams simply contribute zero to the vector."""
    X_ngram = vectorizer.transform(urls)
    return clf.predict_proba(X_ngram)[:, 1]