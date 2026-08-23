"""
Serving API for the phishing classifier.

Fixes vs. the original draft:
  - /health actually reflects reality: it will NOT report "ok" if the model
    isn't loaded, or if Kafka streaming is enabled but the consumer thread
    has died. The original started a daemon thread that could silently die
    on a Kafka connection error while /health kept saying everything was fine.
  - Input is validated (empty string, oversized string, non-string) via
    Pydantic + explicit checks -> 4xx with a clear message, not a raw
    traceback / 500 leaking internals to the client.
  - Structured logging instead of print().
  - Model + feature extraction failures are caught and turned into proper
    HTTP error responses instead of crashing the request handler.
  - Kafka streaming is optional and config-gated (PHISHGUARD_KAFKA_ENABLED),
    since requiring a live Kafka cluster just to serve predictions is
    unnecessary coupling for most deployments.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.features import extract_features, is_known_safe_domain
from app.model_registry import get_current_model_file

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("phishguard.api")

app = FastAPI(title="PhishGuard", version="1.0.0")

# --- Rate limiting: per-client-IP, config-driven limit ---
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{config.RATE_LIMIT_PER_MINUTE}/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- API key auth ---
# Disabled entirely if PHISHGUARD_API_KEYS is unset/empty (local dev default).
# Set PHISHGUARD_API_KEYS (comma-separated) to enable it in any shared env.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str = Security(_api_key_header)) -> None:
    if not config.API_KEYS:
        return
    if api_key not in config.API_KEYS:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


# --- Model state, tracked explicitly so /health can be honest about it ---
_model = None
_decision_threshold = 0.5
_model_load_error: Optional[str] = None
_kafka_consumer_alive = threading.Event()  # only meaningful if Kafka enabled


def load_model():
    global _model, _decision_threshold, _model_load_error
    model_path = get_current_model_file(config.MODEL_DIR, config.MODEL_FILE)
    try:
        bundle = joblib.load(model_path)
        # Backward-compatible: older model files were a bare model, not a
        # {"model", "decision_threshold"} bundle.
        if isinstance(bundle, dict) and "model" in bundle:
            _model = bundle["model"]
            _decision_threshold = bundle.get("decision_threshold", 0.5)
        else:
            _model = bundle
            _decision_threshold = 0.5
        log.info("Model loaded from %s (decision_threshold=%.2f)",
                  model_path, _decision_threshold)
    except FileNotFoundError:
        _model_load_error = f"Model file not found at {model_path}. Run training first."
        log.error(_model_load_error)
    except Exception as e:  # noqa: BLE001 - deliberately broad at the boundary, logged
        _model_load_error = f"Failed to load model: {e}"
        log.exception("Model load failed")


load_model()


class URLInput(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("url must not be empty")
        if len(v) > config.MAX_URL_LENGTH:
            raise ValueError(f"url exceeds max length of {config.MAX_URL_LENGTH}")
        return v


class PredictionResponse(BaseModel):
    is_phishing: bool
    confidence: float
    url_length: int
    source: str = "model"  # "model" or "allowlist" - which decided the verdict


@app.get("/health")
def health():
    """
    Honest health check. Returns 503 (not 200) if the model isn't usable,
    so orchestrators (k8s, load balancers) correctly stop routing traffic
    here instead of trusting a lie.
    """
    checks = {
        "model_loaded": _model is not None,
        "model_load_error": _model_load_error,
    }
    if config.KAFKA_ENABLED:
        checks["kafka_consumer_alive"] = _kafka_consumer_alive.is_set()

    healthy = checks["model_loaded"] and (not config.KAFKA_ENABLED or checks["kafka_consumer_alive"])
    if not healthy:
        raise HTTPException(status_code=503, detail=checks)
    return {"status": "ok", **checks}


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
def predict_url(request: Request, input: URLInput, _auth: None = Security(require_api_key)):
    if _model is None:
        # Fail loudly and specifically, not with a generic 500 traceback.
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {_model_load_error or 'unknown load failure'}",
        )
    try:
        feats = extract_features(input.url)

        # Allowlist check happens BEFORE the model runs. See features.py for
        # why this exists: the model has weak signal on bare root domains
        # and produces false positives on major legitimate sites at the
        # recall-tuned threshold. This is a standard known-good bypass, not
        # a fix to the model itself.
        try:
            parsed = urlparse(input.url if "//" in input.url else f"//{input.url}")
            host = parsed.hostname or ""
        except ValueError:
            host = ""

        if is_known_safe_domain(host):
            return PredictionResponse(
                is_phishing=False,
                confidence=1.0,
                url_length=feats.url_length,
                source="allowlist",
            )

        X = pd.DataFrame([feats.to_dict()])
        proba = float(_model.predict_proba(X)[0, 1])
        pred = int(proba >= _decision_threshold)
    except Exception as e:  # noqa: BLE001 - boundary catch, never leak raw traceback to client
        log.exception("Prediction failed for input url (length=%d)", len(input.url))
        raise HTTPException(status_code=500, detail="Internal error during prediction") from e

    return PredictionResponse(
        is_phishing=bool(pred),
        confidence=round(proba, 4),
        url_length=feats.url_length,
        source="model",
    )


# --- Optional streaming consumer, config-gated and self-reporting ---
def _consume_live_urls():
    """
    Background Kafka consumer. Unlike the original draft, this:
      - only starts if explicitly enabled via config (no hard Kafka dependency
        to serve predictions),
      - sets a threading.Event the /health endpoint checks, so a dead
        consumer is visible instead of silently invisible,
      - retries with backoff instead of dying on the first connection error.
    """
    from kafka import KafkaConsumer  # imported lazily; not a hard dependency

    backoff = 5
    while True:
        try:
            consumer = KafkaConsumer(
                config.KAFKA_TOPIC,
                bootstrap_servers=config.KAFKA_BROKER,
                value_deserializer=lambda m: m.decode("utf-8"),
            )
            _kafka_consumer_alive.set()
            log.info("Kafka consumer connected, listening on topic '%s'", config.KAFKA_TOPIC)
            for msg in consumer:
                if _model is None:
                    continue
                url = msg.value
                feats = extract_features(url)
                X = pd.DataFrame([feats.to_dict()])
                proba = float(_model.predict_proba(X)[0, 1])
                pred = int(proba >= _decision_threshold)
                log.info("live_prediction is_phishing=%s conf=%.3f url=%.80s",
                          bool(pred), proba, url)
        except Exception:  # noqa: BLE001
            _kafka_consumer_alive.clear()
            log.exception("Kafka consumer error, retrying in %ds", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)


if config.KAFKA_ENABLED:
    threading.Thread(target=_consume_live_urls, daemon=True).start()
