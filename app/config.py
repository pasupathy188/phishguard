"""
Centralized configuration. All tunables live here, sourced from environment
variables with sane defaults. Nothing in the rest of the codebase should
hardcode a path, host, or magic number — that's what makes this "one env
per environment" instead of "edit the source to deploy."
"""
import os

# --- Paths ---
DATA_DIR = os.getenv("PHISHGUARD_DATA_DIR", "data")
MODEL_DIR = os.getenv("PHISHGUARD_MODEL_DIR", "model")
RAW_DATA_FILE = os.getenv("PHISHGUARD_RAW_DATA_FILE", os.path.join(DATA_DIR, "urls.csv"))
MODEL_FILE = os.getenv("PHISHGUARD_MODEL_FILE", os.path.join(MODEL_DIR, "phishing_model.joblib"))
METRICS_FILE = os.getenv("PHISHGUARD_METRICS_FILE", os.path.join(MODEL_DIR, "metrics.json"))

# --- Training ---
TRAIN_SAMPLE_ROWS = int(os.getenv("PHISHGUARD_TRAIN_SAMPLE_ROWS", "500000"))
TEST_SIZE = float(os.getenv("PHISHGUARD_TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("PHISHGUARD_RANDOM_STATE", "42"))
N_ESTIMATORS = int(os.getenv("PHISHGUARD_N_ESTIMATORS", "200"))
MIN_ACCEPTABLE_RECALL = float(os.getenv("PHISHGUARD_MIN_RECALL", "0.85"))

# --- N-gram ensemble model ---
# The lexical/structural feature model (LightGBM) has documented weak signal
# on bare/short domains (see README: the false-positive incident). A
# character n-gram model sees raw substrings directly (e.g. "payp4l",
# "arnaz0n") without needing them hand-coded as a feature, so it catches
# patterns the engineered features miss. Combined via probability averaging.
NGRAM_ENABLED = os.getenv("PHISHGUARD_NGRAM_ENABLED", "true").lower() == "true"
NGRAM_RANGE_MIN = int(os.getenv("PHISHGUARD_NGRAM_MIN", "3"))
NGRAM_RANGE_MAX = int(os.getenv("PHISHGUARD_NGRAM_MAX", "5"))
NGRAM_MAX_FEATURES = int(os.getenv("PHISHGUARD_NGRAM_MAX_FEATURES", "20000"))
# Weight given to the LightGBM model's probability vs the n-gram model's
# when combining (0.6 = 60% lexical-feature model, 40% n-gram model).
ENSEMBLE_LGBM_WEIGHT = float(os.getenv("PHISHGUARD_ENSEMBLE_LGBM_WEIGHT", "0.6"))

# --- Serving ---
API_HOST = os.getenv("PHISHGUARD_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PHISHGUARD_API_PORT", "8000"))
MAX_URL_LENGTH = int(os.getenv("PHISHGUARD_MAX_URL_LENGTH", "2048"))

# --- Auth & rate limiting ---
# Empty string = auth disabled (useful for local dev). Set a real value in
# any shared/deployed environment. Comma-separated list allows multiple
# clients each with their own key without a database.
API_KEYS = frozenset(
    k.strip() for k in os.getenv("PHISHGUARD_API_KEYS", "").split(",") if k.strip()
)
RATE_LIMIT_PER_MINUTE = int(os.getenv("PHISHGUARD_RATE_LIMIT_PER_MINUTE", "60"))

# --- Kafka / streaming (optional component) ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_urls")
KAFKA_ENABLED = os.getenv("PHISHGUARD_KAFKA_ENABLED", "false").lower() == "true"

# --- Logging ---
LOG_LEVEL = os.getenv("PHISHGUARD_LOG_LEVEL", "INFO")
