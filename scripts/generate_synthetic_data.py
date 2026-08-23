"""
Generates a synthetic urls.csv for local testing/demo purposes, since we
don't have network access to Kaggle in this environment. Structurally
matches what the real dataset looks like: `url`, `label` columns.
"""
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config

random.seed(42)

BENIGN_DOMAINS = ["wikipedia.org", "github.com", "nytimes.com", "python.org",
                   "amazon.com", "stackoverflow.com", "bbc.com", "cloudflare.com"]
BENIGN_PATHS = ["", "/articles/latest", "/docs/api", "/products/item123",
                "/search?q=test", "/blog/2026/announcement"]

PHISH_DOMAINS = ["paypa1-secure.com", "appleid-verify-account.net",
                  "secure-login-bank0famerica.com", "account-update-chase.info",
                  "192.168.44.12", "verify-wallet-metamask.xyz"]
PHISH_PATHS = ["/login", "/secure/verify", "/account/confirm",
               "/wp-includes/signin.php", "/auth?redirect=1"]


def make_rows(n: int, domains, paths, https_rate: float, label: str):
    rows = []
    for _ in range(n):
        domain = random.choice(domains)
        path = random.choice(paths)
        scheme = "https" if random.random() < https_rate else "http"
        url = f"{scheme}://{domain}{path}"
        rows.append({"url": url, "label": label})
    return rows


def main():
    n_per_class = 4000
    rows = (
        make_rows(n_per_class, BENIGN_DOMAINS, BENIGN_PATHS, https_rate=0.9, label="benign")
        + make_rows(n_per_class, PHISH_DOMAINS, PHISH_PATHS, https_rate=0.3, label="phishing")
    )
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_DATA_FILE, index=False)
    print(f"Wrote {len(df)} synthetic rows to {config.RAW_DATA_FILE}")


if __name__ == "__main__":
    main()
