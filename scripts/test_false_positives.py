"""
Hits the running /predict endpoint with a batch of well-known, unambiguously
legitimate URLs and reports how many get (wrongly) flagged as phishing.

This exists specifically to measure the real-world cost of the recall-biased
decision threshold (0.43) before deciding whether an allowlist is needed -
one anecdotal false positive (github.com) isn't a rate, this is.

Usage:
    python scripts/test_false_positives.py
Assumes the API is already running locally (uvicorn app.main:app).
"""
import sys
import requests

API_URL = "http://localhost:8000/predict"

# A deliberately boring, uncontroversial set of top global sites - if a
# classifier flags a meaningful fraction of these, that's a real precision
# problem, not noise.
KNOWN_LEGITIMATE_URLS = [
    "https://github.com",
    "https://google.com",
    "https://microsoft.com",
    "https://apple.com",
    "https://amazon.com",
    "https://wikipedia.org",
    "https://stackoverflow.com",
    "https://nytimes.com",
    "https://bbc.com",
    "https://python.org",
    "https://linkedin.com",
    "https://reddit.com",
    "https://netflix.com",
    "https://paypal.com",
    "https://chase.com",
    "https://wellsfargo.com",
    "https://instagram.com",
    "https://twitter.com",
    "https://x.com",
    "https://cloudflare.com",
]


def main():
    results = []
    for url in KNOWN_LEGITIMATE_URLS:
        try:
            resp = requests.post(API_URL, json={"url": url}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            results.append((url, data["is_phishing"], data["confidence"]))
        except requests.RequestException as e:
            print(f"ERROR calling API for {url}: {e}")
            sys.exit(1)

    false_positives = [r for r in results if r[1]]

    print(f"{'URL':40} {'Flagged?':10} {'Confidence':10}")
    print("-" * 62)
    for url, flagged, conf in results:
        marker = "FALSE POS" if flagged else "ok"
        print(f"{url:40} {marker:10} {conf:.4f}")

    print()
    print(f"False positive rate: {len(false_positives)}/{len(results)} "
          f"({100 * len(false_positives) / len(results):.1f}%)")


if __name__ == "__main__":
    main()
