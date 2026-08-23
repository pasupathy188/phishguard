"""
Feature extraction for URL classification.

This module is imported by BOTH the training pipeline and the serving API.
That is deliberate: it is the single source of truth for how a raw URL becomes
a feature vector. If training and serving ever compute features differently
(a copy-pasted function that drifts out of sync), the model silently degrades
in production. This is one of the most common real-world MLOps bugs
("train/serve skew") and this module exists specifically to prevent it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from urllib.parse import urlparse

# Keyword list externalized as a constant (not buried inline) so it can be
# swapped, extended, or loaded from config without touching extraction logic.
SUSPICIOUS_KEYWORDS = (
    "login", "signin", "auth", "secure", "bank",
    "verify", "account", "update", "confirm", "wallet",
)
_KEYWORD_PATTERN = re.compile("|".join(SUSPICIOUS_KEYWORDS), re.IGNORECASE)

# A syntactically valid IPv4 octet is 0-255. The naive `\d+\.\d+\.\d+\.\d+`
# regex used in the original draft also matches garbage like "1.2.3.4000".
_IPV4_OCTET = r"(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
_IPV4_PATTERN = re.compile(rf"^{_IPV4_OCTET}(\.{_IPV4_OCTET}){{3}}$")

# TLDs disproportionately abused for phishing/spam because they're cheap or
# unmoderated. This isn't exhaustive - it's a known-risk list, updateable
# without touching extraction logic. Legitimate sites using these TLDs exist,
# so this is a signal, not a verdict.
HIGH_RISK_TLDS = frozenset({
    "xyz", "top", "club", "click", "link", "work", "icu", "cn", "tk", "ml",
    "ga", "cf", "gq", "buzz", "rest", "fit", "loan", "men", "date", "racing",
    "review", "party", "trade", "webcam", "win", "bid", "stream", "download",
})

# Well-known brands frequently impersonated via typosquatting/lookalike
# domains (e.g. "paypa1.com", "amaz0n-secure.com"). Used only to flag when
# a brand name appears WITHOUT being the actual registered brand domain -
# not a blocklist of the brands themselves.
IMPERSONATED_BRANDS = (
    "paypal", "apple", "amazon", "microsoft", "google", "facebook", "netflix",
    "bankofamerica", "wellsfargo", "chase", "citibank", "instagram", "ebay",
)

_LEGIT_BRAND_DOMAINS = {
    "paypal.com", "apple.com", "amazon.com", "microsoft.com", "google.com",
    "facebook.com", "netflix.com", "bankofamerica.com", "wellsfargo.com",
    "chase.com", "citibank.com", "instagram.com", "ebay.com",
}

# Leetspeak-style substitutions used to visually mimic letters in brand names
# (e.g. "0" for "o", "1" for "l"/"i"). Used for detecting a brand name
# hidden via character substitution, e.g. "payp4l" or "amaz0n".
_LEET_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"})


@dataclass(frozen=True)
class URLFeatures:
    """Explicit, typed feature schema. Field order = column order everywhere."""
    url_length: int
    num_dots: int
    num_hyphens: int
    num_underscores: int
    num_slashes: int
    num_digits: int
    num_special_chars: int
    has_ip_host: int
    is_https: int
    has_suspicious_keyword: int
    entropy: float
    subdomain_count: int
    has_high_risk_tld: int
    has_brand_lookalike: int
    has_leet_brand_substitution: int
    domain_digit_ratio: float

    @classmethod
    def column_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _shannon_entropy(s: str) -> float:
    """
    Correct Shannon entropy over the character distribution of `s`.

    The original draft's version had two real bugs:
      1. No guard for empty string -> ZeroDivisionError masked by a bare except.
      2. Because it recomputed count/len per unique char inline, it was easy
         to get subtly wrong; here it's computed explicitly and unit-tested.
    """
    if not s:
        return 0.0
    length = len(s)
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _is_ip_host(host: str) -> int:
    return 1 if _IPV4_PATTERN.match(host) else 0


def _registrable_domain(host: str) -> str:
    """Best-effort last-two-labels extraction (e.g. 'sub.example.com' -> 'example.com')."""
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _get_tld(host: str) -> str:
    parts = host.rsplit(".", 1)
    return parts[-1].lower() if len(parts) == 2 else ""


def _has_brand_lookalike(host: str) -> int:
    """
    Flags when a known brand name appears in the host, but the host is NOT
    that brand's actual registered domain. E.g. 'paypal-secure-login.com'
    contains 'paypal' but isn't paypal.com -> flagged. 'www.paypal.com'
    contains 'paypal' AND is paypal.com -> not flagged.
    """
    if not host:
        return 0
    registrable = _registrable_domain(host)
    if registrable in _LEGIT_BRAND_DOMAINS:
        return 0
    host_lower = host.lower()
    return 1 if any(brand in host_lower for brand in IMPERSONATED_BRANDS) else 0


def _has_leet_brand_substitution(host: str) -> int:
    """
    Flags brand names hidden via leetspeak-style character substitution,
    e.g. 'payp4l.com' or 'amaz0n-verify.net' - normalizes digits back to
    likely letters and checks for a brand match that wasn't already caught
    by the plain substring check.
    """
    if not host:
        return 0
    normalized = host.lower().translate(_LEET_MAP)
    if normalized == host.lower():
        return 0  # no substitution chars present, nothing new to detect
    registrable = _registrable_domain(host)
    if registrable in _LEGIT_BRAND_DOMAINS:
        return 0
    return 1 if any(brand in normalized for brand in IMPERSONATED_BRANDS) else 0


def _domain_digit_ratio(host: str) -> float:
    """High digit density in a hostname (not a subdomain like 'cdn2') is a
    common obfuscation/randomly-generated-domain signal."""
    if not host:
        return 0.0
    digits = sum(c.isdigit() for c in host)
    return round(digits / len(host), 4)


# Well-known, unambiguously legitimate root domains. Checked BEFORE the model
# runs, not after - these bypass classification entirely rather than relying
# on the model to score them correctly. This exists because testing revealed
# the model (trained mostly on URLs with paths/subdomains) has weak signal
# on bare root domains and produces false positives on major legitimate
# sites once the recall-tuned decision threshold is applied. An allowlist is
# a standard production pattern for exactly this kind of known-good bypass -
# it does not fix the underlying training-data distribution gap, it only
# prevents that gap from causing embarrassing false positives on top sites.
KNOWN_SAFE_DOMAINS = frozenset({
    "google.com", "github.com", "microsoft.com", "apple.com", "amazon.com",
    "wikipedia.org", "stackoverflow.com", "nytimes.com", "bbc.com",
    "python.org", "linkedin.com", "reddit.com", "netflix.com", "paypal.com",
    "chase.com", "wellsfargo.com", "instagram.com", "twitter.com", "x.com",
    "cloudflare.com", "facebook.com", "youtube.com",
})


def is_known_safe_domain(host: str) -> bool:
    """True if the URL's registrable domain is on the allowlist."""
    if not host:
        return False
    registrable = _registrable_domain(host.lower())
    return registrable in KNOWN_SAFE_DOMAINS


def extract_features(url: str) -> URLFeatures:
    """
    Extract a fixed-schema feature vector from a raw URL string.

    Never raises on malformed input: unparseable URLs degrade gracefully to
    a "maximally suspicious / minimal information" feature vector rather than
    crashing the caller. Callers (API, training) can still choose to reject
    empty input upstream, but this function itself is total.
    """
    if not isinstance(url, str) or not url.strip():
        # Degenerate input: return a well-defined zero-ish vector rather than
        # raising, so a single bad row can't crash a batch training job.
        return URLFeatures(
            url_length=0, num_dots=0, num_hyphens=0, num_underscores=0,
            num_slashes=0, num_digits=0, num_special_chars=0, has_ip_host=0,
            is_https=0, has_suspicious_keyword=0, entropy=0.0, subdomain_count=0,
            has_high_risk_tld=0, has_brand_lookalike=0,
            has_leet_brand_substitution=0, domain_digit_ratio=0.0,
        )

    url = url.strip()

    try:
        parsed = urlparse(url if "//" in url else f"//{url}")
        host = parsed.hostname or ""
    except ValueError:
        host = ""

    special_chars = sum(1 for c in url if not c.isalnum() and c not in ".-_/:")

    return URLFeatures(
        url_length=len(url),
        num_dots=url.count("."),
        num_hyphens=url.count("-"),
        num_underscores=url.count("_"),
        num_slashes=url.count("/"),
        num_digits=sum(c.isdigit() for c in url),
        num_special_chars=special_chars,
        has_ip_host=_is_ip_host(host),
        is_https=1 if url.lower().startswith("https://") else 0,
        has_suspicious_keyword=1 if _KEYWORD_PATTERN.search(url) else 0,
        entropy=round(_shannon_entropy(url), 4),
        subdomain_count=max(host.count(".") - 1, 0) if host else 0,
        has_high_risk_tld=1 if _get_tld(host) in HIGH_RISK_TLDS else 0,
        has_brand_lookalike=_has_brand_lookalike(host),
        has_leet_brand_substitution=_has_leet_brand_substitution(host),
        domain_digit_ratio=_domain_digit_ratio(host),
    )
