import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.features import extract_features, _shannon_entropy, URLFeatures


def test_empty_string_does_not_crash():
    # This is the exact bug class the original bare `except:` was masking.
    feats = extract_features("")
    assert feats.entropy == 0.0
    assert feats.url_length == 0


def test_none_and_non_string_input_is_safe():
    assert extract_features(None).url_length == 0
    assert extract_features(123).url_length == 0  # type: ignore[arg-type]


def test_entropy_of_single_repeated_char_is_zero():
    # A string with one unique character has zero uncertainty -> entropy 0.
    assert _shannon_entropy("aaaaaa") == 0.0


def test_entropy_matches_known_value():
    # "aabb" -> 2 symbols, each p=0.5 -> entropy = 1.0 bit exactly.
    assert math.isclose(_shannon_entropy("aabb"), 1.0, rel_tol=1e-9)


def test_valid_ipv4_host_detected():
    feats = extract_features("http://192.168.1.1/login")
    assert feats.has_ip_host == 1


def test_invalid_ipv4_like_string_not_flagged():
    # Regression test: the naive regex in the original draft would wrongly
    # match an out-of-range octet like 999 or a version-string-like host.
    feats = extract_features("http://1.2.3.4000.example.com/login")
    assert feats.has_ip_host == 0


def test_scheme_does_not_change_features():
    """The scheme is stripped before feature extraction, so HTTP, HTTPS, and
    scheme-less URLs must produce identical feature vectors. This guards
    against the train/serve skew that caused `is_https` to be a no-op."""
    assert extract_features("https://example.com/a") == extract_features("example.com/a")
    assert extract_features("http://example.com/a") == extract_features("example.com/a")
    assert extract_features("HTTPS://example.com/a") == extract_features("example.com/a")


def test_suspicious_keyword_detected():
    assert extract_features("http://example.com/secure-login").has_suspicious_keyword == 1
    assert extract_features("http://example.com/products").has_suspicious_keyword == 0


def test_column_names_match_dataclass_fields():
    feats = extract_features("http://example.com")
    assert set(feats.to_dict().keys()) == set(URLFeatures.column_names())


def test_subdomain_count():
    assert extract_features("http://a.b.c.example.com").subdomain_count >= 1
    assert extract_features("http://example.com").subdomain_count == 0


def test_high_risk_tld_flagged():
    assert extract_features("http://free-gift-card.xyz").has_high_risk_tld == 1
    assert extract_features("http://example.com").has_high_risk_tld == 0


def test_brand_lookalike_flagged_but_not_real_domain():
    assert extract_features("http://paypal-secure-login.com/verify").has_brand_lookalike == 1
    assert extract_features("http://www.paypal.com/login").has_brand_lookalike == 0
    assert extract_features("http://example.com").has_brand_lookalike == 0


def test_leet_brand_substitution_detected():
    assert extract_features("http://payp4l-verify.net").has_leet_brand_substitution == 1
    assert extract_features("http://amaz0n-account.com").has_leet_brand_substitution == 1
    # real domain with no substitution chars shouldn't false-positive
    assert extract_features("http://example.com").has_leet_brand_substitution == 0


def test_domain_digit_ratio():
    high = extract_features("http://192837465.suspicious-host.com").domain_digit_ratio
    low = extract_features("http://example.com").domain_digit_ratio
    assert high > low


def test_known_safe_domain_allowlist():
    from app.features import is_known_safe_domain
    assert is_known_safe_domain("github.com") is True
    assert is_known_safe_domain("www.github.com") is True
    assert is_known_safe_domain("totally-not-github.com") is False
    assert is_known_safe_domain("") is False

def test_host_parsed_when_url_contains_embedded_scheme():
    feats = extract_features("http://192.168.1.1/go?u=http://x.com")
    assert feats.has_ip_host == 1


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])