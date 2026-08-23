"""
Data validation layer for the raw training CSV.

This is intentionally NOT a Great Expectations integration (that's a real,
heavier dependency choice for a team that wants a shared validation
framework across many datasets). For a single-dataset project, explicit
Python checks are more transparent and just as enforceable - the point of
a validation layer is that it runs and can fail the pipeline, not which
library does it.

Every check here either raises ValueError (hard stop - bad enough that
training should not proceed) or logs a warning (worth knowing, not fatal).
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("data_validation")


class DataValidationError(ValueError):
    """Raised when the raw dataset fails a hard validation check."""


def validate_raw_data(df: pd.DataFrame, min_rows: int = 1000) -> None:
    """
    Runs all validation checks against a loaded raw dataframe. Raises
    DataValidationError on any hard failure. Call this AFTER load_raw_data's
    basic column/null check and BEFORE feature extraction, so a bad dataset
    is rejected before spending time on the expensive part of the pipeline.
    """
    _check_min_rows(df, min_rows)
    _check_label_distribution(df)
    _check_url_sanity(df)
    _check_duplicates(df)


def _check_min_rows(df: pd.DataFrame, min_rows: int) -> None:
    if len(df) < min_rows:
        raise DataValidationError(
            f"Dataset has only {len(df)} rows after cleaning, below the "
            f"minimum of {min_rows}. Refusing to train on too little data."
        )


def _check_label_distribution(df: pd.DataFrame) -> None:
    """
    A dataset with only one class, or a wildly skewed one, will silently
    produce a useless or misleading model. This is the exact failure mode
    that would have occurred earlier in this project if the -1/1 numeric
    label mapping had gone unnoticed - all rows could have silently mapped
    to a single class.
    """
    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise DataValidationError(
            f"Dataset contains only one distinct label value: {counts.to_dict()}. "
            f"A classifier needs both classes present."
        )
    minority_ratio = counts.min() / counts.sum()
    if minority_ratio < 0.02:
        log.warning(
            "Severe class imbalance: minority class is only %.1f%% of the data (%s). "
            "Model quality may suffer even with class-weighting.",
            minority_ratio * 100, counts.to_dict(),
        )


def _check_url_sanity(df: pd.DataFrame) -> None:
    """Flags rows that are technically non-null but structurally useless."""
    blank = (df["url"].astype(str).str.strip() == "").sum()
    if blank > 0:
        log.warning("%d rows have a blank/whitespace-only url after stripping", blank)

    too_short = (df["url"].astype(str).str.len() < 4).sum()
    if too_short > len(df) * 0.01:
        log.warning(
            "%d rows (%.1f%%) have suspiciously short urls (<4 chars) - "
            "check for a parsing or export issue in the source data.",
            too_short, 100 * too_short / len(df),
        )


def _check_duplicates(df: pd.DataFrame) -> None:
    """
    Duplicate URLs with CONFLICTING labels indicate a real data quality
    problem (the same URL labeled both phishing and benign across rows) -
    this actively confuses the model and is worth knowing about even though
    it doesn't block training by itself.
    """
    dup_mask = df.duplicated(subset=["url"], keep=False)
    if not dup_mask.any():
        return
    dup_df = df[dup_mask]
    conflicting = dup_df.groupby("url")["label"].nunique()
    conflicting = conflicting[conflicting > 1]
    if len(conflicting) > 0:
        log.warning(
            "%d distinct URLs appear multiple times with CONFLICTING labels "
            "in the raw data. Example: %s",
            len(conflicting), conflicting.index[0],
        )
