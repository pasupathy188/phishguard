"""
Lightweight model registry.

Not MLflow/a hosted registry - for a single-model project, a timestamped
file naming scheme plus a JSON index is a real, working versioning system
without adding infrastructure. It answers the questions a registry exists
for: which model is currently "production", what were its metrics, and can
I roll back to a previous one.

model/registry.json structure:
{
  "current": "phishing_model_20260823_144210.joblib",
  "versions": [
    {"file": "...", "trained_at": "...", "recall": 0.85, "roc_auc": 0.93, "n_train": ...},
    ...
  ]
}
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone


def register_model(model_dir: str, trained_model_path: str, metrics: dict) -> str:
    """
    Copies the just-trained model into a timestamped, versioned filename,
    updates registry.json to point "current" at it, and appends a history
    entry. Returns the new versioned filename.

    Never overwrites a previous version's file - each trained model that
    passes the quality gate gets its own permanent artifact, so rollback is
    just editing registry.json's "current" pointer back to an old filename.
    """
    os.makedirs(model_dir, exist_ok=True)
    registry_path = os.path.join(model_dir, "registry.json")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    versioned_name = f"phishing_model_{timestamp}.joblib"
    versioned_path = os.path.join(model_dir, versioned_name)
    shutil.copy2(trained_model_path, versioned_path)

    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"current": None, "versions": []}

    registry["current"] = versioned_name
    registry["versions"].append({
        "file": versioned_name,
        "trained_at": timestamp,
        "recall": metrics.get("recall"),
        "precision": metrics.get("precision"),
        "roc_auc": metrics.get("roc_auc"),
        "decision_threshold": metrics.get("decision_threshold"),
        "n_train": metrics.get("n_train"),
    })

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    return versioned_name


def get_current_model_file(model_dir: str, default_path: str) -> str:
    """
    Resolves which model file the serving API should load: the registry's
    "current" pointer if a registry exists, otherwise falls back to the
    plain default path (backward compatible with pre-registry setups).
    """
    registry_path = os.path.join(model_dir, "registry.json")
    if not os.path.exists(registry_path):
        return default_path
    with open(registry_path) as f:
        registry = json.load(f)
    current = registry.get("current")
    if not current:
        return default_path
    resolved = os.path.join(model_dir, current)
    return resolved if os.path.exists(resolved) else default_path
