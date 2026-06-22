import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
from scipy.sparse import hstack

from .keyword_rules import DEFAULT_RULES, build_keyword_contributions
from .text_extract import (
    extract_company_name,
    extract_job_description_preview,
    extract_location,
    extract_requirements,
    extract_salary,
)


MODEL_ARTIFACT_PATH = os.environ.get(
    "JOB_FRAUD_MODEL_PATH", "trained_models/random_forest_job_fraud.joblib"
)


@dataclass(frozen=True)
class InferenceResult:
    prediction: str
    confidence: float
    fraud_score: float
    suspicious_contributions: List[Dict[str, Any]]
    extracted_features: Dict[str, Any]
    model_top_features: List[Dict[str, Any]]


def _compute_confidence_from_proba(proba: Any, fraud_index: int = 1) -> float:
    # proba shape: (1, n_classes)
    try:
        probs = proba[0]
        return round(float(max(probs)) * 100.0, 1)
    except Exception:
        return 0.0


def _safe_location_encode(location_encoder: Any, location: str) -> Tuple[Any, Optional[List[str]]]:
    """Try to encode location; if unknown, return [-1] and suggestions."""
    common_indian_locations = [
        "Mumbai",
        "Delhi",
        "Bangalore",
        "Hyderabad",
        "Ahmedabad",
        "Chennai",
        "Kolkata",
        "Pune",
        "Jaipur",
        "Surat",
        "Lucknow",
        "Kanpur",
        "Nagpur",
        "Visakhapatnam",
        "Patna",
    ]

    if location is None:
        return -1, None

    try:
        encoded = location_encoder.transform([location])
        return encoded.reshape(1, -1), None
    except Exception:
        suggestions = [loc for loc in common_indian_locations if location.lower() in loc.lower()]
        return -1, suggestions


def _estimate_model_top_features(
    job_description: str,
    text_transformer: Any,
    rf_model: Any,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Best-effort explanation.

    RandomForest doesn't provide coefficients. We approximate by:
    - get top TF-IDF terms in the input
    - weight them using feature_importances_ if available

    Returns list of {feature, importance, tfidf_value}
    """
    if not hasattr(text_transformer, "get_feature_names_out"):
        return []

    vocab = text_transformer.get_feature_names_out()
    tfidf_vec = text_transformer.transform([job_description])

    # tfidf_vec is sparse (1, n_features)
    try:
        coo = tfidf_vec.tocoo()
        pairs = list(zip(coo.col.tolist(), coo.data.tolist()))
    except Exception:
        return []

    # Map col -> tfidf
    tfidf_by_col = {col: val for col, val in pairs}

    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
    else:
        importances = None

    scored: List[Dict[str, Any]] = []
    for col, tfidf_val in tfidf_by_col.items():
        if col < len(vocab):
            feature_name = str(vocab[col])
        else:
            continue

        imp = float(importances[col]) if importances is not None and col < len(importances) else 0.0
        scored.append({"feature": feature_name, "importance": imp, "tfidf_value": float(tfidf_val)})

    scored.sort(key=lambda d: (d["importance"] * 0.6 + d["tfidf_value"] * 0.4), reverse=True)
    return scored[:top_k]


def predict_job_post(
    job_description: str,
    location: str,
    salary: Optional[str] = None,
    role: Optional[str] = None,
) -> InferenceResult:
    if not job_description or not job_description.strip():
        raise ValueError("job_description is required")

    salary_text = extract_salary(salary or "")

    # Load model artifacts lazily
    if not os.path.exists(MODEL_ARTIFACT_PATH):
        raise FileNotFoundError(
            f"Model artifact not found at: {MODEL_ARTIFACT_PATH}. "
            "Run training first or set JOB_FRAUD_MODEL_PATH env var."
        )

    model_data = joblib.load(MODEL_ARTIFACT_PATH)
    rf_model = model_data["model"]
    text_transformer = model_data["text_transformer"]
    location_encoder = model_data["location_encoder"]

    # Location encoding
    location_encoded, suggestions = _safe_location_encode(location_encoder, location)

    # Create combined features
    text_features = text_transformer.transform([job_description])
    if isinstance(location_encoded, int):
        # unknown => build correct shape for hstack
        import numpy as np

        loc_arr = np.array([location_encoded]).reshape(1, -1)
        features = hstack([text_features, loc_arr])
    else:
        features = hstack([text_features, location_encoded])

    proba = rf_model.predict_proba(features)

    # By convention, dataset label 1 is fraudulent (as in model.py)
    # proba index [0] -> class0, [1] -> class1
    fraud_index = 1 if len(proba[0]) > 1 else 0
    fraud_prob = float(proba[0][fraud_index])

    threshold = float(os.environ.get("JOB_FRAUD_THRESHOLD", "0.6"))
    prediction = "Fraudulent" if fraud_prob > threshold else "Not Fraudulent"

    confidence = _compute_confidence_from_proba(proba, fraud_index=fraud_index)

    # Suspicious keywords contribution score (rules-based)
    keyword_score, triggered_rules = build_keyword_contributions(job_description, DEFAULT_RULES)
    # Combine with model probability lightly into fraud_score for dashboard display
    fraud_score = round(0.6 * fraud_prob * 100 + 0.4 * min(keyword_score, 20) * 5, 2)

    suspicious_contributions = triggered_rules

    extracted_features = {
        "company_name": extract_company_name(job_description),
        "location": extract_location(job_description, fallback_location=location),
        "salary": salary_text,
        "role": role,
        "requirements": extract_requirements(job_description),
        "job_description_preview": extract_job_description_preview(job_description),
        "location_suggestions": suggestions,
    }

    model_top_features = _estimate_model_top_features(job_description, text_transformer, rf_model)

    return InferenceResult(
        prediction=prediction,
        confidence=confidence,
        fraud_score=fraud_score,
        suspicious_contributions=suspicious_contributions,
        extracted_features=extracted_features,
        model_top_features=model_top_features,
    )

