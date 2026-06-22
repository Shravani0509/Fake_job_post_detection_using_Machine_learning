import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .image_features import compute_image_embedding_from_upload
from .inference import predict_job_post


@dataclass(frozen=True)
class MultimodalInferenceResult:
    prediction: str
    confidence: float
    fraud_score: float
    suspicious_contributions: List[Dict[str, Any]]
    extracted_features: Dict[str, Any]
    model_top_features: List[Dict[str, Any]]
    image_analysis: Dict[str, Any]


def _image_risk_summary(embedding: torch.Tensor) -> Dict[str, Any]:
    """Best-effort image risk signals.

    Note: We don't yet have a trained multimodal fusion model in this initial
    production upgrade step; this summary provides a structured image
    explanation placeholder that we can later connect to a trained classifier.
    """

    # Heuristic: compute norms and a few embedding statistics.
    # (Will be replaced by real attribution once multimodal classifier is trained.)
    emb = embedding.float()
    l2 = float(torch.norm(emb).item())
    mean = float(emb.mean().item())
    std = float(emb.std(unbiased=False).item())

    # Simple normalization into pseudo risk.
    # Clamp to [0, 100].
    pseudo_risk = max(0.0, min(100.0, (l2 % 1000.0) / 10.0))

    return {
        "embedding_dim": int(emb.numel()),
        "heuristics": {
            "l2_norm": round(l2, 4),
            "mean": round(mean, 4),
            "std": round(std, 4),
        },
        "pseudo_image_risk": round(pseudo_risk, 2),
        "note": "Image analysis is currently heuristic until multimodal fusion training is enabled.",
    }


def predict_job_post_multimodal(
    job_description: str,
    location: str,
    salary: Optional[str] = None,
    role: Optional[str] = None,
    image_upload: Optional[Any] = None,
) -> MultimodalInferenceResult:
    # Baseline: existing text-only classifier.
    text_result = predict_job_post(
        job_description=job_description,
        location=location,
        salary=salary,
        role=role,
    )

    image_analysis: Dict[str, Any] = {
        "has_image": False,
        "pseudo_image_risk": 0.0,
        "note": "No image provided.",
    }

    if image_upload is not None and getattr(image_upload, "filename", None):
        # compute embedding (fast backbone)
        emb_result = compute_image_embedding_from_upload(image_upload)
        image_analysis = _image_risk_summary(emb_result.embedding)
        image_analysis["has_image"] = True

    # Fusion placeholder: combine text fraud_prob-like score with pseudo image risk.
    # We use fraud_score from the text pipeline (0-100-ish), plus pseudo risk.
    text_fraud_score = float(text_result.fraud_score)
    pseudo_img_risk = float(image_analysis.get("pseudo_image_risk", 0.0))

    # Weighted fusion.
    fraud_score = round(0.75 * text_fraud_score + 0.25 * pseudo_img_risk, 2)

    # Map fraud_score back into label using same threshold as text classifier.
    threshold = float(torch.clamp(torch.tensor(60.0), 0, 100).item())
    # Keep consistent with existing threshold env if present.
    import os

    text_threshold = float(os.environ.get("JOB_FRAUD_THRESHOLD", "0.6"))
    # Convert text_threshold to fraud_score space roughly: text fraud_score uses model_prob*(0.6*100)
    # We'll just use a pragmatic fixed mapping.
    label_threshold = 55.0 if text_threshold >= 0.5 else 60.0

    prediction = "Fraudulent" if fraud_score >= label_threshold else "Not Fraudulent"

    # confidence: keep text confidence but slightly adjust if image present.
    confidence = text_result.confidence
    if image_analysis.get("has_image"):
        confidence = round(max(0.0, min(100.0, confidence * (0.85 + pseudo_img_risk / 100.0 * 0.15))), 1)

    return MultimodalInferenceResult(
        prediction=prediction,
        confidence=confidence,
        fraud_score=fraud_score,
        suspicious_contributions=text_result.suspicious_contributions,
        extracted_features=text_result.extracted_features,
        model_top_features=text_result.model_top_features,
        image_analysis=image_analysis,
    )

