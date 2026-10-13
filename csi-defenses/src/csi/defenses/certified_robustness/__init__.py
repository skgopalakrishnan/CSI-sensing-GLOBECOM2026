"""Certified robustness utilities (randomized smoothing)."""

from csi.defenses.certified_robustness.noise import add_amplitude_noise, add_gaussian_noise, add_noise
from csi.defenses.certified_robustness.randomized_smoothing import (
    CertificationResult,
    RandomizedSmoother,
    certify_dataset,
    certified_accuracy_at_radius,
    evaluate_smoothed_accuracy,
)

__all__ = [
    "add_amplitude_noise",
    "add_gaussian_noise",
    "add_noise",
    "CertificationResult",
    "RandomizedSmoother",
    "certify_dataset",
    "certified_accuracy_at_radius",
    "evaluate_smoothed_accuracy",
]
