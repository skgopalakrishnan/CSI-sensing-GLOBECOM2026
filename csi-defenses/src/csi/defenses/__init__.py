"""Defense methods for CSI models."""

from csi.defenses.adv_training.pgd_at import PGDATrainer
from csi.defenses.adv_training.randomized_smoothing import RandomizedSmoothingTrainer
from csi.defenses.adv_training.trades import TRADESTrainer
from csi.defenses.certified_robustness.noise import add_amplitude_noise, add_gaussian_noise, add_noise
from csi.defenses.certified_robustness.randomized_smoothing import (
    CertificationResult,
    RandomizedSmoother,
    certify_dataset,
    certified_accuracy_at_radius,
    evaluate_smoothed_accuracy,
)

__all__ = [
    "PGDATrainer",
    "TRADESTrainer",
    "RandomizedSmoothingTrainer",
    "add_amplitude_noise",
    "add_gaussian_noise",
    "add_noise",
    "CertificationResult",
    "RandomizedSmoother",
    "certify_dataset",
    "certified_accuracy_at_radius",
    "evaluate_smoothed_accuracy",
]
