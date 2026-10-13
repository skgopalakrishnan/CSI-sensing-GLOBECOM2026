"""Utilities for attack configuration and CSI correlation constraints."""

from .attack_utils import (
    _compute_budgets,
    _eps2_to_snr_db,
    _snr_db_to_eps2,
    build_label_mapping,
    canonical_project_name,
    compute_effective_eps2,
    enforce_har_hid_attack_policy,
    eps_abs_from_l2_ratio,
    eps_l2_from_snr_db,
    infer_dataset_family,
    normalize_attack_method_name,
    project_l2_ratio,
    resolve_eps2_or_raise,
    sample_l2_ratio_ball,
    snr_db_from_eps_l2,
    view_as_broadcast,
)
from .correlation_utils import (
    build_pdp_correlation_matrix,
    correlation_preservation_pipeline,
    estimate_tau_rms_from_deltafreq,
    process_csi_tensor_for_correlation,
)

__all__ = [
    "build_label_mapping",
    "build_pdp_correlation_matrix",
    "canonical_project_name",
    "compute_effective_eps2",
    "correlation_preservation_pipeline",
    "enforce_har_hid_attack_policy",
    "eps_abs_from_l2_ratio",
    "eps_l2_from_snr_db",
    "estimate_tau_rms_from_deltafreq",
    "infer_dataset_family",
    "normalize_attack_method_name",
    "process_csi_tensor_for_correlation",
    "project_l2_ratio",
    "resolve_eps2_or_raise",
    "sample_l2_ratio_ball",
    "snr_db_from_eps_l2",
    "view_as_broadcast",
    "_compute_budgets",
    "_eps2_to_snr_db",
    "_snr_db_to_eps2",
]
