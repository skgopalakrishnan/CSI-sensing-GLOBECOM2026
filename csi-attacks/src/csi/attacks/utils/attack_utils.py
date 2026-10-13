from math import log10
from typing import Dict, List, Tuple, Optional
import torch


def eps_l2_from_snr_db(snr_db: float) -> float:
    """L2-budget ratio ε such that ||n||₂ ≤ ε·||x||₂ for a target SNR (dB)."""
    return float(10 ** (-snr_db / 20.0))

def snr_db_from_eps_l2(eps: float) -> float:
    """Convert an L2-budget ratio ε back to SNR in dB."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    return float(-20.0 * log10(eps))

def compute_effective_eps2(eps_2: Optional[float], target_snr_db: Optional[float]) -> Optional[float]:
    """Return explicit eps_2 if provided; otherwise derive from target_snr_db if possible."""
    if eps_2 is not None:
        return float(eps_2)
    if target_snr_db is None or eps_l2_from_snr_db is None:  # Check if target SNR is provided
        return None
    try:  # Reaches if eps_2 is None and target_snr_db is provided
        return float(eps_l2_from_snr_db(float(target_snr_db)))
    except Exception:
        return None


def resolve_eps2_or_raise(
    eps_2: Optional[float],
    target_snr_db: Optional[float],
    *,
    context: str = "attack",
) -> float:
    """Resolve eps_2 from explicit value or target_snr_db and validate positivity."""
    value = compute_effective_eps2(eps_2, target_snr_db)
    if value is None:
        raise ValueError(f"{context}: provide either eps_2 or target_snr_db.")
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{context}: eps_2 must be positive, got {value}.")
    return value


def view_as_broadcast(vec: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a (B,) vector to broadcast over `like` with leading batch dim."""
    return vec.reshape(vec.size(0), *([1] * (like.dim() - 1)))


def eps_abs_from_l2_ratio(x: torch.Tensor, eps_2: float) -> torch.Tensor:
    """Per-sample absolute budget eps_abs[b] = eps_2 * ||x_b||_2."""
    b = x.size(0)
    return float(eps_2) * torch.norm(x.reshape(b, -1), p=2, dim=1)


def project_l2_ratio(
    delta: torch.Tensor,
    x_original: torch.Tensor,
    *,
    eps_2: float,
    eps_for_division: float = 1e-10,
) -> torch.Tensor:
    """Project perturbations onto per-sample L2-ratio balls."""
    batch = delta.size(0)
    l2_norms = torch.norm(delta.reshape(batch, -1), p=2, dim=1)
    l2_budgets = float(eps_2) * torch.norm(x_original.reshape(batch, -1), p=2, dim=1)
    scaling = torch.min(l2_budgets / (l2_norms + float(eps_for_division)), torch.ones_like(l2_norms))
    scaling = view_as_broadcast(scaling, delta)
    return delta * scaling


def sample_l2_ratio_ball(
    x_original: torch.Tensor,
    *,
    eps_2: float,
    eps_for_division: float = 1e-10,
) -> torch.Tensor:
    """Sample random perturbations uniformly inside per-sample L2-ratio balls."""
    batch = x_original.size(0)
    g = torch.randn_like(x_original)
    g_flat = g.reshape(batch, -1)
    g_norm = g_flat.norm(p=2, dim=1, keepdim=True).clamp_min(float(eps_for_division))
    direction = (g_flat / g_norm).reshape_as(x_original)

    dim = g_flat.size(1)
    u = torch.rand(batch, device=x_original.device).pow(1.0 / max(dim, 1))
    radius = view_as_broadcast(eps_abs_from_l2_ratio(x_original, eps_2) * u, x_original)
    delta = direction * radius
    return project_l2_ratio(
        delta,
        x_original,
        eps_2=eps_2,
        eps_for_division=eps_for_division,
    )

def _compute_budgets(attack_cfg: dict) -> List[Tuple[str, float]]:
    """Resolve budgets to a list of (label, eps_2) tuples.

    Supports either explicit `eps_2`, `snr_dB` (list or scalar), or `target_snr_db`.
    """
    # Normalize SNR configuration: accept list or single value; fall back to target_snr_db
    raw_snr = attack_cfg.get('snr_dB')
    if raw_snr is None:
        snr_list: List[float] = []
    elif isinstance(raw_snr, (list, tuple)):
        snr_list = [float(s) for s in raw_snr if s is not None]
    else:
        snr_list = [float(raw_snr)]
    if not snr_list and attack_cfg.get('target_snr_db') is not None:
        target_snr = attack_cfg.get('target_snr_db')
        if target_snr is not None:
            snr_list = [float(target_snr)]

    eps_2_cfg = attack_cfg.get('eps_2')

    budgets: List[Tuple[str, float]] = []
    if eps_2_cfg is not None:
        budgets.append((f"eps2={float(eps_2_cfg):.4f}", float(eps_2_cfg)))
    if snr_list:
        for s in snr_list:
            eff = compute_effective_eps2(None, s)
            if eff is not None:
                budgets.append((f"SNR={int(s)}dB", eff))
    if not budgets:  # Reaches if neither eps_2 nor snr_dB/target_snr_db provided
        # Default to a moderate 10 dB budget if nothing provided
        default_eps2 = compute_effective_eps2(None, 10)
        if default_eps2 is not None:
            budgets.append(("SNR=10dB", default_eps2))
    return budgets

# Backwards-compatibility aliases used in some modules
# Map SNR (dB) -> eps_2 ratio and inverse.
_snr_db_to_eps2 = eps_l2_from_snr_db
_eps2_to_snr_db = snr_db_from_eps_l2


def _mean_l2_norm(loader, device: torch.device, max_batches: Optional[int] = 50) -> float:
    total = 0.0
    count = 0
    with torch.no_grad():
        for i, (xb, _) in enumerate(loader):
            xb = xb.to(device)
            b = xb.size(0)
            total += torch.linalg.norm(xb.reshape(b, -1), ord=2, dim=1).sum().item()
            count += b
            if max_batches is not None and i + 1 >= int(max_batches):
                break
    return (total / max(count, 1)) if count > 0 else 1.0


def _project_l2(delta: torch.Tensor, radius: float) -> torch.Tensor:
    # Project onto L2 ball of radius
    flat = delta.reshape(1, -1)
    n = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(radius / n, max=1.0)
    return (flat * scale).reshape_as(delta)


def canonical_project_name(project: str) -> str:
    """Canonicalize dataset/task names used by attack/eval configs."""
    project_raw = str(project).strip()
    key = project_raw.lower().replace(" ", "").replace("_", "").replace("-", "")
    aliases = {
        "har": "HumanActivityRecognition",
        "humanactivityrecognition": "HumanActivityRecognition",
        "hid": "HumanIdentification",
        "humanidentification": "HumanIdentification",
        "ntufihar": "NTU-Fi_HAR",
        "ntufihumanid": "NTU-Fi-HumanID",
        "csibench": "CSI-Bench",
    }
    return aliases.get(key, project_raw)


def infer_dataset_family(project: str) -> str:
    """Return coarse family in {'har', 'hid', 'other'} for policy checks."""
    canonical = canonical_project_name(project)
    if canonical in {"NTU-Fi_HAR", "HumanActivityRecognition"}:
        return "har"
    if canonical in {"NTU-Fi-HumanID", "HumanIdentification"}:
        return "hid"
    return "other"


def normalize_attack_method_name(attack_method: str) -> str:
    method_raw = str(attack_method).strip().lower()
    method_aliases = {
        "pgdl2mmd_corr": "pgdl2mmdcorr",
        "pgd_l2": "pgdl2",
    }
    return method_aliases.get(method_raw, method_raw)


def enforce_har_hid_attack_policy(
    project: str,
    attack_type: str,
    attack_method: str,
    *,
    strict: bool = True,
) -> Tuple[str, str]:
    """Enforce current policy: HAR->untargeted, HID->targeted APGD/APGDT."""
    attack_mode = str(attack_type).strip().lower()
    method_norm = normalize_attack_method_name(attack_method)
    if not strict:
        return attack_mode, method_norm

    family = infer_dataset_family(project)
    if family == "har" and attack_mode != "untargeted":
        raise ValueError(
            f"{project}: HAR experiments must use attack.type='untargeted' (got {attack_type!r})."
        )
    if family == "hid":
        if attack_mode != "targeted":
            raise ValueError(
                f"{project}: HID experiments must use attack.type='targeted' (got {attack_type!r})."
            )
        if method_norm not in {"apgd", "apgdt"}:
            raise ValueError(
                f"{project}: HID targeted experiments must use attack.method in {{'apgd','apgdt'}} "
                f"(got {attack_method!r})."
            )
    return attack_mode, method_norm


# Optional dataset -> num_classes map to support simple label mapping utilities here.
# Keep in sync with experiment configs; safe defaults if unknown dataset is passed.
CLASSES: Dict[str, int] = {
    "NTU-Fi-HumanID": 14,
    "NTU-Fi_HAR": 6,
}


def build_label_mapping(
    project: str,
    attack_type: str,
    *,
    num_classes: Optional[int] = None,
) -> Tuple[Dict[int, List[int]], bool]:
    """Build a class->target(s) label mapping and targeted flag.

    Modes:
    - `untargeted`: identity mapping (ignored by untargeted attacks)
    - `targeted`: deterministic single-target mapping i -> (i+1) mod n
    """
    project_canonical = canonical_project_name(project)
    n = int(num_classes) if num_classes is not None else CLASSES.get(project_canonical)
    if n is None:
        raise ValueError(
            f"Unknown project '{project}'. Known: {sorted(CLASSES.keys())}. "
            "Pass num_classes explicitly for task-specific datasets."
        )
    if int(n) < 2:
        raise ValueError(f"num_classes must be >=2, got {n}.")
    attack_mode = str(attack_type).strip().lower()
    if attack_mode == "untargeted":
        mapping = {i: [i] for i in range(n)}
        targeted = False
        print("  Mode: Untargeted attack")
    elif attack_mode == "targeted":
        mapping = {i: [((i + 1) % n)] for i in range(n)}
        targeted = True
        print("  Mode: Targeted attack")
    else:
        raise ValueError(
            f"Unsupported attack.type={attack_type!r}. Use 'untargeted' or 'targeted'."
        )
    return mapping, targeted
