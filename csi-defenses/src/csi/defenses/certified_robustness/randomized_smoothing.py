from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from csi.defenses.certified_robustness.noise import add_noise

try:
    from csi.models.utils.training import _maybe_tqdm
except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional dependency fallback
    try:
        from csi.models.core import _maybe_tqdm
    except (ModuleNotFoundError, ImportError):  # pragma: no cover - last-resort local shim
        def _maybe_tqdm(loader, use_tqdm, desc=None):
            _ = (use_tqdm, desc)
            return loader


@dataclass
class CertificationResult:
    prediction: int
    radius: float
    is_abstain: bool


class RandomizedSmoother:
    """Cohen-style randomized smoothing classifier for certification in L2."""

    ABSTAIN = -1

    def __init__(
        self,
        base_classifier: torch.nn.Module,
        num_classes: int,
        sigma: float,
        *,
        device: Optional[torch.device] = None,
        noise_type: str = "gaussian",
        amplitude_clamp_min: Optional[float] = 0.0,
    ) -> None:
        if int(num_classes) <= 1:
            raise ValueError("num_classes must be >= 2.")
        if float(sigma) <= 0.0:
            raise ValueError("sigma must be > 0.")

        self.base_classifier = base_classifier
        self.num_classes = int(num_classes)
        self.sigma = float(sigma)
        if device is None:
            try:
                self.device = next(base_classifier.parameters()).device
            except Exception:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.noise_type = str(noise_type).strip().lower()
        self.amplitude_clamp_min = None if amplitude_clamp_min is None else float(amplitude_clamp_min)

    def certify(
        self,
        x: torch.Tensor,
        n0: int,
        n: int,
        alpha: float,
        batch_size: int,
    ) -> Tuple[int, float]:
        """Certify a single input sample.

        Returns `(prediction, radius)` with `prediction == ABSTAIN` and `radius==0.0`
        when certification abstains.
        """
        if int(n0) <= 0:
            raise ValueError("n0 must be > 0.")
        if int(n) <= 0:
            raise ValueError("n must be > 0.")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0.")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1).")

        self.base_classifier.eval()
        x_batched = self._as_single_batched_input(x)

        counts_selection = self._sample_noise(x_batched, int(n0), int(batch_size))
        class_hat = int(torch.argmax(counts_selection).item())

        counts_estimation = self._sample_noise(x_batched, int(n), int(batch_size))
        n_a = int(counts_estimation[class_hat].item())
        p_a_bar = self._lower_confidence_bound(n_a, int(n), float(alpha))

        if p_a_bar < 0.5:
            return self.ABSTAIN, 0.0

        p_a_bar = float(max(min(p_a_bar, 1.0 - 1e-12), 1e-12))
        normal = torch.distributions.Normal(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )
        radius = float(self.sigma * normal.icdf(torch.tensor(p_a_bar, dtype=torch.float64)).item())
        if not torch.isfinite(torch.tensor(radius)):
            return self.ABSTAIN, 0.0
        return class_hat, max(0.0, radius)

    def predict(
        self,
        x: torch.Tensor,
        n: int,
        alpha: float,
        batch_size: int,
    ) -> int:
        """Predict the class of a single sample with abstention."""
        if int(n) <= 0:
            raise ValueError("n must be > 0.")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0.")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1).")

        self.base_classifier.eval()
        x_batched = self._as_single_batched_input(x)
        counts = self._sample_noise(x_batched, int(n), int(batch_size))
        top2 = torch.argsort(counts, descending=True)[:2]
        count1 = int(counts[top2[0]].item())
        count2 = int(counts[top2[1]].item())
        p_value = self._binom_test_pvalue(count1, count1 + count2)
        if p_value > float(alpha):
            return self.ABSTAIN
        return int(top2[0].item())

    def certify_single(
        self,
        x: torch.Tensor,
        *,
        n0: int = 100,
        n: int = 10000,
        alpha: float = 0.001,
        batch_size: int = 512,
    ) -> CertificationResult:
        pred, radius = self.certify(x=x, n0=n0, n=n, alpha=alpha, batch_size=batch_size)
        return CertificationResult(
            prediction=int(pred),
            radius=float(radius),
            is_abstain=bool(int(pred) == self.ABSTAIN),
        )

    def _as_single_batched_input(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim == 0:
            raise ValueError("x must have at least 1 dimension.")
        # Standard Cohen API expects one sample without batch; make batch size one.
        return x.to(self.device, non_blocking=True).unsqueeze(0)

    def _sample_noise(self, x_batched: torch.Tensor, num: int, batch_size: int) -> torch.Tensor:
        with torch.no_grad():
            counts = torch.zeros(self.num_classes, device="cpu", dtype=torch.long)
            remaining = int(num)
            while remaining > 0:
                this_batch = min(batch_size, remaining)
                remaining -= this_batch
                batch = x_batched.expand(this_batch, *x_batched.shape[1:])
                noisy_batch = add_noise(
                    batch,
                    self.sigma,
                    noise_type=self.noise_type,
                    amplitude_clamp_min=self.amplitude_clamp_min,
                )
                logits = self.base_classifier(noisy_batch)
                preds = logits.argmax(dim=1).to("cpu")
                counts = counts + torch.bincount(preds, minlength=self.num_classes)
            return counts

    @staticmethod
    def _lower_confidence_bound(num_successes: int, num_trials: int, alpha: float) -> float:
        if num_trials <= 0:
            raise ValueError("num_trials must be > 0.")
        if num_successes <= 0:
            return 0.0
        if num_successes >= num_trials:
            return 1.0

        a = float(num_successes)
        b = float(num_trials - num_successes + 1)
        q = float(alpha)
        try:
            beta = torch.distributions.Beta(
                concentration1=torch.tensor(a, dtype=torch.float64),
                concentration0=torch.tensor(b, dtype=torch.float64),
            )
            val = float(beta.icdf(torch.tensor(q, dtype=torch.float64)).item())
            if 0.0 <= val <= 1.0:
                return val
        except Exception:
            pass

        # Wilson lower bound fallback (conservative approximation when Beta icdf is unavailable).
        n = float(num_trials)
        phat = float(num_successes) / n
        normal = torch.distributions.Normal(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )
        z = float(normal.icdf(torch.tensor(1.0 - q, dtype=torch.float64)).item())
        denom = 1.0 + (z * z / n)
        center = (phat + (z * z) / (2.0 * n)) / denom
        margin = (z / denom) * ((phat * (1.0 - phat) / n + (z * z) / (4.0 * n * n)) ** 0.5)
        return max(0.0, min(1.0, center - margin))

    @staticmethod
    def _binom_test_pvalue(k: int, n: int) -> float:
        if n <= 0:
            return 1.0
        # Use exact SciPy binomial test when available.
        try:
            from scipy.stats import binomtest  # type: ignore

            return float(binomtest(k=int(k), n=int(n), p=0.5, alternative="two-sided").pvalue)
        except Exception:
            pass

        # Normal approximation fallback.
        phat = float(k) / float(n)
        z = abs(phat - 0.5) / ((0.25 / float(n)) ** 0.5)
        normal = torch.distributions.Normal(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )
        one_side = 1.0 - float(normal.cdf(torch.tensor(z, dtype=torch.float64)).item())
        return max(0.0, min(1.0, 2.0 * one_side))


def certify_dataset(
    smoother: RandomizedSmoother,
    loader,
    *,
    n0: int = 100,
    n: int = 10000,
    alpha: float = 0.001,
    batch_size: int = 512,
    skip: int = 1,
    max_examples: Optional[int] = None,
    use_tqdm: bool = False,
) -> Dict[str, Any]:
    """Run Cohen-style certification over a loader and return per-example rows + summary."""
    if int(skip) <= 0:
        raise ValueError("skip must be >= 1.")

    rows: List[Dict[str, Any]] = []
    total_seen = 0
    total_used = 0
    correct = 0
    abstained = 0

    iterator = _maybe_tqdm(loader, use_tqdm, desc="Randomized smoothing certification")
    for batch in iterator:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise RuntimeError("Expected loader to yield (inputs, labels) tuples.")
        inputs, labels = batch[0], batch[1]
        if not torch.is_tensor(inputs) or not torch.is_tensor(labels):
            raise RuntimeError("Expected tensor inputs and labels.")

        batch_size_local = int(inputs.size(0))
        for i in range(batch_size_local):
            if (total_seen % int(skip)) != 0:
                total_seen += 1
                continue
            if max_examples is not None and total_used >= int(max_examples):
                break

            x_i = inputs[i]
            y_i = int(labels[i].item())
            t0 = perf_counter()
            pred, radius = smoother.certify(
                x_i,
                n0=int(n0),
                n=int(n),
                alpha=float(alpha),
                batch_size=int(batch_size),
            )
            elapsed = perf_counter() - t0
            is_abstain = int(pred) == smoother.ABSTAIN
            is_correct = (int(pred) == y_i) and (not is_abstain)
            correct += int(is_correct)
            abstained += int(is_abstain)
            total_used += 1

            rows.append(
                {
                    "idx": int(total_seen),
                    "label": y_i,
                    "prediction": int(pred),
                    "radius": float(radius),
                    "correct": int(is_correct),
                    "abstain": int(is_abstain),
                    "time_sec": float(elapsed),
                }
            )
            total_seen += 1

        if max_examples is not None and total_used >= int(max_examples):
            break

    accuracy = float(correct) / float(total_used) if total_used > 0 else 0.0
    abstain_rate = float(abstained) / float(total_used) if total_used > 0 else 0.0
    summary = {
        "n_certified": int(total_used),
        "accuracy": float(accuracy),
        "abstain_rate": float(abstain_rate),
        "mean_radius": float(sum(row["radius"] for row in rows) / total_used) if total_used > 0 else 0.0,
    }
    return {"rows": rows, "summary": summary}


def certified_accuracy_at_radius(rows: Sequence[Dict[str, Any]], radius: float) -> float:
    if len(rows) == 0:
        return 0.0
    threshold = float(radius)
    count = 0
    for row in rows:
        if int(row.get("correct", 0)) == 1 and float(row.get("radius", 0.0)) >= threshold:
            count += 1
    return float(count) / float(len(rows))


def evaluate_smoothed_accuracy(
    smoother: RandomizedSmoother,
    loader,
    *,
    n: int = 1000,
    alpha: float = 0.001,
    batch_size: int = 512,
    skip: int = 1,
    max_examples: Optional[int] = None,
    use_tqdm: bool = False,
) -> Dict[str, Any]:
    """Evaluate predictive accuracy of the smoothed classifier without radius certification."""
    if int(skip) <= 0:
        raise ValueError("skip must be >= 1.")

    total_seen = 0
    total_used = 0
    correct = 0
    abstained = 0

    iterator = _maybe_tqdm(loader, use_tqdm, desc="Randomized smoothing prediction eval")
    for batch in iterator:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise RuntimeError("Expected loader to yield (inputs, labels) tuples.")
        inputs, labels = batch[0], batch[1]
        if not torch.is_tensor(inputs) or not torch.is_tensor(labels):
            raise RuntimeError("Expected tensor inputs and labels.")

        for i in range(int(inputs.size(0))):
            if (total_seen % int(skip)) != 0:
                total_seen += 1
                continue
            if max_examples is not None and total_used >= int(max_examples):
                break

            pred = smoother.predict(
                inputs[i],
                n=int(n),
                alpha=float(alpha),
                batch_size=int(batch_size),
            )
            label = int(labels[i].item())
            is_abstain = int(pred) == smoother.ABSTAIN
            if (not is_abstain) and (int(pred) == label):
                correct += 1
            abstained += int(is_abstain)
            total_used += 1
            total_seen += 1

        if max_examples is not None and total_used >= int(max_examples):
            break

    return {
        "n_eval": int(total_used),
        "accuracy": float(correct) / float(total_used) if total_used > 0 else 0.0,
        "abstain_rate": float(abstained) / float(total_used) if total_used > 0 else 0.0,
    }
