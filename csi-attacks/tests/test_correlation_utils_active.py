import pytest
import torch

from csi.attacks.utils.correlation_utils import process_csi_tensor_for_correlation


@pytest.mark.parametrize(
    ("dataset_name", "shape"),
    [
        ("NTU-Fi_HAR", (1, 3, 114, 250)),
        ("NTU-Fi-HumanID", (1, 3, 114, 250)),
        ("HAR", (1, 1, 56, 250)),
        ("HumanActivityRecognition", (1, 1, 56, 250)),
        ("HID", (1, 1, 56, 250)),
        ("HumanIdentification", (1, 1, 56, 250)),
        ("CSI-Bench", (1, 1, 56, 250)),
    ],
)
def test_process_correlation_accepts_active_cft_profiles(dataset_name, shape):
    x = torch.randn(*shape)
    clean = torch.randn_like(x)
    y = process_csi_tensor_for_correlation(x, dataset_name, clean_tensor=clean)
    assert y.shape == x.shape


def test_process_correlation_rejects_wrong_subcarriers():
    x = torch.randn(1, 1, 55, 250)
    clean = torch.randn_like(x)
    with pytest.raises(ValueError, match="expected subcarriers"):
        process_csi_tensor_for_correlation(x, "HumanActivityRecognition", clean_tensor=clean)


def test_process_correlation_rejects_wrong_timesteps():
    x = torch.randn(1, 1, 56, 249)
    clean = torch.randn_like(x)
    with pytest.raises(ValueError, match="expected timesteps"):
        process_csi_tensor_for_correlation(x, "HumanIdentification", clean_tensor=clean)

def test_process_correlation_single_antenna_not_just_scalar_rescale():
    x = torch.randn(1, 1, 56, 250)
    clean = torch.randn_like(x)
    y = process_csi_tensor_for_correlation(x, "HumanActivityRecognition", clean_tensor=clean)

    x_flat = x.reshape(1, -1)
    y_flat = y.reshape(1, -1)
    scale = (x_flat * y_flat).sum(dim=1) / x_flat.square().sum(dim=1).clamp_min(1e-12)
    residual = (y_flat - scale[:, None] * x_flat).norm(dim=1) / y_flat.norm(dim=1).clamp_min(1e-12)

    assert float(residual.item()) > 1e-4
