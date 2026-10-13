import torch
import pytest


@pytest.mark.parametrize(
    "dataset, model_name, input_shape",
    [
        ("NTU-Fi_HAR", "ResNet18", (2, 3, 114, 500)),
        ("NTU-Fi_HAR", "LeNet", (2, 3, 114, 500)),
        ("NTU-Fi_HAR", "AEHeadNTU_TCN", (2, 3, 114, 500)),
        ("NTU-Fi_HAR", "AEHeadNTU_GRU", (2, 3, 114, 500)),
    ],
)
def test_build_model_and_forward(dataset, model_name, input_shape):
    from csi.models.model_factory import build_model

    model, _ = build_model(dataset, model_name)
    model.eval()

    x = torch.randn(*input_shape)
    with torch.no_grad():
        y = model(x)
    assert y.ndim == 2  # (N, C)
    assert y.shape[0] == input_shape[0]


def test_build_plain_autoencoder_and_forward():
    from csi.models.model_factory import build_model

    model, _ = build_model("NTU-Fi_HAR", "tiny_f_stride")
    model.eval()

    x = torch.randn(2, 3, 114, 500)
    with torch.no_grad():
        reconstruction, latent = model(x)
    assert reconstruction.shape == x.shape
    assert latent.ndim == 4


def test_build_csi_bench_supervised_with_metadata():
    from csi.models.model_factory import build_model

    model, _ = build_model(
        "csi-bench",
        "LeNet",
        arch_metadata={
            "num_classes": 5,
            "task_name": "HumanActivityRecognition",
            "input_shape": (1, 232, 500),
        },
    )
    model.eval()

    x = torch.randn(2, 1, 232, 500)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 5)


def test_get_weights_path_roundtrip(tmp_path):
    from csi.models.utils.model_utils import get_weights_path

    weights_dir = tmp_path / "NTU-Fi_HAR" / "weights"
    weights_dir.mkdir(parents=True)
    expected = weights_dir / "NTU-Fi_HAR_ResNet18.pt"
    expected.touch()

    path = get_weights_path("NTU-Fi_HAR", "ResNet18", str(tmp_path), "weights")
    assert path == str(expected)
