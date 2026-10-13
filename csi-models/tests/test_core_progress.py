import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _make_toy_loader(n=8, c=1, h=8, w=8, num_classes=3, batch=4):
    x = torch.randn(n, c, h, w)
    y = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch, shuffle=False)


def _make_toy_model(num_classes=3):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(1 * 8 * 8, 16),
        nn.ReLU(),
        nn.Linear(16, num_classes),
    )


class _ToyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.decoder = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        latent = torch.relu(self.encoder(x))
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def _make_toy_autoencoder_loader(n=8, c=1, h=8, w=8, batch=4):
    x = torch.randn(n, c, h, w)
    ds = TensorDataset(x)
    return DataLoader(ds, batch_size=batch, shuffle=False)


def test_test_progress_bar_runs():
    from csi.models.core import test
    model = _make_toy_model(3)
    loader = _make_toy_loader()
    crit = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    acc, loss = test(model, loader, crit, device, use_tqdm=True)
    assert isinstance(acc, float)
    assert isinstance(loss, float)


def test_train_progress_bar_runs_one_epoch():
    from csi.models.core import train
    model = _make_toy_model(3)
    loader = _make_toy_loader()
    crit = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    train(model, loader, num_epochs=1, learning_rate=1e-3, criterion=crit, device=device, use_tqdm=True)


def test_train_autoencoder_progress_bar_runs_one_epoch():
    from csi.models.core import train_autoencoder

    model = _ToyAutoencoder()
    loader = _make_toy_autoencoder_loader()
    device = torch.device("cpu")
    summary = train_autoencoder(
        model,
        loader,
        num_epochs=1,
        learning_rate=1e-3,
        device=device,
        use_tqdm=True,
    )
    assert summary.best_epoch == 1
