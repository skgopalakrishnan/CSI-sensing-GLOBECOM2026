"""Classifier heads built on autoencoder latent representations."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import prune

from ..utils.datasets import DatasetSpec
from .models import TinyFStrideAutoencoder, TinyFStrideConfig

__all__ = [
    "AutoencoderClassifier",
    "TemporalTCNHead",
    "TemporalGRUHead",
    "TemporalSSMHead",
    "build_classifier_head",
    "build_autoencoder_classifier",
    "HEAD_ALIASES",
    # "apply_head_pruning",
    # "apply_dynamic_quantization",
]


class DepthwisePointwiseConv1d(nn.Module):
    """Depthwise + pointwise Conv1d block with normalisation and activation."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x)


class TemporalResidualBlock(nn.Module):
    """Residual temporal block with optional depthwise separable convolutions."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
        depthwise: bool,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        if depthwise:
            self.main = DepthwisePointwiseConv1d(
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
        else:
            self.main = nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(channels),
                nn.Dropout(dropout),
            )
        self.out_norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.main(x)
        out = self.out_norm(out)
        return self.act(out + residual)


class TemporalTCNHead(nn.Module):
    """Temporal head constructed from lightweight residual Conv1d blocks."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        project_dim: Optional[int] = None,
        dilations: Sequence[int] = (1, 2),
        kernel_size: int = 3,
        dropout: float = 0.1,
        final_dropout: float = 0.0,
        depthwise: bool = True,
        pre_pool_kernel: Optional[int] = None,
        pool_type: str = "avg",
        channel_reduction: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.project = None
        channels = in_channels
        proj = project_dim or 0
        if proj > 0 and proj != in_channels:
            self.project = nn.Sequential(
                nn.Conv1d(in_channels, proj, kernel_size=1, bias=False),
                nn.BatchNorm1d(proj),
                nn.GELU(),
            )
            channels = proj
        pool_kernel = int(pre_pool_kernel or 1)
        if pool_kernel > 1:
            pool_type = pool_type.lower()
            if pool_type == "avg":
                self.pre_pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=False)
            elif pool_type == "max":
                self.pre_pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel)
            else:
                raise ValueError(f"Unknown pool_type '{pool_type}'")
        else:
            self.pre_pool = nn.Identity()

        blocks = [
            TemporalResidualBlock(
                channels,
                kernel_size=kernel_size,
                dilation=int(d),
                dropout=dropout,
                depthwise=depthwise,
            )
            for d in dilations
        ]
        self.blocks = nn.Sequential(*blocks) if blocks else nn.Identity()

        reducers: list[nn.Module] = []
        if channel_reduction:
            prev = channels
            for idx, hidden in enumerate(channel_reduction):
                hidden = int(hidden)
                reducers.append(
                    nn.Sequential(
                        nn.Conv1d(prev, hidden, kernel_size=1, bias=False),
                        nn.BatchNorm1d(hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                )
                prev = hidden
            channels = int(channel_reduction[-1])
        self.channel_reducer = nn.Sequential(*reducers) if reducers else nn.Identity()

        self.final_dropout = nn.Dropout(final_dropout) if final_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, z: Tensor, *, return_features: bool = False):
        if z.dim() != 4:
            raise ValueError(f"Expected latent tensor with 4 dims, got shape {tuple(z.shape)}")
        x = z.mean(dim=2)  # [B, C, T]
        if self.project is not None:
            x = self.project(x)
        x = self.pre_pool(x)
        x = self.blocks(x)
        x = self.channel_reducer(x)
        features = x.mean(dim=-1)
        features = self.final_dropout(features)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class TemporalGRUHead(nn.Module):
    """GRU-based temporal head with optional early pooling and feature MLP."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        project_dim: Optional[int] = None,
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.05,
        pool: str = "last",
        attention_hidden: Optional[int] = None,
        final_dropout: float = 0.0,
        pre_pool_kernel: Optional[int] = None,
        feature_mlp: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.project = None
        channels = in_channels
        proj = project_dim or 0
        if proj > 0 and proj != in_channels:
            self.project = nn.Sequential(
                nn.Conv1d(in_channels, proj, kernel_size=1, bias=False),
                nn.BatchNorm1d(proj),
                nn.GELU(),
            )
            channels = proj

        pool_kernel = int(pre_pool_kernel or 1)
        if pool_kernel > 1:
            self.pre_pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        else:
            self.pre_pool = nn.Identity()

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1
        gru_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=gru_dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        pool_mode = str(pool).lower()
        if pool_mode not in {"last", "attention", "mean"}:
            raise ValueError(f"Unsupported pool mode '{pool}'")
        self.pool = pool_mode
        if pool_mode == "attention":
            attn_hidden = int(attention_hidden) if attention_hidden is not None else self.hidden_size
            self.attn = nn.Sequential(
                nn.Linear(self.hidden_size * self.num_directions, attn_hidden),
                nn.Tanh(),
                nn.Linear(attn_hidden, 1),
            )
        else:
            self.attn = None
        self.final_dropout = nn.Dropout(final_dropout) if final_dropout > 0 else nn.Identity()

        feature_layers: list[nn.Module] = []
        last_dim = self.hidden_size * self.num_directions
        if feature_mlp:
            for hidden in feature_mlp:
                hidden = int(hidden)
                feature_layers.extend(
                    [
                        nn.Linear(last_dim, hidden),
                        nn.GELU(),
                        nn.Dropout(final_dropout),
                    ]
                )
                last_dim = hidden
        self.feature_mlp = nn.Sequential(*feature_layers) if feature_layers else nn.Identity()
        self.classifier = nn.Linear(last_dim, num_classes)

    def forward(self, z: Tensor, *, return_features: bool = False):
        if z.dim() != 4:
            raise ValueError(f"Expected latent tensor with 4 dims, got shape {tuple(z.shape)}")
        x = z.mean(dim=2)
        if self.project is not None:
            x = self.project(x)
        x = self.pre_pool(x)
        x = x.transpose(1, 2)  # [B, T, C]
        outputs, hidden = self.gru(x)
        if self.pool == "last":
            if self.bidirectional:
                hidden = hidden.view(self.num_layers, self.num_directions, x.size(0), self.hidden_size)
                last = hidden[-1]
                features = torch.cat([last[0], last[1]], dim=-1)
            else:
                features = hidden[-1]
        elif self.pool == "mean":
            features = outputs.mean(dim=1)
        elif self.pool == "attention":
            if self.attn is None:
                raise ValueError("Attention pooling selected but attn module is not initialized.")
            energy = self.attn(outputs)  # [B, T, 1]
            weights = torch.softmax(energy, dim=1)
            features = (weights * outputs).sum(dim=1)
        else:
            raise ValueError(f"Unsupported pool mode '{self.pool}'")
        features = self.feature_mlp(features)
        features = self.final_dropout(features)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class LinearStateSpaceBlock(nn.Module):
    """Lightweight state-space update block with learnable transition."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        *,
        dropout: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.transition = nn.Linear(state_dim, state_dim, bias=False)
        self.input_proj = nn.Linear(input_dim, state_dim, bias=True)
        self.bias = nn.Parameter(torch.zeros(state_dim))
        self.dropout = nn.Dropout(dropout)
        activation = activation.lower()
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.GELU()

    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        # x: [B, C, T]
        B, _, T = x.shape
        if state is None:
            state = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            u_t = x[:, :, t]
            state = self.transition(state) + self.input_proj(u_t) + self.bias
            state = self.act(state)
            state = self.dropout(state)
            outputs.append(state.unsqueeze(-1))
        return torch.cat(outputs, dim=-1), state


class TemporalSSMHead(nn.Module):
    """State-space model head with compact state dimension."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        project_dim: Optional[int] = None,
        state_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.05,
        activation: str = "gelu",
        pre_pool_kernel: Optional[int] = None,
        aggregation: str = "last",
        feature_mlp: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.project = None
        channels = in_channels
        proj = project_dim or 0
        if proj > 0 and proj != in_channels:
            self.project = nn.Sequential(
                nn.Conv1d(in_channels, proj, kernel_size=1, bias=False),
                nn.BatchNorm1d(proj),
                nn.GELU(),
            )
            channels = proj
        pool_kernel = int(pre_pool_kernel or 1)
        if pool_kernel > 1:
            self.pre_pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        else:
            self.pre_pool = nn.Identity()

        layers: list[LinearStateSpaceBlock] = []
        current = channels
        state_dim = int(state_dim)
        for _ in range(int(num_layers)):
            block = LinearStateSpaceBlock(
                input_dim=current,
                state_dim=state_dim,
                dropout=dropout,
                activation=activation,
            )
            layers.append(block)
            current = state_dim
        self.layers = nn.ModuleList(layers)

        aggregation = aggregation.lower()
        if aggregation not in {"last", "mean"}:
            raise ValueError(f"Unsupported aggregation '{aggregation}'")
        self.aggregation = aggregation

        feature_layers: list[nn.Module] = []
        last_dim = state_dim
        if feature_mlp:
            for hidden in feature_mlp:
                hidden = int(hidden)
                feature_layers.extend(
                    [
                        nn.Linear(last_dim, hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
                last_dim = hidden
        self.feature_mlp = nn.Sequential(*feature_layers) if feature_layers else nn.Identity()
        self.classifier = nn.Linear(last_dim, num_classes)

    def forward(self, z: Tensor, *, return_features: bool = False):
        if z.dim() != 4:
            raise ValueError(f"Expected latent tensor with 4 dims, got shape {tuple(z.shape)}")
        x = z.mean(dim=2)
        if self.project is not None:
            x = self.project(x)
        x = self.pre_pool(x)
        state = None
        for layer in self.layers:
            x, state = layer(x, state)
        if self.aggregation == "mean":
            features = x.mean(dim=-1)
        else:
            features = state
        features = self.feature_mlp(features)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class AutoencoderClassifier(nn.Module):
    """Wrapper combining an autoencoder encoder with a temporal head."""

    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: Tensor, *, return_features: bool = False):
        latent = None
        encode_fn = getattr(self.encoder, "encode", None)
        if callable(encode_fn):
            latent = encode_fn(x)
        else:
            out = self.encoder(x)
            # encoder may return (reconstruction, latent) or just latent
            if isinstance(out, tuple) and len(out) >= 2:
                _, latent = out
            else:
                latent = out
        if isinstance(latent, tuple):
            if len(latent) == 0:
                raise ValueError("Encoder returned an empty tuple for latent; expected a tensor or a tuple containing the latent tensor.")
            latent = latent[0]
        if not isinstance(latent, Tensor):
            raise ValueError(f"Expected latent to be a Tensor, got {type(latent)}")
        if latent.dim() != 4:
            raise ValueError(f"Expected latent tensor with 4 dims, got shape {tuple(latent.shape)}")
        if return_features:
            return self.head(latent, return_features=True)
        return self.head(latent)


def build_classifier_head(
    *,
    head_type: str,
    in_channels: int,
    num_classes: int,
    project_dim: Optional[int] = None,
    dropout: float = 0.0,
    tcn_cfg: Optional[dict] = None,
    gru_cfg: Optional[dict] = None,
    ssm_cfg: Optional[dict] = None,
) -> nn.Module:
    """Instantiate a temporal head based on configuration."""

    key = head_type.lower()
    if key == "tcn":
        cfg = dict(tcn_cfg or {})
        dilations = cfg.get("dilations", (1, 2))
        if not isinstance(dilations, Sequence):
            dilations = (dilations,)
        dilations = tuple(int(d) for d in dilations)
        kernel_size = int(cfg.get("kernel_size", 3))
        block_dropout = float(cfg.get("dropout", dropout))
        final_dropout = float(cfg.get("final_dropout", dropout))
        depthwise = bool(cfg.get("depthwise", True))
        pre_pool_kernel = cfg.get("pre_pool_kernel")
        pool_type = cfg.get("pool_type", "avg")
        channel_reduction = cfg.get("channel_reduction")
        return TemporalTCNHead(
            in_channels,
            num_classes,
            project_dim=project_dim,
            dilations=dilations,
            kernel_size=kernel_size,
            dropout=block_dropout,
            final_dropout=final_dropout,
            depthwise=depthwise,
            pre_pool_kernel=pre_pool_kernel,
            pool_type=pool_type,
            channel_reduction=channel_reduction,
        )
    if key in {"gru", "bgru", "bigru"}:
        cfg = dict(gru_cfg or {})
        hidden_size = int(cfg.get("hidden_size", project_dim or in_channels))
        num_layers = int(cfg.get("num_layers", 1))
        bidirectional = bool(cfg.get("bidirectional", True))
        gru_dropout = float(cfg.get("dropout", dropout))
        pool = cfg.get("pool", "last")
        attention_hidden = cfg.get("attention_hidden")
        final_dropout = float(cfg.get("final_dropout", dropout))
        pre_pool_kernel = cfg.get("pre_pool_kernel")
        feature_mlp = cfg.get("feature_mlp")
        return TemporalGRUHead(
            in_channels,
            num_classes,
            project_dim=project_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=gru_dropout,
            pool=pool,
            attention_hidden=attention_hidden,
            final_dropout=final_dropout,
            pre_pool_kernel=pre_pool_kernel,
            feature_mlp=feature_mlp,
        )
    if key in {"ssm", "statespace", "state-space"}:
        cfg = dict(ssm_cfg or {})
        state_dim = int(cfg.get("state_dim", project_dim or 64))
        num_layers = int(cfg.get("num_layers", 2))
        dropout_val = float(cfg.get("dropout", dropout))
        activation = cfg.get("activation", "gelu")
        pre_pool_kernel = cfg.get("pre_pool_kernel")
        aggregation = cfg.get("aggregation", "last")
        feature_mlp = cfg.get("feature_mlp")
        return TemporalSSMHead(
            in_channels,
            num_classes,
            project_dim=project_dim,
            state_dim=state_dim,
            num_layers=num_layers,
            dropout=dropout_val,
            activation=activation,
            pre_pool_kernel=pre_pool_kernel,
            aggregation=aggregation,
            feature_mlp=feature_mlp,
        )
    raise ValueError(f"Unsupported head type '{head_type}'")


def build_autoencoder_classifier(
    spec: DatasetSpec,
    num_classes: int,
    *,
    head_type: str,
    head_cfg: Optional[Mapping[str, object]] = None,
) -> AutoencoderClassifier:
    encoder = TinyFStrideAutoencoder(_default_autoencoder_config(spec))
    latent_channels = encoder.latent_channels
    base_defaults = _DEFAULT_HEAD_CFG.get(head_type.lower(), {})
    defaults: Dict[str, Any] = copy.deepcopy(base_defaults) if base_defaults else {}

    dataset_key = _dataset_key(spec)
    try:
        dataset_overrides = _HEAD_DATASET_OVERRIDES.get(dataset_key, {}).get(head_type.lower())
        if dataset_overrides:
            defaults = _merge_nested(defaults, dataset_overrides)
    except NameError:
        # _HEAD_DATASET_OVERRIDES is not defined
        pass

    default_project_dim = int(defaults.get("project_dim", latent_channels))
    default_dropout = float(defaults.get("dropout", 0.0))
    default_tcn_cfg = defaults.get("tcn")
    default_gru_cfg = defaults.get("gru")
    default_ssm_cfg = defaults.get("ssm")

    project_dim = default_project_dim
    dropout = default_dropout
    tcn_cfg = default_tcn_cfg
    gru_cfg = default_gru_cfg
    ssm_cfg = default_ssm_cfg

    if head_cfg:
        project_dim = int(head_cfg.get("project_dim", project_dim))
        dropout = float(head_cfg.get("dropout", dropout))
        if "tcn" in head_cfg and head_cfg.get("tcn") is not None:
            tcn_cfg = head_cfg.get("tcn")
        if "gru" in head_cfg and head_cfg.get("gru") is not None:
            gru_cfg = head_cfg.get("gru")
        if "ssm" in head_cfg and head_cfg.get("ssm") is not None:
            ssm_cfg = head_cfg.get("ssm")

    head = build_classifier_head(
        head_type=head_type,
        in_channels=latent_channels,
        num_classes=num_classes,
        project_dim=project_dim,
        dropout=dropout,
        tcn_cfg=tcn_cfg if head_type.lower() == "tcn" else None,
        gru_cfg=gru_cfg if head_type.lower() == "gru" else None,
        ssm_cfg=ssm_cfg if head_type.lower() == "ssm" else None,
    )
    return AutoencoderClassifier(encoder, head)


HEAD_ALIASES: Dict[str, str] = {
    "aeheadtcn": "tcn",
    "aeheadgru": "gru",
    "aeheadssm": "ssm",
    "aehead_tcn": "tcn",
    "aehead_gru": "gru",
    "aehead_ssm": "ssm",
    "aeheadntu_tcn": "tcn",
    "aeheadntu_gru": "gru",
    "aeheadntu_ssm": "ssm",
}


def _dataset_key(spec: DatasetSpec) -> str:
    return spec.name.strip().lower().replace("-", "_")


def _merge_nested(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_nested(current, value)
        else:
            merged[key] = value
    return merged

def _default_autoencoder_config(spec: DatasetSpec) -> TinyFStrideConfig:
    dataset_key = _dataset_key(spec)
    try:
        dataset_defaults = _AUTOENCODER_DEFAULTS.get(dataset_key, {})
    except NameError:
        # _AUTOENCODER_DEFAULTS is not defined
        dataset_defaults = {}

    hidden_dims = tuple(int(v) for v in dataset_defaults.get("hidden_dims", (16, 32)))
    encoder_kernels = tuple(
        tuple(int(x) for x in kernel)
        for kernel in dataset_defaults.get("encoder_kernels", ((5, 3), (3, 3)))
    )
    encoder_strides = tuple(
        tuple(int(x) for x in stride)
        for stride in dataset_defaults.get("encoder_strides", ((2, 1), (2, 1)))
    )
    temporal_kernel = int(dataset_defaults.get("temporal_kernel", 3))
    temporal_dilations = tuple(
        tuple(int(x) for x in seq)
        for seq in dataset_defaults.get("temporal_dilations", ((1, 2), (2, 4)))
    )
    dropout = float(dataset_defaults.get("dropout", 0.10))

    latent_channels = dataset_defaults.get("latent_channels")
    if latent_channels is None:
        latent_channels = max(6, min(spec.channels, hidden_dims[-1]))
    else:
        latent_channels = int(latent_channels)

    latent_freq = dataset_defaults.get("latent_freq")
    if latent_freq is None:
        latent_freq = max(4, min(spec.freq_bins // 2, 16))
    else:
        latent_freq = int(latent_freq)

    return TinyFStrideConfig(
        in_channels=spec.channels,
        hidden_dims=hidden_dims,  # type: ignore[arg-type]
        encoder_kernels=encoder_kernels,  # type: ignore[arg-type]
        encoder_strides=encoder_strides,  # type: ignore[arg-type]
        temporal_kernel=temporal_kernel,
        temporal_dilations=temporal_dilations,  # type: ignore[arg-type]
        dropout=dropout,
        output_activation=None,
        latent_channels=latent_channels,
        latent_freq=int(latent_freq),
    )

_DEFAULT_HEAD_CFG = {
    "tcn": {
        "project_dim": 64,
        "dropout": 0.1,
        "tcn": {
            "dilations": (1, 2),
            "kernel_size": 3,
            "dropout": 0.1,
            "final_dropout": 0.1,
            "depthwise": True,
            "pre_pool_kernel": 2,
            "channel_reduction": (64,),
        },
    },
    "gru": {
        "project_dim": 64,
        "dropout": 0.1,
        "gru": {
            "hidden_size": 64,
            "num_layers": 1,
            "bidirectional": True,
            "dropout": 0.05,
            "pool": "attention",
            "attention_hidden": 64,
            "final_dropout": 0.1,
            "pre_pool_kernel": 2,
            "feature_mlp": [64],
        },
    },
    "ssm": {
        "project_dim": 64,
        "dropout": 0.1,
        "ssm": {
            "state_dim": 64,
            "num_layers": 2,
            "dropout": 0.05,
            "activation": "gelu",
            "pre_pool_kernel": 2,
            "aggregation": "mean",
            "feature_mlp": [64],
        },
    },
}
