"""Common CSI model builders with dataset-aware configuration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol

import torch
from torch import Tensor, nn
import torchvision

from ..utils.datasets import DatasetSpec


__all__ = [
    "CSIMLP",
    "CSILeNet",
    "CSIResNet18",
    "CSIBiLSTM",
    "CSITimeSformer",
    "MODEL_BUILDERS",
]

####################################################################################################

class CSIMLP(nn.Module):
    """Standalone MLP baseline model for CSI Data."""

    def __init__(
        self,
        *,
        in_channels: int,
        feat_size: int,
        seq_len: int,
        num_classes: int,
        # --- Hyperparameters ---
        hidden_dim: int = 512,   # Capacity of the network
        num_layers: int = 2,     # How deep the MLP is
        dropout: float = 0.3,    # Regularization
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.feat_size = feat_size
        self.seq_len = seq_len

        # The Blueprint
        self.params = {
            "in_channels": in_channels,
            "feat_size": feat_size,
            "seq_len": seq_len,
            "num_classes": num_classes,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }

        # Calculate the massive flattened size of the raw CSI data
        self.flat_size = in_channels * feat_size * seq_len

        # Project the massive flattened input down to our hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(self.flat_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Stack our reusable MLP for depth
        self.hidden_layers = nn.ModuleList([
            MLP(in_dim=hidden_dim, hidden_dim=hidden_dim * 2, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final classification head
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels or x.shape[2] != self.feat_size or x.shape[3] != self.seq_len:
            raise ValueError(f"Shape Mismatch. Expected (B, {self.in_channels}, {self.feat_size}, {self.seq_len})")

        # Flatten all spatial and temporal dimensions: [B, C, F, T] -> [B, C*F*T]
        x = x.reshape(x.size(0), -1)

        # Pass through the network
        x = self.input_projection(x)
        
        # Pass through the stacked reusable MLP blocks
        for layer in self.hidden_layers:
            # We add a residual connection here to help gradients flow in deep MLPs
            x = x + layer(x) 

        return self.head(x)

    def get_checkpoint(self) -> Dict[str, Any]:
        return {"model_state_dict": self.state_dict(), "params": self.params}

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict[str, Any]) -> 'CSIMLP':
        model = cls(**checkpoint["params"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


def build_mlp(
    spec: DatasetSpec,
    *,  # Enforce keyword arguments for clarity, since there are many hyperparameters with default values
    num_classes: int,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.25) -> nn.Module:
    
    return CSIMLP(
        in_channels=spec.channels,
        feat_size=spec.freq_bins,
        seq_len=spec.time_steps,
        num_classes=num_classes,     
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )

####################################################################################################

class CSILeNet(nn.Module):
    """Shared LeNet-style backbone with adaptive pooling. Keeping three layers."""

    def __init__(self,
                 *,  # Enforce keyword arguments for clarity, since there are many hyperparameters with default values
                 in_channels: int,
                 num_classes: int,
                 base_filters: int = 32,    # Width/capacity of network
                 fc_hidden_dim: int = 256,  # Size of classification head
                 pool_size: int = 2,        # Aggressiveness of spatial downsampling
                 dropout: float = 0.25,     # Regularization strength
                 ) -> None:
        
        super().__init__()
        
        # N.b., forward method does not like dicts
        self.in_channels = in_channels
        
        # Blueprint for checkpointing
        self.params = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "base_filters": base_filters,
            "fc_hidden_dim": fc_hidden_dim,
            "pool_size": pool_size,
            "dropout": dropout,
        }
        
        # Scale filter sizes
        f1 = base_filters
        f2 = base_filters * 2
        f3 = base_filters * 4
        
        # Convolutional backbone with batch norm, ReLU, and pooling
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Dynamic pooling to ensure fixed-size features regardless of input dimensions
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        
        # Dynamic classification head
        flattened_size = f3 * pool_size * pool_size
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected input shape (B, {self.in_channels}, F, T), but got {x.shape}")
        x = self.encoder(x)
        x = self.pool(x)
        return self.head(x)

    def get_checkpoint(self) -> Dict:
        """Packages the model's blueprint and current weights."""
        # Use as: 
        # checkpoint_dict = model.get_checkpoint()
        # torch.save(checkpoint_dict, "resnet18_checkpoint.pth")
        return {
            "model_state_dict": self.state_dict(),
            "params": self.params,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict) -> CSILeNet:
        """Factory method: Builds a new model directly from a saved checkpoint."""
        # Use as: 
        # saved_data = torch.load("resnet18_checkpoint.pth")
        # model = CSIResNet18.from_checkpoint(saved_data)
        # model.eval()
        model = cls(**checkpoint["params"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


def build_lenet(spec: DatasetSpec,
                *,  # Enforce keyword arguments for clarity, since there are many hyperparameters with default values
                num_classes: int,
                base_filters: int = 32, 
                fc_hidden_dim: int = 256, 
                pool_size: int = 2, 
                dropout: float = 0.25
                ) -> nn.Module:
    
    return CSILeNet(in_channels=spec.channels, 
                    num_classes=num_classes, 
                    base_filters=base_filters, 
                    fc_hidden_dim=fc_hidden_dim, 
                    pool_size=pool_size, 
                    dropout=dropout)

####################################################################################################

class CSIResNet18(nn.Module):
    """ResNet18 wrapper with dataset-aware input handling."""

    def __init__(self,
                 *,  # Enforce keyword arguments for clarity, since there are many hyperparameters with default values
                 in_channels: int,
                 num_classes: int,
                 pretrained: bool = False
                 ) -> None:
        super().__init__()
        
        # N.b., forward method does not like dicts
        self.in_channels = in_channels
        
        # For easy saving/loading
        self.params = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "pretrained": pretrained
        }
        
        # If pretrained = True, use Transfer Learning with ImageNet weights; otherwise, initialize randomly.
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.resnet18(weights=weights)
        
        # Input adapter
        if self.in_channels != 3:  # ResNet18 expects 3 input channels by default
            self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Output adapter
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected input shape (B, {self.in_channels}, F, T), but got {x.shape}")
        return self.model(x)
    
    def get_checkpoint(self) -> Dict:
        """Packages the model's blueprint and current weights."""
        # Use as: 
        # checkpoint_dict = model.get_checkpoint()
        # torch.save(checkpoint_dict, "resnet18_checkpoint.pth")
        return {
            "model_state_dict": self.state_dict(),
            "params": self.params,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict) -> CSIResNet18:
        """Factory method: Builds a new model directly from a saved checkpoint."""
        # Use as: 
        # saved_data = torch.load("resnet18_checkpoint.pth")
        # model = CSIResNet18.from_checkpoint(saved_data)
        # model.eval()
        model = cls(**checkpoint["params"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    

def build_resnet18(spec: DatasetSpec, 
                   *,  # Enforce keyword arguments for clarity, since there are many hyperparameters with default values
                   num_classes: int, 
                   pretrained: bool = False
                   ) -> nn.Module:
    
    return CSIResNet18(in_channels = spec.channels, 
                       num_classes = num_classes, 
                       pretrained = pretrained)

####################################################################################################

class CSIBiLSTM(nn.Module):
    """BiLSTM classifier over flattened channel-frequency features."""

    def __init__(self,
                 *,  # Enforce keyword arguments for clarity, since there are many hyperparameters with default values
                 in_channels: int,
                 freq_bins: int,
                 num_classes: int,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 bidirectional: bool = True,
                 dropout: float = 0.25,
                 ) -> None:
        super().__init__()
        
        # N.b., forward method does not like dicts
        self.in_channels = in_channels
        self.freq_bins = freq_bins
        self.bidirectional = bidirectional
        
        # Blueprint
        self.params = {
            "in_channels": in_channels,
            "freq_bins": freq_bins,
            "num_classes": num_classes,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "dropout": dropout,
        }
        
        self.lstm = nn.LSTM(
            self.params["in_channels"] * self.params["freq_bins"],
            self.params["hidden_dim"],
            num_layers=self.params["num_layers"],
            bidirectional=self.params["bidirectional"],
            dropout=self.params["dropout"] if self.params["num_layers"] > 1 else 0.0,
            batch_first=False  # Marginally more efficient for sequence processing
        )

        # Determine the size of the features coming out of the LSTM
        lstm_out_dim = self.params["hidden_dim"] * (2 if self.params["bidirectional"] else 1)
        
        # Classification MLP
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, self.params["hidden_dim"]), 
            nn.ReLU(),                           
            nn.Dropout(self.params["dropout"]),                 
            nn.Linear(self.params["hidden_dim"], num_classes)   
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels or x.shape[2] != self.freq_bins:
            raise ValueError(f"Expected input shape (B, {self.in_channels}, {self.freq_bins}, T), but got {x.shape}")
        b, c, f, t = x.shape
        x = x.reshape(b, c * f, t)  # flatten c and f into feature dimension
        x = x.permute(2, 0, 1)  # (T, B, C*F), since batch_first = False
        _, (ht, ct) = self.lstm(x)
        features = torch.cat((ht[-2], ht[-1]), dim=1) if self.bidirectional else ht[-1]  # Use the last hidden state(s) as features
        return self.fc(features)

    # Checkpointing methods for saving and loading model state
    def get_checkpoint(self) -> Dict:
        """Packages the model's blueprint and current weights."""
        # Use as: 
        # checkpoint_dict = model.get_checkpoint()
        # torch.save(checkpoint_dict, "bilstm_checkpoint.pth")
        return {
            "model_state_dict": self.state_dict(),
            "params": self.params,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict) -> CSIBiLSTM:
        """
        Builds a new model directly from a saved checkpoint.
        """
        # Use as: 
        # saved_data = torch.load("bilstm_checkpoint.pth")
        # model = CSIBiLSTM.from_checkpoint(saved_data)
        # model.eval()
        model = cls(**checkpoint["params"])  # Unpack the dictionary to build the model with the exact original dimensions
        model.load_state_dict(checkpoint["model_state_dict"])  # Load the trained weights into the perfectly sized architecture
        return model


def build_bilstm(spec: DatasetSpec,
                 *,  # Enforce keyword arguments for clarity, since there are many hyperparameters with default values
                 num_classes: int,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 bidirectional: bool = True,
                 dropout: float = 0.25,
                 ) -> nn.Module:
    
    return CSIBiLSTM(in_channels=spec.channels,
                     freq_bins=spec.freq_bins,
                     num_classes=num_classes,
                     hidden_dim=hidden_dim,
                     num_layers=num_layers,
                     bidirectional=bidirectional,
                     dropout=dropout)

####################################################################################################

class Rearrange(nn.Module):
    """Flexible module for rearranging tensor dimensions in nn.Sequential."""
    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(*self.dims)  # Dynamically unpack the dims tuple into permute arguments


class MLP(nn.Module):
    """Standard Transformer Feed-Forward Network."""
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 emb_dim: int, 
                 num_heads: int, 
                 dropout: float = 0.0
                 ) -> None:
        
        super().__init__()
        
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads  # Dimension of each attention head
        
        if self.head_dim * num_heads != emb_dim:
            raise ValueError("emb_dim must be divisible by num_heads")
        
        self.qkv = nn.Linear(emb_dim, emb_dim * 3)  # Compute queries, keys, and values in one go
        self.proj = nn.Linear(emb_dim, emb_dim)  # Final output projection. Just mixing heads together; no change in dimension
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        # Assumes input shape is (B, T, E) where B is batch size, T is sequence length, and E is embedding dimension
        bsz, seq_len, emb_dim = x.shape
        
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)  # Before: [Batch, Seq, 3, Heads, Head_Dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # After: [3, Batch, Heads, Seq, Head_Dim]; 
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each (Q, K, V) is now [Batch, Heads, Seq, Head_Dim]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # Apply scaled dot-product attention
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, emb_dim)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class TransformerEncoderBlock(nn.Module):
    """A standard modular Transformer block (Norm -> MHA -> Norm -> MLP)."""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0
                 ) -> None:
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)  # Expands and then contracts to same dimensions

    def forward(self, x: Tensor) -> Tensor:
        # Assumed shape is (B, F, E) where B is bsz, F is seq_len (time or feature patches), and E is emb_dim
        # Residual connection 1
        x = x + self.attn(self.norm1(x))
        # Residual connection 2
        x = x + self.mlp(self.norm2(x))
        return x


class CSITimesFormerBlock(nn.Module):
    """Divided Attention: Sequentially applies Temporal and Feature Attention."""
    def __init__(self, 
                 dim: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4.0, 
                 dropout: float = 0.0):
        super().__init__()
        
        # Two separate Transformer blocks for temporal and feature attention, each with its own normalization and MLP
        self.temporal_block = TransformerEncoderBlock(dim, num_heads, mlp_ratio, dropout)
        self.feature_block = TransformerEncoderBlock(dim, num_heads, mlp_ratio, dropout)

    def forward(self, x: Tensor, f_patches: int, t_patches: int) -> Tensor:
        # Assumed shape is (B, f_patches * t_patches, E) where E is the embedding dimension. We
        B, _, E = x.shape

        # Temporal Attention: Attend only across the same frequency band
        # Reshape to [B * f_patches, t_patches, E]
        xt = x.reshape(B * f_patches, t_patches, E)
        xt = self.temporal_block(xt)

        # Feature Attention: Attend only across the same time step on xt 
        # Permute to [B, t_patches, f_patches, E] then reshape to [B * t_patches, f_patches, E]
        xf = xt.reshape(B, f_patches, t_patches, E).permute(0, 2, 1, 3)  # split f_patches, t_patches and swap axes
        xf = xf.reshape(B * t_patches, f_patches, E)
        xf = self.feature_block(xf)

        # 3. Restore to original sequence shape: [B, f_patches * t_patches, E]
        x_out = xf.reshape(B, t_patches, f_patches, E).permute(0, 2, 1, 3)
        return x_out.reshape(B, f_patches * t_patches, E)


class CSITimeSformer(nn.Module):
    """Divided Attention Transformer for CSI Data with Hyperparameter Hooks."""
    def __init__(
        self,
        *,
        in_channels: int,
        feat_size: int,
        seq_len: int,
        num_classes: int,
        patch_f: int = 10,
        patch_t: int = 20,
        emb_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
        ) -> None:       
        
        super().__init__()
        
        self.in_channels = in_channels
        self.feat_size = feat_size
        self.seq_len = seq_len
                
        # Ensure feature_size (freq_bins) and seq_len (time_steps) are divisible by their respective patch sizes
        if feat_size % patch_f != 0 or seq_len % patch_t != 0:
            raise ValueError(f"freq_bins({feat_size}) or time_steps({seq_len}) not divisible by patches.")

        self.f_patches = self.feat_size // patch_f
        self.t_patches = self.seq_len // patch_t
        self.num_patches = self.f_patches * self.t_patches
        
        # Blueprint
        self.params = {
            "in_channels": in_channels,
            "feat_size": feat_size,
            "seq_len": seq_len,
            "num_classes": num_classes,
            "patch_f": patch_f,
            "patch_t": patch_t,
            "emb_dim": emb_dim,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "dropout": dropout,
        }
        
        # 2D Tokenizer: Creates the grid of patches across freq and time. Input is [B, C, F, T], output is [B, num_patches, emb_dim]
        self.patch_embed = nn.Conv2d(
            in_channels, emb_dim, kernel_size=(patch_f, patch_t), stride=(patch_f, patch_t)
        )
        
        # Tokens to learn spatial relationships between patches. N.b.: Replacing cls_token with Global Average Pooling for simplicity and stability(?)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # TimeSformer blocks with divided attention
        self.blocks = nn.ModuleList([
                    CSITimesFormerBlock(dim=emb_dim, 
                                        num_heads=num_heads, 
                                        mlp_ratio=mlp_ratio, 
                                        dropout=dropout)
                    for _ in range(depth)
                ])
 
        self.norm = nn.LayerNorm(emb_dim)
        
        # Classification head. Using a simple MLP on the global average pooled features from the final block
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels or x.shape[2] != self.feat_size or x.shape[3] != self.seq_len:
            raise ValueError(f"Shape Mismatch. Expected (B, {self.in_channels}, {self.feat_size}, {self.seq_len})")
        
        # Patching and adding pos_embed tokens
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # Output dims: [B, num_patches, emb_dim]
        x = x + self.pos_embed
        
        # Divided Attention blocks
        for block in self.blocks:
            x = block(x, self.f_patches, self.t_patches)
            
        # Global Pooling (taking mean) and Classify
        x = self.norm(x).mean(dim=1)
        return self.head(x)

    def get_checkpoint(self) -> Dict:
        """Packages the model's blueprint and current weights."""
        # Use as: 
        # checkpoint_dict = model.get_checkpoint()
        # torch.save(checkpoint_dict, "csitimesformer_checkpoint.pth")
        return {"model_state_dict": self.state_dict(), "params": self.params}

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict[str, Any]) -> CSITimeSformer:
        """Builds a new model directly from a saved checkpoint."""
        # Use as: 
        # saved_data = torch.load("csitimesformer_checkpoint.pth")
        # model = CSITimeSformer.from_checkpoint(saved_data)
        # model.eval()
        
        model = cls(**checkpoint["params"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


def build_csitimesformer(spec: DatasetSpec,
                         num_classes: int,
                         *,
                         patch_f: int = 10,
                         patch_t: int = 20,
                         emb_dim: int = 128,
                         depth: int = 4,
                         num_heads: int = 8,
                         mlp_ratio: float = 4.0,
                         dropout: float = 0.1
                         ) -> nn.Module:

    return CSITimeSformer(
        in_channels=spec.channels,
        feat_size=spec.freq_bins,
        seq_len=spec.time_steps,
        num_classes=num_classes,
        patch_f=patch_f,
        patch_t=patch_t,
        emb_dim=emb_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout
    )

####################################################################################################

MODEL_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "mlp": build_mlp,
    "lenet": build_lenet,
    "resnet18": build_resnet18,
    "bilstm": build_bilstm,
    "csitimesformer": build_csitimesformer,
}

####################################################################################################
