from pathlib import Path

import torch
import torch.nn as nn

try:
    from pytorchvideo.models.hub import x3d_s, x3d_xs
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("pytorchvideo is required for X3D backbones (install pytorchvideo>=0.1).") from exc


def _load_local_checkpoint(base: nn.Module, weights_path: Path):
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    base.load_state_dict(state, strict=True)


def _resolve_weights_path(variant: str, weights_path: str | None):
    """
    Returns a Path or None. If weights_path is "auto" (default), look for a
    root-level checkpoint matching the variant.
    """
    if weights_path == "auto":
        if variant == "x3d_xs":
            candidate = Path(__file__).resolve().parents[3] / "X3D_XS.pyth"
            if candidate.exists():
                return candidate
        return None
    return Path(weights_path) if weights_path else None


def _build_backbone(variant: str, pretrained: bool, weights_path: str | None):
    resolved_path = _resolve_weights_path(variant, weights_path)

    if variant == "x3d_xs":
        base = x3d_xs(pretrained=pretrained and resolved_path is None)
        if resolved_path:
            _load_local_checkpoint(base, resolved_path)
    elif variant == "x3d_s":
        base = x3d_s(pretrained=pretrained and resolved_path is None)
        if resolved_path:
            _load_local_checkpoint(base, resolved_path)
    else:
        raise ValueError("variant must be x3d_xs|x3d_s")

    # Drop the classification head and keep only the feature extractor blocks.
    stem = nn.Sequential(*list(base.blocks[:-1]))

    # Infer the channel dimension once so the decoder can be wired correctly.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 4, 128, 128)
        feat_channels = stem(dummy).shape[1]

    return stem, feat_channels


def deconv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class X3DBackbone(nn.Module):
    """Lightweight X3D backbone with temporal convolutions preserved.

    The network keeps the temporal dimension until an adaptive temporal pooling step
    just before the 2D decoder, so different window lengths (``temporal_T``) and
    strides are supported without any hard-coded assumptions.
    """

    def __init__(
        self,
        backbone: str = "x3d_xs",
        out_heatmap_ch: int = 1,
        presence_dropout: float = 0.0,
        pretrained: bool = True,
        weights_path: str | None = "auto",
    ):
        super().__init__()
        self.backbone_name = backbone
        self.stem, feat_ch = _build_backbone(backbone, pretrained, weights_path)

        # Preserve temporal dynamics until this pooling collapses only T -> 1.
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))

        self.deconv1 = deconv_block(feat_ch, 256)
        self.deconv2 = deconv_block(256, 256)
        self.deconv3 = deconv_block(256, 256)

        self.head_heatmap = nn.Conv2d(256, out_heatmap_ch, kernel_size=1)
        self.head_presence_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head_presence_dropout = (
            nn.Dropout(p=presence_dropout) if presence_dropout > 0 else nn.Identity()
        )
        self.head_presence_fc = nn.Linear(256, 1)
        self.input_is_video = True

    def forward(self, x: torch.Tensor):
        """Forward on spatio-temporal clips.

        Args:
            x: Tensor with shape (B, C, T, H, W).
        Returns:
            heatmap: (B, 1, H/4, W/4) after decoding.
            presence_logit: (B, 1) logits for presence.
        """

        f = self.stem(x)  # (B, C, T', H', W')
        f = self.temporal_pool(f).squeeze(2)  # -> (B, C, H', W')

        y = self.deconv1(f)
        y = self.deconv2(y)
        y = self.deconv3(y)
        heatmap = self.head_heatmap(y)

        g = self.head_presence_gap(y).flatten(1)
        g = self.head_presence_dropout(g)
        presence_logit = self.head_presence_fc(g)
        return heatmap, presence_logit
