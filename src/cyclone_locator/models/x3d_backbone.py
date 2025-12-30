from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_x3d_builders():
    try:
        from pytorchvideo.models.hub import x3d_s, x3d_xs
        try:
            from pytorchvideo.models.hub import x3d_m  # type: ignore
        except ImportError:  # pragma: no cover - depends on pytorchvideo version
            x3d_m = None
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise ImportError(
            "pytorchvideo is required for X3D backbones (install pytorchvideo>=0.1)."
        ) from exc
    return x3d_xs, x3d_s, x3d_m


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
        if variant == "x3d_m":
            candidate = Path(__file__).resolve().parents[3] / "X3D_M.pyth"
            if candidate.exists():
                return candidate
        if variant == "x3d_xs":
            candidate = Path(__file__).resolve().parents[3] / "X3D_XS.pyth"
            if candidate.exists():
                return candidate
        return None
    return Path(weights_path) if weights_path else None


def _build_backbone(variant: str, pretrained: bool, weights_path: str | None):
    resolved_path = _resolve_weights_path(variant, weights_path)
    x3d_xs, x3d_s, x3d_m = _get_x3d_builders()

    if variant == "x3d_xs":
        base = x3d_xs(pretrained=pretrained and resolved_path is None)
        if resolved_path:
            _load_local_checkpoint(base, resolved_path)
    elif variant == "x3d_s":
        base = x3d_s(pretrained=pretrained and resolved_path is None)
        if resolved_path:
            _load_local_checkpoint(base, resolved_path)
    elif variant == "x3d_m":
        if x3d_m is None:  # pragma: no cover - depends on pytorchvideo version
            raise ImportError(
                "Your pytorchvideo installation does not expose x3d_m; upgrade pytorchvideo."
            )
        base = x3d_m(pretrained=pretrained and resolved_path is None)
        if resolved_path:
            _load_local_checkpoint(base, resolved_path)
    else:
        raise ValueError("variant must be x3d_xs|x3d_s|x3d_m")

    # Drop the classification head and keep only the feature extractor blocks.
    stem = nn.Sequential(*list(base.blocks[:-1]))

    # Infer the channel dimension once so the decoder can be wired correctly.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 4, 128, 128)
        feat_channels = stem(dummy).shape[1]

    return stem, feat_channels


def deconv_block(in_ch, out_ch):
    # Back-compat alias (ConvTranspose2d removed to avoid checkerboard artifacts)
    return resize_conv_block(in_ch, out_ch, mode="bilinear")


def resize_conv_block(in_ch, out_ch, *, mode: str = "bilinear"):
    if mode not in {"bilinear", "nearest"}:
        raise ValueError("mode must be 'bilinear' or 'nearest'")
    upsample = nn.Upsample(
        scale_factor=2,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    return nn.Sequential(
        upsample,
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
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
        heatmap_stride: int = 4,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.stem, feat_ch = _build_backbone(backbone, pretrained, weights_path)
        self.base_heatmap_stride = 4  # decoder porta sempre a /4
        heatmap_stride = int(heatmap_stride)
        if heatmap_stride <= 0:
            raise ValueError("heatmap_stride must be > 0")
        if self.base_heatmap_stride % heatmap_stride != 0:
            raise ValueError(
                f"heatmap_stride={heatmap_stride} not supported (base stride is {self.base_heatmap_stride})"
            )
        self.heatmap_stride = heatmap_stride
        self.heatmap_upsample_factor = self.base_heatmap_stride // heatmap_stride

        # Preserve temporal dynamics until this pooling collapses only T -> 1.
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))

        self.up1 = resize_conv_block(feat_ch, 256, mode="bilinear")
        self.up2 = resize_conv_block(256, 256, mode="bilinear")
        self.up3 = resize_conv_block(256, 256, mode="bilinear")

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

        y = self.up1(f)
        y = self.up2(y)
        y = self.up3(y)
        if self.heatmap_upsample_factor > 1:
            y_hm = F.interpolate(
                y,
                scale_factor=self.heatmap_upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
        else:
            y_hm = y
        heatmap = self.head_heatmap(y_hm)

        g = self.head_presence_gap(y).flatten(1)
        g = self.head_presence_dropout(g)
        presence_logit = self.head_presence_fc(g)
        return heatmap, presence_logit
