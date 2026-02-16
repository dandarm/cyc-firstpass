import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm

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
        nn.ReLU(inplace=True)
    )

class SimpleBaseline(nn.Module):
    """
    ResNet backbone -> 3 deconv -> head heatmap(1 canale) + head presenza(1 logit).
    - out heatmap size: input/4 (se 3 deconv su feature stride 32 -> saliamo a /4)
    """
    def __init__(self, backbone="resnet18", out_heatmap_ch=1, temporal_T: int = 1,
                 presence_dropout: float = 0.0, pretrained: bool = True,
                 heatmap_stride: int = 4):
        super().__init__()
        temporal_T = max(1, int(temporal_T))
        presence_dropout = max(0.0, float(presence_dropout))
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
        if backbone == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            m = tvm.resnet18(weights=weights)
            feat_ch = 512
        elif backbone == "resnet50":
            weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            m = tvm.resnet50(weights=weights)
            feat_ch = 2048
        else:
            raise ValueError("backbone must be resnet18|resnet50")

        in_ch = 3 * temporal_T
        if in_ch != m.conv1.in_channels:
            base_conv = m.conv1
            new_conv = nn.Conv2d(
                in_ch, base_conv.out_channels,
                kernel_size=base_conv.kernel_size,
                stride=base_conv.stride,
                padding=base_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight.copy_(base_conv.weight.repeat(1, temporal_T, 1, 1) / temporal_T)
            m.conv1 = new_conv

        # Prendiamo tutto tranne l'avgpool e fc
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )

        # 3 resize-conv upsampling: evita checkerboard rispetto a ConvTranspose2d
        self.up1 = resize_conv_block(feat_ch, 256, mode="bilinear")
        self.up2 = resize_conv_block(256, 256, mode="bilinear")
        self.up3 = resize_conv_block(256, 256, mode="bilinear")

        # Head heatmap (K=1 canale)
        self.head_heatmap = nn.Conv2d(256, out_heatmap_ch, kernel_size=1)

        # Head presenza: GAP -> FC(1)
        self.head_presence_gap = nn.AdaptiveAvgPool2d((1,1))
        self.head_presence_dropout = nn.Dropout(p=presence_dropout) if presence_dropout > 0 else nn.Identity()
        self.head_presence_fc  = nn.Linear(256, 1)
        self.input_is_video = False

    def forward(self, x):
        """
        x: (B,C,H,W) - H=W=512 (o multipli di 32).
        Return:
          heatmap: (B,1,H/4,W/4)
          presence_logit: (B,1)
        """
        f = self.stem(x)              # (B,feat_ch,H/32,W/32)
        y = self.up1(f)               # /16
        y = self.up2(y)               # /8
        y = self.up3(y)               # /4
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

        g = self.head_presence_gap(y).flatten(1)  # (B,256)
        g = self.head_presence_dropout(g)
        presence_logit = self.head_presence_fc(g) # (B,1)
        return heatmap, presence_logit
