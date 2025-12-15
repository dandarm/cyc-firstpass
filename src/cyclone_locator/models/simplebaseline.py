import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm

def deconv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class SimpleBaseline(nn.Module):
    """
    ResNet backbone -> 3 deconv -> head heatmap(1 canale) + head presenza(1 logit).
    - out heatmap size: input/4 (se 3 deconv su feature stride 32 -> saliamo a /4)
    """
    def __init__(self, backbone="resnet18", out_heatmap_ch=1, temporal_T: int = 1,
                 presence_dropout: float = 0.0):
        super().__init__()
        temporal_T = max(1, int(temporal_T))
        presence_dropout = max(0.0, float(presence_dropout))
        if backbone == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            feat_ch = 512
        elif backbone == "resnet50":
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
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

        # 3 deconv: riportano verso l'alto la risoluzione
        self.deconv1 = deconv_block(feat_ch, 256)
        self.deconv2 = deconv_block(256, 256)
        self.deconv3 = deconv_block(256, 256)

        # Head heatmap (K=1 canale)
        self.head_heatmap = nn.Conv2d(256, out_heatmap_ch, kernel_size=1)

        # Head presenza: GAP -> FC(1)
        self.head_presence_gap = nn.AdaptiveAvgPool2d((1,1))
        self.head_presence_dropout = nn.Dropout(p=presence_dropout) if presence_dropout > 0 else nn.Identity()
        self.head_presence_fc  = nn.Linear(256, 1)

    def forward(self, x):
        """
        x: (B,C,H,W) - H=W=512 (o multipli di 32).
        Return:
          heatmap: (B,1,H/4,W/4)
          presence_logit: (B,1)
        """
        f = self.stem(x)              # (B,feat_ch,H/32,W/32)
        y = self.deconv1(f)           # /16
        y = self.deconv2(y)           # /8
        y = self.deconv3(y)           # /4
        heatmap = self.head_heatmap(y)

        g = self.head_presence_gap(y).flatten(1)  # (B,256)
        g = self.head_presence_dropout(g)
        presence_logit = self.head_presence_fc(g) # (B,1)
        return heatmap, presence_logit
