"""Lab5 自定义 EuroSAT 分类网络。"""

from mindspore import nn, ops
from mindspore.common.initializer import HeNormal, TruncatedNormal


class ConvNormAct(nn.Cell):
    """卷积、批归一化和 SiLU 激活的基础组合。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        activate=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=kernel_size // 2,
            group=groups,
            has_bias=False,
            weight_init=HeNormal(mode="fan_out", nonlinearity="relu"),
        )
        self.norm = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.activate = activate
        self.activation = nn.SiLU() if activate else None

    def construct(self, x):
        x = self.norm(self.conv(x))
        if self.activate:
            x = self.activation(x)
        return x


class ChannelGate(nn.Cell):
    """使用全局上下文生成逐通道权重。"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_channels = max(16, channels // reduction)
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.SequentialCell(
            nn.Dense(channels, hidden_channels),
            nn.SiLU(),
            nn.Dense(
                hidden_channels,
                channels,
                weight_init=TruncatedNormal(sigma=0.02),
            ),
            nn.Sigmoid(),
        )

    def construct(self, x):
        weights = self.pool(x)
        weights = ops.reshape(weights, (weights.shape[0], self.channels))
        weights = self.mlp(weights)
        weights = ops.reshape(weights, (weights.shape[0], self.channels, 1, 1))
        return x * weights


class GeoFusionBlock(nn.Cell):
    """融合局部 3x3 纹理与较大 5x5 上下文的自定义残差块。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        branch_channels = out_channels // 2

        self.local_branch = nn.SequentialCell(
            ConvNormAct(
                in_channels,
                branch_channels,
                kernel_size=3,
                stride=stride,
            ),
            ConvNormAct(
                branch_channels,
                branch_channels,
                kernel_size=3,
            ),
        )
        self.context_branch = nn.SequentialCell(
            ConvNormAct(in_channels, branch_channels, kernel_size=1),
            ConvNormAct(
                branch_channels,
                branch_channels,
                kernel_size=5,
                stride=stride,
                groups=branch_channels,
            ),
            ConvNormAct(
                branch_channels,
                branch_channels,
                kernel_size=1,
            ),
        )
        self.fuse = ConvNormAct(
            out_channels,
            out_channels,
            kernel_size=1,
            activate=False,
        )
        self.gate = ChannelGate(out_channels)
        self.needs_projection = stride != 1 or in_channels != out_channels
        if self.needs_projection:
            self.shortcut = ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                activate=False,
            )
        self.activation = nn.SiLU()

    def construct(self, x):
        residual = self.shortcut(x) if self.needs_projection else x
        local_features = self.local_branch(x)
        context_features = self.context_branch(x)
        fused = ops.concat((local_features, context_features), axis=1)
        fused = self.gate(self.fuse(fused))
        return self.activation(fused + residual)


class GeoFusionNet(nn.Cell):
    """面向 64x64 遥感图像设计的四阶段分类网络。"""

    def __init__(self, num_classes=10, dropout=0.25):
        super().__init__()
        self.stem = nn.SequentialCell(
            ConvNormAct(3, 32, kernel_size=3),
            ConvNormAct(32, 40, kernel_size=3),
        )
        self.stage1 = self._make_stage(40, 48, blocks=2, stride=1)
        self.stage2 = self._make_stage(48, 80, blocks=2, stride=2)
        self.stage3 = self._make_stage(80, 144, blocks=3, stride=2)
        self.stage4 = self._make_stage(144, 240, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Dense(
            240,
            num_classes,
            weight_init=TruncatedNormal(sigma=0.02),
        )

    @staticmethod
    def _make_stage(in_channels, out_channels, blocks, stride):
        layers = [GeoFusionBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(GeoFusionBlock(out_channels, out_channels))
        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = ops.reshape(x, (x.shape[0], -1))
        return self.classifier(self.dropout(x))


def create_model(num_classes=10):
    return GeoFusionNet(num_classes=num_classes)
