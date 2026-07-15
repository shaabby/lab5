"""Lab5 简单 EuroSAT 分类网络。"""

from mindspore import nn
from mindspore.common.initializer import HeNormal, TruncatedNormal


class ConvBlock(nn.Cell):
    """使用基础算子完成卷积、激活和下采样。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            pad_mode="pad",
            padding=1,
            has_bias=True,
            weight_init=HeNormal(mode="fan_out", nonlinearity="relu"),
        )
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        return self.pool(self.activation(self.conv(x)))


class SimpleEuroSATCNN(nn.Cell):
    """面向 64x64 输入的四层卷积分类网络。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.SequentialCell(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 96),
            ConvBlock(96, 128),
        )
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()
        self.classifier = nn.Dense(
            128,
            num_classes,
            weight_init=TruncatedNormal(sigma=0.02),
        )

    def construct(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)


def create_model(num_classes=10):
    return SimpleEuroSATCNN(num_classes=num_classes)
