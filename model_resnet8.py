"""面向 EuroSAT 64x64 图像的宽通道 ResNet-8。"""

from mindspore import nn
from mindspore.common.initializer import HeNormal, TruncatedNormal


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        pad_mode="pad",
        padding=1,
        has_bias=False,
        weight_init=HeNormal(mode="fan_out", nonlinearity="relu"),
    )


class BasicBlock(nn.Cell):
    """包含两层 3x3 卷积的基础残差块。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.norm1 = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.activation = nn.ReLU()
        self.needs_projection = stride != 1 or in_channels != out_channels
        if self.needs_projection:
            self.projection = nn.SequentialCell(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    pad_mode="valid",
                    has_bias=False,
                    weight_init=HeNormal(mode="fan_out", nonlinearity="relu"),
                ),
                nn.BatchNorm2d(out_channels, momentum=0.9),
            )

    def construct(self, x):
        residual = self.projection(x) if self.needs_projection else x
        features = self.activation(self.norm1(self.conv1(x)))
        features = self.norm2(self.conv2(features))
        return self.activation(features + residual)


class WideResNet8(nn.Cell):
    """由一个 stem 和三个 BasicBlock 构成的 ResNet-8。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.SequentialCell(
            conv3x3(3, 32),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.ReLU(),
        )
        self.block1 = BasicBlock(32, 32)
        self.block2 = BasicBlock(32, 64, stride=2)
        self.block3 = BasicBlock(64, 128, stride=2)
        self.global_pool = nn.AvgPool2d(kernel_size=16, stride=16)
        self.flatten = nn.Flatten()
        self.classifier = nn.Dense(
            128,
            num_classes,
            weight_init=TruncatedNormal(sigma=0.02),
        )

    def construct(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return self.classifier(x)


def create_model(num_classes=10):
    return WideResNet8(num_classes=num_classes)
