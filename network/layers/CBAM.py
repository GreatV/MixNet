import paddle


class ChannelAttention(paddle.nn.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = paddle.nn.AdaptiveAvgPool2D(output_size=1)
        self.max_pool = paddle.nn.AdaptiveMaxPool2D(output_size=1)
        self.fc = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=in_planes,
                out_channels=in_planes // ratio,
                kernel_size=1,
                bias_attr=False,
            ),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=in_planes // ratio,
                out_channels=in_planes,
                kernel_size=1,
                bias_attr=False,
            ),
        )
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(paddle.nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias_attr=False,
        )
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        avg_out = paddle.mean(x=x, axis=1, keepdim=True)
        max_out, _ = (
            paddle.max(x=x, axis=1, keepdim=True),
            paddle.argmax(x=x, axis=1, keepdim=True),
        )
        x = paddle.concat(x=[avg_out, max_out], axis=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(paddle.nn.Layer):
    def __init__(self, inplane, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(inplane)
        self.sp = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sp(x) * x
        return x
