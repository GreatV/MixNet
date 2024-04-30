import paddle


class basisblock(paddle.nn.Layer):
    def __init__(self, inplanes, planes, groups=1):
        super(basisblock, self).__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
        )
        self.bn1 = paddle.nn.BatchNorm2D(num_features=planes)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
        )
        self.bn2 = paddle.nn.BatchNorm2D(num_features=planes)
        self.relu = paddle.nn.ReLU()
        self.resid = None
        if inplanes != planes:
            self.resid = paddle.nn.Conv2D(
                in_channels=inplanes,
                out_channels=planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False,
            )

    def forward(self, x):
        residual = x.clone()
        if self.resid:
            residual = self.resid(residual)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x


class bottleneck(paddle.nn.Layer):
    def __init__(self, inplanes, planes, groups=1):
        super(bottleneck, self).__init__()
        self.resid = None
        if inplanes != planes:
            self.resid = paddle.nn.Conv2D(
                in_channels=inplanes,
                out_channels=planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False,
            )
            hidplanes = inplanes
        else:
            hidplanes = inplanes // 2
        self.relu = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=inplanes,
            out_channels=hidplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False,
        )
        self.bn1 = paddle.nn.BatchNorm2D(num_features=hidplanes)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=hidplanes,
            out_channels=hidplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias_attr=False,
        )
        self.bn2 = paddle.nn.BatchNorm2D(num_features=hidplanes)
        self.conv3 = paddle.nn.Conv2D(
            in_channels=hidplanes,
            out_channels=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False,
        )
        self.bn3 = paddle.nn.BatchNorm2D(num_features=planes)

    def forward(self, x):
        residual = x.clone()
        if self.resid:
            residual = self.resid(residual)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += residual
        x = self.relu(x)
        return x


def switchLayer(channels, xs):
    numofeature = len(xs)
    splitxs = []
    for i in range(numofeature):
        splitxs.append(list(paddle.chunk(x=xs[i], chunks=numofeature, axis=1)))
    for i in range(numofeature):
        h, w = tuple(splitxs[i][i].shape)[2:]
        tmp = []
        for j in range(numofeature):
            if i > j:
                splitxs[j][i] = paddle.nn.functional.avg_pool2d(
                    kernel_size=2 ** (i - j), x=splitxs[j][i], exclusive=False
                )
            elif i < j:
                splitxs[j][i] = paddle.nn.functional.interpolate(
                    x=splitxs[j][i], size=(h, w)
                )
            tmp.append(splitxs[j][i])
        xs[i] = paddle.concat(x=tmp, axis=1)
    return xs


class FeatureShuffleNet(paddle.nn.Layer):
    def __init__(self, block, channels=64, numofblocks=None, groups=1):
        super(FeatureShuffleNet, self).__init__()
        self.channels = channels
        self.numofblocks = numofblocks
        self.stem = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=3,
                out_channels=channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=channels),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=channels),
            paddle.nn.ReLU(),
        )
        Layerplanes = [
            self.channels,
            self.channels,
            self.channels * 2,
            self.channels * 3,
            self.channels * 4,
        ]
        self.downSteps = paddle.nn.LayerList()
        for planes in Layerplanes[:-1]:
            self.downSteps.append(
                paddle.nn.Sequential(
                    paddle.nn.Conv2D(
                        in_channels=planes,
                        out_channels=planes,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False,
                    ),
                    paddle.nn.BatchNorm2D(num_features=planes),
                    paddle.nn.ReLU(),
                )
            )
        self.blocks_1 = paddle.nn.LayerList()
        self.blocks_2 = paddle.nn.LayerList()
        self.blocks_3 = paddle.nn.LayerList()
        for l in range(4):
            for i, num in enumerate(self.numofblocks[l]):
                tmp = [block(Layerplanes[i + l], Layerplanes[i + 1 + l], groups=groups)]
                for j in range(num - 1):
                    tmp.append(
                        block(
                            Layerplanes[i + 1 + l],
                            Layerplanes[i + 1 + l],
                            groups=groups,
                        )
                    )
                if l == 0:
                    self.blocks_1.append(paddle.nn.Sequential(*tmp))
                elif l == 1:
                    self.blocks_2.append(paddle.nn.Sequential(*tmp))
                elif l == 2:
                    self.blocks_3.append(paddle.nn.Sequential(*tmp))
                else:
                    self.blocks_4 = paddle.nn.Sequential(*tmp)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.downSteps[0](x)
        x1 = self.blocks_1[0](x1)
        x2 = self.downSteps[1](x1)
        x1 = self.blocks_1[1](x1)
        x2 = self.blocks_2[0](x2)
        x3 = self.downSteps[2](x2)
        x1, x2 = switchLayer(self.channels, [x1, x2])
        x1 = self.blocks_1[2](x1)
        x2 = self.blocks_2[1](x2)
        x3 = self.blocks_3[0](x3)
        x4 = self.downSteps[3](x3)
        x1, x2, x3 = switchLayer(self.channels, [x1, x2, x3])
        x1 = self.blocks_1[3](x1)
        x2 = self.blocks_2[2](x2)
        x3 = self.blocks_3[1](x3)
        x4 = self.blocks_4(x4)
        return x1, x2, x3, x4


def count_parameters(model):
    return sum(p.size for p in model.parameters() if not p.stop_gradient)


def FSNet_Splus(pretrained=True):
    numofblocks = [[4, 2, 2, 2], [2, 2, 2], [8, 8], [4]]
    model = FeatureShuffleNet(basisblock, channels=64, numofblocks=numofblocks)
    print("FSNet_M parameter size: ", count_parameters(model))
    if pretrained:
        print("FSNet_M does not have pretrained weight yet. ")
    return model


def FSNet_M(pretrained=True):
    numofblocks = [[4, 2, 2, 2], [4, 4, 4], [10, 10], [10]]
    model = FeatureShuffleNet(bottleneck, channels=64, numofblocks=numofblocks)
    print("FSNet_M now with bottleneck.")
    print("FSNet_M parameter size: ", count_parameters(model))
    if pretrained:
        print("FSNet_M does not have pretrained weight yet. ")
    return model


def FSNeXt_M(pretrained=True):
    numofblocks = [[4, 2, 2, 2], [4, 4, 4], [10, 10], [10]]
    model = FeatureShuffleNet(
        bottleneck, channels=64, numofblocks=numofblocks, groups=32
    )
    print("FSNeXt_M parameter size: ", count_parameters(model))
    if pretrained:
        print("FSNeXt_M does not have pretrained weight yet. ")
    return model


def FSNet_S(pretrained=True):
    numofblocks = [[4, 1, 1, 1], [4, 2, 2], [8, 8], [4]]
    model = FeatureShuffleNet(basisblock, channels=64, numofblocks=numofblocks)
    print("FSNet_S parameter size: ", count_parameters(model))
    if pretrained:
        print("FSNet_S does not have pretrained weight yet. ")
    return model


def FSNeXt_S(pretrained=True):
    numofblocks = [[4, 1, 1, 1], [4, 2, 2], [8, 8], [4]]
    model = FeatureShuffleNet(
        bottleneck, channels=128, numofblocks=numofblocks, groups=32
    )
    print("FSNeXt_S parameter size: ", count_parameters(model))
    if pretrained:
        print("FSNeXt_S does not have pretrained weight yet. ")
    return model


def FSNet_T(pretrained=True):
    numofblocks = [[1, 1, 1, 1], [2, 1, 1], [3, 3], [3]]
    model = FeatureShuffleNet(basisblock, channels=64, numofblocks=numofblocks)
    print("FSNet_T parameter size: ", count_parameters(model))
    if pretrained:
        print("FSNet_T does not have pretrained weight yet. ")
    return model
