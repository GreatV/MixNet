import paddle


class block(paddle.nn.Layer):
    def __init__(self, inplanes, planes, dcn=False):
        super(block, self).__init__()
        self.dcn = dcn
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
                    kernel_size=2 * (i - j), x=splitxs[j][i], exclusive=False
                )
            elif i < j:
                splitxs[j][i] = paddle.nn.functional.interpolate(
                    x=splitxs[j][i], size=(h, w)
                )
            tmp.append(splitxs[j][i])
        xs[i] = paddle.concat(x=tmp, axis=1)
    return xs


class FSNet(paddle.nn.Layer):
    def __init__(self, channels=64, numofblocks=4, layers=[1, 2, 3, 4], dcn=False):
        super(FSNet, self).__init__()
        self.channels = channels
        self.numofblocks = numofblocks
        self.layers = layers
        self.blocks = paddle.nn.LayerList()
        self.steps = paddle.nn.LayerList()
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
        for l in layers:
            self.steps.append(
                paddle.nn.Sequential(
                    paddle.nn.Conv2D(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False,
                    ),
                    paddle.nn.BatchNorm2D(num_features=channels),
                    paddle.nn.ReLU(),
                )
            )
            next_channels = self.channels * l
            for i in range(l):
                tmp = [block(channels, next_channels, dcn=False)]
                for j in range(self.numofblocks - 1):
                    tmp.append(block(next_channels, next_channels, dcn=dcn))
                self.blocks.append(paddle.nn.Sequential(*tmp))
            channels = next_channels

    def forward(self, x):
        x = self.stem(x)
        x1 = self.steps[0](x)
        x1 = self.blocks[0](x1)
        x2 = self.steps[1](x1)
        x1 = self.blocks[1](x1)
        x2 = self.blocks[2](x2)
        x3 = self.steps[2](x2)
        x1, x2 = switchLayer(self.channels, [x1, x2])
        x1 = self.blocks[3](x1)
        x2 = self.blocks[4](x2)
        x3 = self.blocks[5](x3)
        x4 = self.steps[3](x3)
        x1, x2, x3 = switchLayer(self.channels, [x1, x2, x3])
        x1 = self.blocks[6](x1)
        x2 = self.blocks[7](x2)
        x3 = self.blocks[8](x3)
        x4 = self.blocks[9](x4)
        return x1, x2, x3, x4


def count_parameters(model):
    return sum(p.size for p in model.parameters() if not p.stop_gradient)


def FSNet_M(pretrained=True):
    model = FSNet()
    print("MixNet backbone parameter size: ", count_parameters(model))
    if pretrained:
        load_path = "./pretrained/triHRnet_Synth_weight.pth"
        cpt = paddle.load(path=load_path)
        model.set_state_dict(state_dict=cpt, use_structured_name=True)
        print("load pretrain weight from {}. ".format(load_path))
    return model
