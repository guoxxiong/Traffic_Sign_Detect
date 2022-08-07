from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1), 
            nn.BatchNorm2d(num_features=out_channels), 
            nn.LeakyReLU(negative_slope=0.1, inplace=True), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(num_features=out_channels), 
        )
        self.shortcut = shortcut
        self.leaky = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.layer(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return self.leaky(out)


class CLASSIFIER(nn.Module):
    """
    ResNet34
    """
    def __init__(self, classes=50):
        super(CLASSIFIER, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(num_features=16), 
            nn.LeakyReLU(negative_slope=0.1, inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), 
        )

        self.layer1 = self._make_layer(16, 16, 3)
        self.layer2 = self._make_layer(16, 32, 4, stride=2)
        self.layer3 = self._make_layer(32, 64, 6, stride=2)
        self.layer4 = self._make_layer(64, 128, 3, stride=2)

        # freeze
        # for p in self.parameters():
        #     p.requires_gard = False

        self.fc = nn.Linear(128, classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, in_channels, out_channels, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), 
            nn.BatchNorm2d(num_features=out_channels), 
        )
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, shortcut=shortcut))
        for _ in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels))
        layers.append(nn.Dropout2d(p=0.2))
        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        avgpool = nn.AvgPool2d(kernel_size=x.shape[2:])
        x = avgpool(x).view(x.size(0), -1)
        x = self.fc(x)

        if target is not None:
            loss = nn.functional.cross_entropy(x, target)
        x = self.softmax(x)

        return x if target is None else (x, loss)