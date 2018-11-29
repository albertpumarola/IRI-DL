from src.networks.networks import NetworkBase
import torch.nn as nn

class VGG11(NetworkBase):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self._features = self._make_layers(config)
        self._classifier = nn.Linear(512, num_classes)

        # TODO this is just an example, this should load pytorch pretrained weights instead random
        self.init_weights(self)

    def forward(self, x):
        out = self._features(x)
        out = out.view(out.size(0), -1)
        out = self._classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
