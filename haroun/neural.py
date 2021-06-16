import torch


class ConvPool(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(ConvPool, self).__init__()
        self.conv = torch.nn.Conv2d(in_features, out_features, kernel_size=3,
                                    stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.norm = torch.nn.BatchNorm2d(out_features)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv(X)
        X = self.relu(X)
        X = self.norm(X)
        X = self.pool(X)
        return X


class Conv(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(ConvPool, self).__init__()
        self.conv = torch.nn.Conv2d(in_features, out_features, kernel_size=3,
                                    stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.norm = torch.nn.BatchNorm2d(out_features)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv(X)
        X = self.relu(X)
        X = self.norm(X)
        return X


def downward_layers(channels):
    layers = torch.nn.ModuleList()
    layers.append(torch.nn.BatchNorm2d(channels[0], affine=False))
    input_channels = channels[:-1]
    output_channels = channels[1:]
    total = len(channels) - 1
    features = [(input_channels[i], output_channels[i]) for i in range(total)]
    for feature in features:
        inp, out = feature
        layers.append(ConvPool(in_features=inp, out_features=out))
    # layers = torch.nn.Sequential(*layers)
    return layers


def dense_layers(input, output, div):
    channels = []
    layers = torch.nn.ModuleList()
    while input > output:
        channels.append(input)
        input = input // div
    channels.append(output)
    input_channels = channels[:-1]
    output_channels = channels[1:]
    total = len(channels) - 1
    features = [(input_channels[i], output_channels[i]) for i in range(total)]
    for feature in features:
        inp, outp = feature
        layers.append(torch.nn.Linear(in_features=inp, out_features=outp))
        if feature != features[-1]:
            layers.append(torch.nn.BatchNorm1d(outp))
        else:
            pass
    # layers = torch.nn.Sequential(*layers)
    return layers


class Network(torch.nn.Module):
    def __init__(self, downward, output, div, activation):
        super(Network, self).__init__()
        self.downward = downward
        self.output = output
        self.div = div
        self.activation = activation

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = downward_layers(self.downward)(X)
        X = X.reshape(X.size(0), -1)
        X = dense_layers(X.size(1), self.output, self.div)(X)
        if self.activation == "softmax":
            X = torch.nn.functional.softmax(X, dim=0)
        elif self.activation == "elu":
            X = torch.nn.functional.elu(X, alpha=1.0, inplace=False)
        return X
