import torch


class ConvPool(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(ConvPool, self).__init__()
        self.conv = torch.nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1)
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
        self.conv = torch.nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.norm = torch.nn.BatchNorm2d(out_features)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv(X)
        X = self.relu(X)
        X = self.norm(X)
        return X
