import torch


def criterion(target, pred):
    MSE = torch.nn.functional.mse_loss(target, pred, reduction="sum")
    return torch.sqrt(MSE)
