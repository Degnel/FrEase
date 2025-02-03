import math
import torch.nn as nn


def floor_power2(n: float) -> int:
    """
    Renvoie la plus grande puissance de 2 inférieure ou égale à n.
    Par exemple, floor_power2(35) renverra 32.
    """
    return 2 ** int(math.floor(math.log(n, 2))) if n > 0 else 0


def freeze_layers(model: nn.Module, except_layers: list = None):
    """
    Gèle les paramètres du modèle sauf pour ceux dont le nom contient un des éléments de except_layers.
    """
    for name, param in model.named_parameters():
        if except_layers and any(except_name in name for except_name in except_layers):
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_layers(model: nn.Module):
    """
    Décongèle tous les paramètres du modèle.
    """
    for param in model.parameters():
        param.requires_grad = True
