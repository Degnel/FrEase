import torch.nn as nn

class IceCube(nn.Module):
    def __init__(self, module: nn.Module, freeze_others: bool = True):
        """
        Un bloc d'architecture à ajouter progressivement.
        
        Args:
            module (nn.Module): Le sous-module (bloc) à encapsuler.
            freeze_others (bool): Si True, lors de l'ajout de ce bloc, on peut geler les autres parties du modèle.
        """
        super(IceCube, self).__init__()
        self.module = module
        self.freeze_others = freeze_others

    def forward(self, x):
        return self.module(x)
