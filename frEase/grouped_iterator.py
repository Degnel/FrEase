import torch

class GroupedIterator:
    def __init__(self, dataloader, group_size):
        """
        Itérateur qui regroupe les éléments d'un DataLoader par groupes de taille group_size.

        Args:
            dataloader: Un DataLoader PyTorch.
            group_size (int): Le nombre d'éléments à regrouper (typiquement, une puissance de 2 multipliée par la batch_size initiale).
        """
        self.dataloader = dataloader
        self.group_size = group_size
        self.iterator = iter(dataloader)
        self.batch_size = None

    def __iter__(self):
        return self

    def __next__(self):
        inputs = []
        outputs = []
        for _ in range(self.group_size):
            try:
                x, y = next(self.iterator)
                inputs.append(x)
                outputs.append(y)
            except StopIteration:
                break
        if not self.batch_size:
            self.batch_size = self.group_size * len(inputs[0])
        if inputs and outputs:
            concat_inputs = torch.cat(inputs, dim=0)
            concat_outputs = torch.cat(outputs, dim=0)
            return concat_inputs, concat_outputs
        raise StopIteration