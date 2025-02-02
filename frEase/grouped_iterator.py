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

    def __iter__(self):
        return self

    def __next__(self):
        group = []
        for _ in range(self.group_size):
            try:
                group.append(next(self.iterator))
            except StopIteration:
                break
        if group:
            return group
        raise StopIteration