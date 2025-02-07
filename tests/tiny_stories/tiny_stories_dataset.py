from torch.utils.data import IterableDataset
import torch


class TinyStoriesDataset(IterableDataset):
    def __init__(self, dataset: list, sequence_length: int, max_batch_count: int):
        """
        Dataset itératif pour générer des séquences (entrée, cible).

        Args:
            dataset (list): Liste des séquences de tokens.
            sequence_length (int): Longueur des séquences à générer.
            batch_count (int): Nombre maximum de batchs à renvoyer.
        """
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.max_batch_count = max_batch_count

    def __iter__(self):
        """
        Générateur qui produit les couples (entrée, cible) séquentiellement.
        Chaque couple est constitué de deux listes de longueur sequence_length,
        en sautant de sequence_length en sequence_length dans la séquence de tokens.
        """
        batch_counter = 0
        for tokens in self.dataset:
            if len(tokens) >= self.sequence_length + 1:
                n_batches = (len(tokens) - 1) // self.sequence_length

                for i in range(n_batches):
                    if batch_counter >= self.max_batch_count:
                        return
                    start = i * self.sequence_length
                    X = tokens[start : start + self.sequence_length]
                    y = tokens[start + 1 : start + self.sequence_length + 1]
                    yield torch.tensor(X, dtype=torch.long), torch.tensor(
                        y, dtype=torch.long
                    )
                    batch_counter += 1


# from torch.utils.data import IterableDataset
# import torch


# class TinyStoriesDataset(IterableDataset):
#     def __init__(self, dataset: list, sequence_length: int, max_batch_count: int):
#         """
#         Dataset itératif pour générer des séquences (entrée, cible).

#         Args:
#             dataset (list): Liste des séquences de tokens.
#             sequence_length (int): Longueur des séquences à générer.
#             batch_count (int): Nombre maximum de batchs à renvoyer.
#         """
#         self.dataset = dataset
#         self.sequence_length = sequence_length
#         self.max_batch_count = max_batch_count

#     def __iter__(self):
#         """
#         Générateur qui produit les couples (entrée, cible) séquentiellement.
#         """
#         batch_counter = 0
#         for tokens in self.dataset:
#             if len(tokens) > self.sequence_length:
#                 for i in range(len(tokens) - self.sequence_length):
#                     if batch_counter >= self.max_batch_count:
#                         return
#                     X = tokens[i : i + self.sequence_length]
#                     y = tokens[i + 1 : i + self.sequence_length + 1]
#                     yield torch.tensor(X, dtype=torch.long), torch.tensor(
#                         y, dtype=torch.long
#                     )
#                     batch_counter += 1
