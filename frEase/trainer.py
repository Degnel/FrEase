import torch
import os
import torch.nn as nn
from torch.utils.data.dataloader import default_collate

# from .utils import freeze_layers, unfreeze_layers, floor_power2
from .grouped_iterator import GroupedIterator

# from .model_parser import ice_cube_dicer
from frEase.recipes import ProgressiveRecipes
from frEase.freezer import Freezer
import pickle as pkl


class ProgressiveTrainer:
    def __init__(self, recipe: ProgressiveRecipes | str):
        if isinstance(recipe, str):
            with open(recipe, "rb") as f:
                self.recipe = pkl.load(f)
        else:
            self.recipe = recipe

        assert isinstance(
            recipe, ProgressiveRecipes
        ), "recipe must be a progressive recipe or a path to a progressive recipe"
        self.freezer = Freezer(recipe)

    def train_step(self, optimizer, criterion, grouped_batches):
        """
        Effectue une passe d'entraînement sur les groupes issus du GroupedIterator.
        Si le groupe n'est pas complet (dernier groupe d'un epoch), ajuste le lr proportionnellement.
        """
        total_loss = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recipe.model.train()
        for i, group in enumerate(grouped_batches):
            expected_size = grouped_batches.group_size
            current_size = len(group)
            current_lr = optimizer.param_groups[0]["lr"]
            if current_size < expected_size:
                adjusted_lr = current_lr * (current_size / expected_size)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = adjusted_lr
                print(
                    f"  Groupe incomplet (taille {current_size}/{expected_size}) → lr ajusté à {adjusted_lr:.6f}"
                )
            batch = default_collate(group)
            optimizer.zero_grad()
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            outputs = self.recipe.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss /= i + 1
        return total_loss

    def train(
        self,
        data_loader,
        optimizer,
        criterion,
        test_loader=None,
        test_criterion=None,
        save_checkpoints=True,
        saving_path="./checkpoints",
    ):
        cycles = len(self.recipe.frozen_cubes)
        if save_checkpoints:
            file_path = os.path.join(saving_path, "recipe.pkl")
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)
            with open(file_path, "wb") as fichier:
                pkl.dump(self.recipe, fichier)

        for cycle in range(cycles):
            next(self.freezer)
            num_epoch = self.recipe.epochs[cycle]
            print(f"--- Cycle {cycle+1}/{cycles} ---")
            loss = 0
            for epoch in range(num_epoch):
                current_lr = self.recipe.lr[cycle][epoch]
                current_gs = self.recipe.group_size[cycle][epoch]
                optim = optimizer(self.recipe.model.parameters(), current_lr)
                print(
                    f"Epoch {epoch+1}/{num_epoch} - group_size effectif: {current_gs}, lr: {current_lr:.6f}"
                )
                grouped_batches = GroupedIterator(data_loader, group_size=current_gs)
                loss += self.train_step(optim, criterion, grouped_batches)

            loss /= num_epoch
            print(f"Loss: {loss:.4f}")

            if save_checkpoints:
                file_name = f"checkpoint_{cycle}.pt"
                file_path = os.path.join(saving_path, file_name)
                self.save_architecture(file_path)

            if test_loader:
                if test_criterion:
                    test_loss = self.test_model(test_loader, test_criterion)
                else:
                    self.test_model(test_loader, criterion)
                print(f"Test - Loss: {test_loss:.4f}")

        return

    def test_model(self, test_loader, criterion):
        """
        Teste le modèle sur un jeu de test et affiche la perte moyenne et la précision (pour la classification).
        On suppose que le batch est un dictionnaire contenant 'input' et 'target'.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recipe.model.to(device)
        self.recipe.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                outputs = self.recipe.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

        total_loss /= i + 1
        return total_loss

    def save_architecture(self, file_path: str):
        """
        Sauvegarde l'état du modèle dans un fichier via torch.save.
        """
        torch.save(self.recipe.model.state_dict(), file_path)
        print(f"Architecture sauvegardée dans {file_path}")

    def load_architecture(self, file_path: str, stage: int):
        """
        Recharge l'état du modèle à partir d'un fichier via torch.load.
        """
        state = torch.load(file_path)
        for _ in range(stage):
            next(self.freezer)
        self.recipe.model.load_state_dict(state)
        print(f"Architecture rechargée depuis {file_path}")
