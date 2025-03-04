import time
import torch
import os
from frEase.grouped_iterator import GroupedIterator
from frEase.recipes import ProgressiveRecipes
from frEase.freezer import Freezer
from frEase.thermal_camera import ThermalCamera
import pickle as pkl


class ProgressiveTrainer:
    def __init__(self, recipe: ProgressiveRecipes | str):
        if isinstance(recipe, str):
            with open(recipe, "rb") as f:
                self.recipe = pkl.load(f)
        else:
            self.recipe = recipe

        assert isinstance(recipe, ProgressiveRecipes), (
            "recipe must be a progressive recipe or a path to a progressive recipe"
        )
        self.freezer = Freezer(recipe)
        self.time_tracking = []

    def train_step(
        self, optimizer, criterion, grouped_batches: GroupedIterator, show_batch_score
    ):
        """
        Effectue une passe d'entraînement sur les groupes issus du GroupedIterator.
        Si le groupe n'est pas complet (dernier groupe d'un epoch), ajuste le lr proportionnellement.
        """
        total_loss = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recipe.model.train()
        self.recipe.model.to(device)
        for i, (inputs, targets) in enumerate(grouped_batches):
            expected_size = grouped_batches.batch_size
            current_size = inputs.shape[0]
            if current_size < expected_size:
                current_lr = optimizer.param_groups[0]["lr"]
                adjusted_lr = current_lr * (current_size / expected_size)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = adjusted_lr
                # print(
                #     f"  Groupe incomplet (taille {current_size}/{expected_size}) → lr ajusté à {adjusted_lr:.6f}"
                # )
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.recipe.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.recipe.model.parameters(), 1)
            optimizer.step()
            total_loss += loss.item()
            if show_batch_score:
                print(f"  Batch {i + 1} - Loss: {loss.item():.4f}")

        total_loss /= i + 1
        return total_loss

    def train(
        self,
        data_loader,
        optimizer,
        criterion,
        test_loader=None,
        test_criterion=None,
        show_batch_score=False,
        save_checkpoints=True,
        checkpoints_saving_path="./checkpoints",
        results_saving_name="./results/time_tracking.pkl",
    ):
        cycles = len(self.recipe.frozen_cubes)
        if save_checkpoints:
            file_path = os.path.join(checkpoints_saving_path, "recipe.pkl")
            if not os.path.exists(checkpoints_saving_path):
                os.makedirs(checkpoints_saving_path)
            with open(file_path, "wb") as fichier:
                pkl.dump(self.recipe, fichier)

        thermal_camera = ThermalCamera()
        start_time = time.time()

        for cycle in range(cycles):
            next(self.freezer)
            num_epoch = self.recipe.epochs[cycle]
            print(f"--- Cycle {cycle + 1}/{cycles} ---")
            thermal_camera.display_network_state(self.recipe.frozen_cubes[cycle])
            optim = optimizer(self.recipe.model.parameters())

            for epoch in range(num_epoch):
                current_lr = self.recipe.lr[cycle][epoch]
                for param_group in optim.param_groups:
                    param_group["lr"] = current_lr
                current_gs = self.recipe.group_size[cycle][epoch]
                grouped_batches = GroupedIterator(data_loader, group_size=current_gs)
                # print(
                #     f"Epoch {epoch+1}/{num_epoch} - group_size effectif: {current_gs}, lr: {current_lr:.6f}"
                # )
                loss = self.train_step(
                    optim, criterion, grouped_batches, show_batch_score
                )

                elapsed_time = time.time() - start_time
                test_loss = None
                if test_loader is not None:
                    test_loss = self.test_model(
                        test_loader, test_criterion or criterion, show_batch_score
                    )
                    print(
                        f"Epoch: {epoch + 1}/{num_epoch} - Loss: {loss:.4f} - Test: {test_loss:.4f} - Time: {elapsed_time:.2f}s"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}/{num_epoch} - Loss {loss:.4f} - Time: {elapsed_time:.2f}s"
                    )

                self.time_tracking.append(
                    {
                        "epoch": epoch + 1,
                        "cycle": cycle + 1,
                        "time": elapsed_time,
                        "loss": loss,
                        "test_loss": test_loss,
                    }
                )

            if save_checkpoints:
                self.save_architecture(checkpoints_saving_path, cycle)

        dir_name = os.path.dirname(results_saving_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(results_saving_name, "wb") as f:
            pkl.dump(self.time_tracking, f)

        if test_loader is None:
            return loss
        else:
            return loss, test_loss

    def test_model(self, test_loader, criterion, show_batch_score):
        """
        Teste le modèle sur un jeu de test et affiche la perte moyenne et la précision (pour la classification).
        On suppose que le batch est un dictionnaire contenant 'input' et 'target'.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recipe.model.to(device)
        self.recipe.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = outputs.to(device)
                outputs = self.recipe.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                if show_batch_score:
                    print(f"  Test Batch {i + 1} - Loss: {loss.item():.4f}")

        total_loss /= i + 1
        return total_loss

    def save_architecture(self, path: str, stage: int):
        """
        Sauvegarde l'état du modèle dans un fichier via torch.save.
        """

        file_name = f"checkpoint_{stage}.pt"
        file_path = os.path.join(path, file_name)
        params = (
            {
                k: v
                for k, v in self.recipe.model.state_dict().items()
                if self.recipe.model.get_parameter(k).requires_grad
            }
            if stage > 0
            else self.recipe.model.state_dict()
        )
        torch.save(params, file_path)
        print(f"Architecture sauvegardée dans {file_path}")

    def load_architecture(self, path: str, stage: int):
        """
        Recharge l'état du modèle à partir d'un fichier via torch.load.
        """
        self.freezer = Freezer(self.recipe)
        for s in range(stage):
            next(self.freezer)
            file_name = f"checkpoint_{s}.pt"
            file_path = os.path.join(path, file_name)
            new_params = torch.load(file_path)
            current_params = self.recipe.model.state_dict()
            params = {**current_params, **new_params}
            self.recipe.model.load_state_dict(params)
            print(f"Architecture rechargée depuis {file_path}")