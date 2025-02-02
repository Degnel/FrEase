import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
# from .utils import freeze_layers, unfreeze_layers, floor_power2
from .grouped_iterator import GroupedIterator
# from .model_parser import ice_cube_dicer
from frEase.recipes import ProgressiveRecipes
from frEase.freezer import Freezer 

class ProgressiveTrainer:
    def __init__(self, recipe: ProgressiveRecipes):
        self.recipe = recipe
        self.freezer = Freezer(recipe)
    
    def train_step(self, optimizer, criterion, grouped_batches):
        """
        Effectue une passe d'entraînement sur les groupes issus du GroupedIterator.
        Si le groupe n'est pas complet (dernier groupe d'un epoch), ajuste le lr proportionnellement.
        """
        self.recipe.model.train()
        for group in grouped_batches:
            expected_size = grouped_batches.group_size
            current_size = len(group)
            current_lr = optimizer.param_groups[0]['lr']
            if current_size < expected_size:
                adjusted_lr = current_lr * (current_size / expected_size)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = adjusted_lr
                print(f"  Groupe incomplet (taille {current_size}/{expected_size}) → lr ajusté à {adjusted_lr:.6f}")
            batch = default_collate(group)
            optimizer.zero_grad()
            X = batch[0]
            y = batch[1]
            outputs = self.recipe.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    
    def train(self, data_loader, optimizer, criterion):
        cycles = len(self.recipe.frozen_cubes)
        for cycle in range(cycles):
            num_epoch = self.recipe.epochs[cycle]
            print(f"--- Cycle {cycle+1}/{cycles} ---")
            for epoch in range(num_epoch):
                current_lr = self.recipe.lr[cycle][epoch]
                current_gs = self.recipe.group_size[cycle][epoch]
                optim = optimizer(self.recipe.model.parameters(), current_lr)
                print(f"Epoch {epoch+1}/{num_epoch} - group_size effectif: {current_gs}, lr: {current_lr:.6f}")
                grouped_batches = GroupedIterator(data_loader, group_size=current_gs)
                self.train_step(optim, criterion, grouped_batches)
    
    def test_model(self, test_loader, loss_fn):
        """
        Teste le modèle sur un jeu de test et affiche la perte moyenne et la précision (pour la classification).
        On suppose que le batch est un dictionnaire contenant 'input' et 'target'.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0  # Pour la classification
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples if total_samples > 0 else 0
        print(f"Test - Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def save_architecture(self, filepath: str):
        """
        Sauvegarde l'état du modèle dans un fichier via torch.save.
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"Architecture sauvegardée dans {filepath}")

    def load_architecture(self, filepath: str, recipe: ProgressiveRecipes, stage: int):
        """
        Recharge l'état du modèle à partir d'un fichier via torch.load.
        """
        state = torch.load(filepath)
        model = nn.Module()
        model.load_state_dict(state)
        self.freezer = Freezer(recipe, model, stage)
        print(f"Architecture rechargée depuis {filepath}")