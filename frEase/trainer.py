import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate  # Pour reconstituer le batch
from .utils import freeze_layers, unfreeze_layers, floor_power2
from .grouped_iterator import GroupedIterator
from .model_parser import ice_cube_dicer

class ProgressiveTrainer:
    def __init__(self, base_model: nn.Module):
        """
        Initialise le trainer avec le modèle de base.
        Si un nn.Module est passé, il est automatiquement passé à travers ice_cube_dicer
        pour extraire les IceCubes.
        """
        self.model = base_model
        # Découpage automatique du modèle pour extraire les IceCubes
        self.icecube_list = ice_cube_dicer(base_model)
    
    def add_icecube(self, name: str, icecube: nn.Module, freeze_others: bool = True):
        """
        Ajoute un IceCube au modèle et, si freeze_others est True, gèle les autres parties.
        Cette méthode peut être utilisée pour ajouter des IceCubes supplémentaires.
        """
        if freeze_others:
            freeze_layers(self.model, except_layers=[name])
        self.icecube_list.append((name, icecube))
        self.model.add_module(name, icecube)
    
    def train_step(self, optimizer, loss_fn, grouped_iter):
        """
        Effectue une passe d'entraînement sur les groupes issus du GroupedIterator.
        Si le groupe n'est pas complet (dernier groupe d'un epoch), ajuste le lr proportionnellement.
        """
        self.model.train()
        for group in grouped_iter:
            expected_size = grouped_iter.group_size
            current_size = len(group)
            current_lr = optimizer.param_groups[0]['lr']
            if current_size < expected_size:
                adjusted_lr = current_lr * (current_size / expected_size)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = adjusted_lr
                print(f"  Groupe incomplet (taille {current_size}/{expected_size}) → lr ajusté à {adjusted_lr:.6f}")
            batch = default_collate(group)
            optimizer.zero_grad()
            outputs = self.model(batch['input'])
            loss = loss_fn(outputs, batch['target'])
            loss.backward()
            optimizer.step()
    
    def train(self, data_loader, optimizer, loss_fn, epochs, scaling_factor, lr, bs, training_recipe):
        """
        Méthode d'entraînement principale.
        
        Paramètres :
          - data_loader : DataLoader PyTorch.
          - optimizer : Optimiseur.
          - loss_fn : Fonction de perte.
          - epochs : soit un int (nombre d'epochs pour le cycle unique)
                     soit une liste d'int (un entier par cycle de freezing).
          - lr : peut être
                • une valeur unique (float) → planning dynamique calculé sur epochs avec scaling_factor,
                • une liste de float (pour un cycle unique) dont la longueur doit être égale à epochs,
                • ou une liste de listes de float (pour plusieurs cycles) dont la première dimension a
                  la taille du nombre de cycles et chaque sous-liste une longueur égale au nombre d'epochs du cycle.
          - bs : même logique que pour lr (mais pour la taille de batch de base).
          - training_recipe : Fonction de recette d'entraînement (par exemple ProgressiveRecipes.progressive_simple).
          
        Le nombre de cycles attendu dépend de la recette utilisée :
          - Par exemple, en progressive_simple, si le modèle comporte n IceCubes,
            on effectuera n+1 cycles (une fois par IceCube puis une phase globale).
          - En iterative_freeze_defreeze, il y aura n * iterations cycles.
        """
        # Cas où lr et bs sont des valeurs uniques : on construit des listes dynamiques
        if not isinstance(lr, list):
            # On crée une liste pour chaque epoch pour un cycle unique
            schedule_lr = [lr * (scaling_factor ** i) for i in range(epochs)]
            schedule_bs = [floor_power2(bs * (scaling_factor ** i)) for i in range(epochs)]
            cycles = 1
        # Cas où lr est une liste de nombres (cycle unique)
        elif isinstance(lr, list) and (not lr or isinstance(lr[0], (int, float))):
            if len(lr) != epochs:
                raise ValueError("La longueur de la liste lr doit être égale à epochs pour un cycle unique.")
            if not (isinstance(bs, list) and len(bs) == epochs):
                raise ValueError("La longueur de la liste bs doit être égale à epochs pour un cycle unique.")
            schedule_lr = lr
            schedule_bs = [floor_power2(x) for x in bs]
            cycles = 1
        # Cas où lr est une liste de listes (plusieurs cycles)
        elif isinstance(lr, list) and isinstance(lr[0], list):
            if not isinstance(epochs, list):
                raise ValueError("Lorsque lr est une liste de listes, epochs doit être une liste d'entiers.")
            if len(lr) != len(epochs):
                raise ValueError("Le nombre de cycles (taille de lr) doit être égal à la longueur de epochs.")
            if not (isinstance(bs, list) and isinstance(bs[0], list) and len(bs) == len(epochs)):
                raise ValueError("bs doit être une liste de listes de même longueur que lr et epochs.")
            # On s'assure que chaque sous-liste a la longueur attendue
            for i, num_epoch in enumerate(epochs):
                if len(lr[i]) != num_epoch or len(bs[i]) != num_epoch:
                    raise ValueError(f"Pour le cycle {i}, la longueur des sous-listes lr et bs doit être égale à {num_epoch}.")
            schedule_lr = lr
            schedule_bs = [[floor_power2(x) for x in sub_bs] for sub_bs in bs]
            cycles = len(epochs)
        else:
            raise ValueError("Format inattendu pour lr et bs.")
        
        # Exécution de l'entraînement en fonction du nombre de cycles
        if cycles == 1:
            print("Cycle unique d'entraînement")
            for epoch in range(epochs):
                current_lr = schedule_lr[epoch] if cycles == 1 and isinstance(schedule_lr, list) else schedule_lr[0][epoch]
                current_bs = schedule_bs[epoch] if cycles == 1 and isinstance(schedule_bs, list) else schedule_bs[0][epoch]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                print(f"Epoch {epoch+1}/{epochs} - batch_size effectif: {current_bs}, lr: {current_lr:.6f}")
                grouped_iter = GroupedIterator(data_loader, group_size=current_bs)
                self.train_step(optimizer, loss_fn, grouped_iter)
        else:
            for cycle in range(cycles):
                num_epoch = epochs[cycle]
                print(f"--- Cycle {cycle+1}/{cycles} ---")
                for epoch in range(num_epoch):
                    current_lr = schedule_lr[cycle][epoch]
                    current_bs = schedule_bs[cycle][epoch]
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                    print(f"Epoch {epoch+1}/{num_epoch} - batch_size effectif: {current_bs}, lr: {current_lr:.6f}")
                    grouped_iter = GroupedIterator(data_loader, group_size=current_bs)
                    self.train_step(optimizer, loss_fn, grouped_iter)

        # Appel de la recette en lui passant le trainer (self) et les paramètres
        training_recipe(self, data_loader, optimizer, loss_fn, epochs, scaling_factor, lr, bs)
    
    def test_model(self, test_loader, loss_fn, device='cpu'):
        """
        Teste le modèle sur un jeu de test et affiche la perte moyenne et la précision (pour la classification).
        On suppose que le batch est un dictionnaire contenant 'input' et 'target'.
        """
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

    def load_architecture(self, filepath: str, map_location=None):
        """
        Recharge l'état du modèle à partir d'un fichier via torch.load.
        """
        state = torch.load(filepath, map_location=map_location)
        self.model.load_state_dict(state)
        print(f"Architecture rechargée depuis {filepath}")
