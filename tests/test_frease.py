from frEase.trainer import ProgressiveTrainer
from frEase.recipes import ProgressiveRecipes
import torch.nn as nn

# Création du modèle (par exemple un nn.Sequential ou tout autre nn.Module)
model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

# Initialisation du trainer (le modèle sera automatiquement découpé via ice_cube_dicer)
trainer = ProgressiveTrainer()

# Choix de la recette d'entraînement
training_recipe = ProgressiveRecipes.progressive_simple(model, epochs, scaling_factor, lr, bs, optimizer, criterion)
# On découpe le modèle
# On met les lr et bs sous la bonne forme
# On setup les optimizers

# Définition des autres éléments nécessaires à l'entraînement
# data_loader, optimizer et loss_fn doivent être définis par l'utilisateur.
epochs = 5                      # ou une liste d'entiers pour plusieurs cycles
scaling_factor = 1.3
lr = 0.0001                     # peut être une valeur unique, une liste ou une liste de listes
bs = 32                         # même logique pour la batch_size de base

# Lancement de l'entraînement
trainer.train(data_loader, training_recipe)