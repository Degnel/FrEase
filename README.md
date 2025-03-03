IceCubes

How to use it:

training_recipe = ProgressiveRecipes(model) # this split your model main module list into ice cubes
training_recipe.progressive_simple(
    epochs, lr, group_size, global_trainning, scaling_factor
) # this gives the recipe to follow during training (typically adding layers progressively and freezing the ones that have already been trained)
trainer = ProgressiveTrainer(training_recipe) # this creates a trainer aware of the recipe (it will maybe be called 'cooker' in the future)
trainer.train(data_loader, optimizer, criterion) # this trains the model using (you need to have a data_loader giving pairs of X and y values)

You can also have the following recipe which starts with all layers and unfreeze only one layer at a time:
training_recipe.iterative_freeze_defreeze(
   epochs, lr, group_size, global_trainning, scaling_factor
)

I'm starting to love this package. It's the ultimate way to make large nn converge super fast. Probably making it reach local minima, but hell it is quick.

Les avantages du freezing :
- convergence plus rapide
- sauvegarde des poids moins lourde (car on peut se permettre de n'enregistrer que les poids entrainés à chaque étape)
- passe avant plus rapide si l'on enregistre les valeurs en sortie des icebergs (les blocs freeze au début du modèle) -> cela évite de refaire la passe avant dans les parties freeze que l'on connait déjà
- l'entrainement progressif permet d'incentiver naturellement le réseau à prédire le prochain token dès la première couche -> cela ouvre une porte royale vers l'early exiting
- il permet également une certaine forme de régularisation que je présens accélérer la convergence vers la phase de grokking

TODO:
Recipe: Faire en sorte d'ajuster le recipe si on a des layers sans poids -> dans ce cas il faut les ajouter avec le layer précédent car il est inutile d'apprendre

WARNING: Attention le loading d'un modèle ne fonctionne probablement pas (il n'a pas été testé en condition réelle)
