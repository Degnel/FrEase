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


TODO:
Recipe: Faire en sorte d'ajuster le recipe si on a des layers sans poids -> dans ce cas il faut les ajouter avec le layer précédent car il est inutile d'apprendre

WARNING: Attention le loading d'un modèle ne fonctionne probablement pas (il n'a pas été testé en condition réelle)