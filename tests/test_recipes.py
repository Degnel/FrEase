from frEase.recipes import ProgressiveRecipes
from transformer.transformer import Transformer

model = Transformer(
    d_model=256,
    n_heads=6,
    d_ff=512,
    depth=8,
    vocab_size=10000,
)

transfomer_recipe = ProgressiveRecipes(model)
transfomer_recipe.progressive_simple(
    epochs=[2, 3, 4, 5, 6, 7, 8, 9, 10],
    lr=0.001,
    group_size=1,
    global_trainning=3,
    scaling_factor=1.2,
)

print(transfomer_recipe)


transfomer_recipe.iterative_freeze_defreeze(
    epochs=[2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10],
    lr=0.001,
    group_size=[1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2],
    iterations=2,
    scaling_factor=1.5,
)

print(transfomer_recipe)
