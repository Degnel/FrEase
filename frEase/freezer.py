from frEase.recipes import ProgressiveRecipes


class Freezer:
    def __init__(self, recipe: ProgressiveRecipes):
        # if model:
        #     self.model = model
        # else:
        # for _ in range(len(recipe.ml_ptr)):
        #     recipe.ml_ptr.pop(0)
        # checker s'il n'y a pas d'effet de bord en faisant de manière brutale
        # sinon on untilise la boucle ci-dessus
        recipe.ml_ptr._modules = {}
        self.recipe = recipe
        self.current_stage = 0
        # for _ in range(start_at_stage):
        #     next(self)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_stage < len(self.recipe.frozen_cubes):
            frozen_cubes = self.recipe.frozen_cubes[self.current_stage]
            frozen_cubes_len = len(frozen_cubes)
            if self.current_stage == 0:
                last_config_len = 0
            else:
                last_config_len = len(self.recipe.frozen_cubes[self.current_stage - 1])
            for i in range(last_config_len, frozen_cubes_len):
                self.recipe.ml_ptr.append(self.recipe.ice_cubes[i])

            for module, freeze in zip(self.recipe.ml_ptr, frozen_cubes):
                for param in module.parameters():
                    param.requires_grad = not freeze

            self.current_stage += 1
            return self.recipe.model
        else:
            raise StopIteration
