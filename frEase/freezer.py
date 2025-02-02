from torch import nn
from frEase.recipes import ProgressiveRecipes

class Freezer():
    def __init__(self, recipe: ProgressiveRecipes, model: nn.Module = None, start_at_stage: int = 0):
        if model:
            self.model = model
        else:
            recipe.ml_ptr = nn.ModuleList([])
            self.model = recipe.model
        self.recipe = recipe
        self.current_stage = start_at_stage
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_stage < len(self.recipe.frozen_cubes):
            frozen_cubes = self.recipe.frozen_cubes[self.current_stage]
            frozen_cubes_len = len(frozen_cubes)
            if self.current_stage == 0:
                last_config_len = 0
            else:
                last_config_len = len(self.recipe.frozen_cubes[self.current_stage-1])
            for i in range(last_config_len, frozen_cubes_len):
                self.recipe.ml_ptr.append(self.recipe.ice_cubes[i])

            for module, freeze in zip(self.recipe.ml_ptr, self.recipe.frozen_cubes):
                for param in module.parameters():
                    param.requires_grad = not freeze

            self.current_stage += 1
            return self.model
        else:
            raise StopIteration