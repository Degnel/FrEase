from torch import nn
from .utils import floor_power2

class ProgressiveRecipes:
    def __init__(self, model):
        self.model = model
        self.ice_cubes = self.extract_ice_cubes(model)

    def _init_params(self, epochs, lr, batch_size, scaling_factor):
        self.epochs = self.create_epochs(epochs)
        self.lr = self.create_hyperparam(lr, scaling_factor)
        self.batch_size = self.create_hyperparam(batch_size, scaling_factor)
        
    def progressive_simple(self, epochs=10, lr=0.001, batch_size=16, global_trainning=0, scaling_factor=1):

        """
        Recette "progressive_simple" :
          - Le modèle passe par chaque IceCube (chaque freezing constitue un cycle) puis une phase globale.
          - Si le modèle comporte n IceCubes, on effectue n + global_trainning cycles.
        """
        ic_len = len(self.ice_cubes)
        frozen_cubes = [[j != i for j in range(i+1)] for i in range(ic_len)]
        frozen_cubes.append([[False] * ic_len] * global_trainning)
        self.frozen_cubes = frozen_cubes
        self._init_params(epochs, lr, batch_size, scaling_factor)


    def iterative_freeze_defreeze(self, epochs, lr, batch_size, iterations=1, scaling_factor=1):
        """
        Recette "iterative_freeze_defreeze" :
          - Le modèle est complet dès le départ, mais les IceCube ne sont dégelé qu'un part un.
          - Si le modèle comporte n IceCubes, on effectue n * iterations cycles.
        """
        ic_len = len(self.ice_cubes)
        frozen_cubes = [[i == j for j in range(ic_len)] for i in range(ic_len)] * iterations
        self.frozen_cubes = frozen_cubes
        self._init_params(epochs, lr, batch_size, scaling_factor)

    def create_hyperparam(self, param, scaling_factor=1):
        """
        Génère la structure de lr ou batch_size selon le format donné.
        """
        length = len(self.frozen_cubes)
        if isinstance(param, (int, float)):
            return [[param * floor_power2(scaling_factor ** j) for j in range(self.epochs[i])] for i in range(length)]
        elif isinstance(param, list):
            if all(isinstance(sublist, list) for sublist in param):
                assert len(param) == length and all(len(sublist) == self.epochs[i] for i, sublist in enumerate(param))
                return param
            else:
                assert len(param) == length
                return [[param[i] * floor_power2(scaling_factor ** j) for j in range(self.epochs[i])] for i in range(length)]
        else:
            raise ValueError("lr and batch_size must be integers, floats, lists or lists of lists.")
    
    def create_epochs(self, epochs):
        """
        Génère la structure des epochs selon le format donné.
        """
        if isinstance(epochs, int):
            return [epochs] * len(self.frozen_cubes)
        elif isinstance(epochs, list):
            assert isinstance(epochs, list) and len(epochs) == len(self.frozen_cubes), "epochs must be the same size as frozen cubes"
            return epochs
        else:
            raise ValueError("epochs must be an integer or a list of integers.")
    
    def extract_ice_cubes(self, model):
        # Vérification des enfants directs
        for name, module in model.named_children():
            if isinstance(module, nn.ModuleList):
                return [(f"{name}.{i}", sub_module)
                        for i, sub_module in enumerate(module)]
        
        # Sinon, recherche dans le premier niveau d'imbrication
        for name, module in model.named_children():
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, nn.ModuleList):
                    full_name = f"{name}.{sub_name}"
                    return [(f"{full_name}.{i}", sub_module_i)
                            for i, sub_module_i in enumerate(sub_module)]
        
        # Aucun ModuleList trouvé
        return []
    
    def __repr__(self):
        return f"""
            ice_cubes: {self.ice_cubes}
            epochs: {self.epochs}
            lr: {self.lr}
            batch_size: {self.batch_size}
            frozen_cubes: {self.frozen_cubes}
        """
