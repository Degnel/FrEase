from .trainer import ProgressiveTrainer
from .utils import unfreeze_layers

class ProgressiveRecipes:
    @staticmethod
    def progressive_simple(trainer: ProgressiveTrainer, data_loader, optimizer, loss_fn,
                           epochs, scaling_factor, lr, bs):
        """
        Recette "progressive_simple" :
          - Le modèle passe par chaque IceCube (chaque freezing constitue un cycle) puis une phase globale.
          - Par exemple, si le modèle comporte n IceCubes, on effectue n+1 cycles.
        """
        # Pour chaque IceCube, on entraîne le modèle dans un cycle de freezing
        for name, _ in trainer.icecube_list:
            print(f"Cycle de freezing pour l'IceCube {name}")
            trainer.train(data_loader, optimizer, loss_fn, epochs, scaling_factor, lr, bs, lambda *args: None)
            # Ici, le lambda est un placeholder puisque la dynamique se réinitialise à chaque cycle.
            # On suppose que chaque appel à trainer.train réalise le cycle de freezing.
        # Phase globale : décongélation complète
        print("Phase globale (décongélation complète du modèle)")
        unfreeze_layers(trainer.model)
        trainer.train(data_loader, optimizer, loss_fn, epochs, scaling_factor, lr, bs, lambda *args: None)

    @staticmethod
    def iterative_freeze_defreeze(trainer: ProgressiveTrainer, data_loader, optimizer, loss_fn,
                                  iterations, epochs, scaling_factor, lr, bs):
        """
        Recette "iterative_freeze_defreeze" :
          - On suppose que tous les IceCubes ont déjà été ajoutés.
          - On répète 'iterations' fois le cycle suivant :
              - Gel de tous les IceCubes sauf le premier, puis entraînement.
          - Ensuite, une phase globale de décongélation.
        """
        if not trainer.icecube_list:
            print("Aucun IceCube n'a été ajouté au modèle.")
            return
        for i in range(iterations):
            print(f"Cycle itératif {i+1}/{iterations} : Geler tous les blocs sauf le premier")
            first_cube_name = trainer.icecube_list[0][0]
            from .utils import freeze_layers
            freeze_layers(trainer.model, except_layers=[first_cube_name])
            trainer.train(data_loader, optimizer, loss_fn, epochs, scaling_factor, lr, bs, lambda *args: None)
        print("Phase globale après cycles itératifs (décongélation complète du modèle)")
        from .utils import unfreeze_layers
        unfreeze_layers(trainer.model)
        trainer.train(data_loader, optimizer, loss_fn, epochs, scaling_factor, lr, bs, lambda *args: None)