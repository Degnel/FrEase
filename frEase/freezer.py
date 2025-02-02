class Freezer():
    # Lors de l'initialisation du freezer, on lui donne un plan d'attaque (soit un model et une recipe soit juste une recipe)
    # Il donne le premier model
    # Lors de l'itération (next_stage), on copie-colle le model donné en paramètre, et on avance d'un étage dans le recipe

    # Le but de cette classe est de prendre un modèle d'analyser sa structure
    # On lui passe une recette (contenant la liste des modules et des modules à freeze)

    pass


# On me donne un model
# Je décompose son architecture
# Je trouve la liste des modules de plus haut niveau
# J'ai une méthode