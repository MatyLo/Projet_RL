# Projet de Renforcement Learning

Ce projet implémente plusieurs algorithmes de renforcement learning classiques et les applique à différents environnements.

## Structure du Projet

```
l_project/
├── algorithms/                 # Bibliothèque d'algorithmes
│   ├── dynamic_programming/    # Algorithmes de programmation dynamique
│   ├── monte_carlo/           # Méthodes Monte Carlo
│   ├── temporal_difference/   # Méthodes de différence temporelle
│   └── planning/              # Algorithmes de planification
├── environments/              # Bibliothèque d'environnements
│   └── visualization/         # Modules de visualisation
├── experiments/               # Scripts d'expérimentation
├── saved_models/              # Modèles sauvegardés
├── notebooks/                 # Notebooks Jupyter
└── tests/                     # Tests unitaires
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Les exemples d'utilisation se trouvent dans le dossier `notebooks/`.

## Tests

Pour exécuter les tests :

```bash
python -m pytest tests/
``` 