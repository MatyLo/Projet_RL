# Scripts de Test pour Monty Hall avec PyGame

Ce dossier contient plusieurs scripts pour tester et visualiser les algorithmes de reinforcement learning sur l'environnement Monty Hall avec PyGame.

## Scripts Disponibles

### 1. Tests d'Algorithmes

#### `test_montyhall_policy_iteration.py`
- **Description**: Test de Policy Iteration sur Monty Hall avec visualisation PyGame
- **Fonctionnalités**:
  - Entraînement de l'algorithme Policy Iteration
  - Visualisation de l'entraînement
  - Test de la politique apprise
  - Affichage des résultats et statistiques
- **Contrôles**: ESPACE (pause), ENTRÉE (étape), ÉCHAP (quitter)

#### `test_montyhall_value_iteration.py`
- **Description**: Test de Value Iteration sur Monty Hall avec visualisation PyGame
- **Fonctionnalités**:
  - Entraînement de l'algorithme Value Iteration
  - Visualisation de l'entraînement
  - Test de la politique optimale
  - Affichage des résultats et statistiques
- **Contrôles**: ESPACE (pause), ENTRÉE (étape), ÉCHAP (quitter)

### 2. Visualiseurs d'Épisodes

#### `monty_hall_episode_viewer.py`
- **Description**: Visualiseur étape par étape des épisodes
- **Fonctionnalités**:
  - Contrôle manuel du déroulement des épisodes
  - Comparaison entre agent entraîné et agent aléatoire
  - Affichage détaillé de chaque étape
  - Statistiques en temps réel
- **Contrôles**: ENTRÉE (étape suivante), ESPACE (pause), ÉCHAP (quitter)

#### `monty_hall_continuous_viewer.py`
- **Description**: Visualiseur continu de plusieurs épisodes
- **Fonctionnalités**:
  - Visualisation automatique de plusieurs épisodes
  - Contrôles de vitesse
  - Statistiques en temps réel
  - Comparaison avec la théorie
- **Contrôles**: ESPACE (pause), + (vitesse +), - (vitesse -), ÉCHAP (quitter)

### 3. Démonstrations

#### `monty_hall_demo.py`
- **Description**: Démonstration interactive complète
- **Fonctionnalités**:
  - Menu interactif
  - Possibilité de jouer soi-même
  - Comparaison avec l'agent
  - Statistiques détaillées

## Comment Utiliser

### Installation des Dépendances
```bash
pip install pygame numpy
```

### Exécution des Scripts
```bash
# Test Policy Iteration
python experiments/test_montyhall_policy_iteration.py

# Test Value Iteration
python experiments/test_montyhall_value_iteration.py

# Visualiseur étape par étape
python experiments/monty_hall_episode_viewer.py

# Visualiseur continu
python experiments/monty_hall_continuous_viewer.py

# Démonstration interactive
python experiments/monty_hall_demo.py
```

## Contrôles PyGame

### Contrôles Généraux
- **ÉCHAP**: Quitter le programme
- **ESPACE**: Pause/Reprendre (selon le script)
- **ENTRÉE**: Passer à l'étape suivante (selon le script)

### Contrôles Spécifiques
- **+/-**: Ajuster la vitesse (visualiseur continu)
- **0, 1, 2**: Choisir une porte (démonstration interactive)
- **0, 1**: Garder/Changer de porte (démonstration interactive)

## Interprétation des Résultats

### Politique Optimale
L'algorithme apprend généralement que:
- **État 0** (choix initial): L'action n'a pas d'importance car toutes les portes sont équivalentes
- **État 1** (après révélation): **Changer** de porte est optimal (2/3 de chance de gagner)

### Fonction de Valeur
- **État 0**: V(s) ≈ 0.600 (valeur espérée du jeu)
- **État 1**: V(s) ≈ 0.667 (valeur espérée si on change)

### Performance Attendue
- **Avec changement**: ~66.67% de réussite
- **Sans changement**: ~33.33% de réussite
- **Agent aléatoire**: ~50% de réussite

## Structure de l'Environnement Monty Hall

### États
- **État 0**: Choix initial de porte
- **État 1**: Après révélation d'une chèvre
- **État 2**: Terminal - Gagné
- **État 3**: Terminal - Perdu
- **État 4**: Terminal - Abandon

### Actions
- **Actions 0, 1, 2**: Choix de porte (état 0)
- **Action 0**: Garder sa porte (état 1)
- **Action 1**: Changer de porte (état 1)

### Récompenses
- **0.0**: Perte ou étape intermédiaire
- **1.0**: Victoire

## Visualisation PyGame

### Interface
- **Portes**: Représentées par des rectangles colorés
- **Couleurs**:
  - Bleu clair: Portes fermées
  - Bleu: Porte choisie
  - Rouge: Porte ouverte (chèvre)
  - Vert: Voiture
  - Marron: Chèvre révélée

### Informations Affichées
- État actuel du jeu
- Actions prises
- Récompenses
- Statistiques
- Instructions pour l'utilisateur

## Conseils d'Utilisation

1. **Commencez par** `monty_hall_episode_viewer.py` pour comprendre le déroulement
2. **Utilisez** `monty_hall_continuous_viewer.py` pour voir les performances sur plusieurs épisodes
3. **Testez** les algorithmes individuels pour voir les différences
4. **Expérimentez** avec les contrôles pour ajuster la vitesse d'affichage

## Dépannage

### Problèmes Courants
- **Fenêtre ne s'affiche pas**: Vérifiez que PyGame est installé
- **Erreur d'import**: Vérifiez que vous êtes dans le bon répertoire
- **Performance lente**: Réduisez la vitesse ou fermez d'autres applications

### Support
Pour toute question ou problème, consultez la documentation PyGame ou les logs d'erreur dans le terminal. 