# Visualisation PyGame pour le Reinforcement Learning

Ce module ajoute une visualisation graphique interactive à vos environnements de reinforcement learning en utilisant PyGame.

## Installation

1. Installez PyGame :
```bash
pip install pygame>=2.0.0
```

Ou installez toutes les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Test basique
Pour tester que la visualisation fonctionne :
```bash
python experiments/test_pygame_visualization.py
```

### Visualisation complète des algorithmes
Pour voir l'entraînement et le test des algorithmes de programmation dynamique :
```bash
python experiments/pygame_dynamic_programming_example.py
```

## Contrôles

- **ESPACE** : Pause/Reprendre l'animation
- **ENTRÉE** : Mode étape par étape
- **ÉCHAP** : Quitter la visualisation

## Fonctionnalités

### Visualisation de l'entraînement
- Affichage en temps réel de l'évolution des fonctions de valeur
- Visualisation de la convergence des algorithmes
- Affichage des politiques apprises

### Visualisation du test
- Animation de l'agent utilisant la politique apprise
- Affichage des actions, récompenses et statistiques
- Contrôle de la vitesse d'animation

### Environnements supportés
- **LineWorld** : Environnement linéaire 1D
- **GridWorld** : Environnement en grille 2D

## Exemples d'utilisation

### LineWorld avec Policy Iteration
```python
from environments.line_world import LineWorld
from environments.visualization.pygame_visualizer import LineWorldVisualizer
from algorithms.dynamic_programming.policy_iteration import PolicyIteration

# Création de l'environnement et du visualiseur
env = LineWorld(length=5, start_pos=0, goal_pos=4)
visualizer = LineWorldVisualizer(env)

# Création et entraînement de l'algorithme
agent = PolicyIteration(environment=env, discount_factor=0.99)
agent.train()

# Test avec visualisation
state = env.reset()
while not done:
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)
    visualizer.render_state(state, agent.value_function)
```

### GridWorld avec Value Iteration
```python
from environments.grid_world import GridWorld
from environments.visualization.pygame_visualizer import GridWorldVisualizer
from algorithms.dynamic_programming.value_iteration import ValueIteration

# Création de l'environnement et du visualiseur
env = GridWorld(width=4, height=4)
visualizer = GridWorldVisualizer(env)

# Création et entraînement de l'algorithme
agent = ValueIteration(environment=env, discount_factor=0.99)
agent.train()

# Test avec visualisation
state = env.reset()
while not done:
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)
    visualizer.render_state(state, agent.value_function)
```

## Personnalisation

### Couleurs
Vous pouvez modifier les couleurs dans `pygame_visualizer.py` :
```python
self.RED = (255, 0, 0)      # Agent
self.GREEN = (0, 255, 0)    # Objectif
self.BLUE = (0, 0, 255)     # Actions
self.LIGHT_BLUE = (173, 216, 230)  # Fond
```

### Vitesse d'animation
Modifiez les valeurs `time.sleep()` pour ajuster la vitesse :
```python
time.sleep(0.1)  # Plus rapide
time.sleep(1.0)  # Plus lent
```

### Taille de fenêtre
Modifiez les paramètres lors de la création du visualiseur :
```python
visualizer = LineWorldVisualizer(env, width=1200, height=600)
```

## Dépannage

### Erreur "pygame module not found"
```bash
pip install pygame
```

### Fenêtre qui ne s'affiche pas
- Vérifiez que vous avez un environnement graphique
- Sur Linux, installez les paquets X11 nécessaires

### Performance lente
- Réduisez la fréquence de mise à jour (augmentez `time.sleep()`)
- Diminuez la taille de la fenêtre

## Prochaines étapes

Pour étendre la visualisation à d'autres algorithmes :
1. Créez un nouveau visualiseur pour votre environnement
2. Intégrez-le dans vos scripts d'expérimentation
3. Ajoutez des fonctionnalités spécifiques (graphiques, statistiques, etc.)

## Support

Pour toute question ou problème, consultez la documentation PyGame ou ouvrez une issue dans le projet. 