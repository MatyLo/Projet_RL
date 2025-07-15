# Nouvelle Interface Interactive Monty Hall

## 🎮 Fonctionnalités

La nouvelle interface PyGame offre une expérience utilisateur moderne et intuitive pour le problème de Monty Hall :

### 1. **Sélection d'Environnement**
- **Monty Hall 1** : Version classique avec 3 portes
- **Monty Hall 2** : Version étendue avec 5 portes et étapes multiples

### 2. **Choix du Mode de Jeu**
- **Mode Humain** : Jouez vous-même en cliquant sur les portes
- **Mode Agent** : Laissez l'IA jouer pour vous

### 3. **Sélection d'Algorithme** (Mode Agent uniquement)
- **Q-Learning** : Apprentissage par renforcement avec exploration
- **SARSA** : State-Action-Reward-State-Action
- **Value Iteration** : Itération de valeurs pour politique optimale

## 🚀 Comment Lancer

### Lancement Simple
```bash
python src/visualization/run_monty_hall.py
```

### Tests
```bash
python test_new_interface.py
```

## 🎯 Workflow Utilisateur

1. **Écran d'accueil** : Sélectionnez l'environnement (Monty Hall 1 ou 2)
2. **Choix du mode** : 
   - Cliquez sur "MODE HUMAIN" pour jouer vous-même
   - Cliquez sur "MODE AGENT" pour l'IA
3. **Sélection d'algorithme** (si mode agent) : Choisissez l'algorithme d'apprentissage
4. **Jeu** : Observez l'agent jouer ou jouez vous-même

## 🎨 Interface Utilisateur

### Design Moderne
- **Fond dégradé** : Interface visuellement attrayante
- **Boutons interactifs** : Effets de survol et couleurs dynamiques
- **Navigation intuitive** : Boutons "Retour" et raccourcis clavier (Échap)

### Informations en Temps Réel
- **Statut de l'épisode** : Numéro d'épisode actuel
- **Compteur de victoires** : Suivi des performances
- **Mode et algorithme** : Affichage des paramètres actuels

## 🔧 Architecture Technique

### Structure des Fichiers
```
src/
├── visualization/
│   ├── monty_hall_visualizer.py    # Interface PyGame principale
│   └── run_monty_hall.py          # Script de lancement
├── rl_environments/
│   ├── monty_hall_interactive.py   # Monty Hall 1
│   └── monty_hall2_stepbystep.py   # Monty Hall 2
└── rl_algorithms/
    ├── q_learning.py              # Algorithme Q-Learning
    ├── sarsa.py                   # Algorithme SARSA
    └── value_iteration.py         # Algorithme Value Iteration
```

### Fonctionnalités Clés

#### Détection Automatique des Algorithmes
```python
def get_available_algorithms(self) -> List[str]:
    """Détecte automatiquement les algorithmes disponibles."""
    # Import dynamique et détection des classes disponibles
```

#### Menus Interactifs
- **Menu principal** : Sélection mode humain/agent
- **Menu algorithmes** : Choix de l'algorithme d'apprentissage
- **Menu environnements** : Sélection Monty Hall 1 ou 2

#### Gestion des États
- **États de jeu** : Gestion des différentes phases du jeu
- **États d'interface** : Navigation entre les menus
- **États d'agent** : Entraînement et sélection d'actions

## 🎯 Exemples d'Utilisation

### Mode Humain
1. Lancez le script
2. Sélectionnez l'environnement
3. Choisissez "MODE HUMAIN"
4. Jouez en cliquant sur les portes

### Mode Agent
1. Lancez le script
2. Sélectionnez l'environnement
3. Choisissez "MODE AGENT"
4. Sélectionnez l'algorithme (Q-Learning, SARSA, ou Value Iteration)
5. Observez l'agent jouer automatiquement

## 🔍 Détails Techniques

### Algorithmes Supportés
- **Q-Learning** : 1000 épisodes d'entraînement pour MH1, 2000 pour MH2
- **SARSA** : 1000 épisodes d'entraînement pour MH1, 2000 pour MH2
- **Value Iteration** : Convergence automatique

### Paramètres par Défaut
- **Learning Rate** : 0.1
- **Gamma (discount factor)** : 0.9
- **Epsilon (exploration)** : 0.1
- **Nombre d'épisodes de jeu** : 3

### Gestion des Erreurs
- **Import dynamique** : Gestion gracieuse des modules manquants
- **Fallback** : Valeurs par défaut en cas d'erreur
- **Fermeture propre** : Gestion des événements de fermeture

## 🚀 Améliorations Futures

### Fonctionnalités Possibles
- [ ] Sauvegarde/chargement des agents entraînés
- [ ] Comparaison de performances entre algorithmes
- [ ] Paramètres configurables (learning rate, epsilon, etc.)
- [ ] Visualisation des matrices Q-values
- [ ] Mode compétition entre agents

### Optimisations
- [ ] Interface plus fluide avec animations
- [ ] Support pour d'autres environnements
- [ ] Export des résultats en CSV/JSON
- [ ] Mode batch pour tests automatisés

## 📝 Notes de Développement

### Changements Majeurs
1. **Refactoring complet** : Séparation claire des responsabilités
2. **Interface unifiée** : Un seul script pour tous les environnements
3. **Détection automatique** : Plus besoin de spécifier les algorithmes manuellement
4. **UX améliorée** : Navigation intuitive et design moderne

### Compatibilité
- **Python 3.7+** : Compatible avec les versions récentes
- **Pygame 2.0+** : Interface graphique moderne
- **Systèmes** : Windows, macOS, Linux

---

**Développé avec ❤️ pour l'apprentissage par renforcement** 