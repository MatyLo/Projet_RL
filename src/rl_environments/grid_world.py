import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from .base_environment import BaseEnvironment

class GridWorld(BaseEnvironment):
    """
    Environnement GridWorld

    Environnement compatible avec les différents types d'algorithmes,
    qu'ils soient basés sur un modèle (Policy/Value Iteration) ou par expérience (Q-Learning, SARSA).

    Spécifications :
    - Grille : 5x5 par défaut.
    - Actions : 0=Haut, 1=Bas, 2=Gauche, 3=Droite.
    - Récompenses : +1.0 sur la case d'arrivée (G), -3.0 sur la case piège (L), 0.0 sinon.
    - Comportement : Tenter de sortir de la grille maintient l'agent sur place.
    """
    ACTIONS = [0, 1, 2, 3]
    ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

    def __init__(self, width: int = 5, height: int = 5):
        """Initialise l'environnement GridWorld."""
        super().__init__("GridWorld")
        self.width = width
        self.height = height

        # Définition positions clés
        self.start_pos = (0, 0)
        self.goal_pos = (self.width - 1, self.height - 1)  # (4, 4) -> "Tout en bas à droite"
        self.losing_pos = (self.width - 1, 0)             # (4, 0) -> "Tout en haut à droite"

        self.current_pos = None

        # Coordonnées des états qui terminent un épisode
        self.terminal_positions = [self.goal_pos, self.losing_pos]

        self.reset()

    # --- Propriétés Abstraites de BaseEnvironment ---

    @property
    def state_space_size(self) -> int:
        """Retourne la taille de l'espace d'états."""
        return self.width * self.height

    @property
    def action_space_size(self) -> int:
        """Retourne la taille de l'espace d'actions."""
        return len(self.ACTIONS)

    @property
    def valid_actions(self) -> List[int]:
        """Retourne les actions valides. Dans ce GridWorld, toutes sont toujours possibles."""
        return self.ACTIONS.copy()

    # --- Méthodes Abstraites BaseEnvironment ---

    def reset(self) -> int:
        """Remet l'environnement à sa position de départ."""
        self.current_pos = self.start_pos
        self._reset_episode_stats()
        return self._to_state(self.current_pos)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Exécute une action et retourne le résultat de la transition."""
        if self.current_pos in self.terminal_positions:
            return self._to_state(self.current_pos), 0.0, True, {"info": "Episode is already finished."}

        # Calcule la nouvelle position en restant dans les limites de la grille
        x, y = self.current_pos
        if action == 0:    # Haut
            y = max(0, y - 1)
        elif action == 1:  # Bas
            y = min(self.height - 1, y + 1)
        elif action == 2:  # Gauche
            x = max(0, x - 1)
        elif action == 3:  # Droite
            x = min(self.width - 1, x + 1)
        new_pos = (x, y)

        # Mise à jour de l'état
        self.current_pos = new_pos
        next_state = self._to_state(new_pos)

        # Détermination de la récompense et de la fin de l'épisode
        done = (new_pos in self.terminal_positions)
        if new_pos == self.goal_pos:
            reward = 1.0
        elif new_pos == self.losing_pos:
            reward = -3.0
        else:
            reward = 0.0

        # Mise à jour des stats et infos
        self._update_episode_stats(action, reward, next_state, done)
        info = {
            "pos": new_pos,
            "action_name": self.ACTION_NAMES[action],
            "done": done,
            "target_reached": new_pos == self.goal_pos
        }
        return next_state, reward, done, info

    def render(self, mode: str = 'console') -> Optional[Dict[str, Any]]:
        """Affiche une représentation de la grille."""
        if mode == 'console':
            grid = np.full((self.height, self.width), '.', dtype=str)
            gx, gy = self.goal_pos
            lx, ly = self.losing_pos
            ax, ay = self.current_pos

            grid[gy, gx] = 'G'  # Goal
            grid[ly, lx] = 'L'  # Losing
            grid[ay, ax] = 'A'  # Agent

            print("\n".join([" ".join(row) for row in grid]))
            print(f"Agent à la position : {self.current_pos}, État : {self._to_state(self.current_pos)}")
            
        elif mode == 'pygame':
            # Retourne les données nécessaires pour le rendu PyGame
            return self._get_pygame_render_data()
        else:
            raise ValueError(f"Mode de rendu non supporté: {mode}")
        
        return None

    def get_state_description(self, state: int) -> str:
        """Retourne une description d'un état donné."""
        pos = self._from_state(state)
        if pos == self.goal_pos:
            return f"Case d'arrivée {pos}"
        elif pos == self.losing_pos:
            return f"Case piège {pos}"
        else:
            return f"Case normale {pos}"

    # --- Méthodes pour Algorithmes Basés sur un Modèle (Dynamic Programming) ---

    def get_transition_probabilities(self, state: int, action: int) -> Dict[int, float]:
        """
        PROGRAMMATION DYNAMIQUE
        Calcule les probabilités de transition pour un état et une action donnés.
        L'environnement étant déterministe, la probabilité est toujours de 1.0.
        """
        pos = self._from_state(state)

        if pos in self.terminal_positions:
            # Si on est sur un état terminal, on y reste pour toujours
            return {state: 1.0}

        # Calcule la position suivante
        x, y = pos
        if action == 0: y = max(0, y - 1)
        elif action == 1: y = min(self.height - 1, y + 1)
        elif action == 2: x = max(0, x - 1)
        elif action == 3: x = min(self.width - 1, x + 1)
        
        next_state = self._to_state((x, y))
        return {next_state: 1.0}

    def get_reward_function(self, state: int, action: int, next_state: int) -> float:
        """
        POUR PROGRAMMATION DYNAMIQUE
        Retourne la récompense pour une transition.
        Dans ce GridWorld, la récompense ne dépend que de l'état d'arrivée.
        """
        next_pos = self._from_state(next_state)

        if next_pos == self.goal_pos:
            return 1.0
        elif next_pos == self.losing_pos:
            return -3.0
        else:
            return 0.0
    
    def _get_pygame_render_data(self) -> Dict[str, Any]:
        """Retourne les données nécessaires pour le rendu PyGame."""
        return {
            'width': self.width,
            'height': self.height,
            'current_pos': self.current_pos,
            'goal_pos': self.goal_pos,
            'losing_pos': self.losing_pos,
            'start_pos': self.start_pos,
            'current_state': self._to_state(self.current_pos),
            'total_reward': self.total_reward,
            'episode_step': self.episode_step,
            'valid_actions': self.valid_actions,
            'action_names': self.ACTION_NAMES,
            'terminal_positions': self.terminal_positions,
            'done': self.current_pos in self.terminal_positions
        }

    # --- Méthodes Utilitaires ---

    def _to_state(self, pos: Tuple[int, int]) -> int:
        """Convertit une position (x, y) en un état entier."""
        x, y = pos
        return y * self.width + x

    def _from_state(self, state: int) -> Tuple[int, int]:
        """Convertit un état entier en une position (x, y)."""
        x = state % self.width
        y = state // self.width
        return (x, y)
    
    def get_terminal_states(self) -> List[int]:
        """Retourne la liste des ID des états terminaux."""
        return [self._to_state(pos) for pos in self.terminal_positions]