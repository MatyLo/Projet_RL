import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from .base_environment import BaseEnvironment

class GridWorld(BaseEnvironment):
    """
    Environnement GridWorld 2D classique.
    - Grille 5x5 par défaut
    - Actions : 0=Haut, 1=Bas, 2=Gauche, 3=Droite
    - Récompense -1 par mouvement, +10 à l'arrivée (en bas à droite)
    - États = (x, y) indexés de 0 à n-1
    """
    ACTIONS = [0, 1, 2, 3]
    ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

    def __init__(self, width: int = 5, height: int = 5):
        super().__init__("GridWorld")
        self.width = width
        self.height = height
        self.start_pos = (0, 0)
        self.goal_pos = (width-1, height-1)
        self.current_pos = None
        self.terminal_states = [self._to_state(self.goal_pos)]
        self.reset()

    @property
    def state_space_size(self) -> int:
        return self.width * self.height

    @property
    def action_space_size(self) -> int:
        return 4

    @property
    def valid_actions(self) -> List[int]:
        x, y = self.current_pos
        actions = []
        if y > 0:
            actions.append(0)  # Up
        if y < self.height - 1:
            actions.append(1)  # Down
        if x > 0:
            actions.append(2)  # Left
        if x < self.width - 1:
            actions.append(3)  # Right
        return actions

    def reset(self) -> int:
        self.current_pos = self.start_pos
        self._reset_episode_stats()
        return self._to_state(self.current_pos)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        if action not in self.valid_actions:
            raise ValueError(f"Action invalide: {action} depuis {self.current_pos}")
        x, y = self.current_pos
        if action == 0:
            y -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1
        new_pos = (x, y)
        self.current_pos = new_pos
        state = self._to_state(new_pos)
        done = (new_pos == self.goal_pos)
        reward = 10.0 if done else -1.0
        self._update_episode_stats(action, reward, state, done)
        return state, reward, done, {"pos": new_pos, "action_name": self.ACTION_NAMES[action], "done": done}

    def _to_state(self, pos: Tuple[int, int]) -> int:
        x, y = pos
        return y * self.width + x

    def _from_state(self, state: int) -> Tuple[int, int]:
        x = state % self.width
        y = state // self.width
        return (x, y)

    def render(self, mode: str = 'console') -> Optional[Any]:
        if mode == 'console':
            grid = np.full((self.height, self.width), '.', dtype=str)
            x, y = self.current_pos
            gx, gy = self.goal_pos
            grid[gy, gx] = 'G'
            grid[y, x] = 'A'
            print("\n".join([" ".join(row) for row in grid]))
            print(f"Position agent: {self.current_pos}")
        return None

    def get_state_description(self, state: int) -> str:
        x, y = self._from_state(state)
        if (x, y) == self.goal_pos:
            return f"Case arrivée ({x},{y})"
        else:
            return f"Case ({x},{y})"

    def get_state_space(self):
        return list(range(self.state_space_size))

    def get_action_space(self):
        return self.ACTIONS.copy()

    def get_rewards(self):
        return [-1.0, 10.0]

    def get_terminal_states(self):
        return self.terminal_states.copy()

    def get_transition_matrix(self):
        # Optionnel : à implémenter pour algos modèle si besoin
        return None 