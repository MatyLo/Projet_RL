import numpy as np
from .base_environment import BaseEnvironment

class GridWorld(BaseEnvironment):
    """
    Environnement GridWorld 4x4 classique.
    L'agent commence en haut à gauche (0,0) et doit atteindre le coin en bas à droite (3,3).
    Les actions sont : 0=haut, 1=bas, 2=gauche, 3=droite.
    Récompense -1 à chaque pas, 0 à l'arrivée.
    """
    def __init__(self, width=4, height=4, start_pos=(0,0), goal_pos=(3,3)):
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.S = [i for i in range(width * height)]
        self.A = [0, 1, 2, 3]  # haut, bas, gauche, droite
        self.R = [-1, 0]
        self.T = [self._to_state(goal_pos)]
        self.state = self._to_state(start_pos)
        self.p = self._build_transition_matrix()
        super().__init__("GridWorld")

    def _to_state(self, pos):
        return pos[0] * self.width + pos[1]

    def _to_pos(self, state):
        return (state // self.width, state % self.width)

    def _build_transition_matrix(self):
        p = np.zeros((self.width * self.height, 4, self.width * self.height, 2))
        for s in self.S:
            for a in self.A:
                pos = self._to_pos(s)
                if s in self.T:
                    p[s, a, s, 1] = 1.0  # reste sur place, reward 0
                    continue
                next_pos = list(pos)
                if a == 0:   # haut
                    next_pos[0] = max(0, pos[0] - 1)
                elif a == 1: # bas
                    next_pos[0] = min(self.height - 1, pos[0] + 1)
                elif a == 2: # gauche
                    next_pos[1] = max(0, pos[1] - 1)
                elif a == 3: # droite
                    next_pos[1] = min(self.width - 1, pos[1] + 1)
                next_state = self._to_state(tuple(next_pos))
                reward_idx = 1 if next_state in self.T else 0
                p[s, a, next_state, reward_idx] = 1.0
        return p

    def reset(self):
        self.state = self._to_state(self.start_pos)
        return self.state

    def step(self, action):
        pos = self._to_pos(self.state)
        if self.state in self.T:
            return self.state, 0.0, True, {}
        next_pos = list(pos)
        if action == 0:
            next_pos[0] = max(0, pos[0] - 1)
        elif action == 1:
            next_pos[0] = min(self.height - 1, pos[0] + 1)
        elif action == 2:
            next_pos[1] = max(0, pos[1] - 1)
        elif action == 3:
            next_pos[1] = min(self.width - 1, pos[1] + 1)
        next_state = self._to_state(tuple(next_pos))
        reward = 0.0 if next_state in self.T else -1.0
        done = next_state in self.T
        self.state = next_state
        return next_state, reward, done, {}

    def render(self):
        grid = np.full((self.height, self.width), '.', dtype=str)
        pos = self._to_pos(self.state)
        goal = self._to_pos(self.T[0])
        grid[goal] = 'G'
        grid[pos] = 'A'
        for row in grid:
            print(' '.join(row))
        print()

    def get_transition_matrix(self):
        return self.p
    
    def get_state_space(self):
        """Retourne l'espace des états."""
        return self.S
    
    def get_action_space(self):
        """Retourne l'espace des actions."""
        return self.A
    
    def get_rewards(self):
        """Retourne la liste des récompenses possibles."""
        return self.R
    
    def get_terminal_states(self):
        """Retourne la liste des états terminaux."""
        return self.T
    
    # Méthodes requises par BaseEnvironment
    @property
    def state_space_size(self) -> int:
        return len(self.S)
    
    @property
    def action_space_size(self) -> int:
        return len(self.A)
    
    @property
    def valid_actions(self):
        return self.A
    
    def get_state_description(self, state: int) -> str:
        pos = self._to_pos(state)
        return f"Position ({pos[0]}, {pos[1]})" 