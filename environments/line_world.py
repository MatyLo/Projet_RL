import numpy as np

class LineWorldEnvDP:
    def __init__(self):
        self.S = [0, 1, 2, 3, 4]       # États
        self.A = [0, 1]                # Actions: 0=left, 1=right
        self.R = [-1.0, 0.0, 1.0]      # Récompenses possibles
        self.T = [0, 4]                # États terminaux

        # Initialisation de la matrice de transition
        self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))

        # Transitions définies (même logique que ton code)
        self.p[3, 0, 2, 1] = 1.0
        self.p[2, 0, 1, 1] = 1.0
        self.p[1, 0, 0, 0] = 1.0
        self.p[3, 1, 4, 2] = 1.0
        self.p[2, 1, 3, 1] = 1.0
        self.p[1, 1, 2, 1] = 1.0

    def get_states(self):
        return self.S

    def get_actions(self):
        return self.A

    def get_rewards(self):
        return self.R

    def get_terminal_states(self):
        return self.T

    def get_transition_probabilities(self):
        return self.p

    def reset(self) -> int:
        """Resets the environment to initial state"""
        # Start in the middle position
        self.current_position = self.size // 2
            
        return self.current_position