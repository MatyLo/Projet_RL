import numpy as np

class GridWorldEnvDP:
    def __init__(self):
        self.S = list(range(25))  # États : 5x5 grille
        self.A = [0, 1, 2, 3]      # Actions : Gauche, Droite, Haut, Bas
        self.R = [-3.0, 0.0, 1.0]  # Récompenses possibles
        self.T = [4, 24]           # États terminaux
        self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))
        self._init_transitions()

    def _init_transitions(self):
        # Transitions codées à la main depuis ton code (collé directement ici)
        p = self.p

        def set(s, a, sp, r_idx): p[s, a, sp, r_idx] = 1.0

        set(0, 0, 0, 1); set(0, 1, 1, 1); set(0, 2, 0, 1); set(0, 3, 5, 1)
        set(1, 0, 0, 1); set(1, 1, 2, 1); set(1, 2, 1, 1); set(1, 3, 6, 1)
        set(2, 0, 1, 1); set(2, 1, 3, 1); set(2, 2, 2, 1); set(2, 3, 7, 1)
        set(3, 0, 2, 1); set(3, 1, 4, 0); set(3, 2, 3, 1); set(3, 3, 8, 1)
        set(5, 0, 5, 1); set(5, 1, 6, 1); set(5, 2, 1, 1); set(5, 3, 10, 1)
        set(6, 0, 5, 1); set(6, 1, 7, 1); set(6, 2, 1, 1); set(6, 3, 11, 1)
        set(7, 0, 6, 1); set(7, 1, 8, 1); set(7, 2, 2, 1); set(7, 3, 12, 1)
        set(8, 0, 7, 1); set(8, 1, 9, 1); set(8, 2, 3, 1); set(8, 3, 13, 1)
        set(9, 0, 8, 1); set(9, 1, 9, 1); set(9, 2, 4, 0); set(9, 3, 14, 1)
        set(10, 0, 10, 1); set(10, 1, 11, 1); set(10, 2, 5, 1); set(10, 3, 15, 1)
        set(11, 0, 10, 1); set(11, 1, 12, 1); set(11, 2, 6, 1); set(11, 3, 16, 1)
        set(12, 0, 11, 1); set(12, 1, 13, 1); set(12, 2, 7, 1); set(12, 3, 17, 1)
        set(13, 0, 12, 1); set(13, 1, 14, 1); set(13, 2, 8, 1); set(13, 3, 18, 1)
        set(14, 0, 13, 1); set(14, 1, 14, 1); set(14, 2, 9, 1); set(14, 3, 19, 1)
        set(15, 0, 15, 1); set(15, 1, 16, 1); set(15, 2, 10, 1); set(15, 3, 20, 1)
        set(16, 0, 15, 1); set(16, 1, 17, 1); set(16, 2, 11, 1); set(16, 3, 21, 1)
        set(17, 0, 16, 1); set(17, 1, 18, 1); set(17, 2, 12, 1); set(17, 3, 22, 1)
        set(18, 0, 17, 1); set(18, 1, 19, 1); set(18, 2, 13, 1); set(18, 3, 23, 1)
        set(19, 0, 18, 1); set(19, 1, 19, 1); set(19, 2, 14, 1); set(19, 3, 24, 2)
        set(20, 0, 20, 1); set(20, 1, 21, 1); set(20, 2, 15, 1); set(20, 3, 20, 1)
        set(21, 0, 20, 1); set(21, 1, 22, 1); set(21, 2, 16, 1); set(21, 3, 21, 1)
        set(22, 0, 21, 1); set(22, 1, 23, 1); set(22, 2, 17, 1); set(22, 3, 22, 1)
        set(23, 0, 22, 1); set(23, 1, 24, 2); set(23, 2, 18, 1); set(23, 3, 23, 1)

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
