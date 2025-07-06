from typing import Tuple, Dict, Any, List
import numpy as np
from .base_environment import BaseEnvironment

class LineWorld(BaseEnvironment):
    """
    Un environnement simple où l'agent doit se déplacer sur une ligne
    pour atteindre un objectif.
    """
    
    def __init__(self, length: int = 5, start_pos: int = 0, goal_pos: int = 4):
        """
        Initialise l'environnement LineWorld.
        
        Args:
            length: Longueur de la ligne
            start_pos: Position de départ
            goal_pos: Position objectif
        """
        self.length = length
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.current_pos = start_pos
        
        # Définition des états, actions et récompenses
        self.S = list(range(length))  # États possibles
        self.A = [0, 1]  # 0: Gauche, 1: Droite
        self.R = [-1.0, 0.0, 1.0]  # Récompenses possibles
        self.T = [0, goal_pos]  # États terminaux
        
        # Initialisation de la matrice de transition
        self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))
        self._initialize_transition_matrix()
        super().__init__()
        
    def _initialize_transition_matrix(self):
        """Initialise la matrice de transition p(s, a, s', r)."""
        # Définition des transitions
        for s in range(1, self.length - 1):
            # Action gauche
            if s-1 == 0:
                self.p[s, 0, s-1, 0] = 1.0  # Déplacement à gauche vers état terminal, récompense -1
            else:
                self.p[s, 0, s-1, 1] = 1.0  # Déplacement à gauche, récompense 0
            # Action droite
            self.p[s, 1, s+1, 1] = 1.0  # Déplacement à droite, récompense 0
        
        # Cas spéciaux
        # État initial (0)
        self.p[0, 0, 0, 0] = 1.0  # Rester à gauche, récompense -1
        self.p[0, 1, 1, 1] = 1.0  # Aller à droite, récompense 0
        
        # État final (goal_pos)
        self.p[self.goal_pos, 0, self.goal_pos-1, 1] = 1.0  # Aller à gauche, récompense 0
        self.p[self.goal_pos, 1, self.goal_pos, 2] = 1.0  # Rester à droite, récompense 1
        
    def reset(self) -> int:
        """Réinitialise l'environnement à sa position de départ."""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Exécute une action dans l'environnement.
        
        Args:
            action: 0 pour gauche, 1 pour droite
            
        Returns:
            Tuple (next_state, reward, done, info)
        """
        if action not in self.A:
            raise ValueError(f"Action invalide: {action}")
        
        # Trouver la prochaine transition
        next_state_probs = self.p[self.current_pos, action].sum(axis=1)
        next_state = np.random.choice(self.S, p=next_state_probs)
        
        # Trouver la récompense
        reward_probs = self.p[self.current_pos, action, next_state]
        reward = self.R[np.random.choice(len(self.R), p=reward_probs/reward_probs.sum())]
        
        self.current_pos = next_state
        done = self.current_pos in self.T
        
        return self.current_pos, reward, done, {}
    
    def render(self) -> None:
        """Affiche l'état actuel de l'environnement."""
        line = ['-'] * self.length
        line[self.current_pos] = 'A'  # Agent
        line[self.goal_pos] = 'G'     # Goal
        print(''.join(line))
    
    def get_state_space(self) -> List[int]:
        """Retourne la liste des états possibles."""
        return self.S
    
    def get_action_space(self) -> List[int]:
        """Retourne la liste des actions possibles."""
        return self.A
    
    def get_transition_matrix(self) -> np.ndarray:
        """Retourne la matrice de transition."""
        return self.p
    
    def get_rewards(self) -> List[float]:
        """Retourne la liste des récompenses possibles."""
        return self.R
    
    def get_terminal_states(self) -> List[int]:
        """Retourne la liste des états terminaux."""
        return self.T 