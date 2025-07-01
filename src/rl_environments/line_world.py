"""
Line World Environment - Environnement en ligne simple pour l'apprentissage par renforcement.

L'agent se déplace sur une ligne de positions numérotées. L'objectif est d'atteindre
une position cible en un minimum d'étapes.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import sys
import os

# Ajouter le chemin vers la classe de base
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.rl_environments.base_environment import BaseEnvironment


class LineWorld(BaseEnvironment):
    """
    Environnement Line World - L'agent se déplace sur une ligne.
    
    L'agent peut se déplacer vers la gauche (action 0) ou vers la droite (action 1).
    L'objectif est d'atteindre la position cible.
    """
    
    # Actions possibles
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_NAMES = {ACTION_LEFT: "Left", ACTION_RIGHT: "Right"}
    
    def __init__(self, 
                 line_length: int = 7,
                 start_position: int = None,
                 target_position: int = None,
                 reward_target: float = 10.0,
                 reward_step: float = -0.1,
                 reward_boundary: float = -1.0,
                 max_steps: int = 100):
        """
        Initialise l'environnement Line World.
        
        Args:
            line_length (int): Longueur de la ligne (nombre de positions)
            start_position (int): Position de départ (None = position aléatoire)
            target_position (int): Position cible (None = dernière position)
            reward_target (float): Récompense pour atteindre la cible
            reward_step (float): Récompense pour chaque étape
            reward_boundary (float): Pénalité pour sortir des limites
            max_steps (int): Nombre maximum d'étapes par épisode
        """
        super().__init__("LineWorld")
        
        # Configuration de l'environnement
        self.line_length = max(2, line_length)  # Au minimum 2 positions
        self.start_position = start_position
        self.target_position = target_position if target_position is not None else line_length - 1
        
        # Validation des positions
        if self.target_position >= self.line_length:
            self.target_position = self.line_length - 1
        
        # Récompenses
        self.reward_target = reward_target
        self.reward_step = reward_step
        self.reward_boundary = reward_boundary
        
        # Contraintes d'épisode
        self.max_steps = max_steps
        self.steps_taken = 0
        
        # État initial
        self.current_state = None
        self.reset()
    
    @property
    def state_space_size(self) -> int:
        """Retourne la taille de l'espace d'états."""
        return self.line_length
    
    @property
    def action_space_size(self) -> int:
        """Retourne la taille de l'espace d'actions."""
        return 2  # Gauche, Droite
    
    @property
    def valid_actions(self) -> List[int]:
        """Retourne la liste des actions valides dans l'état actuel."""
        return [self.ACTION_LEFT, self.ACTION_RIGHT]
    
    def reset(self) -> int:
        """
        Remet l'environnement à l'état initial.
        
        Returns:
            int: Position initiale de l'agent
        """
        # Détermine la position de départ
        if self.start_position is not None:
            self.current_state = self.start_position
        else:
            # Position aléatoire (mais pas la cible pour rendre le problème intéressant)
            possible_starts = [i for i in range(self.line_length) if i != self.target_position]
            self.current_state = np.random.choice(possible_starts)
        
        # Remet à zéro les statistiques
        self.steps_taken = 0
        self._reset_episode_stats()
        
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Exécute une action dans l'environnement.
        
        Args:
            action (int): Action à exécuter (0=Gauche, 1=Droite)
            
        Returns:
            Tuple[int, float, bool, Dict[str, Any]]: 
                - next_state: Nouvelle position
                - reward: Récompense obtenue
                - done: True si l'épisode est terminé
                - info: Informations supplémentaires
        """
        if not self.is_valid_action(action):
            raise ValueError(f"Action invalide: {action}. Actions valides: {self.valid_actions}")
        
        self.steps_taken += 1
        old_state = self.current_state
        
        # Calcule la nouvelle position
        if action == self.ACTION_LEFT:
            new_position = self.current_state - 1
        else:  # ACTION_RIGHT
            new_position = self.current_state + 1
        
        # Vérifie les limites et calcule la récompense
        reward = self.reward_step  # Récompense de base pour chaque étape
        done = False
        info = {"action_name": self.ACTION_NAMES[action]}
        
        # Gestion des limites
        if new_position < 0 or new_position >= self.line_length:
            # L'agent sort des limites - reste à sa position actuelle
            new_position = self.current_state
            reward = self.reward_boundary
            info["boundary_hit"] = True
        else:
            # Mouvement valide
            self.current_state = new_position
            info["boundary_hit"] = False
        
        # Vérifie si la cible est atteinte
        if self.current_state == self.target_position:
            reward = self.reward_target
            done = True
            info["target_reached"] = True
        else:
            info["target_reached"] = False
        
        # Vérifie la limite d'étapes
        if self.steps_taken >= self.max_steps:
            done = True
            info["max_steps_reached"] = True
        else:
            info["max_steps_reached"] = False
        
        # Distance à la cible (pour l'analyse)
        info["distance_to_target"] = abs(self.current_state - self.target_position)
        info["steps_taken"] = self.steps_taken
        
        # Met à jour les statistiques
        self._update_episode_stats(action, reward, self.current_state, done)
        
        return self.current_state, reward, done, info
    
    def render(self, mode: str = 'console') -> Optional[Any]:
        """
        Affiche l'état actuel de l'environnement.
        
        Args:
            mode (str): Mode d'affichage ('console' ou 'pygame')
        """
        if mode == 'console':
            self._render_console()
        elif mode == 'pygame':
            return self._render_pygame()
        else:
            raise ValueError(f"Mode de rendu non supporté: {mode}")
    
    def _render_console(self):
        """Affichage console de l'environnement."""
        print(f"\n=== Line World (Step {self.steps_taken}) ===")
        
        # Ligne supérieure avec numéros de position
        position_line = "Pos: "
        for i in range(self.line_length):
            position_line += f"{i:2d} "
        print(position_line)
        
        # Ligne avec la représentation de l'environnement
        env_line = "     "
        for i in range(self.line_length):
            if i == self.current_state and i == self.target_position:
                env_line += "[A]"  # Agent sur la cible
            elif i == self.current_state:
                env_line += "[A]"  # Agent
            elif i == self.target_position:
                env_line += "(T)"  # Cible
            else:
                env_line += " . "  # Position vide
        print(env_line)
        
        # Informations supplémentaires
        distance = abs(self.current_state - self.target_position)
        print(f"Agent: pos {self.current_state} | Target: pos {self.target_position} | Distance: {distance}")
        print(f"Total reward: {self.total_reward:.2f}")
        print("Actions: [0] Left ← | [1] Right →")
    
    def _render_pygame(self):
        """
        Affichage pygame de l'environnement.
        Note: Implémentation basique - sera améliorée plus tard.
        """
        # Pour l'instant, retourne les données pour un rendu pygame externe
        return {
            'positions': list(range(self.line_length)),
            'agent_position': self.current_state,
            'target_position': self.target_position,
            'steps_taken': self.steps_taken,
            'total_reward': self.total_reward
        }
    
    def get_state_description(self, state: int) -> str:
        """
        Retourne une description textuelle d'un état.
        
        Args:
            state (int): État à décrire
            
        Returns:
            str: Description de l'état
        """
        if state == self.target_position:
            return f"Position {state} (CIBLE)"
        else:
            distance = abs(state - self.target_position)
            return f"Position {state} (distance à cible: {distance})"
    
    def get_transition_probabilities(self, state: int, action: int) -> Dict[int, float]:
        """
        Retourne les probabilités de transition depuis un état avec une action.
        
        Args:
            state (int): État de départ
            action (int): Action exécutée
            
        Returns:
            Dict[int, float]: Dictionnaire {next_state: probability}
        """
        if action == self.ACTION_LEFT:
            next_state = max(0, state - 1)  # Ne peut pas aller en dessous de 0
        else:  # ACTION_RIGHT
            next_state = min(self.line_length - 1, state + 1)  # Ne peut pas dépasser la limite
        
        return {next_state: 1.0}  # Transition déterministe
    
    def get_reward_function(self, state: int, action: int, next_state: int) -> float:
        """
        Retourne la récompense pour une transition donnée.
        
        Args:
            state (int): État de départ
            action (int): Action exécutée
            next_state (int): État d'arrivée
            
        Returns:
            float: Récompense de la transition
        """
        # Récompense de base
        reward = self.reward_step
        
        # Pénalité pour collision avec les limites
        if action == self.ACTION_LEFT and state == 0 and next_state == 0:
            reward = self.reward_boundary
        elif action == self.ACTION_RIGHT and state == self.line_length - 1 and next_state == self.line_length - 1:
            reward = self.reward_boundary
        
        # Récompense pour atteindre la cible
        if next_state == self.target_position:
            reward = self.reward_target
        
        return reward
    
    def get_optimal_policy(self) -> Dict[int, int]:
        """
        Retourne la politique optimale (pour validation des algorithmes).
        
        Returns:
            Dict[int, int]: Politique optimale {state: action}
        """
        policy = {}
        for state in range(self.line_length):
            if state < self.target_position:
                policy[state] = self.ACTION_RIGHT
            elif state > self.target_position:
                policy[state] = self.ACTION_LEFT
            else:
                # À la cible, n'importe quelle action (on reste sur place)
                policy[state] = self.ACTION_RIGHT
        return policy
    
    def get_optimal_value_function(self, gamma: float = 0.9) -> Dict[int, float]:
        """
        Retourne la fonction de valeur optimale (pour validation).
        
        Args:
            gamma (float): Facteur d'actualisation
            
        Returns:
            Dict[int, float]: Fonction de valeur optimale
        """
        value_function = {}
        for state in range(self.line_length):
            if state == self.target_position:
                value_function[state] = 0.0  # Déjà à la cible
            else:
                # Distance minimale à la cible
                distance = abs(state - self.target_position)
                # Valeur = récompense target - coût des étapes pour y arriver
                value_function[state] = self.reward_target + (distance * self.reward_step)
        return value_function
    
    def create_random_start_variant(self):
        """
        Crée une variante avec position de départ aléatoire.
        
        Returns:
            LineWorld: Nouvelle instance avec départ aléatoire
        """
        return LineWorld(
            line_length=self.line_length,
            start_position=None,  # Position aléatoire
            target_position=self.target_position,
            reward_target=self.reward_target,
            reward_step=self.reward_step,
            reward_boundary=self.reward_boundary,
            max_steps=self.max_steps
        )


# Fonction utilitaire pour créer des variantes pré-configurées
def create_simple_lineworld() -> LineWorld:
    """Crée un Line World simple pour les tests."""
    return LineWorld(line_length=5, start_position=0, target_position=4)


def create_challenging_lineworld() -> LineWorld:
    """Crée un Line World plus challengeant."""
    return LineWorld(
        line_length=10, 
        start_position=None,  # Position aléatoire
        target_position=9,
        reward_step=-0.01,    # Pénalité plus faible
        max_steps=50
    )


if __name__ == "__main__":
    # Test rapide de l'environnement
    print("Test de Line World")
    env = create_simple_lineworld()
    
    print("État initial:")
    env.render()
    
    print("\nTest de quelques actions:")
    for i in range(3):
        action = np.random.choice([0, 1])
        state, reward, done, info = env.step(action)
        print(f"\nAction: {env.ACTION_NAMES[action]} -> Reward: {reward:.2f}")
        env.render()
        
        if done:
            print("Épisode terminé!")
            break