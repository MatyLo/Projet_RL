"""
Classe abstraite de base pour tous les environnements d'apprentissage par renforcement.

Cette classe définit l'interface commune que tous les environnements doivent respecter
pour être compatibles avec les algorithmes d'apprentissage par renforcement.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
import numpy as np


class BaseEnvironment(ABC):
    """
    Classe abstraite pour les environnements de RL.
    
    Tous les environnements doivent hériter de cette classe et implémenter
    les méthodes abstraites pour assurer la compatibilité avec les algorithmes.
    """
    
    def __init__(self, env_name: str):
        """
        Initialise l'environnement de base.
        
        Args:
            env_name (str): Nom de l'environnement pour identification
        """
        self.env_name = env_name
        self.current_state = None
        self.episode_step = 0
        self.total_reward = 0.0
        self.episode_history = []  # Historique de l'épisode en cours
        
    @property
    @abstractmethod
    def state_space_size(self) -> int:
        """Retourne la taille de l'espace d'états (nombre d'états possibles)."""
        pass
    
    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Retourne la taille de l'espace d'actions (nombre d'actions possibles)."""
        pass
    
    @property
    @abstractmethod
    def valid_actions(self) -> List[int]:
        """Retourne la liste des actions valides dans l'état actuel."""
        pass
    
    @abstractmethod
    def reset(self) -> int:
        """
        Remet l'environnement à l'état initial.
        
        Returns:
            int: État initial de l'environnement
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Exécute une action dans l'environnement.
        
        Args:
            action (int): Action à exécuter
            
        Returns:
            Tuple[int, float, bool, Dict[str, Any]]: 
                - next_state: Nouvel état
                - reward: Récompense obtenue
                - done: True si l'épisode est terminé
                - info: Informations supplémentaires
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = 'console') -> Optional[Any]:
        """
        Affiche l'état actuel de l'environnement.
        
        Args:
            mode (str): Mode d'affichage ('console' ou 'pygame')
            
        Returns:
            Optional[Any]: Données de rendu si nécessaire
        """
        pass
    
    @abstractmethod
    def get_state_description(self, state: int) -> str:
        """
        Retourne une description textuelle d'un état.
        
        Args:
            state (int): État à décrire
            
        Returns:
            str: Description de l'état
        """
        pass
    
    def get_current_state(self) -> int:
        """Retourne l'état actuel de l'environnement."""
        return self.current_state
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de l'épisode en cours.
        
        Returns:
            Dict[str, Any]: Statistiques de l'épisode
        """
        return {
            'episode_length': self.episode_step,
            'total_reward': self.total_reward,
            'history': self.episode_history.copy()
        }
    
    def is_valid_action(self, action: int) -> bool:
        """
        Vérifie si une action est valide dans l'état actuel.
        
        Args:
            action (int): Action à vérifier
            
        Returns:
            bool: True si l'action est valide
        """
        return action in self.valid_actions
    
    def get_transition_probabilities(self, state: int, action: int) -> Dict[int, float]:
        """
        Retourne les probabilités de transition depuis un état avec une action.
        Méthode optionnelle - utile pour les algorithmes de programmation dynamique.
        
        Args:
            state (int): État de départ
            action (int): Action exécutée
            
        Returns:
            Dict[int, float]: Dictionnaire {next_state: probability}
        """
        # Par défaut, retourne une transition déterministe vers l'état actuel
        # Les environnements stochastiques peuvent overrider cette méthode
        return {state: 1.0}
    
    def get_reward_function(self, state: int, action: int, next_state: int) -> float:
        """
        Retourne la récompense pour une transition donnée.
        Méthode optionnelle - utile pour les algorithmes de programmation dynamique.
        
        Args:
            state (int): État de départ
            action (int): Action exécutée
            next_state (int): État d'arrivée
            
        Returns:
            float: Récompense de la transition
        """
        # Par défaut, effectue un step pour obtenir la récompense
        # Les environnements peuvent overrider pour plus d'efficacité
        current_state_backup = self.current_state
        self.current_state = state
        _, reward, _, _ = self.step(action)
        self.current_state = current_state_backup
        return reward
    
    def _update_episode_stats(self, action: int, reward: float, next_state: int, done: bool):
        """
        Met à jour les statistiques de l'épisode en cours.
        
        Args:
            action (int): Action exécutée
            reward (float): Récompense obtenue
            next_state (int): Nouvel état
            done (bool): True si l'épisode est terminé
        """
        self.episode_step += 1
        self.total_reward += reward
        
        step_info = {
            'step': self.episode_step,
            'state': self.current_state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.episode_history.append(step_info)
    
    def _reset_episode_stats(self):
        """Remet à zéro les statistiques de l'épisode."""
        self.episode_step = 0
        self.total_reward = 0.0
        self.episode_history = []
    
    def seed(self, seed: int = None):
        """
        Définit la graine aléatoire pour la reproductibilité.
        
        Args:
            seed (int): Graine aléatoire
        """
        np.random.seed(seed)
    
    def __str__(self) -> str:
        """Représentation textuelle de l'environnement."""
        return f"{self.env_name}(state={self.current_state}, step={self.episode_step})"
    
    def __repr__(self) -> str:
        """Représentation détaillée de l'environnement."""
        return (f"{self.__class__.__name__}("
                f"name='{self.env_name}', "
                f"state_space={self.state_space_size}, "
                f"action_space={self.action_space_size})")