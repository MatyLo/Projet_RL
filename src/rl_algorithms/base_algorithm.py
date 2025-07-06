"""
Classe abstraite de base pour tous les algorithmes d'apprentissage par renforcement.

Version simplifiée - focus sur l'essentiel pour faciliter la compréhension.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import numpy as np
import time
import json
import pickle


class BaseAlgorithm(ABC):
    """
    Classe abstraite pour les algorithmes.
    
    Tous les algorithmes (Q-Learning, SARSA, etc.) héritent de cette classe.
    
    Workflow simple:
    1. Créer algorithme avec from_config()
    2. Entraîner avec train()
    3. Utiliser avec Agent wrapper pour évaluation/démo
    """
    
    def __init__(self, 
                 algo_name: str,
                 state_space_size: int, 
                 action_space_size: int):
        """
        Initialise l'algorithme de base.
        
        Args:
            algo_name: Nom de l'algorithme pour identification
            state_space_size: Nombre d'états dans l'environnement
            action_space_size: Nombre d'actions possibles
        """
        self.algo_name = algo_name
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        
        # État d'entraînement
        self.is_trained = False
        self.training_history = []  # Liste simple de dict {episode, reward, steps}
        
        # Structures apprises (initialisées par les sous-classes)
        self.policy = None
        self.q_function = None
        self.value_function = None
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any], environment):
        """
        Crée un algorithme depuis une configuration JSON.
        
        Args:
            config: Configuration de l'algorithme
            environment: Environnement d'entraînement
            
        Returns:
            Instance configurée de l'algorithme
        """
        pass
    
    @abstractmethod
    def train(self, environment, num_episodes: int, verbose: bool = False):
        """
        Entraîne l'algorithme de manière autonome.
        
        Args:
            environment: Environnement d'entraînement
            num_episodes: Nombre d'épisodes d'entraînement
            verbose: Affichage des informations de progression
            
        Returns:
            Dict avec statistiques d'entraînement
        """
        pass
    
    @abstractmethod
    def select_action(self, state: int, training: bool = False):
        """
        Sélectionne une action pour un état donné.
        
        Args:
            state: État actuel
            training: True si en mode entraînement (avec exploration)
            
        Returns:
            Action sélectionnée
        """
        pass
    
    @abstractmethod
    def get_policy(self):
        """Retourne la politique apprise."""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> bool:
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            filepath: Chemin de sauvegarde
            
        Returns:
            True si succès
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> bool:
        """
        Charge un modèle pré-entraîné.
        
        Args:
            filepath: Chemin du modèle
            
        Returns:
            True si succès
        """
        pass
    
    # ============ MÉTHODES non abstraites ============
    
    def get_training_results(self):
        """Retourne les résultats d'entraînement pour l'Agent wrapper."""
        if not self.training_history:
            return {"message": "Aucune donnée d'entraînement"}
        
        rewards = [episode['reward'] for episode in self.training_history]
        steps = [episode['steps'] for episode in self.training_history]
        
        return {
            "algorithm": self.algo_name,
            "episodes_trained": len(self.training_history),
            "is_trained": self.is_trained,
            "avg_reward": np.mean(rewards),
            "final_reward": rewards[-1],
            "avg_steps": np.mean(steps),
            "training_history": self.training_history
        }
    
    def add_training_episode(self, episode: int, reward: float, steps: int):
        """Ajoute un épisode à l'historique d'entraînement."""
        self.training_history.append({
            'episode': episode,
            'reward': reward,
            'steps': steps
        })
    
    def select_greedy_action(self, state: int):
        """Sélectionne l'action gloutonne (meilleure) pour un état."""
        if self.q_function is None:
            raise ValueError("Algorithme non entraîné")
        
        if isinstance(self.q_function, np.ndarray):
            return np.argmax(self.q_function[state])
        else:
            # Pour les dictionnaires
            q_values = [self.q_function.get((state, action), 0.0) 
                       for action in range(self.action_space_size)]
            return np.argmax(q_values)
    
    def select_epsilon_greedy_action(self, state: int, epsilon: float):
        """Sélectionne une action epsilon-greedy."""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_space_size)
        else:
            return self.select_greedy_action(state)
    
    def __str__(self):
        """Représentation textuelle simple."""
        status = "entraîné" if self.is_trained else "non entraîné"
        episodes = len(self.training_history)
        return f"{self.algo_name}({status}, {episodes} épisodes)"