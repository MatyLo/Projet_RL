"""
Classe abstraite de base pour tous les algorithmes d'apprentissage par renforcement.

Cette classe définit l'interface commune que tous les algorithmes doivent respecter
pour être compatibles avec les environnements et le système d'expérimentation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class TrainingStats:
    """Statistiques d'entraînement pour un algorithme."""
    episode: int
    total_reward: float
    steps: int
    convergence_metric: Optional[float] = None
    additional_metrics: Optional[Dict[str, float]] = None


class BaseAlgorithm(ABC):
    """
    Classe abstraite pour les algorithmes d'apprentissage par renforcement.
    
    Tous les algorithmes (Q-Learning, SARSA, Policy Iteration, etc.) doivent
    hériter de cette classe et implémenter les méthodes abstraites.
    """
    
    def __init__(self, 
                 algo_name: str,
                 state_space_size: int, 
                 action_space_size: int,
                 **kwargs):
        """
        Initialise l'algorithme de base.
        
        Args:
            algo_name (str): Nom de l'algorithme pour identification
            state_space_size (int): Taille de l'espace d'états
            action_space_size (int): Taille de l'espace d'actions
            **kwargs: Paramètres spécifiques à l'algorithme
        """
        self.algo_name = algo_name
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        
        # Hyperparamètres communs
        self.gamma = kwargs.get('gamma', 0.9)  # Facteur d'actualisation
        self.epsilon = kwargs.get('epsilon', 0.1)  # Exploration
        
        # Statistiques d'entraînement
        self.training_stats: List[TrainingStats] = []
        self.is_trained = False
        self.episode_count = 0
        
        # Politiques et fonctions de valeur (initialisées par les sous-classes)
        self.policy = None
        self.value_function = None
        self.q_function = None
        
    @property
    @abstractmethod
    def algorithm_type(self) -> str:
        """Retourne le type d'algorithme ('dp', 'mc', 'td', 'planning')."""
        pass
    
    @property
    @abstractmethod
    def required_environment_features(self) -> List[str]:
        """
        Retourne les fonctionnalités requises de l'environnement.
        
        Returns:
            List[str]: Liste des fonctionnalités requises
                      ('transitions', 'rewards', 'episodes', etc.)
        """
        pass
    
    @abstractmethod
    def train(self, environment, num_episodes: int, **kwargs) -> Dict[str, Any]:
        """
        Entraîne l'algorithme sur un environnement.
        
        Args:
            environment: Environnement d'entraînement
            num_episodes (int): Nombre d'épisodes d'entraînement
            **kwargs: Paramètres spécifiques à l'entraînement
            
        Returns:
            Dict[str, Any]: Statistiques d'entraînement et résultats
        """
        pass
    
    @abstractmethod
    def select_action(self, state: int, **kwargs) -> int:
        """
        Sélectionne une action pour un état donné.
        
        Args:
            state (int): État actuel
            **kwargs: Paramètres pour la sélection (exploration, etc.)
            
        Returns:
            int: Action sélectionnée
        """
        pass
    
    @abstractmethod
    def get_policy(self) -> Union[np.ndarray, Dict[int, int]]:
        """
        Retourne la politique apprise.
        
        Returns:
            Union[np.ndarray, Dict[int, int]]: Politique optimale
        """
        pass
    
    @abstractmethod
    def get_value_function(self) -> Union[np.ndarray, Dict[int, float]]:
        """
        Retourne la fonction de valeur d'état.
        
        Returns:
            Union[np.ndarray, Dict[int, float]]: Fonction de valeur
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> bool:
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            filepath (str): Chemin de sauvegarde
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> bool:
        """
        Charge un modèle pré-entraîné.
        
        Args:
            filepath (str): Chemin du modèle à charger
            
        Returns:
            bool: True si le chargement a réussi
        """
        pass
    
    def get_q_function(self) -> Union[np.ndarray, Dict[Tuple[int, int], float], None]:
        """
        Retourne la fonction de valeur action-état (Q-function).
        
        Returns:
            Union[np.ndarray, Dict, None]: Q-function si disponible
        """
        return self.q_function
    
    def select_greedy_action(self, state: int) -> int:
        """
        Sélectionne l'action gloutonne (meilleure) pour un état donné.
        
        Args:
            state (int): État actuel
            
        Returns:
            int: Action gloutonne
        """
        if self.q_function is None:
            raise ValueError("Q-function not available. Train the algorithm first.")
        
        if isinstance(self.q_function, np.ndarray):
            return np.argmax(self.q_function[state])
        else:
            # Pour les dictionnaires, trouve la meilleure action
            q_values = [self.q_function.get((state, action), 0.0) 
                       for action in range(self.action_space_size)]
            return np.argmax(q_values)
    
    def select_epsilon_greedy_action(self, state: int, epsilon: Optional[float] = None) -> int:
        """
        Sélectionne une action selon la stratégie epsilon-greedy.
        
        Args:
            state (int): État actuel
            epsilon (float, optional): Taux d'exploration. Utilise self.epsilon par défaut.
            
        Returns:
            int: Action sélectionnée
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() < epsilon:
            # Exploration : action aléatoire
            return np.random.randint(0, self.action_space_size)
        else:
            # Exploitation : action gloutonne
            return self.select_greedy_action(state)
    
    def update_training_stats(self, episode: int, total_reward: float, 
                            steps: int, convergence_metric: Optional[float] = None,
                            additional_metrics: Optional[Dict[str, float]] = None):
        """
        Met à jour les statistiques d'entraînement.
        
        Args:
            episode (int): Numéro de l'épisode
            total_reward (float): Récompense totale de l'épisode
            steps (int): Nombre d'étapes de l'épisode
            convergence_metric (float, optional): Métrique de convergence
            additional_metrics (Dict[str, float], optional): Métriques supplémentaires
        """
        stats = TrainingStats(
            episode=episode,
            total_reward=total_reward,
            steps=steps,
            convergence_metric=convergence_metric,
            additional_metrics=additional_metrics or {}
        )
        self.training_stats.append(stats)
        self.episode_count = episode
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé des statistiques d'entraînement.
        
        Returns:
            Dict[str, Any]: Résumé des statistiques
        """
        if not self.training_stats:
            return {"message": "No training statistics available"}
        
        rewards = [stats.total_reward for stats in self.training_stats]
        steps = [stats.steps for stats in self.training_stats]
        
        return {
            "algorithm": self.algo_name,
            "episodes_trained": len(self.training_stats),
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "avg_steps": np.mean(steps),
            "final_episode_reward": rewards[-1],
            "is_converged": self.is_trained
        }
    
    def evaluate_policy(self, environment, num_episodes: int = 100, 
                       max_steps: int = 1000) -> Dict[str, float]:
        """
        Évalue la politique apprise sur un environnement.
        
        Args:
            environment: Environnement d'évaluation
            num_episodes (int): Nombre d'épisodes d'évaluation
            max_steps (int): Nombre maximum d'étapes par épisode
            
        Returns:
            Dict[str, float]: Statistiques d'évaluation
        """
        if not self.is_trained:
            raise ValueError("Algorithm must be trained before evaluation")
        
        total_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0.0
            steps = 0
            
            for step in range(max_steps):
                # Sélection d'action sans exploration (politique gloutonne)
                action = self.select_greedy_action(state)
                next_state, reward, done, _ = environment.step(action)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        return {
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "min_reward": np.min(total_rewards),
            "max_reward": np.max(total_rewards),
            "avg_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths),
            "num_episodes": num_episodes
        }
    
    def reset_training(self):
        """Remet à zéro l'entraînement de l'algorithme."""
        self.training_stats = []
        self.is_trained = False
        self.episode_count = 0
        # Les sous-classes peuvent override pour réinitialiser leurs structures
    
    def set_hyperparameters(self, **kwargs):
        """
        Met à jour les hyperparamètres de l'algorithme.
        
        Args:
            **kwargs: Nouveaux hyperparamètres
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Hyperparameter '{key}' not recognized for {self.algo_name}")
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Retourne les hyperparamètres actuels de l'algorithme.
        
        Returns:
            Dict[str, Any]: Hyperparamètres
        """
        # Hyperparamètres de base - les sous-classes peuvent étendre
        return {
            "gamma": self.gamma,
            "epsilon": self.epsilon
        }
    
    def __str__(self) -> str:
        """Représentation textuelle de l'algorithme."""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.algo_name}({status}, episodes={self.episode_count})"
    
    def __repr__(self) -> str:
        """Représentation détaillée de l'algorithme."""
        return (f"{self.__class__.__name__}("
                f"name='{self.algo_name}', "
                f"states={self.state_space_size}, "
                f"actions={self.action_space_size}, "
                f"trained={self.is_trained})")