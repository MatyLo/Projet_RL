"""
Classe abstraite de base pour tous les algorithmes d'apprentissage par renforcement.

Cette classe définit l'interface commune que tous les algorithmes doivent respecter
pour être compatibles avec les environnements et le système d'expérimentation.

Cette version améliorée supporte :
- Configuration via JSON avec from_config()
- Entraînement autonome (style professeur)
- Interface standardisée pour Agent wrapper
- Métriques de convergence avancées
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import time
from dataclasses import dataclass
from datetime import datetime


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
    
    Architecture Hybride :
    - Phase 1 : Entraînement autonome avec configuration JSON
    - Phase 2 : Compatible avec Agent wrapper pour post-entraînement
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
        
        # Métriques de convergence
        self.convergence_threshold = kwargs.get('convergence_threshold', 1e-4)
        
        # Statistiques d'entraînement
        self.training_stats: List[TrainingStats] = []
        self.is_trained = False
        self.episode_count = 0
        self.training_start_time = None
        self.training_end_time = None
        
        # Politiques et fonctions de valeur (initialisées par les sous-classes)
        self.policy = None
        self.value_function = None
        self.q_function = None
        
        # Configuration source (pour debugging et reproductibilité)
        self._source_config = kwargs.copy()
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any], environment) -> 'BaseAlgorithm':
        """
        NOUVELLE MÉTHODE : Crée un algorithme depuis une configuration JSON.
        
        Cette méthode permet l'approche hybride : configuration JSON -> entraînement autonome.
        
        Args:
            config (Dict[str, Any]): Configuration de l'algorithme depuis JSON
            environment: Environnement d'entraînement (pour les tailles d'espaces)
            
        Returns:
            BaseAlgorithm: Instance configurée de l'algorithme
            
        Raises:
            ValueError: Si la configuration est invalide
        """
        pass
    
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
        MODIFIÉE : Entraîne l'algorithme de manière autonome (style professeur).
        
        Cette méthode implémente l'entraînement autonome pour l'approche hybride.
        L'algorithme se débrouille tout seul, sans dépendre d'un Agent.
        
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
            **kwargs: Paramètres pour la sélection (training=True/False, etc.)
            
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
    
    # ==================== NOUVELLES MÉTHODES POUR APPROCHE HYBRIDE ====================
    
    def to_config(self) -> Dict[str, Any]:
        """
        NOUVELLE : Exporte la configuration actuelle de l'algorithme.
        
        Utile pour sauvegarder les hyperparamètres utilisés et reproduire les expériences.
        
        Returns:
            Dict[str, Any]: Configuration exportable en JSON
        """
        base_config = {
            'algorithm_type': self.algorithm_type,
            'algo_name': self.algo_name,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'convergence_threshold': self.convergence_threshold,
            'state_space_size': self.state_space_size,
            'action_space_size': self.action_space_size
        }
        
        # Ajout de la configuration source si disponible
        if hasattr(self, '_source_config'):
            base_config.update(self._source_config)
        
        return base_config
    
    def get_training_results(self) -> Dict[str, Any]:
        """
        NOUVELLE : Retourne un résumé complet des résultats d'entraînement.
        
        Conçu pour l'Agent wrapper qui a besoin d'analyser les performances
        sans connaître les détails internes de l'algorithme.
        
        Returns:
            Dict[str, Any]: Résultats d'entraînement formatés
        """
        if not self.training_stats:
            return {"message": "Aucune donnée d'entraînement disponible"}
        
        rewards = [stats.total_reward for stats in self.training_stats]
        steps = [stats.steps for stats in self.training_stats]
        convergence_metrics = [stats.convergence_metric for stats in self.training_stats 
                              if stats.convergence_metric is not None]
        
        results = {
            "algorithm": self.algo_name,
            "episodes_trained": len(self.training_stats),
            "is_trained": self.is_trained,
            "training_time": self._get_training_duration(),
            
            # Statistiques de récompenses
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "final_episode_reward": rewards[-1],
            
            # Statistiques d'épisodes
            "avg_episode_length": np.mean(steps),
            "std_episode_length": np.std(steps),
            "min_episode_length": np.min(steps),
            "max_episode_length": np.max(steps),
            
            # Convergence
            "convergence_data": convergence_metrics,
            "converged": self._check_convergence(),
            
            # Métadonnées
            "hyperparameters": self.get_hyperparameters(),
            "config_source": getattr(self, '_source_config', {}),
        }
        
        return results
    
    def _get_training_duration(self) -> float:
        """Calcule la durée d'entraînement."""
        if self.training_start_time and self.training_end_time:
            return self.training_end_time - self.training_start_time
        return 0.0
    
    def is_ready_for_evaluation(self) -> bool:
        """
        NOUVELLE : Vérifie si l'algorithme est prêt pour l'évaluation post-entraînement.
        
        L'Agent wrapper peut utiliser cette méthode pour valider que l'algorithme
        est dans un état utilisable.
        
        Returns:
            bool: True si prêt pour évaluation
        """
        return (self.is_trained and 
                self.policy is not None and 
                len(self.training_stats) > 0)
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        NOUVELLE : Retourne les informations essentielles de l'algorithme.
        
        Utile pour l'Agent wrapper et les comparaisons d'algorithmes.
        
        Returns:
            Dict[str, Any]: Informations de l'algorithme
        """
        return {
            "name": self.algo_name,
            "type": self.algorithm_type,
            "state_space_size": self.state_space_size,
            "action_space_size": self.action_space_size,
            "is_trained": self.is_trained,
            "episode_count": self.episode_count,
            "convergence_threshold": self.convergence_threshold,
            "required_features": self.required_environment_features
        }
    
    # ==================== MÉTHODES EXISTANTES AMÉLIORÉES ====================
    
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
        AMÉLIORÉE : Retourne un résumé des statistiques d'entraînement.
        
        Returns:
            Dict[str, Any]: Résumé des statistiques
        """
        return self.get_training_results()  # Utilise la nouvelle méthode complète
    
    def evaluate_policy(self, environment, num_episodes: int = 100, 
                       max_steps: int = 1000) -> Dict[str, float]:
        """
        AMÉLIORÉE : Évalue la politique apprise sur un environnement.
        
        Cette méthode permet à l'algorithme d'auto-évaluer ses performances,
        compatible avec l'Agent wrapper.
        
        Args:
            environment: Environnement d'évaluation
            num_episodes (int): Nombre d'épisodes d'évaluation
            max_steps (int): Nombre maximum d'étapes par épisode
            
        Returns:
            Dict[str, float]: Statistiques d'évaluation
        """
        if not self.is_ready_for_evaluation():
            raise ValueError("Algorithm must be trained before evaluation")
        
        total_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0.0
            steps = 0
            
            for step in range(max_steps):
                # Sélection d'action sans exploration (politique gloutonne)
                action = self.select_action(state, training=False)
                next_state, reward, done, info = environment.step(action)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    # Vérifie si c'est un succès
                    if info.get("target_reached", False):
                        success_count += 1
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
            "success_rate": success_count / num_episodes,
            "num_episodes": num_episodes
        }
    
    def _check_convergence(self, window_size: int = 50) -> bool:
        """
        AMÉLIORÉE : Vérifie si l'algorithme a convergé.
        
        Args:
            window_size (int): Taille de la fenêtre pour le calcul
            
        Returns:
            bool: True si convergé
        """
        if len(self.training_stats) < window_size:
            return False
        
        # Vérifie la convergence sur les métriques disponibles
        recent_stats = self.training_stats[-window_size:]
        convergence_metrics = [s.convergence_metric for s in recent_stats 
                              if s.convergence_metric is not None]
        
        if convergence_metrics:
            avg_convergence = np.mean(convergence_metrics)
            return avg_convergence < self.convergence_threshold
        
        return False
    
    def reset_training(self):
        """AMÉLIORÉE : Remet à zéro l'entraînement de l'algorithme."""
        self.training_stats = []
        self.is_trained = False
        self.episode_count = 0
        self.training_start_time = None
        self.training_end_time = None
        # Les sous-classes peuvent override pour réinitialiser leurs structures
    
    def set_hyperparameters(self, **kwargs):
        """
        AMÉLIORÉE : Met à jour les hyperparamètres de l'algorithme.
        
        Args:
            **kwargs: Nouveaux hyperparamètres
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # Met à jour la config source aussi
                if hasattr(self, '_source_config'):
                    self._source_config[key] = value
            else:
                print(f"Warning: Hyperparameter '{key}' not recognized for {self.algo_name}")
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        AMÉLIORÉE : Retourne les hyperparamètres actuels de l'algorithme.
        
        Returns:
            Dict[str, Any]: Hyperparamètres
        """
        # Hyperparamètres de base - les sous-classes peuvent étendre
        base_params = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "convergence_threshold": self.convergence_threshold
        }
        
        # Ajoute les hyperparamètres de la configuration source
        if hasattr(self, '_source_config'):
            base_params.update(self._source_config)
        
        return base_params
    
    # ==================== MÉTHODES UTILITAIRES ====================
    
    def _start_training_timer(self):
        """Démarre le chronométrage d'entraînement."""
        self.training_start_time = time.time()
    
    def _end_training_timer(self):
        """Arrête le chronométrage d'entraînement."""
        self.training_end_time = time.time()
    
    def _validate_environment_compatibility(self, environment):
        """
        Valide que l'environnement est compatible avec l'algorithme.
        
        Args:
            environment: Environnement à valider
            
        Raises:
            ValueError: Si incompatible
        """
        if self.state_space_size != environment.state_space_size:
            raise ValueError(f"Incompatibilité d'espace d'états: "
                           f"algo={self.state_space_size}, env={environment.state_space_size}")
        
        if self.action_space_size != environment.action_space_size:
            raise ValueError(f"Incompatibilité d'espace d'actions: "
                           f"algo={self.action_space_size}, env={environment.action_space_size}")
    
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