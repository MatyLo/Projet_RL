"""
Classe abstraite de base pour tous les algorithmes d'apprentissage par renforcement.

Version simplifiée - focus sur l'essentiel pour faciliter la compréhension.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import time
import json
import pickle
import matplotlib.pyplot as plt


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

    def plot_training_curves(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Affiche les courbes d'entraînement."""
        if not hasattr(self, 'training_history') or not self.training_history:
            print("Aucun historique d'entraînement disponible. Entraînez d'abord l'algorithme.")
            return
        
        # Extraction des données depuis training_history
        rewards = [episode['reward'] for episode in self.training_history]
        steps = [episode['steps'] for episode in self.training_history]
        episodes = [episode['episode'] for episode in self.training_history]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Récompenses par épisode
        axes[0, 0].plot(episodes, rewards, alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Récompenses par épisode')
        axes[0, 0].set_xlabel('Épisode')
        axes[0, 0].set_ylabel('Récompense')
        axes[0, 0].grid(True)
        
        # Moyenne mobile des récompenses
        if len(rewards) > 10:
            window = min(100, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            # Ajustement des indices pour la moyenne mobile
            moving_episodes = episodes[window-1:]
            
            axes[0, 1].plot(episodes, rewards, alpha=0.3, color='lightblue', label='Récompenses')
            axes[0, 1].plot(moving_episodes, moving_avg, color='red', linewidth=2, label=f'Moyenne mobile ({window})')
            axes[0, 1].set_title(f'Moyenne mobile (fenêtre {window})')
            axes[0, 1].set_xlabel('Épisode')
            axes[0, 1].set_ylabel('Récompense moyenne')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Longueur des épisodes
        axes[1, 0].plot(episodes, steps, alpha=0.7, linewidth=1)
        axes[1, 0].set_title('Longueur des épisodes')
        axes[1, 0].set_xlabel('Épisode')
        axes[1, 0].set_ylabel('Nombre de pas')
        axes[1, 0].grid(True)
        
        # Analyse de convergence (variance glissante)
        if len(rewards) > 50:
            window_var = min(50, len(rewards) // 20)
            variance = []
            var_episodes = []
            
            for i in range(window_var, len(rewards)):
                window_rewards = rewards[i-window_var:i]
                variance.append(np.var(window_rewards))
                var_episodes.append(episodes[i])
            
            axes[1, 1].plot(var_episodes, variance, color='green', linewidth=2)
            axes[1, 1].set_title(f'Variance des récompenses (fenêtre {window_var})')
            axes[1, 1].set_xlabel('Épisode')
            axes[1, 1].set_ylabel('Variance')
            axes[1, 1].grid(True)
        else:
            # Si pas assez de données pour la variance, afficher les pertes si disponibles
            if hasattr(self, 'metrics') and self.metrics.get('training_losses'):
                axes[1, 1].plot(self.metrics['training_losses'])
                axes[1, 1].set_title('Perte d\'entraînement')
                axes[1, 1].set_xlabel('Épisode')
                axes[1, 1].set_ylabel('Perte')
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    #Nouvelles méthodes ajoutées
    def get_value_function(self):
        """
        Retourne la fonction de valeur apprise.
        
        Returns:
            Copy de la fonction de valeur ou None si non entraîné
        """
        if not self.is_trained or self.value_function is None:
            return None
        
        return self.value_function.copy()
    
    def update_value_function(self):
        """
        Met à jour la fonction de valeur V(s) = max_a Q(s, a).
        
        Méthode par défaut qui peut être redéfinie par les sous-classes.
        """
        if self.q_function is not None:
            if isinstance(self.q_function, np.ndarray):
                self.value_function = np.max(self.q_function, axis=1)
            else:
                # Pour les dictionnaires Q
                self.value_function = np.zeros(self.state_space_size)
                for state in range(self.state_space_size):
                    q_values = [self.q_function.get((state, action), 0.0) 
                               for action in range(self.action_space_size)]
                    self.value_function[state] = np.max(q_values)
    
    def reset_training(self):
        """
        Remet à zéro l'entraînement.
        
        Méthode par défaut qui peut être redéfinie par les sous-classes.
        """
        self.is_trained = False
        self.training_history = []
        self.policy = None
        self.q_function = None
        self.value_function = None
    
    def visualize_q_table(self, precision: int = 2):
        """
        Affiche la Q-table de manière lisible.
        
        Args:
            precision: Nombre de décimales
            
        Returns:
            String formaté de la Q-table
        """
        if not self.is_trained or self.q_function is None:
            return "❌ Algorithme non entraîné"
        
        output = f"\n{'='*50}\n"
        output += f"Q-TABLE - {self.algo_name}\n"
        output += f"{'='*50}\n"
        output += f"{'État':<6}"
        
        for action in range(self.action_space_size):
            output += f"Action{action:<8}"
        output += f"{'Politique':<10}{'Valeur':<10}\n"
        output += "-" * 50 + "\n"
        
        for state in range(self.state_space_size):
            output += f"{state:<6}"
            
            # Gestion des Q-functions en array ou dict
            if isinstance(self.q_function, np.ndarray):
                q_values = self.q_function[state]
                best_action = np.argmax(q_values)
                state_value = np.max(q_values)
            else:
                q_values = [self.q_function.get((state, action), 0.0) 
                           for action in range(self.action_space_size)]
                best_action = np.argmax(q_values)
                state_value = np.max(q_values)
            
            # Affichage des Q-values
            for action in range(self.action_space_size):
                if isinstance(self.q_function, np.ndarray):
                    q_val = self.q_function[state, action]
                else:
                    q_val = self.q_function.get((state, action), 0.0)
                output += f"{q_val:<12.{precision}f} "
            
            # Politique et valeur
            if self.value_function is not None:
                state_value = self.value_function[state]
            
            output += f"{best_action:<10}{state_value:<10.{precision}f}\n"
        
        output += "=" * 50 + "\n"
        return output
    