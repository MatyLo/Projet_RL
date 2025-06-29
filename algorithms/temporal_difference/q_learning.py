from typing import Dict, Any, Tuple
import numpy as np
from ..base_algorithm import BaseAlgorithm

class QLearning(BaseAlgorithm):
    """
    Implémentation de l'algorithme Q-Learning.
    """
    
    def __init__(self, environment: Any, learning_rate: float = 0.1,
                 discount_factor: float = 0.99, epsilon: float = 0.1):
        """
        Initialise l'algorithme Q-Learning.
        
        Args:
            environment: L'environnement sur lequel l'algorithme va s'entraîner
            learning_rate: Taux d'apprentissage
            discount_factor: Facteur d'actualisation
            epsilon: Paramètre d'exploration
        """
        super().__init__(environment)
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Initialisation de la table Q
        n_states = len(environment.get_state_space())
        n_actions = len(environment.get_action_space())
        self.q_table = np.zeros((n_states, n_actions))
        
    def get_action(self, state: int) -> int:
        """
        Retourne l'action à prendre dans un état donné.
        
        Args:
            state: L'état actuel
            
        Returns:
            L'action à prendre
        """
        if np.random.random() < self.epsilon:
            # Exploration
            return np.random.choice(self.environment.get_action_space())
        else:
            # Exploitation
            return np.argmax(self.q_table[state])
    
    def train(self, n_episodes: int) -> Dict[str, Any]:
        """
        Entraîne l'algorithme sur l'environnement.
        
        Args:
            n_episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Dict contenant les métriques d'entraînement
        """
        rewards = []
        
        for episode in range(n_episodes):
            state = self.environment.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.environment.step(action)
                
                # Mise à jour de la table Q
                best_next_action = np.argmax(self.q_table[next_state])
                td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_error
                
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
            
        return {
            "rewards": rewards,
            "mean_reward": np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        }
    
    def save(self, path: str) -> None:
        """
        Sauvegarde la table Q.
        
        Args:
            path: Chemin où sauvegarder la table Q
        """
        np.save(path, self.q_table)
    
    def load(self, path: str) -> None:
        """
        Charge la table Q.
        
        Args:
            path: Chemin de la table Q à charger
        """
        self.q_table = np.load(path) 