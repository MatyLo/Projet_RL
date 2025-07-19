"""
Q-Learning Algorithm - Implémentation simplifiée

Q-Learning utilise la mise à jour:
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

Workflow:
1. Créer avec from_config()
2. Entraîner avec train()
3. Utiliser avec Agent pour évaluation/démo
"""

import numpy as np
import pickle
import json
import time
from typing import Dict, Any
import sys
import os

# Ajout des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'utils'))

from src.rl_algorithms.base_algorithm import BaseAlgorithm


class QLearning(BaseAlgorithm):
    """
    Implémentation Q-Learning simplifiée.
    
    Exemple d'utilisation:
    >>> config = {'learning_rate': 0.1, 'gamma': 0.9, 'epsilon': 0.1}
    >>> q_algo = QLearning.from_config(config, environment)
    >>> q_algo.train(environment, num_episodes=1000)
    >>> # Ensuite utiliser avec Agent wrapper
    >>> tout le workflow peut être fait en lancant un des tests dans demo_scripts (v4)
    """
    
    def __init__(self, 
                 state_space_size: int,
                 action_space_size: int,
                 learning_rate: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialise Q-Learning.
        
        Args:
            state_space_size: Nombre d'états
            action_space_size: Nombre d'actions
            learning_rate: Taux d'apprentissage (α)
            gamma: Facteur d'actualisation
            epsilon: Taux d'exploration initial
            epsilon_decay: Facteur de décroissance d'epsilon
            epsilon_min: Valeur minimale d'epsilon
        """
        super().__init__(
            algo_name="Q-Learning",
            state_space_size=state_space_size,
            action_space_size=action_space_size
        )
        
        # Hyperparamètres
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_epsilon = epsilon  # Pour reset
        
        # Q-table: matrice [états x actions]
        self.q_function = np.zeros((state_space_size, action_space_size))
        
        # Politique et fonction de valeur (calculées après entraînement)
        self.policy = None
        self.value_function = None
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], environment):
        """
        Crée Q-Learning depuis une configuration JSON.
        
        Args:
            config: Dict avec learning_rate, gamma, epsilon, etc.
            environment: Environnement d'entraînement
            
        Returns:
            Instance configurée de Q-Learning
        """
        return cls(
            state_space_size=environment.state_space_size,
            action_space_size=environment.action_space_size,
            learning_rate=config.get('learning_rate', 0.1),
            gamma=config.get('gamma', 0.9),
            epsilon=config.get('epsilon', 0.1),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            epsilon_min=config.get('epsilon_min', 0.01)
        )
    
    def train(self, environment, num_episodes: int, verbose: bool = False):
        """
        Entraîne Q-Learning de manière autonome.
        
        Args:
            environment: Environnement d'entraînement
            num_episodes: Nombre d'épisodes
            verbose: Affichage des informations
            
        Returns:
            Dict avec statistiques d'entraînement
        """
        if verbose:
            print(f"🚀 Entraînement Q-Learning sur {num_episodes} épisodes")
            print(f"Paramètres: α={self.learning_rate}, γ={self.gamma}, ε={self.epsilon}")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_reward, episode_steps = self._run_episode(environment)
            
            # Décroissance d'epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
            
            # Enregistrement
            self.add_training_episode(episode + 1, episode_reward, episode_steps)
            
            # Affichage périodique
            if verbose and (episode + 1) % 100 == 0:
                print(f"Épisode {episode + 1}: Reward={episode_reward:.2f}, ε={self.epsilon:.3f}")
        
        # Finalisation
        self._compute_final_policy()
        self.is_trained = True
        training_time = time.time() - start_time
        
        if verbose:
            results = self.get_training_results()
            print(f"✅ Entraînement terminé en {training_time:.1f}s")
            print(f"Récompense moyenne: {results['avg_reward']:.2f}")
        
        return self.get_training_results()
    
    def _run_episode(self, environment):
        """Exécute un épisode d'entraînement."""
        state = environment.reset()
        total_reward = 0.0
        steps = 0
        max_steps = getattr(environment, 'max_steps', 1000)
        
        for step in range(max_steps):
            # Sélection action epsilon-greedy
            action = self.select_action(state, training=True)
            
            # Exécution
            next_state, reward, done, info = environment.step(action)
            
            # Mise à jour Q-Learning
            self._update_q_value(state, action, reward, next_state, done)
            
            # Transition
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        return total_reward, steps
    
    def _update_q_value(self, state: int, action: int, reward: float, 
                       next_state: int, done: bool):
        """
        Met à jour une Q-value selon la règle Q-Learning.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_function[state, action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_function[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Mise à jour
        self.q_function[state, action] = current_q + self.learning_rate * (target_q - current_q)
    
    def select_action(self, state: int, training: bool = False):
        """
        Sélectionne une action.
        
        Args:
            state: État actuel
            training: True pour epsilon-greedy, False pour greedy
            
        Returns:
            Action sélectionnée
        """
        if training:
            return self.select_epsilon_greedy_action(state, self.epsilon)
        else:
            return self.select_greedy_action(state)
    
    def get_policy(self):
        """Retourne la politique optimale (action pour chaque état)."""
        if self.policy is None:
            self._compute_final_policy()
        return self.policy
    
    def _compute_final_policy(self):
        """Calcule la politique et fonction de valeur finales."""
        self.policy = np.argmax(self.q_function, axis=1)
        self.value_function = np.max(self.q_function, axis=1)
    
    def save_model(self, filepath: str) -> bool:
        """
        Sauvegarde le modèle Q-Learning.
        
        Args:
            filepath: Chemin de sauvegarde (sans extension)
            
        Returns:
            True si succès
        """
        try:
            # Données à sauvegarder
            model_data = {
                'q_function': self.q_function.tolist(),
                'policy': self.policy.tolist() if self.policy is not None else None,
                'value_function': self.value_function.tolist() if self.value_function is not None else None,
                'hyperparameters': {
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma,
                    'epsilon_min': self.epsilon_min,
                    'epsilon_decay': self.epsilon_decay
                },
                'training_results': self.get_training_results(),
                'algorithm_info': {
                    'name': self.algo_name,
                    'state_space_size': self.state_space_size,
                    'action_space_size': self.action_space_size,
                    'is_trained': self.is_trained
                }
            }
            
            # Sauvegarde JSON
            with open(f"{filepath}.json", 'w') as f:
                json.dump(model_data, f, indent=2)
            
            # Sauvegarde pickle (pour les objets NumPy)
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(self, f)
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Charge un modèle Q-Learning.
        
        Args:
            filepath: Chemin du modèle (sans extension)
            
        Returns:
            True si succès
        """
        try:
            # Essaie le fichier JSON d'abord
            try:
                with open(f"{filepath}.json", 'r') as f:
                    model_data = json.load(f)
                
                self.q_function = np.array(model_data['q_function'])
                self.policy = np.array(model_data['policy']) if model_data['policy'] else None
                self.value_function = np.array(model_data['value_function']) if model_data['value_function'] else None
                self.is_trained = model_data['algorithm_info']['is_trained']
                
                # Restaure hyperparamètres
                hyperparams = model_data['hyperparameters']
                self.learning_rate = hyperparams['learning_rate']
                self.gamma = hyperparams['gamma']
                self.epsilon_min = hyperparams['epsilon_min']
                self.epsilon_decay = hyperparams['epsilon_decay']
                
            except FileNotFoundError:
                # Essaie le fichier pickle
                with open(f"{filepath}.pkl", 'rb') as f:
                    loaded_model = pickle.load(f)
                
                self.q_function = loaded_model.q_function
                self.policy = loaded_model.policy
                self.value_function = loaded_model.value_function
                self.is_trained = loaded_model.is_trained
                self.training_history = loaded_model.training_history
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
            return False
    
    def visualize_q_table(self, precision: int = 2):
        """
        Affiche la Q-table de manière lisible.
        
        Args:
            precision: Nombre de décimales
            
        Returns:
            String formaté de la Q-table
        """
        if not self.is_trained:
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
            for action in range(self.action_space_size):
                q_val = self.q_function[state, action]
                output += f"{q_val:<12.{precision}f} "
            
            best_action = self.policy[state] if self.policy is not None else np.argmax(self.q_function[state])
            state_value = self.value_function[state] if self.value_function is not None else np.max(self.q_function[state])
            output += f"{best_action:<10}{state_value:<10.{precision}f}\n"
        
        output += "=" * 50 + "\n"
        return output
    
    def reset_training(self):
        """Remet à zéro l'entraînement."""
        self.q_function = np.zeros((self.state_space_size, self.action_space_size))
        self.policy = None
        self.value_function = None
        self.epsilon = self.initial_epsilon
        self.is_trained = False
        self.training_history = []