"""
Algorithme Q-Learning pour l'apprentissage par renforcement.

Implémente l'algorithme Q-Learning off-policy pour l'apprentissage de politiques
dans des environnements avec modèle inconnu.
"""

import numpy as np
import json
import pickle
from typing import Dict, List, Any, Optional
from .base_algorithm import BaseAlgorithm


class QLearning(BaseAlgorithm):
    """
    Algorithme Q-Learning.
    
    Algorithme off-policy qui apprend une politique optimale en utilisant
    l'action optimale pour la mise à jour, indépendamment de la politique suivie.
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
            learning_rate: Taux d'apprentissage (alpha)
            gamma: Facteur d'actualisation
            epsilon: Paramètre d'exploration initial
            epsilon_decay: Facteur de décroissance d'epsilon
            epsilon_min: Valeur minimale d'epsilon
        """
        super().__init__("QLearning", state_space_size, action_space_size)
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialiser la fonction Q
        self.q_function = np.zeros((state_space_size, action_space_size))
        
        # Politique dérivée de Q
        self.policy = None
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], environment):
        """Crée Q-Learning depuis une configuration."""
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
        Entraîne Q-Learning sur l'environnement.
        
        Args:
            environment: Environnement d'entraînement
            num_episodes: Nombre d'épisodes d'entraînement
            verbose: Affichage des informations de progression
        """
        if verbose:
            print(f"Entraînement Q-Learning sur {num_episodes} épisodes...")
            print(f"États: {self.state_space_size}, Actions: {self.action_space_size}")
            print(f"Learning rate: {self.learning_rate}, Gamma: {self.gamma}")
            print(f"Epsilon initial: {self.epsilon}")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = environment.reset()
            total_reward = 0
            steps = 0
            
            done = False
            while not done:
                # Sélectionner une action epsilon-greedy parmi les actions valides
                valid_actions = environment.valid_actions
                if np.random.random() < self.epsilon:
                    action = np.random.choice(valid_actions)
                else:
                    # Action gloutonne parmi les actions valides
                    q_values = [self.q_function[state, a] for a in valid_actions]
                    best_action_idx = np.argmax(q_values)
                    action = valid_actions[best_action_idx]
                
                # Exécuter l'action
                next_state, reward, done, _ = environment.step(action)
                total_reward += reward
                steps += 1
                
                # Mettre à jour Q(s,a) avec Q-Learning
                current_q = self.q_function[state, action]
                
                if not done:
                    # Q-Learning: utiliser le maximum des Q-values du prochain état
                    max_next_q = np.max(self.q_function[next_state])
                    td_target = reward + self.gamma * max_next_q
                else:
                    # Épisode terminé
                    td_target = reward
                
                # Mise à jour: Q(s,a) = Q(s,a) + α[r + γmax Q(s',a') - Q(s,a)]
                td_error = td_target - current_q
                self.q_function[state, action] = current_q + self.learning_rate * td_error
                
                state = next_state
            
            # Mettre à jour epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Enregistrer l'épisode
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            self.add_training_episode(episode + 1, total_reward, steps)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Épisode {episode + 1}: reward moyen = {avg_reward:.3f}, epsilon = {self.epsilon:.3f}")
        
        # Mettre à jour la politique
        self.policy = np.argmax(self.q_function, axis=1)
        self.is_trained = True
        
        if verbose:
            final_avg_reward = np.mean(episode_rewards[-100:])
            print(f"Entraînement terminé. Reward moyen final: {final_avg_reward:.3f}")
            print(f"Epsilon final: {self.epsilon:.3f}")
        
        return {
            'episodes': num_episodes,
            'final_avg_reward': np.mean(episode_rewards[-100:]),
            'final_epsilon': self.epsilon,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
    
    def select_action(self, state: int, training: bool = False):
        """Sélectionne une action selon la politique apprise."""
        if not self.is_trained:
            raise ValueError("Algorithme non entraîné")
        
        if training:
            return self.select_epsilon_greedy_action(state, self.epsilon)
        else:
            return self.select_greedy_action(state)
    
    def get_policy(self):
        """Retourne la politique apprise."""
        if not self.is_trained:
            return None
        
        return self.policy.copy()
    
    def save_model(self, filepath: str) -> bool:
        """Sauvegarde le modèle entraîné."""
        try:
            model_data = {
                'algorithm': self.algo_name,
                'state_space_size': self.state_space_size,
                'action_space_size': self.action_space_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'is_trained': self.is_trained,
                'q_function': self.q_function,
                'policy': self.policy,
                'training_history': self.training_history
            }
            
            # Sauvegarder en JSON et pickle
            json_file = filepath.replace('.pkl', '.json')
            with open(json_file, 'w') as f:
                json_data = model_data.copy()
                json_data['q_function'] = self.q_function.tolist()
                json_data['policy'] = self.policy.tolist() if self.policy is not None else None
                json.dump(json_data, f, indent=2)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
            
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Charge un modèle pré-entraîné."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Vérifier la compatibilité
            if (model_data['state_space_size'] != self.state_space_size or
                model_data['action_space_size'] != self.action_space_size):
                raise ValueError("Incompatibilité des dimensions")
            
            # Charger les données
            self.q_function = model_data['q_function']
            self.policy = model_data['policy']
            self.is_trained = model_data['is_trained']
            self.epsilon = model_data['epsilon']
            self.training_history = model_data.get('training_history', [])
            
            return True
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return False 