"""
Algorithme Value Iteration.

Implémente l'algorithme de programmation dynamique Value Iteration
pour résoudre des MDP avec modèle connu.
"""

import numpy as np
import json
import pickle
from typing import Dict, List, Any, Optional
from .base_algorithm import BaseAlgorithm


class ValueIteration(BaseAlgorithm):
    """
    Algorithme Value Iteration.
    
    Résout un MDP en itérant sur la fonction de valeur jusqu'à convergence.
    Nécessite un modèle de l'environnement (probabilités de transition et récompenses).
    """
    
    def __init__(self, 
                 state_space_size: int, 
                 action_space_size: int,
                 gamma: float = 0.9,
                 theta: float = 1e-6,
                 max_iterations: int = 1000):
        """
        Initialise Value Iteration.
        
        Args:
            state_space_size: Nombre d'états
            action_space_size: Nombre d'actions
            gamma: Facteur d'actualisation
            theta: Seuil de convergence
            max_iterations: Nombre maximum d'itérations
        """
        super().__init__("ValueIteration", state_space_size, action_space_size)
        
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        
        # Initialiser les fonctions
        self.value_function = np.zeros(state_space_size)
        self.q_function = np.zeros((state_space_size, action_space_size))
        self.policy = np.zeros(state_space_size, dtype=int)
        
        # Historique de convergence
        self.convergence_history = []
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], environment):
        """Crée Value Iteration depuis une configuration."""
        return cls(
            state_space_size=environment.state_space_size,
            action_space_size=environment.action_space_size,
            gamma=config.get('gamma', 0.9),
            theta=config.get('theta', 1e-6),
            max_iterations=config.get('max_iterations', 1000)
        )
    
    def train(self, environment, num_episodes: int = None, verbose: bool = False):
        """
        Entraîne Value Iteration.
        
        Note: num_episodes n'est pas utilisé pour Value Iteration
        car l'algorithme converge naturellement.
        """
        if verbose:
            print(f"Entraînement Value Iteration...")
            print(f"États: {self.state_space_size}, Actions: {self.action_space_size}")
            print(f"Gamma: {self.gamma}, Theta: {self.theta}")
        
        # Value Iteration
        iteration = 0
        delta = float('inf')
        
        while delta > self.theta and iteration < self.max_iterations:
            delta = 0
            
            for state in range(self.state_space_size):
                v_old = self.value_function[state]
                # Utiliser uniquement les actions valides pour cet état
                if hasattr(environment, 'get_valid_actions'):
                    valid_actions = environment.get_valid_actions(state)
                elif hasattr(environment, 'valid_actions'):
                    valid_actions = environment.valid_actions if isinstance(environment.valid_actions, list) else list(environment.valid_actions)
                else:
                    valid_actions = range(self.action_space_size)
                q_values = []
                for action in valid_actions:
                    q_value = 0
                    # Obtenir les probabilités de transition
                    transitions = environment.get_transition_probabilities(state, action)
                    for next_state, prob in transitions.items():
                        reward = environment.get_reward_function(state, action, next_state)
                        q_value += prob * (reward + self.gamma * self.value_function[next_state])
                    q_values.append(q_value)
                    self.q_function[state, action] = q_value
                # Mettre à jour la fonction de valeur
                if q_values:
                    self.value_function[state] = max(q_values)
                    # Mettre à jour la politique
                    # On retrouve l'action correspondant à la valeur max
                    best_action = valid_actions[np.argmax(q_values)]
                    self.policy[state] = best_action
                else:
                    # Aucun action valide, garder la valeur précédente
                    self.value_function[state] = v_old
                # Calculer le delta pour la convergence
                delta = max(delta, abs(v_old - self.value_function[state]))
            
            # Enregistrer l'historique
            self.convergence_history.append({
                'iteration': iteration,
                'delta': delta,
                'max_value': np.max(self.value_function)
            })
            
            if verbose and iteration % 100 == 0:
                print(f"Itération {iteration}: delta = {delta:.6f}")
            
            iteration += 1
        
        # Marquer comme entraîné
        self.is_trained = True
        
        # Ajouter un épisode factice pour la compatibilité
        self.add_training_episode(1, np.mean(self.value_function), iteration)
        
        if verbose:
            print(f"Convergence en {iteration} itérations")
            print(f"Valeur maximale: {np.max(self.value_function):.4f}")
        
        return {
            'iterations': iteration,
            'final_delta': delta,
            'converged': delta <= self.theta,
            'max_value': np.max(self.value_function),
            'convergence_history': self.convergence_history
        }
    
    def select_action(self, state: int, training: bool = False):
        """Sélectionne une action selon la politique optimale."""
        if not self.is_trained:
            raise ValueError("Algorithme non entraîné")
        
        # Value Iteration utilise toujours la politique optimale
        return self.policy[state]
    
    def get_policy(self):
        """Retourne la politique optimale."""
        if not self.is_trained:
            return None
        
        return self.policy.copy()
    
    def get_value_function(self):
        """Retourne la fonction de valeur optimale."""
        if not self.is_trained:
            return None
        
        return self.value_function.copy()
    
    def save_model(self, filepath: str) -> bool:
        """Sauvegarde le modèle entraîné."""
        try:
            model_data = {
                'algorithm': self.algo_name,
                'state_space_size': self.state_space_size,
                'action_space_size': self.action_space_size,
                'gamma': self.gamma,
                'theta': self.theta,
                'is_trained': self.is_trained,
                'value_function': self.value_function,
                'q_function': self.q_function,
                'policy': self.policy,
                'convergence_history': self.convergence_history,
                'training_history': self.training_history
            }
            
            # Sauvegarder en JSON et pickle
            json_file = filepath.replace('.pkl', '.json')
            with open(json_file, 'w') as f:
                # Convertir les arrays numpy en listes pour JSON
                json_data = model_data.copy()
                json_data['value_function'] = self.value_function.tolist()
                json_data['q_function'] = self.q_function.tolist()
                json_data['policy'] = self.policy.tolist()
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
            self.value_function = model_data['value_function']
            self.q_function = model_data['q_function']
            self.policy = model_data['policy']
            self.is_trained = model_data['is_trained']
            self.convergence_history = model_data.get('convergence_history', [])
            self.training_history = model_data.get('training_history', [])
            
            return True
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return False 