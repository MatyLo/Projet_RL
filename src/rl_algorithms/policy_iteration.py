"""
Algorithme Policy Iteration pour l'apprentissage par renforcement.

Version harmonisée avec l'architecture du projet :
- Hérite de BaseAlgorithm
- Interface from_config() standardisée
- Méthodes select_action() et save/load_model compatibles
- train(environment, ...) avec signature cohérente
"""

import numpy as np
import json
import pickle
import time
from typing import Dict, List, Any, Optional
from .base_algorithm import BaseAlgorithm


class PolicyIteration(BaseAlgorithm):
    """
    Algorithme Policy Iteration harmonisé.
    
    Résout un MDP en alternant entre évaluation de politique et amélioration de politique
    jusqu'à convergence vers la politique optimale.
    """
    
    def __init__(self, 
                 state_space_size: int, 
                 action_space_size: int,
                 gamma: float = 0.999999,
                 theta: float = 0.00001,
                 max_iterations: int = 1000):
        """
        Initialise Policy Iteration.
        
        Args:
            state_space_size: Nombre d'états
            action_space_size: Nombre d'actions
            gamma: Facteur d'actualisation
            theta: Seuil de convergence pour l'évaluation de politique
            max_iterations: Nombre maximum d'itérations
        """
        super().__init__("PolicyIteration", state_space_size, action_space_size)
        
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        
        # Initialisation des fonctions
        self.value_function = np.random.random((state_space_size,))
        self.policy = np.ones((state_space_size, action_space_size)) / action_space_size  # Politique uniforme
        self.q_function = np.zeros((state_space_size, action_space_size))
        
        # Historique de convergence
        self.convergence_history = []
        
        # Variables pour l'environnement (définies dans train)
        self.S = None
        self.A = None
        self.R = None
        self.T = None
        self.p = None
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], environment):
        """
        Crée Policy Iteration depuis une configuration.
        
        Args:
            config: Configuration avec gamma, theta, max_iterations
            environment: Environnement pour dimensionnement
            
        Returns:
            Instance configurée de PolicyIteration
        """
        return cls(
            state_space_size=environment.state_space_size,
            action_space_size=environment.action_space_size,
            gamma=config.get('gamma', 0.999999),
            theta=config.get('theta', 0.00001),
            max_iterations=config.get('max_iterations', 1000)
        )
    
    def train(self, environment, num_episodes: int = None, verbose: bool = False):
        """
        Entraîne Policy Iteration.
        
        Args:
            environment: Environnement avec modèle de transition
            num_episodes: Non utilisé pour Policy Iteration (convergence naturelle)
            verbose: Affichage des informations de progression
            
        Returns:
            Dict avec résultats d'entraînement
        """
        if verbose:
            print(f"Entraînement Policy Iteration...")
            print(f"États: {self.state_space_size}, Actions: {self.action_space_size}")
            print(f"Gamma: {self.gamma}, Theta: {self.theta}")
        
        start_time = time.time()
        
        # Récupération des paramètres de l'environnement
        self.S = list(range(self.state_space_size))
        self.A = list(range(self.action_space_size))
        self.T = environment.get_terminal_states()
        
        # Initialisation de la fonction de valeur pour les états terminaux
        for terminal_state in self.T:
            self.value_function[terminal_state] = 0.0
        
        iteration = 0
        policy_stable = False
        
        while not policy_stable and iteration < self.max_iterations:
            # 1. ÉVALUATION DE POLITIQUE
            self._policy_evaluation(environment)
            
            # 2. AMÉLIORATION DE POLITIQUE
            policy_stable = self._policy_improvement(environment)
            
            # Enregistrement de l'historique
            self.convergence_history.append({
                'iteration': iteration,
                'policy_stable': policy_stable,
                'max_value': np.max(self.value_function)
            })
            
            if verbose and iteration % 10 == 0:
                print(f"Itération {iteration}: Politique stable = {policy_stable}")
            
            iteration += 1
        
        # Calcul final de la Q-function
        self._compute_q_function(environment)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        # Ajouter un épisode factice pour la compatibilité
        self.add_training_episode(1, np.mean(self.value_function), iteration)
        
        if verbose:
            print(f"Convergence en {iteration} itérations")
            print(f"Politique stable: {policy_stable}")
            print(f"Valeur maximale: {np.max(self.value_function):.4f}")
        
        return {
            'iterations': iteration,
            'converged': policy_stable,
            'max_value': np.max(self.value_function),
            'training_time': training_time,
            'convergence_history': self.convergence_history
        }
    
    def _policy_evaluation(self, environment):
        """
        Évaluation de la politique courante.
        
        Args:
            environment: Environnement avec modèle de transition
        """
        while True:
            delta = 0.0
            
            for s in self.S:
                if s in self.T:  # États terminaux
                    continue
                    
                v_old = self.value_function[s]
                total = 0.0
                
                for a in self.A:
                    # Calcul de la somme pondérée selon la politique
                    sub_total = 0.0
                    
                    # Obtenir les transitions possibles
                    transitions = environment.get_transition_probabilities(s, a)
                    
                    for s_prime, prob in transitions.items():
                        reward = environment.get_reward_function(s, a, s_prime)
                        sub_total += prob * (reward + self.gamma * self.value_function[s_prime])
                    
                    total += self.policy[s, a] * sub_total
                
                self.value_function[s] = total
                delta = max(delta, abs(v_old - self.value_function[s]))
            
            if delta < self.theta:
                break
    
    def _policy_improvement(self, environment) -> bool:
        """
        Amélioration de la politique.
        
        Args:
            environment: Environnement avec modèle de transition
            
        Returns:
            bool: True si la politique est stable (pas de changement)
        """
        policy_stable = True
        
        for s in self.S:
            if s in self.T:  # États terminaux
                continue
                
            # Ancienne action (politique actuelle)
            old_action = np.argmax(self.policy[s])
            
            # Trouver la meilleure action
            best_action = None
            best_action_value = float('-inf')
            
            for a in self.A:
                action_value = 0.0
                
                # Calcul de la valeur de l'action
                transitions = environment.get_transition_probabilities(s, a)
                
                for s_prime, prob in transitions.items():
                    reward = environment.get_reward_function(s, a, s_prime)
                    action_value += prob * (reward + self.gamma * self.value_function[s_prime])
                
                if action_value > best_action_value:
                    best_action_value = action_value
                    best_action = a
            
            # Mise à jour de la politique (politique déterministe)
            if best_action != old_action:
                policy_stable = False
            
            # Nouvelle politique déterministe
            self.policy[s] = np.zeros(self.action_space_size)
            self.policy[s, best_action] = 1.0
        
        return policy_stable
    
    def _compute_q_function(self, environment):
        """
        Calcule la Q-function finale à partir de la politique optimale.
        
        Args:
            environment: Environnement avec modèle de transition
        """
        for s in self.S:
            for a in self.A:
                q_value = 0.0
                
                transitions = environment.get_transition_probabilities(s, a)
                
                for s_prime, prob in transitions.items():
                    reward = environment.get_reward_function(s, a, s_prime)
                    q_value += prob * (reward + self.gamma * self.value_function[s_prime])
                
                self.q_function[s, a] = q_value
    
    def select_action(self, state: int, training: bool = False):
        """
        Sélectionne une action selon la politique optimale.
        
        Args:
            state: État actuel
            training: Non utilisé pour Policy Iteration (déterministe)
            
        Returns:
            Action sélectionnée
        """
        if not self.is_trained:
            raise ValueError("Algorithme non entraîné")
        
        return np.argmax(self.policy[state])
    
    def get_policy(self):
        """
        Retourne la politique optimale.
        
        Returns:
            np.ndarray: Politique déterministe [état] = action
        """
        if not self.is_trained:
            return None
        
        return np.argmax(self.policy, axis=1)
    
    def get_value_function(self):
        """
        Retourne la fonction de valeur optimale.
        
        Returns:
            np.ndarray: Fonction de valeur
        """
        if not self.is_trained:
            return None
        
        return self.value_function.copy()
    
    def save_model(self, filepath: str) -> bool:
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            filepath: Chemin de sauvegarde (sans extension)
            
        Returns:
            True si succès
        """
        try:
            model_data = {
                'algorithm': self.algo_name,
                'state_space_size': self.state_space_size,
                'action_space_size': self.action_space_size,
                'gamma': self.gamma,
                'theta': self.theta,
                'is_trained': self.is_trained,
                'value_function': self.value_function,
                'policy': self.policy,
                'q_function': self.q_function,
                'convergence_history': self.convergence_history,
                'training_history': self.training_history
            }
            
            # Sauvegarde JSON avec conversion des arrays
            json_file = f"{filepath}.json"
            with open(json_file, 'w') as f:
                json_data = model_data.copy()
                json_data['value_function'] = self.value_function.tolist()
                json_data['policy'] = self.policy.tolist()
                json_data['q_function'] = self.q_function.tolist()
                json.dump(json_data, f, indent=2)
            
            # Sauvegarde pickle
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
            
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Charge un modèle pré-entraîné.
        
        Args:
            filepath: Chemin du modèle (sans extension)
            
        Returns:
            True si succès
        """
        try:
            # Essaie d'abord le fichier pickle
            try:
                with open(f"{filepath}.pkl", 'rb') as f:
                    model_data = pickle.load(f)
            except FileNotFoundError:
                # Sinon le fichier JSON
                with open(f"{filepath}.json", 'r') as f:
                    json_data = json.load(f)
                    model_data = json_data.copy()
                    model_data['value_function'] = np.array(json_data['value_function'])
                    model_data['policy'] = np.array(json_data['policy'])
                    model_data['q_function'] = np.array(json_data['q_function'])
            
            # Vérification de compatibilité
            if (model_data['state_space_size'] != self.state_space_size or
                model_data['action_space_size'] != self.action_space_size):
                raise ValueError("Incompatibilité des dimensions")
            
            # Chargement des données
            self.value_function = model_data['value_function']
            self.policy = model_data['policy']
            self.q_function = model_data['q_function']
            self.is_trained = model_data['is_trained']
            self.convergence_history = model_data.get('convergence_history', [])
            self.training_history = model_data.get('training_history', [])
            
            return True
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return False