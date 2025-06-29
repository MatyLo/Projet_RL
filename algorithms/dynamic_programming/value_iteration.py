from typing import Dict, Any, Tuple, List
import numpy as np
from ..base_algorithm import BaseAlgorithm

class ValueIteration(BaseAlgorithm):
    """
    Implémentation de l'algorithme Value Iteration.
    Cet algorithme calcule directement la fonction de valeur optimale
    puis en déduit la politique optimale.
    """
    
    def __init__(self, environment: Any, discount_factor: float = 0.999999,
                 theta: float = 0.00001, max_iterations: int = 1000):
        """
        Initialise l'algorithme Value Iteration.
        
        Args:
            environment: L'environnement sur lequel l'algorithme va s'entraîner
            discount_factor: Facteur d'actualisation
            theta: Seuil de convergence
            max_iterations: Nombre maximum d'itérations
        """
        super().__init__(environment)
        self.gamma = discount_factor
        self.theta = theta
        self.max_iterations = max_iterations
        
        # Récupération des paramètres de l'environnement
        self.S = environment.get_state_space()
        self.A = environment.get_action_space()
        self.R = environment.get_rewards()
        self.T = environment.get_terminal_states()
        self.p = environment.get_transition_matrix()
        
        # Initialisation de la fonction de valeur et de la politique
        self.V = np.random.random((len(self.S),))
        self.V[self.T] = 0.0
        self.policy = np.zeros((len(self.S), len(self.A)))
        
    def train(self, n_episodes: int = None) -> Dict[str, Any]:
        """
        Entraîne l'algorithme sur l'environnement.
        
        Args:
            n_episodes: Non utilisé pour cet algorithme
            
        Returns:
            Dict contenant les métriques d'entraînement
        """
        iterations = 0
        
        while True:
            delta = 0.0
            
            # Mise à jour de la fonction de valeur
            for s in self.S:
                v = self.V[s]
                best_value = float('-inf')
                
                for a in self.A:
                    value = 0.0
                    for s_p in self.S:
                        for r_index in range(len(self.R)):
                            r = self.R[r_index]
                            value += self.p[s, a, s_p, r_index] * (r + self.gamma * self.V[s_p])
                    best_value = max(best_value, value)
                
                self.V[s] = best_value
                delta = max(delta, abs(v - self.V[s]))
            
            iterations += 1
            if delta < self.theta or iterations >= self.max_iterations:
                break
        
        # Calcul de la politique optimale
        for s in self.S:
            best_a = None
            best_a_score = float('-inf')
            
            for a in self.A:
                score = 0.0
                for s_p in self.S:
                    for r_index in range(len(self.R)):
                        r = self.R[r_index]
                        score += self.p[s, a, s_p, r_index] * (r + self.gamma * self.V[s_p])
                
                if score > best_a_score:
                    best_a = a
                    best_a_score = score
            
            self.policy[s] = np.zeros_like(self.policy[s])
            self.policy[s][best_a] = 1.0
        
        return {
            "iterations": iterations,
            "converged": delta < self.theta,
            "final_value_function": self.V.copy()
        }
    
    def get_action(self, state: int) -> int:
        """
        Retourne l'action à prendre dans un état donné.
        
        Args:
            state: L'état actuel
            
        Returns:
            L'action à prendre
        """
        return np.argmax(self.policy[state])
    
    def save(self, path: str) -> None:
        """
        Sauvegarde la politique et la fonction de valeur.
        
        Args:
            path: Chemin où sauvegarder les données
        """
        np.savez(path, policy=self.policy, value_function=self.V)
    
    def load(self, path: str) -> None:
        """
        Charge la politique et la fonction de valeur.
        
        Args:
            path: Chemin des données à charger
        """
        data = np.load(path)
        self.policy = data['policy']
        self.V = data['value_function'] 