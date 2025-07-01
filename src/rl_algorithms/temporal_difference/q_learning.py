"""
Q-Learning Algorithm - Algorithme d'apprentissage par renforcement de type Temporal Difference.

Q-Learning est un algorithme off-policy qui apprend la fonction de valeur action-état optimale Q*(s,a)
directement sans connaître la politique suivie.
"""

import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import sys
import os

# Ajouter le chemin vers la classe de base
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.rl_algorithms.base_algorithm import BaseAlgorithm, TrainingStats


class QLearning(BaseAlgorithm):
    """
    Implémentation de l'algorithme Q-Learning.
    
    Q-Learning utilise la mise à jour:
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(self, 
                 state_space_size: int,
                 action_space_size: int,
                 learning_rate: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 initial_q_value: float = 0.0):
        """
        Initialise l'algorithme Q-Learning.
        
        Args:
            state_space_size (int): Nombre d'états dans l'environnement
            action_space_size (int): Nombre d'actions possibles
            learning_rate (float): Taux d'apprentissage (α)
            gamma (float): Facteur d'actualisation
            epsilon (float): Taux d'exploration initial
            epsilon_decay (float): Facteur de décroissance d'epsilon
            epsilon_min (float): Valeur minimale d'epsilon
            initial_q_value (float): Valeur initiale des Q-values
        """
        super().__init__(
            algo_name="Q-Learning",
            state_space_size=state_space_size,
            action_space_size=action_space_size,
            gamma=gamma,
            epsilon=epsilon
        )
        
        # Hyperparamètres spécifiques à Q-Learning
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_q_value = initial_q_value
        
        # Initialisation de la Q-table
        self.q_function = np.full(
            (state_space_size, action_space_size), 
            initial_q_value, 
            dtype=np.float64
        )
        
        # Politique dérivée de la Q-function
        self.policy = None
        self.value_function = None
        
        # Métriques de convergence
        self.q_value_changes = []
        self.convergence_threshold = 1e-4
        
    @property
    def algorithm_type(self) -> str:
        """Retourne le type d'algorithme."""
        return "td"  # Temporal Difference
    
    @property
    def required_environment_features(self) -> List[str]:
        """Retourne les fonctionnalités requises de l'environnement."""
        return ["episodes", "step_by_step"]  # Pas besoin de modèle complet
    
    def train(self, environment, num_episodes: int, 
              verbose: bool = False, 
              convergence_check_interval: int = 100,
              **kwargs) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Q-Learning sur un environnement.
        
        Args:
            environment: Environnement d'entraînement
            num_episodes (int): Nombre d'épisodes d'entraînement
            verbose (bool): Affichage des informations de progression
            convergence_check_interval (int): Intervalle pour vérifier la convergence
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Dict[str, Any]: Statistiques d'entraînement et résultats
        """
        if verbose:
            print(f"Démarrage de l'entraînement Q-Learning pour {num_episodes} épisodes")
            print(f"Paramètres: α={self.learning_rate}, γ={self.gamma}, ε={self.epsilon}")
        
        # Réinitialisation pour un nouvel entraînement
        self.training_stats = []
        self.q_value_changes = []
        
        for episode in range(num_episodes):
            episode_reward, episode_steps, q_change = self._run_episode(environment)
            
            # Décroissance d'epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
            
            # Enregistrement des statistiques
            self.q_value_changes.append(q_change)
            convergence_metric = q_change if q_change is not None else 0.0
            
            self.update_training_stats(
                episode=episode + 1,
                total_reward=episode_reward,
                steps=episode_steps,
                convergence_metric=convergence_metric,
                additional_metrics={"epsilon": self.epsilon, "q_change": q_change}
            )
            
            # Affichage périodique
            if verbose and (episode + 1) % 100 == 0:
                print(f"Épisode {episode + 1}/{num_episodes}: "
                      f"Récompense={episode_reward:.2f}, "
                      f"Étapes={episode_steps}, "
                      f"ε={self.epsilon:.3f}, "
                      f"ΔQ={q_change:.6f}")
            
            # Vérification de convergence
            if (episode + 1) % convergence_check_interval == 0:
                if self._check_convergence():
                    if verbose:
                        print(f"Convergence détectée à l'épisode {episode + 1}")
                    break
        
        # Calcul des politiques finales
        self._compute_final_policies()
        self.is_trained = True
        
        if verbose:
            summary = self.get_training_summary()
            print(f"\nEntraînement terminé:")
            print(f"- Récompense moyenne: {summary['avg_reward']:.2f}")
            print(f"- Récompense finale: {summary['final_episode_reward']:.2f}")
            print(f"- Epsilon final: {self.epsilon:.3f}")
        
        return self.get_training_summary()
    
    def _run_episode(self, environment) -> Tuple[float, int, Optional[float]]:
        """
        Exécute un épisode d'entraînement.
        
        Args:
            environment: Environnement d'entraînement
            
        Returns:
            Tuple[float, int, Optional[float]]: Récompense totale, nombre d'étapes, changement Q-values
        """
        state = environment.reset()
        total_reward = 0.0
        steps = 0
        max_steps = getattr(environment, 'max_steps', 1000)
        
        # Pour mesurer le changement des Q-values
        q_values_before = self.q_function.copy()
        
        for step in range(max_steps):
            # Sélection d'action avec epsilon-greedy
            action = self.select_action(state, training=True)
            
            # Exécution de l'action
            next_state, reward, done, info = environment.step(action)
            
            # Mise à jour Q-Learning
            self._update_q_value(state, action, reward, next_state, done)
            
            # Transition vers l'état suivant
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Calcul du changement des Q-values
        q_change = np.mean(np.abs(self.q_function - q_values_before))
        
        return total_reward, steps, q_change
    
    def _update_q_value(self, state: int, action: int, reward: float, 
                       next_state: int, done: bool):
        """
        Met à jour une Q-value selon la règle Q-Learning.
        
        Args:
            state (int): État actuel
            action (int): Action exécutée
            reward (float): Récompense reçue
            next_state (int): État suivant
            done (bool): True si l'épisode est terminé
        """
        current_q = self.q_function[state, action]
        
        if done:
            # Si l'épisode est terminé, pas de valeur future
            target_q = reward
        else:
            # Q-Learning: utilise le maximum des Q-values de l'état suivant
            max_next_q = np.max(self.q_function[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Mise à jour avec le taux d'apprentissage
        self.q_function[state, action] = current_q + self.learning_rate * (target_q - current_q)
    
    def select_action(self, state: int, training: bool = False, **kwargs) -> int:
        """
        Sélectionne une action pour un état donné.
        
        Args:
            state (int): État actuel
            training (bool): True si en mode entraînement (avec exploration)
            **kwargs: Paramètres supplémentaires
            
        Returns:
            int: Action sélectionnée
        """
        if training:
            # Mode entraînement: epsilon-greedy avec epsilon courant
            return self.select_epsilon_greedy_action(state, self.epsilon)
        else:
            # Mode évaluation: action gloutonne
            return self.select_greedy_action(state)
    
    def get_policy(self) -> np.ndarray:
        """
        Retourne la politique dérivée de la Q-function.
        
        Returns:
            np.ndarray: Politique optimale (action pour chaque état)
        """
        if self.policy is None:
            self._compute_final_policies()
        return self.policy
    
    def get_value_function(self) -> np.ndarray:
        """
        Retourne la fonction de valeur d'état dérivée de la Q-function.
        
        Returns:
            np.ndarray: Fonction de valeur d'état
        """
        if self.value_function is None:
            self._compute_final_policies()
        return self.value_function
    
    def get_q_function(self) -> np.ndarray:
        """
        Retourne la Q-function apprise.
        
        Returns:
            np.ndarray: Q-function (matrice états x actions)
        """
        return self.q_function
    
    def _compute_final_policies(self):
        """Calcule les politiques et fonctions de valeur finales."""
        # Politique: action avec la plus haute Q-value pour chaque état
        self.policy = np.argmax(self.q_function, axis=1)
        
        # Fonction de valeur: maximum des Q-values pour chaque état
        self.value_function = np.max(self.q_function, axis=1)
    
    def _check_convergence(self, window_size: int = 50) -> bool:
        """
        Vérifie si l'algorithme a convergé.
        
        Args:
            window_size (int): Taille de la fenêtre pour le calcul de convergence
            
        Returns:
            bool: True si convergé
        """
        if len(self.q_value_changes) < window_size:
            return False
        
        # Moyenne des changements récents
        recent_changes = self.q_value_changes[-window_size:]
        avg_change = np.mean(recent_changes)
        
        return avg_change < self.convergence_threshold
    
    def save_model(self, filepath: str) -> bool:
        """
        Sauvegarde le modèle Q-Learning.
        
        Args:
            filepath (str): Chemin de sauvegarde (sans extension)
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            # Sauvegarde des données du modèle
            model_data = {
                'q_function': self.q_function.tolist(),
                'policy': self.policy.tolist() if self.policy is not None else None,
                'value_function': self.value_function.tolist() if self.value_function is not None else None,
                'hyperparameters': self.get_hyperparameters(),
                'state_space_size': self.state_space_size,
                'action_space_size': self.action_space_size,
                'is_trained': self.is_trained,
                'episode_count': self.episode_count
            }
            
            # Sauvegarde JSON pour lisibilité
            with open(f"{filepath}.json", 'w') as f:
                json.dump(model_data, f, indent=2)
            
            # Sauvegarde pickle pour les objets NumPy
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(self, f)
            
            return True
            
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Charge un modèle Q-Learning pré-entraîné.
        
        Args:
            filepath (str): Chemin du modèle (sans extension)
            
        Returns:
            bool: True si le chargement a réussi
        """
        try:
            # Essaie d'abord le fichier JSON
            try:
                with open(f"{filepath}.json", 'r') as f:
                    model_data = json.load(f)
                
                self.q_function = np.array(model_data['q_function'])
                self.policy = np.array(model_data['policy']) if model_data['policy'] else None
                self.value_function = np.array(model_data['value_function']) if model_data['value_function'] else None
                self.is_trained = model_data['is_trained']
                self.episode_count = model_data['episode_count']
                
                # Restaure les hyperparamètres
                hyperparams = model_data['hyperparameters']
                self.set_hyperparameters(**hyperparams)
                
            except FileNotFoundError:
                # Essaie le fichier pickle
                with open(f"{filepath}.pkl", 'rb') as f:
                    loaded_model = pickle.load(f)
                
                # Copie les attributs importants
                self.q_function = loaded_model.q_function
                self.policy = loaded_model.policy
                self.value_function = loaded_model.value_function
                self.is_trained = loaded_model.is_trained
                self.episode_count = loaded_model.episode_count
                self.training_stats = loaded_model.training_stats
            
            return True
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return False
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Retourne les hyperparamètres de Q-Learning.
        
        Returns:
            Dict[str, Any]: Hyperparamètres actuels
        """
        base_params = super().get_hyperparameters()
        q_params = {
            "learning_rate": self.learning_rate,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "initial_q_value": self.initial_q_value,
            "convergence_threshold": self.convergence_threshold
        }
        return {**base_params, **q_params}
    
    def set_hyperparameters(self, **kwargs):
        """
        Met à jour les hyperparamètres de Q-Learning.
        
        Args:
            **kwargs: Nouveaux hyperparamètres
        """
        # Hyperparamètres spécifiques à Q-Learning
        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']
        if 'epsilon_decay' in kwargs:
            self.epsilon_decay = kwargs['epsilon_decay']
        if 'epsilon_min' in kwargs:
            self.epsilon_min = kwargs['epsilon_min']
        if 'convergence_threshold' in kwargs:
            self.convergence_threshold = kwargs['convergence_threshold']
        
        # Appel de la méthode parent pour les hyperparamètres de base
        super().set_hyperparameters(**kwargs)
    
    def get_action_probabilities(self, state: int, temperature: float = 1.0) -> np.ndarray:
        """
        Retourne les probabilités d'actions pour un état (softmax sur Q-values).
        
        Args:
            state (int): État pour lequel calculer les probabilités
            temperature (float): Paramètre de température pour softmax
            
        Returns:
            np.ndarray: Probabilités d'actions
        """
        q_values = self.q_function[state] / temperature
        exp_q = np.exp(q_values - np.max(q_values))  # Stabilité numérique
        return exp_q / np.sum(exp_q)
    
    def get_q_value(self, state: int, action: int) -> float:
        """
        Retourne la Q-value pour une paire état-action.
        
        Args:
            state (int): État
            action (int): Action
            
        Returns:
            float: Q-value
        """
        return self.q_function[state, action]
    
    def reset_training(self):
        """Remet à zéro l'entraînement de Q-Learning."""
        super().reset_training()
        
        # Réinitialise la Q-table
        self.q_function = np.full(
            (self.state_space_size, self.action_space_size),
            self.initial_q_value,
            dtype=np.float64
        )
        
        # Remet les politiques à None
        self.policy = None
        self.value_function = None
        
        # Remet epsilon à sa valeur initiale
        # Note: la valeur initiale doit être stockée lors de l'initialisation
        self.epsilon = getattr(self, 'initial_epsilon', 0.1)
        
        # Réinitialise les métriques
        self.q_value_changes = []
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """
        Retourne des informations sur la convergence de l'algorithme.
        
        Returns:
            Dict[str, Any]: Informations de convergence
        """
        if not self.q_value_changes:
            return {"message": "Aucune donnée de convergence disponible"}
        
        return {
            "total_episodes": len(self.q_value_changes),
            "final_q_change": self.q_value_changes[-1],
            "min_q_change": min(self.q_value_changes),
            "max_q_change": max(self.q_value_changes),
            "avg_q_change": np.mean(self.q_value_changes),
            "converged": self._check_convergence(),
            "convergence_threshold": self.convergence_threshold
        }
    
    def visualize_q_table(self, precision: int = 3) -> str:
        """
        Retourne une représentation textuelle de la Q-table.
        
        Args:
            precision (int): Nombre de décimales à afficher
            
        Returns:
            str: Q-table formatée
        """
        if not self.is_trained:
            return "Algorithme non entraîné - Q-table non disponible"
        
        output = "\n=== Q-Table ===\n"
        output += f"{'État':<6}"
        for action in range(self.action_space_size):
            output += f"Action{action:<7}"
        output += f"{'Politique':<10}\n"
        output += "-" * (10 + self.action_space_size * 12) + "\n"
        
        for state in range(self.state_space_size):
            output += f"{state:<6}"
            for action in range(self.action_space_size):
                q_val = self.q_function[state, action]
                output += f"{q_val:<11.{precision}f} "
            
            best_action = self.policy[state] if self.policy is not None else np.argmax(self.q_function[state])
            output += f"{best_action:<10}\n"
        
        return output
    
    def compare_with_optimal(self, optimal_policy: Dict[int, int], 
                           optimal_values: Dict[int, float] = None) -> Dict[str, Any]:
        """
        Compare la politique apprise avec une politique optimale.
        
        Args:
            optimal_policy (Dict[int, int]): Politique optimale de référence
            optimal_values (Dict[int, float], optional): Valeurs optimales de référence
            
        Returns:
            Dict[str, Any]: Résultats de la comparaison
        """
        if not self.is_trained:
            return {"error": "Algorithme non entraîné"}
        
        policy = self.get_policy()
        
        # Comparaison des politiques
        total_states = len(optimal_policy)
        correct_actions = sum(1 for state in optimal_policy.keys() 
                             if policy[state] == optimal_policy[state])
        policy_accuracy = correct_actions / total_states
        
        results = {
            "policy_accuracy": policy_accuracy,
            "correct_actions": correct_actions,
            "total_states": total_states,
            "incorrect_states": [state for state in optimal_policy.keys() 
                               if policy[state] != optimal_policy[state]]
        }
        
        # Comparaison des valeurs si disponibles
        if optimal_values is not None:
            value_function = self.get_value_function()
            value_errors = []
            for state in optimal_values.keys():
                error = abs(value_function[state] - optimal_values[state])
                value_errors.append(error)
            
            results.update({
                "avg_value_error": np.mean(value_errors),
                "max_value_error": np.max(value_errors),
                "min_value_error": np.min(value_errors),
                "std_value_error": np.std(value_errors)
            })
        
        return results


# Fonctions utilitaires pour créer des configurations pré-définies
def create_standard_qlearning(state_space_size: int, action_space_size: int) -> QLearning:
    """Crée un Q-Learning avec paramètres standards."""
    return QLearning(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.1,
        gamma=0.9,
        epsilon=0.1,
        epsilon_decay=0.995
    )


def create_fast_learning_qlearning(state_space_size: int, action_space_size: int) -> QLearning:
    """Crée un Q-Learning pour apprentissage rapide."""
    return QLearning(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.3,
        gamma=0.95,
        epsilon=0.3,
        epsilon_decay=0.99,
        epsilon_min=0.05
    )


def create_conservative_qlearning(state_space_size: int, action_space_size: int) -> QLearning:
    """Crée un Q-Learning conservateur pour environnements complexes."""
    return QLearning(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.05,
        gamma=0.99,
        epsilon=0.05,
        epsilon_decay=0.9995,
        epsilon_min=0.001
    )


if __name__ == "__main__":
    # Test rapide de Q-Learning
    print("Test de Q-Learning")
    
    # Création d'un algorithme simple
    q_agent = create_standard_qlearning(state_space_size=5, action_space_size=2)
    print(f"Q-Learning créé: {q_agent}")
    
    # Affichage des hyperparamètres
    print("Hyperparamètres:")
    for key, value in q_agent.get_hyperparameters().items():
        print(f"  {key}: {value}")
    
    # Test de sélection d'action
    test_state = 0
    print(f"\nTest de sélection d'action pour l'état {test_state}:")
    for i in range(5):
        action = q_agent.select_action(test_state, training=True)
        print(f"  Action {i+1}: {action}")
    
    print("\nQ-table initiale:")
    print(q_agent.visualize_q_table())