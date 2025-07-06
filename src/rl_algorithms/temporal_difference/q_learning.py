"""
Q-Learning Algorithm - Implémentation avec Approche Hybride

Cette version supporte :
1. Configuration via JSON avec from_config()
2. Entraînement autonome (style professeur)
3. Compatible avec Agent wrapper pour post-entraînement
4. Métriques de convergence avancées
"""

import numpy as np
import pickle
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import sys
import os

from src.rl_algorithms.base_algorithm import BaseAlgorithm, TrainingStats


class QLearning(BaseAlgorithm):
    """
    Implémentation Q-Learning avec architecture hybride.
    
    Q-Learning utilise la mise à jour:
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    
    Nouvelle Architecture :
    - Phase 1 : Configuration JSON -> Entraînement autonome
    - Phase 2 : Compatible Agent wrapper -> Évaluation/Démonstration
    """
    
    def __init__(self, 
                 state_space_size: int,
                 action_space_size: int,
                 learning_rate: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 initial_q_value: float = 0.0,
                 convergence_threshold: float = 1e-4,
                 **kwargs):
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
            convergence_threshold (float): Seuil de convergence
            **kwargs: Paramètres supplémentaires
        """
        super().__init__(
            algo_name="Q-Learning",
            state_space_size=state_space_size,
            action_space_size=action_space_size,
            gamma=gamma,
            epsilon=epsilon,
            convergence_threshold=convergence_threshold,
            **kwargs
        )
        
        # Hyperparamètres spécifiques à Q-Learning
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_q_value = initial_q_value
        self.initial_epsilon = epsilon  # Sauvegarde valeur initiale
        
        # Initialisation de la Q-table
        self.q_function = np.full(
            (state_space_size, action_space_size), 
            initial_q_value, 
            dtype=np.float64
        )
        
        # Politique dérivée de la Q-function
        self.policy = None
        self.value_function = None
        
        # Métriques de convergence spécifiques
        self.q_value_changes = []
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], environment) -> 'QLearning':
        """
         Crée un algorithme Q-Learning depuis une configuration JSON.
        
        Cette méthode implémente l'approche hybride : configuration JSON -> algorithme autonome.
        
        Args:
            config (Dict[str, Any]): Configuration de l'algorithme
            environment: Environnement d'entraînement
            
        Returns:
            QLearning: Instance configurée de Q-Learning
            
        Raises:
            ValueError: Si la configuration est invalide
        """
        # Validation de la configuration
        required_keys = ['learning_rate', 'gamma', 'epsilon']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Paramètre manquant dans la configuration: {key}")
        
        # Création de l'instance avec la configuration
        return cls(
            state_space_size=environment.state_space_size,
            action_space_size=environment.action_space_size,
            learning_rate=config.get('learning_rate', 0.1),
            gamma=config.get('gamma', 0.9),
            epsilon=config.get('epsilon', 0.1),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            epsilon_min=config.get('epsilon_min', 0.01),
            initial_q_value=config.get('initial_q_value', 0.0),
            convergence_threshold=config.get('convergence_threshold', 1e-4)
        )
    
    def to_config(self) -> Dict[str, Any]:
        """
        Exporte la configuration actuelle de l'algorithme.
        
        Returns:
            Dict[str, Any]: Configuration de l'algorithme
        """
        base_config = super().to_config()
        q_learning_config = {
            'algorithm_type': 'q_learning',
            'learning_rate': self.learning_rate,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'initial_q_value': self.initial_q_value,
            'initial_epsilon': self.initial_epsilon
        }
        base_config.update(q_learning_config)
        return base_config
    
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
        Entraîne l'algorithme Q-Learning de manière autonome.
        
        Implémente l'entraînement autonome pour l'approche hybride.
        L'algorithme gère tout lui-même sans dépendre d'un Agent.
        
        Args:
            environment: Environnement d'entraînement
            num_episodes (int): Nombre d'épisodes d'entraînement
            verbose (bool): Affichage des informations de progression
            convergence_check_interval (int): Intervalle pour vérifier la convergence
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Dict[str, Any]: Statistiques d'entraînement et résultats
        """
        # Validation de compatibilité
        self._validate_environment_compatibility(environment)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"ENTRAÎNEMENT AUTONOME Q-LEARNING")
            print(f"{'='*60}")
            print(f"Épisodes: {num_episodes}")
            print(f"Paramètres: α={self.learning_rate}, γ={self.gamma}, ε={self.epsilon}")
            print(f"Environnement: {environment.env_name}")
            print(f"{'='*60}\n")
        
        # Démarrage du chronométrage
        self._start_training_timer()
        
        # Réinitialisation pour un nouvel entraînement
        self.reset_training()
        
        for episode in range(num_episodes):
            episode_reward, episode_steps, q_change = self._run_training_episode(environment)
            
            # Décroissance d'epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
            
            # Enregistrement des statistiques
            self.q_value_changes.append(q_change)
            
            self.update_training_stats(
                episode=episode + 1,
                total_reward=episode_reward,
                steps=episode_steps,
                convergence_metric=q_change,
                additional_metrics={
                    "epsilon": self.epsilon, 
                    "q_change": q_change,
                    "avg_q_value": np.mean(self.q_function)
                }
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
                        print(f"✅ Convergence détectée à l'épisode {episode + 1}")
                    break
        
        # Finalisation de l'entraînement
        self._end_training_timer()
        self._compute_final_policies()
        self.is_trained = True
        
        # Résultats d'entraînement
        training_results = self.get_training_results()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"ENTRAÎNEMENT TERMINÉ")
            print(f"Temps: {training_results['training_time']:.2f}s")
            print(f"Récompense moyenne: {training_results['avg_reward']:.2f}")
            print(f"Récompense finale: {training_results['final_episode_reward']:.2f}")
            print(f"Convergé: {training_results['converged']}")
            print(f"{'='*60}\n")
        
        return training_results
    
    def _run_training_episode(self, environment) -> Tuple[float, int, float]:
        """
        Exécute un épisode d'entraînement autonome.
        
        Args:
            environment: Environnement d'entraînement
            
        Returns:
            Tuple[float, int, float]: Récompense totale, nombre d'étapes, changement Q-values
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
        MODIFIÉE : Sélectionne une action pour un état donné.
        
        Compatible avec l'Agent wrapper qui peut spécifier training=False
        pour l'évaluation post-entraînement.
        
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
    
    def save_model(self, filepath: str) -> bool:
        """
        AMÉLIORÉE : Sauvegarde le modèle Q-Learning avec métadonnées.
        
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
                'algorithm_info': self.get_algorithm_info(),
                'training_results': self.get_training_results(),
                'config_export': self.to_config(),
                'q_value_changes': self.q_value_changes
            }
            
            # Sauvegarde JSON pour lisibilité
            with open(f"{filepath}.json", 'w') as f:
                json.dump(model_data, f, indent=2)
            
            # Sauvegarde pickle pour les objets NumPy
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(self, f)
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        AMÉLIORÉE : Charge un modèle Q-Learning pré-entraîné.
        
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
                self.is_trained = model_data['algorithm_info']['is_trained']
                self.episode_count = model_data['algorithm_info']['episode_count']
                
                # Restaure les métriques de convergence si disponibles
                if 'q_value_changes' in model_data:
                    self.q_value_changes = model_data['q_value_changes']
                
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
                self.q_value_changes = getattr(loaded_model, 'q_value_changes', [])
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        AMÉLIORÉE : Retourne les hyperparamètres de Q-Learning.
        
        Returns:
            Dict[str, Any]: Hyperparamètres actuels
        """
        base_params = super().get_hyperparameters()
        q_params = {
            "learning_rate": self.learning_rate,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "initial_q_value": self.initial_q_value,
            "initial_epsilon": self.initial_epsilon
        }
        return {**base_params, **q_params}
    
    def set_hyperparameters(self, **kwargs):
        """
        AMÉLIORÉE : Met à jour les hyperparamètres de Q-Learning.
        
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
        if 'initial_q_value' in kwargs:
            self.initial_q_value = kwargs['initial_q_value']
        if 'initial_epsilon' in kwargs:
            self.initial_epsilon = kwargs['initial_epsilon']
        
        # Appel de la méthode parent pour les hyperparamètres de base
        super().set_hyperparameters(**kwargs)
    
    # ==================== MÉTHODES SPÉCIFIQUES Q-LEARNING ====================
    
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
        """AMÉLIORÉE : Remet à zéro l'entraînement de Q-Learning."""
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
        self.epsilon = self.initial_epsilon
        
        # Réinitialise les métriques spécifiques
        self.q_value_changes = []
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """
        NOUVELLE : Retourne des informations sur la convergence de l'algorithme.
        
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
            "convergence_threshold": self.convergence_threshold,
            "q_change_history": self.q_value_changes[-10:]  # Derniers 10 valeurs
        }
    
    def visualize_q_table(self, precision: int = 3) -> str:
        """
        NOUVELLE : Retourne une représentation textuelle de la Q-table.
        
        Args:
            precision (int): Nombre de décimales à afficher
            
        Returns:
            str: Q-table formatée
        """
        if not self.is_trained:
            return "❌ Algorithme non entraîné - Q-table non disponible"
        
        output = "\n" + "="*60 + "\n"
        output += "Q-TABLE APPRISE\n"
        output += "="*60 + "\n"
        output += f"{'État':<6}"
        for action in range(self.action_space_size):
            output += f"Action{action:<7}"
        output += f"{'Politique':<10}{'Valeur':<10}\n"
        output += "-" * 60 + "\n"
        
        for state in range(self.state_space_size):
            output += f"{state:<6}"
            for action in range(self.action_space_size):
                q_val = self.q_function[state, action]
                output += f"{q_val:<11.{precision}f} "
            
            best_action = self.policy[state] if self.policy is not None else np.argmax(self.q_function[state])
            state_value = self.value_function[state] if self.value_function is not None else np.max(self.q_function[state])
            output += f"{best_action:<10}{state_value:<10.{precision}f}\n"
        
        output += "="*60 + "\n"
        return output
    
    def compare_with_optimal(self, optimal_policy: Dict[int, int], 
                           optimal_values: Dict[int, float] = None) -> Dict[str, Any]:
        """
        NOUVELLE : Compare la politique apprise avec une politique optimale.
        
        Args:
            optimal_policy (Dict[int, int]): Politique optimale de référence
            optimal_values (Dict[int, float], optional): Valeurs optimales de référence
            
        Returns:
            Dict[str, Any]: Résultats de la comparaison
        """
        if not self.is_ready_for_evaluation():
            return {"error": "Algorithme non entraîné ou non prêt"}
        
        policy = self.get_policy()
        
        # Comparaison des politiques
        total_states = len(optimal_policy)
        correct_actions = sum(1 for state in optimal_policy.keys() 
                             if policy[state] == optimal_policy[state])
        policy_accuracy = correct_actions / total_states
        
        results = {
            "algorithm": self.algo_name,
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
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """
        NOUVELLE : Retourne des statistiques détaillées pour l'analyse.
        
        Returns:
            Dict[str, Any]: Statistiques complètes
        """
        if not self.is_trained:
            return {"message": "Algorithme non entraîné"}
        
        stats = {
            "algorithm_info": self.get_algorithm_info(),
            "training_results": self.get_training_results(),
            "convergence_info": self.get_convergence_info(),
            "hyperparameters": self.get_hyperparameters(),
            "q_table_stats": {
                "shape": self.q_function.shape,
                "min_q_value": float(np.min(self.q_function)),
                "max_q_value": float(np.max(self.q_function)),
                "mean_q_value": float(np.mean(self.q_function)),
                "std_q_value": float(np.std(self.q_function)),
                "zero_q_values": int(np.sum(self.q_function == 0))
            }
        }
        
        return stats


# ==================== FONCTIONS UTILITAIRES ====================

def create_qlearning_from_config(config_path: str, environment) -> QLearning:
    """
    NOUVELLE : Fonction utilitaire pour créer Q-Learning depuis un fichier de config.
    
    Args:
        config_path (str): Chemin vers le fichier de configuration
        environment: Environnement d'entraînement
        
    Returns:
        QLearning: Instance configurée
    """
    from utils.config_loader import load_config
    
    config = load_config(config_path)
    q_learning_config = config['algorithms']['q_learning']
    
    return QLearning.from_config(q_learning_config, environment)


def quick_train_qlearning(environment, config_name: str = "q_learning", 
                         config_path: str = None, verbose: bool = True) -> QLearning:
    """
    NOUVELLE : Fonction utilitaire pour entraînement rapide de Q-Learning.
    
    Args:
        environment: Environnement d'entraînement
        config_name (str): Nom de la configuration dans le fichier JSON
        config_path (str): Chemin vers le fichier de configuration
        verbose (bool): Affichage des détails
        
    Returns:
        QLearning: Algorithme entraîné
    """
    from utils.config_loader import load_config, create_default_config
    
    if config_path:
        config = load_config(config_path)
    else:
        # Configuration par défaut
        config = create_default_config("lineworld", "q_learning")
    
    # Création et entraînement
    algorithm = QLearning.from_config(config['algorithms'][config_name], environment)
    num_episodes = config['algorithms'][config_name].get('num_episodes', 1000)
    
    algorithm.train(environment, num_episodes=num_episodes, verbose=verbose)
    
    return algorithm


# Fonctions de compatibilité avec l'ancienne API
def create_standard_qlearning(state_space_size: int, action_space_size: int) -> QLearning:
    """Fonction de compatibilité : Crée un Q-Learning avec paramètres standards."""
    return QLearning(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.1,
        gamma=0.9,
        epsilon=0.1,
        epsilon_decay=0.995
    )


def create_fast_learning_qlearning(state_space_size: int, action_space_size: int) -> QLearning:
    """Fonction de compatibilité : Crée un Q-Learning pour apprentissage rapide."""
    return QLearning(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.3,
        gamma=0.95,
        epsilon=0.3,
        epsilon_decay=0.99,
        epsilon_min=0.05
    )


if __name__ == "__main__":
    # Test de la version hybride
    print("🧪 Test de Q-Learning avec Architecture Hybride")
    
    # Test 1: Création standard
    print("\n1. Test création standard:")
    q_agent = create_standard_qlearning(state_space_size=5, action_space_size=2)
    print(f"✅ {q_agent}")
    
    # Test 2: Configuration JSON (simulée)
    print("\n2. Test configuration JSON:")
    test_config = {
        'learning_rate': 0.2,
        'gamma': 0.95,
        'epsilon': 0.15,
        'num_episodes': 500
    }
    
    # Simulation d'un environnement simple
    class MockEnvironment:
        def __init__(self):
            self.state_space_size = 5
            self.action_space_size = 2
            self.env_name = "MockEnv"
        
        def reset(self): return 0
        def step(self, action): return 1, 0.1, False, {}
    
    mock_env = MockEnvironment()
    q_agent_config = QLearning.from_config(test_config, mock_env)
    print(f"✅ {q_agent_config}")
    print(f"Hyperparamètres: {q_agent_config.get_hyperparameters()}")
    
    # Test 3: Export de configuration
    print("\n3. Test export configuration:")
    exported_config = q_agent_config.to_config()
    print(f"✅ Configuration exportée: {len(exported_config)} paramètres")
    
    print("\n🎉 Tests de l'architecture hybride réussis !")