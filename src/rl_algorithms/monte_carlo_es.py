"""
Monte Carlo Exploring Starts (MC ES) pour l'apprentissage par renforcement.

Version harmonisée avec l'architecture du projet :
- Hérite de BaseAlgorithm
- Interface from_config() standardisée
- Méthodes select_action() et save/load_model compatibles
- train(environment, num_episodes, verbose) avec signature cohérente
"""

import numpy as np
import json
import pickle
import time
import random
from typing import Dict, List, Any, Optional
from .base_algorithm import BaseAlgorithm


class MonteCarloES(BaseAlgorithm):
    """
    Monte Carlo Exploring Starts harmonisé.
    
    Algorithme d'apprentissage par renforcement qui utilise des starts exploratoires
    pour assurer l'exploration de toutes les paires état-action.
    """
    
    def __init__(self, 
                 state_space_size: int,
                 action_space_size: int,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialise Monte Carlo ES.
        
        Args:
            state_space_size: Nombre d'états
            action_space_size: Nombre d'actions
            gamma: Facteur d'actualisation
            epsilon: Taux d'exploration initial
            epsilon_decay: Facteur de décroissance d'epsilon
            epsilon_min: Valeur minimale d'epsilon
        """
        super().__init__("MonteCarloES", state_space_size, action_space_size)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_epsilon = epsilon  # Pour reset
        
        # Structures de données MC
        self.q_function = np.zeros((state_space_size, action_space_size))
        self.returns = [[[] for _ in range(action_space_size)] for _ in range(state_space_size)]
        self.policy = np.ones((state_space_size, action_space_size)) / action_space_size
        
        # Compteurs pour moyenne incrémentale
        self.visit_counts = np.zeros((state_space_size, action_space_size))
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], environment):
        """
        Crée Monte Carlo ES depuis une configuration.
        
        Args:
            config: Configuration avec gamma, epsilon, etc.
            environment: Environnement pour dimensionnement
            
        Returns:
            Instance configurée de MonteCarloES
        """
        return cls(
            state_space_size=environment.state_space_size,
            action_space_size=environment.action_space_size,
            gamma=config.get('gamma', 0.99),
            epsilon=config.get('epsilon', 0.1),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            epsilon_min=config.get('epsilon_min', 0.01)
        )
    
    def train(self, environment, num_episodes: int, verbose: bool = False):
        """
        Entraîne Monte Carlo ES.
        
        Args:
            environment: Environnement d'entraînement
            num_episodes: Nombre d'épisodes d'entraînement
            verbose: Affichage des informations de progression
            
        Returns:
            Dict avec résultats d'entraînement
        """
        if verbose:
            print(f"Entraînement Monte Carlo ES sur {num_episodes} épisodes...")
            print(f"États: {self.state_space_size}, Actions: {self.action_space_size}")
            print(f"Gamma: {self.gamma}, Epsilon initial: {self.epsilon}")
        
        start_time = time.time()
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            # EXPLORING STARTS: Choix aléatoire d'état et action de départ
            total_reward, episode_length = self._run_episode_with_exploring_starts(environment)
            
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            
            # Mise à jour epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
            
            # Enregistrement de l'épisode
            self.add_training_episode(episode + 1, total_reward, episode_length)
            
            if verbose and (episode + 1) % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Épisode {episode + 1}: reward moyen = {avg_reward:.3f}, epsilon = {self.epsilon:.3f}")
        
        # Finalisation
        training_time = time.time() - start_time
        self.is_trained = True
        
        if verbose:
            final_avg_reward = np.mean(episode_rewards[-100:])
            print(f"Entraînement terminé. Reward moyen final: {final_avg_reward:.3f}")
            print(f"Epsilon final: {self.epsilon:.3f}")
        
        return {
            'episodes': num_episodes,
            'final_avg_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0.0,
            'final_epsilon': self.epsilon,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'training_time': training_time
        }
    
    def _run_episode_with_exploring_starts(self, environment):
        """
        Exécute un épisode avec exploring starts.
        
        Args:
            environment: Environnement d'entraînement
            
        Returns:
            Tuple[float, int]: (total_reward, episode_length)
        """
        # EXPLORING STARTS: État de départ aléatoire
        start_state = np.random.randint(0, self.state_space_size)
        
        # Reset de l'environnement et forcer l'état de départ
        environment.reset()
        if hasattr(environment, 'current_state'):
            environment.current_state = start_state
        
        # Action de départ aléatoire parmi les actions valides
        valid_actions = environment.valid_actions
        if not valid_actions:
            return 0.0, 0
        
        first_action = random.choice(valid_actions)
        
        # Génération de l'épisode
        episode_states = [start_state]
        episode_actions = [first_action]
        episode_rewards = []
        
        # Premier pas avec l'action de départ
        state = start_state
        action = first_action
        done = False
        max_steps = getattr(environment, 'max_steps', 1000)
        
        for step in range(max_steps):
            if done:
                break
                
            # Exécution de l'action
            next_state, reward, done, _ = environment.step(action)
            episode_rewards.append(reward)
            
            if not done:
                # Sélection de l'action suivante selon la politique courante
                valid_actions = environment.valid_actions
                if not valid_actions:
                    break
                    
                action = self._select_action_epsilon_greedy(next_state, valid_actions)
                episode_states.append(next_state)
                episode_actions.append(action)
            
            state = next_state
        
        # Mise à jour des valeurs Q avec les returns
        self._update_q_values_from_episode(episode_states, episode_actions, episode_rewards)
        
        # Mise à jour de la politique
        self._update_policy()
        
        total_reward = sum(episode_rewards)
        episode_length = len(episode_rewards)
        
        return total_reward, episode_length
    
    def _select_action_epsilon_greedy(self, state: int, valid_actions: List[int]) -> int:
        """
        Sélection d'action epsilon-greedy restreinte aux actions valides.
        
        Args:
            state: État actuel
            valid_actions: Actions valides dans cet état
            
        Returns:
            Action sélectionnée
        """
        if np.random.random() < self.epsilon:
            # Exploration: action aléatoire
            return random.choice(valid_actions)
        else:
            # Exploitation: meilleure action selon Q
            q_values = [self.q_function[state, a] for a in valid_actions]
            best_action_idx = np.argmax(q_values)
            return valid_actions[best_action_idx]
    
    def _update_q_values_from_episode(self, states: List[int], actions: List[int], rewards: List[float]):
        """
        Met à jour les Q-values à partir d'un épisode complet.
        
        Args:
            states: Séquence d'états
            actions: Séquence d'actions
            rewards: Séquence de récompenses
        """
        G = 0.0
        visited_sa_pairs = set()
        
        # Parcours inverse de l'épisode (first-visit)
        for t in reversed(range(len(states))):
            if t < len(rewards):
                G = self.gamma * G + rewards[t]
            
            state = states[t]
            action = actions[t]
            sa_pair = (state, action)
            
            # First-visit: mise à jour seulement si première visite
            if sa_pair not in visited_sa_pairs:
                visited_sa_pairs.add(sa_pair)
                
                # Mise à jour avec moyenne incrémentale pour efficacité
                self.visit_counts[state, action] += 1
                n = self.visit_counts[state, action]
                
                # Q(s,a) = Q(s,a) + (1/n) * [G - Q(s,a)]
                self.q_function[state, action] += (G - self.q_function[state, action]) / n
    
    def _update_policy(self):
        """Met à jour la politique epsilon-greedy basée sur Q."""
        for state in range(self.state_space_size):
            # Politique epsilon-greedy
            best_action = np.argmax(self.q_function[state])
            
            # Politique uniforme avec boost pour la meilleure action
            self.policy[state] = np.full(self.action_space_size, self.epsilon / self.action_space_size)
            self.policy[state, best_action] += 1.0 - self.epsilon
    
    def select_action(self, state: int, training: bool = False):
        """
        Sélectionne une action selon la politique apprise.
        
        Args:
            state: État actuel
            training: Si True, utilise epsilon-greedy, sinon greedy pur
            
        Returns:
            Action sélectionnée
        """
        if not self.is_trained:
            raise ValueError("Algorithme non entraîné")
        
        if training:
            return self.select_epsilon_greedy_action(state, self.epsilon)
        else:
            return self.select_greedy_action(state)
    
    def get_policy(self):
        """
        Retourne la politique apprise.
        
        Returns:
            np.ndarray: Politique déterministe [état] = action
        """
        if not self.is_trained:
            return None
        
        return np.argmax(self.q_function, axis=1)
    
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
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'is_trained': self.is_trained,
                'q_function': self.q_function,
                'policy': self.policy,
                'visit_counts': self.visit_counts,
                'training_history': self.training_history
            }
            
            # Sauvegarde JSON avec conversion des arrays
            json_file = f"{filepath}.json"
            with open(json_file, 'w') as f:
                json_data = model_data.copy()
                json_data['q_function'] = self.q_function.tolist()
                json_data['policy'] = self.policy.tolist()
                json_data['visit_counts'] = self.visit_counts.tolist()
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
                    model_data['q_function'] = np.array(json_data['q_function'])
                    model_data['policy'] = np.array(json_data['policy'])
                    model_data['visit_counts'] = np.array(json_data['visit_counts'])
            
            # Vérification de compatibilité
            if (model_data['state_space_size'] != self.state_space_size or
                model_data['action_space_size'] != self.action_space_size):
                raise ValueError("Incompatibilité des dimensions")
            
            # Chargement des données
            self.q_function = model_data['q_function']
            self.policy = model_data['policy']
            self.visit_counts = model_data['visit_counts']
            self.is_trained = model_data['is_trained']
            self.epsilon = model_data['epsilon']
            self.training_history = model_data.get('training_history', [])
            
            return True
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return False
    
    def reset_training(self):
        """Remet à zéro l'entraînement."""
        super().reset_training()
        self.q_function = np.zeros((self.state_space_size, self.action_space_size))
        self.returns = [[[] for _ in range(self.action_space_size)] for _ in range(self.state_space_size)]
        self.policy = np.ones((self.state_space_size, self.action_space_size)) / self.action_space_size
        self.visit_counts = np.zeros((self.state_space_size, self.action_space_size))
        self.epsilon = self.initial_epsilon