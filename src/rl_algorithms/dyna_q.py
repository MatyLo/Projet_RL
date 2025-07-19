"""
Algorithme Dyna-Q

architecture :
- Hérite de BaseAlgorithm
- Interface from_config()
- Méthodes select_action() et save/load_model
- train(environment, num_episodes, verbose)
- Combine Q-Learning avec planification par modèle
"""

import numpy as np
import json
import pickle
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from .base_algorithm import BaseAlgorithm


class DynaQ(BaseAlgorithm):
    """
    Algorithme Dyna-Q .
    
    Combine apprentissage direct (Q-Learning) avec planification en utilisant
    un modèle appris de l'environnement pour effectuer des mises à jour supplémentaires.
    """
    
    def __init__(self, 
                 state_space_size: int,
                 action_space_size: int,
                 learning_rate: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 n_planning_steps: int = 10):
        """
        Initialise Dyna-Q.
        
        Args:
            state_space_size: Nombre d'états
            action_space_size: Nombre d'actions
            learning_rate: Taux d'apprentissage (alpha)
            gamma: Facteur d'actualisation
            epsilon: Taux d'exploration initial
            epsilon_decay: Facteur de décroissance d'epsilon
            epsilon_min: Valeur minimale d'epsilon
            n_planning_steps: Nombre d'étapes de planification par step réel
        """
        super().__init__("DynaQ", state_space_size, action_space_size)
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_planning_steps = n_planning_steps
        self.initial_epsilon = epsilon
        
        # Q-function
        self.q_function = np.zeros((state_space_size, action_space_size))
        
        # Modèle de l'environnement: (state, action) -> (next_state, reward)
        self.model = {}
        
        # Ensemble des paires (état, action) observées pour la planification
        self.observed_sa_pairs = set()
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], environment):
        """
        Crée Dyna-Q depuis une configuration.
        
        Args:
            config: Configuration avec learning_rate, gamma, epsilon, n_planning_steps
            environment: Environnement pour dimensionnement
            
        Returns:
            Instance configurée de DynaQ
        """
        return cls(
            state_space_size=environment.state_space_size,
            action_space_size=environment.action_space_size,
            learning_rate=config.get('learning_rate', 0.1),
            gamma=config.get('gamma', 0.99),
            epsilon=config.get('epsilon', 0.1),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            epsilon_min=config.get('epsilon_min', 0.01),
            n_planning_steps=config.get('n_planning_steps', 10)
        )
    
    def train(self, environment, num_episodes: int, verbose: bool = False):
        """
        Entraîne Dyna-Q.
        
        Args:
            environment: Environnement d'entraînement
            num_episodes: Nombre d'épisodes d'entraînement
            verbose: Affichage des informations de progression
            
        Returns:
            Dict avec résultats d'entraînement
        """
        if verbose:
            print(f"Entraînement Dyna-Q sur {num_episodes} épisodes...")
            print(f"États: {self.state_space_size}, Actions: {self.action_space_size}")
            print(f"Learning rate: {self.learning_rate}, Gamma: {self.gamma}")
            print(f"Epsilon initial: {self.epsilon}, Planning steps: {self.n_planning_steps}")
        
        start_time = time.time()
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            total_reward, episode_length = self._run_episode(environment)
            
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            
            # Mise à jour epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
            
            # Enregistrement de l'épisode
            self.add_training_episode(episode + 1, total_reward, episode_length)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Épisode {episode + 1}: reward moyen = {avg_reward:.3f}, epsilon = {self.epsilon:.3f}")
        
        # Finalisation
        training_time = time.time() - start_time
        self.is_trained = True
        
        if verbose:
            final_avg_reward = np.mean(episode_rewards[-100:])
            print(f"Entraînement terminé. Reward moyen final: {final_avg_reward:.3f}")
            print(f"Epsilon final: {self.epsilon:.3f}")
            print(f"Paires (état, action) dans le modèle: {len(self.model)}")
        
        return {
            'episodes': num_episodes,
            'final_avg_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0.0,
            'final_epsilon': self.epsilon,
            'model_size': len(self.model),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'training_time': training_time
        }
    
    def _run_episode(self, environment):
        """
        Exécute un épisode avec apprentissage direct et planification.
        
        Args:
            environment: Environnement d'entraînement
            
        Returns:
            Tuple[float, int]: (total_reward, episode_length)
        """
        state = environment.reset()
        total_reward = 0.0
        steps = 0
        done = False
        max_steps = getattr(environment, 'max_steps', 1000)
        
        while not done and steps < max_steps:
            # 1. SÉLECTION D'ACTION (epsilon-greedy)
            valid_actions = environment.valid_actions
            if not valid_actions:
                break
            
            action = self._select_action_epsilon_greedy(state, valid_actions)
            
            # 2. EXÉCUTION DE L'ACTION
            next_state, reward, done, _ = environment.step(action)
            total_reward += reward
            steps += 1
            
            # 3. APPRENTISSAGE DIRECT (Q-Learning)
            self._update_q_value_direct(state, action, reward, next_state, done)
            
            # 4. MISE À JOUR DU MODÈLE
            self._update_model(state, action, next_state, reward)
            
            # 5. PLANIFICATION (n steps)
            self._planning_step()
            
            state = next_state
        
        return total_reward, steps
    
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
    
    def _update_q_value_direct(self, state: int, action: int, reward: float, 
                              next_state: int, done: bool):
        """
        Mise à jour Q-Learning directe.
        
        Args:
            state: État actuel
            action: Action exécutée
            reward: Récompense reçue
            next_state: État suivant
            done: True si épisode terminé
        """
        current_q = self.q_function[state, action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_function[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Mise à jour Q-Learning
        self.q_function[state, action] = current_q + self.learning_rate * (target_q - current_q)
    
    def _update_model(self, state: int, action: int, next_state: int, reward: float):
        """
        Met à jour le modèle de l'environnement.
        
        Args:
            state: État actuel
            action: Action exécutée
            next_state: État suivant
            reward: Récompense reçue
        """
        # Stockage de la transition dans le modèle
        self.model[(state, action)] = (next_state, reward)
        
        # Ajout à l'ensemble des paires observées
        self.observed_sa_pairs.add((state, action))
    
    def _planning_step(self):
        """
        Effectue n étapes de planification en utilisant le modèle appris.
        """
        if not self.model:
            return  # Pas de modèle disponible
        
        for _ in range(self.n_planning_steps):
            # Sélection aléatoire d'une paire (état, action) observée
            if self.observed_sa_pairs:
                state, action = random.choice(list(self.observed_sa_pairs))
                
                # Récupération de la transition depuis le modèle
                if (state, action) in self.model:
                    next_state, reward = self.model[(state, action)]
                    
                    # Mise à jour Q avec la transition simulée
                    current_q = self.q_function[state, action]
                    max_next_q = np.max(self.q_function[next_state])
                    target_q = reward + self.gamma * max_next_q
                    
                    self.q_function[state, action] = current_q + self.learning_rate * (target_q - current_q)
    
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
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur le modèle appris.
        
        Returns:
            Dict avec statistiques du modèle
        """
        return {
            'total_transitions': len(self.model),
            'states_covered': len(set(s for s, a in self.model.keys())),
            'sa_pairs_observed': len(self.observed_sa_pairs),
            'coverage_ratio': len(self.observed_sa_pairs) / (self.state_space_size * self.action_space_size)
        }
    
    def save_model(self, filepath: str) -> bool:
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            filepath: Chemin de sauvegarde (sans extension)
            
        Returns:
            True si succès
        """
        try:
            # Conversion du modèle pour JSON (clés tuple -> string)
            model_serializable = {f"{s}_{a}": {"next_state": ns, "reward": r} 
                                for (s, a), (ns, r) in self.model.items()}
            
            observed_pairs_serializable = [{"state": s, "action": a} 
                                         for s, a in self.observed_sa_pairs]
            
            model_data = {
                'algorithm': self.algo_name,
                'state_space_size': self.state_space_size,
                'action_space_size': self.action_space_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'n_planning_steps': self.n_planning_steps,
                'is_trained': self.is_trained,
                'q_function': self.q_function,
                'model': model_serializable,
                'observed_sa_pairs': observed_pairs_serializable,
                'training_history': self.training_history
            }
            
            # Sauvegarde JSON avec conversion des arrays
            json_file = f"{filepath}.json"
            with open(json_file, 'w') as f:
                json_data = model_data.copy()
                json_data['q_function'] = self.q_function.tolist()
                json.dump(json_data, f, indent=2)
            
            # Sauvegarde pickle (plus simple pour les structures complexes)
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
                    
                    # Reconstruction du modèle
                    model_dict = {}
                    for key, value in json_data['model'].items():
                        s, a = map(int, key.split('_'))
                        model_dict[(s, a)] = (value['next_state'], value['reward'])
                    model_data['model'] = model_dict
                    
                    # Reconstruction des paires observées
                    observed_pairs = set()
                    for pair_data in json_data['observed_sa_pairs']:
                        observed_pairs.add((pair_data['state'], pair_data['action']))
                    model_data['observed_sa_pairs'] = observed_pairs
            
            # Vérification de compatibilité
            if (model_data['state_space_size'] != self.state_space_size or
                model_data['action_space_size'] != self.action_space_size):
                raise ValueError("Incompatibilité des dimensions")
            
            # Chargement des données
            self.q_function = model_data['q_function']
            self.model = model_data['model']
            self.observed_sa_pairs = model_data['observed_sa_pairs']
            self.is_trained = model_data['is_trained']
            self.epsilon = model_data['epsilon']
            self.learning_rate = model_data['learning_rate']
            self.n_planning_steps = model_data['n_planning_steps']
            self.training_history = model_data.get('training_history', [])
            
            return True
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return False
    
    def reset_training(self):
        """Remet à zéro l'entraînement."""
        super().reset_training()
        self.q_function = np.zeros((self.state_space_size, self.action_space_size))
        self.model = {}
        self.observed_sa_pairs = set()
        self.epsilon = self.initial_epsilon