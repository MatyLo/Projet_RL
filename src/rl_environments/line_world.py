"""
Line World Environment

Spécifications:
- États: [0, 1, 2, 3, 4]
- Actions: [0, 1] (0: Left, 1: Right)  
- Récompenses: [-1.0, 0.0, 1.0]
- États terminaux: [0, 4]

Transitions déterministes:
- Position 1: Left → 0 (reward -1), Right → 2 (reward 0)
- Position 2: Left → 1 (reward 0), Right → 3 (reward 0)  
- Position 3: Left → 2 (reward 0), Right → 4 (reward 1)

Compatible avec TOUS les algorithmes RL :
- Algorithmes basés expérience : Q-Learning, SARSA, Monte Carlo
- Algorithmes basés modèle : Policy Iteration, Value Iteration, Dynamic Programming
"""

import numpy as np
from typing import Tuple, List, Dict, Any

from src.rl_environments.base_environment import BaseEnvironment


class LineWorld(BaseEnvironment):
    """
    Environnement Line World selon les spécifications du projet.
    
    Compatible avec TOUS les algorithmes d'apprentissage par renforcement :
    - Expérience : Q-Learning, SARSA, Monte Carlo (via step/reset)
    - Modèle : Policy Iteration, Value Iteration (via matrices de transition)
    """
    
    # Actions
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_NAMES = {ACTION_LEFT: "Left", ACTION_RIGHT: "Right"}
    
    REWARD_NEGATIVE = -1.0  # Position 0
    REWARD_NEUTRAL = 0.0    # Déplacements normaux
    REWARD_POSITIVE = 1.0   # Position 4
    
    def __init__(self, max_steps: int = 100):
        """
        Initialise LineWorld avec les règles fixes du projet.
        
        Args:
            max_steps: Nombre maximum d'étapes par épisode
        """
        super().__init__("LineWorld")
        
        self.max_steps = max_steps
        self.steps_taken = 0
        
        # Configuration fixe selon les spécifications
        self.states = [0, 1, 2, 3, 4]
        self.actions = [0, 1]  # Left, Right
        self.terminal_states = [0, 4]
        self.start_position = 2 
        
        # Matrices pour les algorithmes basés modèle
        self._setup_model_matrices()
        
        # Transitions pour les algorithmes basés expérience
        self._setup_transitions()
        
        self.current_state = None
        self.reset()
    
    def _setup_model_matrices(self):
        """Configure les matrices complètes pour algorithmes basés modèle."""
        num_states = len(self.states)
        num_actions = len(self.actions)
        
        # Matrice de transition P(s'|s,a) : [state][action][next_state]
        self.transition_matrix = np.zeros((num_states, num_actions, num_states))
        
        # Matrice de récompenses R(s,a,s') : [state][action][next_state]
        self.reward_matrix = np.zeros((num_states, num_actions, num_states))
        
        # Remplissage selon les transitions définies
        # Position 1: Left → 0 (reward -1), Right → 2 (reward 0)
        self.transition_matrix[1, 0, 0] = 1.0
        self.reward_matrix[1, 0, 0] = self.REWARD_NEGATIVE
        
        self.transition_matrix[1, 1, 2] = 1.0
        self.reward_matrix[1, 1, 2] = self.REWARD_NEUTRAL
        
        # Position 2: Left → 1 (reward 0), Right → 3 (reward 0)
        self.transition_matrix[2, 0, 1] = 1.0
        self.reward_matrix[2, 0, 1] = self.REWARD_NEUTRAL
        
        self.transition_matrix[2, 1, 3] = 1.0
        self.reward_matrix[2, 1, 3] = self.REWARD_NEUTRAL
        
        # Position 3: Left → 2 (reward 0), Right → 4 (reward 1)
        self.transition_matrix[3, 0, 2] = 1.0
        self.reward_matrix[3, 0, 2] = self.REWARD_NEUTRAL
        
        self.transition_matrix[3, 1, 4] = 1.0
        self.reward_matrix[3, 1, 4] = self.REWARD_POSITIVE
        
        # États terminaux (restent sur place)
        for terminal_state in self.terminal_states:
            for action in self.actions:
                self.transition_matrix[terminal_state, action, terminal_state] = 1.0
                self.reward_matrix[terminal_state, action, terminal_state] = 0.0
    
    def _setup_transitions(self):
        """Configure les transitions pour algorithmes basés expérience."""
        # Matrice de transition simple: [état][action] = (next_state, reward)
        self.transitions = {
            1: {0: (0, self.REWARD_NEGATIVE), 1: (2, self.REWARD_NEUTRAL)},
            2: {0: (1, self.REWARD_NEUTRAL), 1: (3, self.REWARD_NEUTRAL)},
            3: {0: (2, self.REWARD_NEUTRAL), 1: (4, self.REWARD_POSITIVE)}
        }
    
    @property
    def state_space_size(self) -> int:
        """Retourne la taille de l'espace d'états."""
        return len(self.states)
    
    @property
    def action_space_size(self) -> int:
        """Retourne la taille de l'espace d'actions."""
        return len(self.actions)
    
    @property
    def valid_actions(self) -> List[int]:
        """Retourne les actions valides dans l'état actuel."""
        if self.current_state in self.terminal_states:
            return []
        return self.actions.copy()
    
    def reset(self) -> int:
        """
        Remet l'environnement à l'état initial.
        
        Returns:
            État initial
        """
        self.current_state = self.start_position
        self.steps_taken = 0
        self._reset_episode_stats()
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Exécute une action selon les règles du professeur.
        
        Args:
            action: 0 (Left) ou 1 (Right)
            
        Returns:
            (next_state, reward, done, info)
        """
        if action not in self.valid_actions:
            raise ValueError(f"Action invalide: {action}. Actions valides: {self.valid_actions}")
        
        if self.current_state in self.terminal_states:
            raise ValueError(f"Épisode terminé, impossible d'agir depuis l'état {self.current_state}")
        
        self.steps_taken += 1
        old_state = self.current_state
        
        # Transition selon les règles
        if self.current_state in self.transitions:
            next_state, reward = self.transitions[self.current_state][action]
        else:
            next_state, reward = self.current_state, 0.0
        
        self.current_state = next_state
        
        # Vérification fin d'épisode
        done = (next_state in self.terminal_states) or (self.steps_taken >= self.max_steps)
        
        # Informations supplémentaires
        info = {
            "action_name": self.ACTION_NAMES[action],
            "old_state": old_state,
            "terminal_reached": next_state in self.terminal_states,
            "max_steps_reached": self.steps_taken >= self.max_steps,
            "steps_taken": self.steps_taken,
            "target_reached": next_state == 4  # Pour compatibilité avec Agent
        }
        
        # Mise à jour statistiques
        self._update_episode_stats(action, reward, next_state, done)
        
        return next_state, reward, done, info
    
    def render(self, mode: str = 'console'):
        """
        Affiche l'état actuel de l'environnement.
        
        Args:
            mode: 'console' ou 'pygame'
        """
        if mode == 'console':
            self._render_console()
        elif mode == 'pygame':
            return self._render_pygame()
        else:
            raise ValueError(f"Mode non supporté: {mode}")
    
    def _render_console(self):
        """Affichage console simple et clair."""
        print(f"\n=== LineWorld (Step {self.steps_taken}) ===")
        
        # Ligne des positions
        position_line = "Pos: "
        visual_line = "     "
        
        for pos in self.states:
            position_line += f"{pos:2d} "
            
            if pos == self.current_state:
                if pos == 0:
                    visual_line += "[A]"  # Agent sur case perdante
                elif pos == 4:
                    visual_line += "[A]"  # Agent sur case gagnante  
                else:
                    visual_line += "[A]"  # Agent en position normale
            elif pos == 0:
                visual_line += "(-)"   # Case perdante
            elif pos == 4:
                visual_line += "(+)"   # Case gagnante
            else:
                visual_line += " . "   # Case normale
        
        print(position_line)
        print(visual_line)
        
        # État et récompenses
        print(f"Agent en position: {self.current_state}")
        print(f"Récompense totale: {self.total_reward:.1f}")
        
        # Actions disponibles
        if self.valid_actions:
            print("Actions: [0] Left ← | [1] Right →")
        else:
            print("État terminal - Aucune action possible")
    
    def _render_pygame(self):
        """Retourne les données pour rendu PyGame."""
        return {
            'positions': self.states,
            'current_position': self.current_state,
            'terminal_states': self.terminal_states,
            'steps_taken': self.steps_taken,
            'total_reward': self.total_reward,
            'valid_actions': self.valid_actions,
            'max_steps': self.max_steps
        }
    
    def get_state_description(self, state: int) -> str:
        """
        Description textuelle d'un état.
        
        Args:
            state: État à décrire
            
        Returns:
            Description de l'état
        """
        if state == 0:
            return f"Position {state} (TERMINAL - Perte)"
        elif state == 4:
            return f"Position {state} (TERMINAL - Victoire)"
        else:
            return f"Position {state} (Normal)"
    
    # ============ MÉTHODES POUR ALGORITHMES BASÉS MODÈLE ============
    
    def get_transition_matrix(self):
        """
        Retourne la matrice de transition complète P(s'|s,a).
        
        Utilisée par : Policy Iteration, Value Iteration, Dynamic Programming
        
        Returns:
            np.ndarray: Matrice [state][action][next_state] = probabilité
        """
        return self.transition_matrix.copy()
    
    def get_reward_matrix(self):
        """
        Retourne la matrice de récompenses complète R(s,a,s').
        
        Utilisée par : Policy Iteration, Value Iteration, Dynamic Programming
        
        Returns:
            np.ndarray: Matrice [state][action][next_state] = récompense
        """
        return self.reward_matrix.copy()
    
    def get_terminal_states(self):
        """
        Retourne les états terminaux.
        
        Utilisée par : Tous les algorithmes basés modèle
        
        Returns:
            List[int]: Liste des états terminaux
        """
        return self.terminal_states.copy()
    
    def get_all_states(self):
        """Retourne tous les états possibles."""
        return self.states.copy()
    
    def get_all_actions(self):
        """Retourne toutes les actions possibles."""
        return self.actions.copy()
    
    # ============ MÉTHODES POUR ALGORITHMES BASÉS EXPÉRIENCE ============
    
    def get_optimal_policy(self) -> Dict[int, int]:
        """
        Retourne la politique optimale.
        
        Returns:
            Dict {state: optimal_action}
        """
        # Politique optimale: toujours aller vers la droite (position 4)
        return {
            1: self.ACTION_RIGHT,  # 1 → 2
            2: self.ACTION_RIGHT,  # 2 → 3  
            3: self.ACTION_RIGHT   # 3 → 4 (victoire)
        }
    
    def get_reward_function(self, state: int, action: int, next_state: int) -> float:
        """
        Fonction de récompense.
        
        Args:
            state: État de départ
            action: Action exécutée  
            next_state: État d'arrivée
            
        Returns:
            Récompense de la transition
        """
        if state in self.transitions and action in self.transitions[state]:
            _, reward = self.transitions[state][action]
            return reward
        else:
            return 0.0
    
    def is_terminal(self, state: int) -> bool:
        """Vérifie si un état est terminal."""
        return state in self.terminal_states


# Fonction utilitaire pour créer l'environnement standard
def create_lineworld():
    """Crée l'environnement LineWorld standard."""
    return LineWorld()


if __name__ == "__main__":
    # Test rapide
    print("🧪 Test LineWorld")
    env = create_lineworld()
    
    print("État initial:")
    env.render()
    
    print("\nTest de quelques actions:")
    actions_test = [1, 1, 0, 1]  # Right, Right, Left, Right
    
    for i, action in enumerate(actions_test):
        if not env.valid_actions:
            print("Épisode terminé!")
            break
            
        print(f"\n--- Action {i+1}: {env.ACTION_NAMES[action]} ---")
        state, reward, done, info = env.step(action)
        print(f"Résultat: État {state}, Reward {reward}")
        env.render()
        
        if done:
            print(f"Épisode terminé! Success: {info['target_reached']}")
            break
    
    # Test des matrices pour algorithmes basés modèle
    print("\n🔍 Test matrices pour Dynamic Programming:")
    print("Matrice de transition shape:", env.get_transition_matrix().shape)
    print("Matrice de récompenses shape:", env.get_reward_matrix().shape)
    print("États terminaux:", env.get_terminal_states())