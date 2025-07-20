"""
Environnement Two Round Rock Paper Scissors pour l'apprentissage par renforcement.

L'agent joue une partie de 2 rounds de Pierre Feuille Ciseaux contre un adversaire spécial:
- Round 1: L'adversaire joue aléatoirement
- Round 2: L'adversaire joue FORCEMENT le choix de l'agent au round 1

Récompenses: +1 victoire, -1 défaite, 0 égalité par round
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import os
import sys

# Ajout des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'utils'))

from rl_environments.base_environment import BaseEnvironment
#from src.rl_environments.base_environment import BaseEnvironment
from agent import Agent

class TwoRoundRPSEnvironment(BaseEnvironment):
    """
    Environnement Two Round Rock Paper Scissors.
    
    États:
    - 0: Début du jeu (round 1)
    - 1-9: Après round 1 (encodage: 3*agent_choice + opponent_choice)
    - 10: Fin du jeu
    
    Actions:
    - 0: Rock (Pierre)
    - 1: Paper (Feuille)
    - 2: Scissors (Ciseaux)
    """
    
    # Constantes
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    
    ACTION_NAMES = {ROCK: "Rock", PAPER: "Paper", SCISSORS: "Scissors"}
    
    def __init__(self, env_name: str = "TwoRoundRPS"):
        super().__init__(env_name)
        self.agent_round1_choice = None
        self.opponent_round1_choice = None
        self.round1_result = None
        self.round2_result = None
        self.current_round = 1
        
    @property
    def state_space_size(self) -> int:
        """11 états: 0 (début), 1-9 (après round 1), 10 (fin)"""
        return 11
    
    @property
    def action_space_size(self) -> int:
        """3 actions: Rock, Paper, Scissors"""
        return 3
    
    @property
    def valid_actions(self) -> List[int]:
        """Actions valides selon l'état actuel"""
        if self.current_state == 10 or self.current_round >= 3:  # Jeu terminé
            return []
        return [self.ROCK, self.PAPER, self.SCISSORS]

    def get_valid_actions(self, state: int = None) -> List[int]:
        """Retourne les actions valides pour un état donné"""
        if state is None:
            state = self.current_state
        
        if state == 10:  # État terminal
            return []
        return [self.ROCK, self.PAPER, self.SCISSORS]
    
    def reset(self) -> int:
        """Remet l'environnement à l'état initial"""
        super()._reset_episode_stats()
        self.current_state = 0
        self.agent_round1_choice = None
        self.opponent_round1_choice = None
        self.round1_result = None
        self.round2_result = None
        self.current_round = 1
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Exécute une action dans l'environnement"""
        if not self.is_valid_action(action):
            return self.current_state, 0.0, True, {"error": "Invalid action"}
    
        if self.current_round >= 3:  # Jeu terminé
            return self.current_state, 0.0, True, {"info": "Game already finished"}
    
        
        reward = 0.0
        done = False
        info = {}
        
        if self.current_round == 1:
            # Round 1: L'adversaire joue aléatoirement
            self.agent_round1_choice = action
            self.opponent_round1_choice = np.random.randint(0, 3)
            
            # Calculer le résultat du round 1
            self.round1_result = self._get_round_result(action, self.opponent_round1_choice)
            reward = self.round1_result
            
            # Passer au round 2 - encoder l'état
            self.current_state = 1 + (3 * self.agent_round1_choice + self.opponent_round1_choice)
            self.current_round = 2
            
            info = {
                'round': 1,
                'agent_choice': self.ACTION_NAMES[action],
                'opponent_choice': self.ACTION_NAMES[self.opponent_round1_choice],
                'result': self.round1_result,
                'round_description': self._get_round_description(action, self.opponent_round1_choice, self.round1_result)
            }
            
        elif self.current_round == 2:
            # Round 2: L'adversaire joue le choix de l'agent au round 1
            opponent_round2_choice = self.agent_round1_choice
            
            # Calculer le résultat du round 2
            self.round2_result = self._get_round_result(action, opponent_round2_choice)
            reward = self.round2_result
            
            # Fin du jeu après le round 2
            self.current_state = 10
            self.current_round = 3  # Indique que le jeu est terminé
            done = True
            
            info = {
                'round': 2,
                'agent_choice': self.ACTION_NAMES[action],
                'opponent_choice': self.ACTION_NAMES[opponent_round2_choice],
                'result': self.round2_result,
                'round_description': self._get_round_description(action, opponent_round2_choice, self.round2_result),
                'game_summary': self._get_game_summary(),
                'target_reached': self.current_round==3
            }
        
        else:
            # Jeu déjà terminé - aucune action possible
            raise ValueError(f"Le jeu est déjà terminé. Utilisez reset() pour recommencer.")
        
        # Mettre à jour les statistiques
        next_state = self.current_state
        self._update_episode_stats(action, reward, next_state, done)
        
        return next_state, reward, done, info
    
    def _get_round_result(self, agent_choice: int, opponent_choice: int) -> float:
        """Calcule le résultat d'un round"""
        if agent_choice == opponent_choice:
            return 0.0  # Égalité
        elif ((agent_choice == self.ROCK and opponent_choice == self.SCISSORS) or
              (agent_choice == self.PAPER and opponent_choice == self.ROCK) or
              (agent_choice == self.SCISSORS and opponent_choice == self.PAPER)):
            return 1.0  # Victoire
        else:
            return -1.0  # Défaite
    
    def _get_round_description(self, agent_choice: int, opponent_choice: int, result: float) -> str:
        """Génère une description textuelle du round"""
        agent_name = self.ACTION_NAMES[agent_choice]
        opponent_name = self.ACTION_NAMES[opponent_choice]
        
        if result == 1.0:
            return f"Agent: {agent_name} vs Opponent: {opponent_name} → Agent wins!"
        elif result == -1.0:
            return f"Agent: {agent_name} vs Opponent: {opponent_name} → Agent loses!"
        else:
            return f"Agent: {agent_name} vs Opponent: {opponent_name} → Tie!"
    
    def _get_game_summary(self) -> Dict[str, Any]:
        """Génère un résumé du jeu complet"""
        total_score = (self.round1_result or 0) + (self.round2_result or 0)
        
        if total_score > 0:
            game_result = "Agent wins the game!"
        elif total_score < 0:
            game_result = "Agent loses the game!"
        else:
            game_result = "Game is a tie!"
            
        return {
            'round1_score': self.round1_result,
            'round2_score': self.round2_result,
            'total_score': total_score,
            'game_result': game_result,
            'agent_round1_choice': self.ACTION_NAMES[self.agent_round1_choice] if self.agent_round1_choice is not None else None,
            'opponent_round1_choice': self.ACTION_NAMES[self.opponent_round1_choice] if self.opponent_round1_choice is not None else None
        }
    
    def render(self, mode: str = 'console') -> Optional[Any]:
        """Affiche l'état actuel de l'environnement"""
        if mode == 'console':
            print(f"\n=== {self.env_name} ===")
            #print(f"Current State: {self.current_state}")
            #print(f"Current Round: {self.current_round}")
            
            if self.current_round == 1:
                print("Round 1 - Opponent plays randomly")
            elif self.current_round == 2:
                print(f"Round 2 - Opponent will play: {self.ACTION_NAMES[self.agent_round1_choice]}")
                print(f"Round 1 result: {self.round1_result}")
            else:
                print("Game finished!")
                if hasattr(self, 'round1_result') and hasattr(self, 'round2_result'):
                    summary = self._get_game_summary()
                    print(f"Final score: {summary['total_score']}")
                    print(f"Result: {summary['game_result']}")
            
            if self.valid_actions:
                print(f"Valid actions: {[self.ACTION_NAMES[a] for a in self.valid_actions]}")
            
            return None
            
        elif mode == 'pygame':
            # Retourne les données nécessaires pour le rendu pygame
            return self._get_pygame_render_data()
        else:
            raise NotImplementedError(f"Render mode '{mode}' not implemented")
    
    def _get_pygame_render_data(self) -> Dict[str, Any]:
        """Retourne les données nécessaires pour le rendu pygame"""
        render_data = {
            'current_state': self.current_state,
            'current_round': self.current_round,
            'valid_actions': self.valid_actions,
            'action_names': self.ACTION_NAMES,
            'episode_step': self.episode_step,
            'total_reward': self.total_reward,
            'game_finished': self.current_state == 10 or self.current_round >= 3
        }
        
        # Informations spécifiques selon le round
        if self.current_round == 1:
            render_data.update({
                'round_info': "Round 1 - Opponent plays randomly",
                'agent_round1_choice': None,
                'opponent_round1_choice': None,
                'round1_result': None,
                'round2_result': None
            })
        elif self.current_round == 2:
            render_data.update({
                'round_info': f"Round 2 - Opponent will play: {self.ACTION_NAMES[self.agent_round1_choice]}",
                'agent_round1_choice': self.agent_round1_choice,
                'opponent_round1_choice': self.opponent_round1_choice,
                'round1_result': self.round1_result,
                'round2_result': None
            })
        else:  # Game finished (current_round >= 3)
            summary = self._get_game_summary()
            render_data.update({
                'round_info': "Game finished!",
                'agent_round1_choice': self.agent_round1_choice,
                'opponent_round1_choice': self.opponent_round1_choice,
                'round1_result': self.round1_result,
                'round2_result': self.round2_result,
                'game_summary': summary
            })
        
        return render_data
    
    def get_action_symbols(self) -> Dict[int, str]:
        """Retourne les symboles visuels pour chaque action (utile pour pygame)"""
        return {
            self.ROCK: "🪨",
            self.PAPER: "📄", 
            self.SCISSORS: "✂️"
        }
    
    def get_result_color(self, result: float) -> str:
        """Retourne une couleur correspondant au résultat (utile pour pygame)"""
        if result > 0:
            return "GREEN"  # Victoire
        elif result < 0:
            return "RED"    # Défaite
        else:
            return "YELLOW" # Égalité
    
    def get_state_description(self, state: int) -> str:
        """Retourne une description textuelle d'un état"""
        if state == 0:
            return "Game start - Round 1"
        elif 1 <= state <= 9:
            # Décoder l'état après round 1
            agent_choice = (state - 1) // 3
            opponent_choice = (state - 1) % 3
            return (f"After Round 1 - Agent: {self.ACTION_NAMES[agent_choice]}, "
                   f"Opponent: {self.ACTION_NAMES[opponent_choice]}")
        elif state == 10:
            return "Game finished"
        else:
            return f"Unknown state: {state}"
    
    def get_transition_probabilities(self, state: int, action: int) -> Dict[int, float]:
        """Retourne les probabilités de transition depuis un état avec une action"""
        if state == 0:
            # Round 1: transition vers états 1-9 avec probabilité 1/3 chacun
            # car l'adversaire joue aléatoirement
            next_states = {}
            for opponent_choice in range(3):
                next_state = 1 + (3 * action + opponent_choice)
                next_states[next_state] = 1.0 / 3.0
            return next_states
        elif 1 <= state <= 9:
            # Round 2: transition déterministe vers état 10
            return {10: 1.0}
        else:
            # État terminal ou invalide
            return {state: 1.0}
    
    def get_reward_function(self, state: int, action: int, next_state: int) -> float:
        """Retourne la récompense pour une transition donnée"""
        if state == 0:
            # Round 1: calculer la récompense basée sur l'action et l'état suivant
            opponent_choice = (next_state - 1) % 3
            return self._get_round_result(action, opponent_choice)
        elif 1 <= state <= 9:
            # Round 2: l'adversaire joue le choix de l'agent au round 1
            agent_round1_choice = (state - 1) // 3
            return self._get_round_result(action, agent_round1_choice)
        else:
            return 0.0

    def _validate_state(self) -> bool:
        """Valide la cohérence de l'état interne"""
        if self.current_state == 0:
            return self.current_round == 1
        elif 1 <= self.current_state <= 9:
            return self.current_round == 2
        elif self.current_state == 10:
            return self.current_round >= 3
        return False

    def get_terminal_states(self) -> List[int]:
        """
        Retourne les états terminaux de l'environnement.
        
        Pour Two Round RPS:
        - État 10: Fin du jeu (après les 2 rounds)
        
        Returns:
            List[int]: Liste des états terminaux
        """
        return [10]


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer l'environnement
    env = TwoRoundRPSEnvironment()
    
    # Simulation d'une partie
    print("=== Simulation d'une partie ===")
    
    # Initialiser l'environnement
    state = env.reset()
    env.render()
    
    # Round 1: L'agent joue Rock
    print("\n--- Round 1 ---")
    print("Agent joue: Rock")
    next_state, reward, done, info = env.step(env.ROCK)
    print(f"Info: {info['round_description']}")
    print(f"Récompense: {reward}")
    env.render()
    
    if not done:
        # Round 2: L'agent joue Paper
        print("\n--- Round 2 ---")
        print("Agent joue: Paper")
        next_state, reward, done, info = env.step(env.PAPER)
        print(f"Info: {info['round_description']}")
        print(f"Récompense: {reward}")
        print(f"Résumé du jeu: {info['game_summary']}")
        env.render()
    
    # Afficher les statistiques finales
    print("\n=== Statistiques de l'épisode ===")
    stats = env.get_episode_stats()
    print(f"Longueur de l'épisode: {stats['episode_length']}")
    print(f"Récompense totale: {stats['total_reward']}")