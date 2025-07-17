import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from src.rl_environments.base_environment import BaseEnvironment

class MontyHallInteractive(BaseEnvironment):
    """
    Environnement Monty Hall autonome pour démonstration interactive.
    Version simplifiée qui hérite directement de BaseEnvironment.
    """
    def __init__(self):
        super().__init__("MontyHallInteractive")
        self.num_doors = 3
        self.winning_door = None
        self.chosen_door = None
        self.eliminated_door = None
        self.final_choice = None
        self.state = 0
        self.valid_actions_list = [0, 1, 2]
        self.opened_door = None
        self.doors = [True] * 3
        self.agent_first_choice = None
        self.agent_final_choice = None
    
    @property
    def state_space_size(self) -> int:
        return 3
    
    @property
    def action_space_size(self) -> int:
        return 3
    
    @property
    def valid_actions(self) -> List[int]:
        if self.state == 0:
            return [0, 1, 2] # Choisir porte 0, 1, ou 2
        elif self.state == 1:
            return [0, 1] # 0 = rester, 1 = changer
        else:
            return []
    
    def reset(self) -> int:
        self.winning_door = random.randint(0, 2)
        self.chosen_door = None
        self.eliminated_door = None
        self.final_choice = None
        self.state = 0
        self.opened_door = None
        self.doors = [True] * 3
        self.agent_first_choice = None
        self.agent_final_choice = None
        self._reset_episode_stats()
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        if not self.is_valid_action(action):
            raise ValueError(f"Action invalide {action} dans l'état {self.state}")
        reward = 0.0
        done = False
        info = {}
        if self.state == 0:
            self.chosen_door = action
            self.agent_first_choice = action
            available_doors = [i for i in range(3) if i != self.chosen_door and i != self.winning_door]
            self.eliminated_door = random.choice(available_doors)
            self.opened_door = self.eliminated_door
            self.doors[self.eliminated_door] = False
            self.state = 1
            info['eliminated_door'] = self.eliminated_door
        elif self.state == 1:
            if action == 0:
                self.final_choice = self.chosen_door
            else:
                remaining_doors = [i for i in range(3) if i != self.chosen_door and i != self.eliminated_door]
                self.final_choice = remaining_doors[0]
            #self.agent_final_choice = self.final_choice
            won = (self.final_choice == self.winning_door)
            #reward = 1.0 if won else 0.0
            # Récompense différentielle pour encourager le changement
            if action == 1 and won:  # Changer et gagner
                reward = 2.0
            elif action == 0 and won:  # Rester et gagner
                reward = 1.0
            else:  # Perdre
                reward = -1.0
            done = True
            self.state = 2
            info['result'] = 'win' if won else 'lose'
            info['winning_door'] = self.winning_door
            info['final_choice'] = self.final_choice
        self._update_episode_stats(action, reward, self.state, done)
        return self.state, reward, done, info
    
    def render(self, mode: str = 'console') -> Optional[Any]:
        if mode == 'console':
            print(f"État: {self.state}")
            print(f"Porte choisie: {self.chosen_door}")
            print(f"Porte éliminée: {self.eliminated_door}")
            print(f"Choix final: {self.final_choice}")
            if self.state == 2:
                print(f"Porte gagnante: {self.winning_door}")
        return None
    
    def get_state_description(self, state: int) -> str:
        descriptions = {
            0: "Choix initial de porte",
            1: "Choix final: rester ou changer",
            2: "Partie terminée"
        }
        return descriptions.get(state, "État inconnu")
    
    def get_game_state(self):
        return {
            'state': self.state,
            'chosen_door': self.chosen_door,
            'eliminated_door': self.eliminated_door,
            'winning_door': self.winning_door if self.state == 2 else None,
            'final_choice': self.final_choice
        }
    
    def get_state_space(self):
        return list(range(self.state_space_size))
    
    def get_action_space(self):
        return list(range(self.action_space_size))
    
    def get_rewards(self):
        return [0.0, 1.0]
    
    def get_terminal_states(self):
        return [2]
    
    def get_transition_matrix(self):
        p = np.zeros((3, 3, 3, 2))
        for a in range(3):
            p[0, a, 1, 0] = 1.0
        for a in range(2):
            p[1, a, 2, 0] = 0.5
            p[1, a, 2, 1] = 0.5
        p[2, 0, 2, 0] = 1.0
        return p
    
    def get_reward_function2(self, state: int, action: int, next_state: int) -> float:
        # Retourne 0.0 si action invalide dans l'état donné
        valid_actions = []
        if state == 0:
            valid_actions = [0, 1, 2]
        elif state == 1:
            valid_actions = [0, 1]
        else:
            valid_actions = []
        if action not in valid_actions:
            return 0.0
        # Sinon, comportement normal
        current_state_backup = self.state
        self.state = state
        won = (self.final_choice == self.winning_door)
        reward = 1.0 if won else 0.0  # Toujours 1.0 ou 0.0
        self.state = current_state_backup
        return reward 

    def get_reward_function(self, state: int, action: int, next_state: int) -> float:
        """Version simplifiée sans appel à step()"""
        if state == 1:  # Seul état où on donne des récompenses
            # Probabilité théorique de gagner
            if action == 1:  # Changer
                return 2.0 * (2/3) + (-1.0) * (1/3)  # ≈ 1.0
            else:  # Rester
                return 1.0 * (1/3) + (-1.0) * (2/3)  # ≈ -0.33
        return 0.0