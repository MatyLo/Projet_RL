import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from rl_environments.base_environment import BaseEnvironment

class MontyHall2StepByStep(BaseEnvironment):
    """
    Environnement Monty Hall 2 (5 portes) pas à pas, hybride.
    """
    def __init__(self, n_doors: int = 5):
        super().__init__("MontyHall2StepByStep")
        self.n_doors = n_doors
        self.state = 0
        self.doors = list(range(n_doors))
        self.winning_door = None
        self.agent_choice = None
        self.agent_first_choice = None
        self.eliminated_doors = []
        self.remaining_doors = list(range(n_doors))
        self.steps = 0
        self.final_choice = None
        self.done = False
        self.last_eliminated = None
        self.choice_at_3 = None
    def reset(self) -> int:
        self.state = 0
        self.winning_door = random.randint(0, self.n_doors - 1)
        self.agent_choice = None
        self.agent_first_choice = None
        self.eliminated_doors = []
        self.remaining_doors = list(range(self.n_doors))
        self.steps = 0
        self.final_choice = None
        self.done = False
        self.last_eliminated = None
        self.choice_at_3 = None
        return self.state
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        info = {}
        reward = 0.0
        if self.state == 0:
            self.agent_choice = action
            self.agent_first_choice = action
            self.state = 1
            self.steps = 1
            elim = self._eliminate_one_door()
            self.eliminated_doors.append(elim)
            self.remaining_doors.remove(elim)
            self.last_eliminated = elim
            info['eliminated'] = elim
            info['remaining'] = self.remaining_doors.copy()
            return self.state, reward, False, info
        elif self.state == 1:
            if action in self.remaining_doors:
                self.agent_choice = action
            if len(self.remaining_doors) > 3:
                elim = self._eliminate_one_door()
                self.eliminated_doors.append(elim)
                self.remaining_doors.remove(elim)
                self.last_eliminated = elim
                info['eliminated'] = elim
                info['remaining'] = self.remaining_doors.copy()
                return self.state, reward, False, info
            elif len(self.remaining_doors) == 3:
                self.choice_at_3 = self.agent_choice
                candidates = [d for d in self.remaining_doors if d != self.choice_at_3 and d != self.winning_door]
                elim = random.choice(candidates)
                self.eliminated_doors.append(elim)
                self.remaining_doors.remove(elim)
                self.last_eliminated = elim
                self.state = 3
                return self.state, reward, False, {'eliminated': elim, 'remaining': self.remaining_doors.copy()}
        elif self.state == 3:
            if action == 0:
                self.final_choice = self.choice_at_3
            else:
                other = [d for d in self.remaining_doors if d != self.choice_at_3][0]
                self.final_choice = other
            self.done = True
            self.state = 4
            reward = 1.0 if self.final_choice == self.winning_door else 0.0
            info['result'] = 'win' if reward == 1.0 else 'lose'
            info['winning_door'] = self.winning_door
            info['final_choice'] = self.final_choice
            info['remaining'] = self.remaining_doors.copy()
            return self.state, reward, True, info
        else:
            return self.state, reward, True, info
    def _eliminate_one_door(self) -> int:
        candidates = [d for d in self.remaining_doors if d != self.agent_choice and d != self.winning_door]
        return random.choice(candidates)
    @property
    def valid_actions(self) -> List[int]:
        if self.state in [0, 1]:
            return self.remaining_doors.copy()
        elif self.state == 3:
            return [0, 1]
        else:
            return []
    @property
    def action_space_size(self) -> int:
        return self.n_doors
    @property
    def state_space_size(self) -> int:
        return 5
    def render(self, mode: str = 'console'):
        if mode == 'console':
            print(f"Portes restantes: {self.remaining_doors}")
            print(f"Portes éliminées: {self.eliminated_doors}")
            print(f"Votre choix: {self.agent_choice}")
            if self.state == 4:
                print(f"Porte gagnante: {self.winning_door}")
                print(f"Gagné ? {self.final_choice == self.winning_door}")
    def get_state_description(self, state: int) -> str:
        descriptions = {
            0: "Choix initial de porte",
            1: "Choix/garde à chaque étape",
            2: "Choix spécial Monty Hall 1 (3 portes)",
            3: "Choix final garder/changer",
            4: "Partie terminée"
        }
        return descriptions.get(state, "État inconnu")
    def get_state_space(self):
        return list(range(self.state_space_size))
    def get_action_space(self):
        return list(range(self.action_space_size))
    def get_rewards(self):
        return [0.0, 1.0]
    def get_terminal_states(self):
        return [4]
    def get_transition_matrix(self):
        p = np.zeros((5, 5, 5, 2))
        for a in range(5):
            p[0, a, 1, 0] = 1.0
        for a in range(5):
            p[1, a, 3, 0] = 1.0
        for a in range(2):
            p[3, a, 4, 0] = 0.5
            p[3, a, 4, 1] = 0.5
        p[4, 0, 4, 0] = 1.0
        return p 