# envs/two_round_rps.py

import numpy as np

class TwoRoundRPSEnv:
    def __init__(self):
        self.actions = [0, 1, 2]  # Rock, Paper, Scissors
        self.action_names = ['Rock', 'Paper', 'Scissors']
        self.state_space = []
        self._init_states()
        self.reset()

    def _init_states(self):
        # Encode les états comme (round, agent_first_action)
        self.state_space = [(0, None)]  # round 0, aucun historique
        for a1 in self.actions:
            self.state_space.append((1, a1))
        self.state_to_index = {s: i for i, s in enumerate(self.state_space)}
        self.index_to_state = {i: s for s, i in self.state_to_index.items()}

    def reset(self):
        self.current_round = 0
        self.agent_history = []
        return self.state_to_index[(0, None)]

    def step(self, action):
        assert action in self.actions
        reward = 0
        done = False

        if self.current_round == 0:
            self.agent_history.append(action)
            opponent_action = np.random.choice(self.actions)
            reward = self._get_reward(action, opponent_action)
            next_state = (1, action)
            self.current_round += 1
            return self.state_to_index[next_state], reward, done, {}
        
        elif self.current_round == 1:
            a1 = self.agent_history[0]
            self.agent_history.append(action)
            opponent_action = a1  # copie l’action du round 1
            reward = self._get_reward(action, opponent_action)
            done = True
            return None, reward, done, {}

    def _get_reward(self, agent_action, opponent_action):
        # Règles classiques de RPS
        if agent_action == opponent_action:
            return 0 #0 en cas d’égalité
        elif (agent_action - opponent_action) % 3 == 1:
            return 1 #+1 si l’agent gagne (P bat C, C bat F, F bat P)
        else:
            return -1 #–1 si l’agent perd

    def get_states(self):
        return list(self.state_to_index.values())

    def get_actions(self):
        return self.actions

    def is_terminal(self, state_idx):
        return state_idx is None

    def render(self):
        print("Historique de l'agent :", [self.action_names[a] for a in self.agent_history])
