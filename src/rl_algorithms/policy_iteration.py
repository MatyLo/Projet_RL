from typing import Dict, Any, Tuple, List
import numpy as np
from .base_algorithm import BaseAlgorithm

class PolicyIteration(BaseAlgorithm):
    """
    Implémentation de l'algorithme Policy Iteration.
    Compatible avec tous les environnements BaseEnvironment.
    """
    def __init__(self, environment: Any, discount_factor: float = 0.999999,
                 theta: float = 0.00001, max_iterations: int = 1000):
        state_space_size = environment.state_space_size if hasattr(environment, 'state_space_size') else len(environment.get_state_space())
        action_space_size = environment.action_space_size if hasattr(environment, 'action_space_size') else len(environment.get_action_space())
        super().__init__("PolicyIteration", state_space_size, action_space_size)
        self.env = environment
        self.gamma = discount_factor
        self.theta = theta
        self.max_iterations = max_iterations
        # Récupération des paramètres de l'environnement
        self.S = environment.get_state_space()
        self.A = environment.get_action_space()
        self.R = environment.get_rewards()
        self.T = environment.get_terminal_states()
        self.p = environment.get_transition_matrix()
        # Initialisation de la politique et de la fonction de valeur
        self.V = np.random.random((len(self.S),))
        self.V[self.T] = 0.0
        self.policy = np.ones((len(self.S), len(self.A))) / len(self.A)  # Politique uniforme au départ
    def policy_evaluation(self) -> None:
        while True:
            delta = 0.0
            for s in self.S:
                v = self.V[s]
                total = 0.0
                for a in self.A:
                    sub_total = 0.0
                    for s_p in self.S:
                        for r_index in range(len(self.R)):
                            r = self.R[r_index]
                            sub_total += self.p[s, a, s_p, r_index] * (r + self.gamma * self.V[s_p])
                    total += self.policy[s, a] * sub_total
                self.V[s] = total
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta:
                break
    def policy_improvement(self) -> bool:
        policy_stable = True
        for s in self.S:
            old_action = np.argmax(self.policy[s])
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
            if best_a != old_action:
                policy_stable = False
            # Mise à jour de la politique
            self.policy[s] = np.zeros_like(self.policy[s])
            self.policy[s][best_a] = 1.0
        return policy_stable
    def train(self, n_episodes: int = None) -> Dict[str, Any]:
        iterations = 0
        policy_stable = False
        while not policy_stable and iterations < self.max_iterations:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            iterations += 1
        return {
            "iterations": iterations,
            "converged": policy_stable,
            "final_value_function": self.V.copy()
        }
    def get_action(self, state: int) -> int:
        return int(np.argmax(self.policy[state]))
    def save(self, path: str) -> None:
        np.savez(path, policy=self.policy, value_function=self.V)
    def load(self, path: str) -> None:
        data = np.load(path)
        self.policy = data['policy']
        self.V = data['value_function']

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError("from_config n'est pas implémenté pour PolicyIteration")

    def get_policy(self):
        return self.policy

    def load_model(self, path):
        self.load(path)

    def save_model(self, path):
        self.save(path)

    def select_action(self, state):
        return self.get_action(state) 