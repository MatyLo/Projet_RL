import numpy as np
import random
import os

class QLearning:
    """
    Q-Learning (off-policy TD control).
    Compatible avec les environnements de type LineWorld.
    """
    def __init__(self, environment, num_episodes=5000, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.env = environment
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.nS = len(self.env.S)
        self.nA = len(self.env.A)
        self.Q = np.zeros((self.nS, self.nA))
        self.policy = np.ones((self.nS, self.nA)) / self.nA

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.nA)
        else:
            return np.argmax(self.policy[state])

    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                td_target = reward + self.gamma * np.max(self.Q[next_state]) * (not done)
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
                # Politique epsilon-greedy
                best_a = np.argmax(self.Q[state])
                self.policy[state] = self.epsilon / self.nA
                self.policy[state, best_a] += 1 - self.epsilon
                state = next_state
        return {
            'Q': self.Q,
            'policy': self.policy
        }

    def save(self, path):
        np.savez(path, Q=self.Q, policy=self.policy)

    def load(self, path):
        data = np.load(path)
        self.Q = data['Q']
        self.policy = data['policy'] 