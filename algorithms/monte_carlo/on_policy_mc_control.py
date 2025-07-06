import numpy as np
import random
import os

class OnPolicyFirstVisitMCControl:
    """
    On-policy First Visit Monte Carlo Control (epsilon-greedy).
    Compatible avec les environnements de type LineWorld.
    """
    def __init__(self, environment, num_episodes=5000, gamma=0.99, epsilon=0.1):
        self.env = environment
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.nS = len(self.env.S)
        self.nA = len(self.env.A)
        self.Q = np.zeros((self.nS, self.nA))
        self.returns = [[[] for _ in range(self.nA)] for _ in range(self.nS)]
        self.policy = np.ones((self.nS, self.nA)) / self.nA

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.nA)
        else:
            return np.argmax(self.policy[state])

    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            done = False
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                state = next_state
            
            G = 0
            visited = set()
            for t in reversed(range(len(episode_states))):
                s = episode_states[t]
                a = episode_actions[t]
                r = episode_rewards[t]
                G = self.gamma * G + r
                if (s, a) not in visited:
                    self.returns[s][a].append(G)
                    self.Q[s, a] = np.mean(self.returns[s][a])
                    best_a = np.argmax(self.Q[s])
                    self.policy[s] = self.epsilon / self.nA
                    self.policy[s, best_a] += 1 - self.epsilon
                    visited.add((s, a))
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