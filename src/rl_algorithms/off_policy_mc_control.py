import numpy as np
import random
import os

class OffPolicyMCControl:
    """
    Off-policy Monte Carlo Control (avec importance sampling).
    Compatible avec tous les environnements BaseEnvironment.
    """
    def __init__(self, environment, num_episodes=5000, gamma=0.99, epsilon=0.1):
        self.env = environment
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.nS = self.env.state_space_size
        self.nA = self.env.action_space_size
        self.Q = np.zeros((self.nS, self.nA))
        self.C = np.zeros((self.nS, self.nA))  # Cumulative sum of weights
        self.target_policy = np.ones((self.nS, self.nA)) / self.nA
        self.behavior_policy = np.ones((self.nS, self.nA)) / self.nA

    def get_action(self, state):
        valid_actions = self.env.valid_actions
        if not valid_actions:
            return 0  # fallback
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_vals = self.behavior_policy[state]
            best_a = max(valid_actions, key=lambda a: q_vals[a])
            return best_a

    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            done = False
            
            while not done:
                valid_actions = self.env.valid_actions
                if not valid_actions:
                    break
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                state = next_state
            
            G = 0
            W = 1.0
            for t in reversed(range(len(episode_states))):
                s = episode_states[t]
                a = episode_actions[t]
                r = episode_rewards[t]
                G = self.gamma * G + r
                self.C[s, a] += W
                self.Q[s, a] += (W / self.C[s, a]) * (G - self.Q[s, a])
                # Met à jour la target policy de manière greedy
                best_a = np.argmax(self.Q[s])
                self.target_policy[s] = 0
                self.target_policy[s, best_a] = 1
                if a != best_a:
                    break
                W = W * 1.0 / (self.behavior_policy[s, a])
        return {
            'Q': self.Q,
            'policy': self.target_policy
        }

    def save(self, path):
        np.savez(path, Q=self.Q, policy=self.target_policy)

    def load(self, path):
        data = np.load(path)
        self.Q = data['Q']
        self.target_policy = data['policy']

    def get_policy(self):
        return self.target_policy

    def get_Q(self):
        return self.Q 

    def save_model(self, filepath: str) -> bool:
        import pickle
        try:
            with open(filepath + '.pkl', 'wb') as f:
                pickle.dump(self.__dict__, f)
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return False 