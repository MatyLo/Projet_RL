import numpy as np
import random
import os

class MonteCarloES:
    """
    Monte Carlo Exploring Starts pour le contrôle de politique.
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
        self.returns = [[[] for _ in range(self.nA)] for _ in range(self.nS)]
        self.policy = np.ones((self.nS, self.nA)) / self.nA

    def get_action(self, state):
        """Choisit une action selon la politique epsilon-greedy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.nA)
        else:
            return np.argmax(self.policy[state])

    def train(self):
        for episode in range(self.num_episodes):
            # Exploring starts: choisir un état et une action de départ aléatoires
            state = np.random.choice(range(self.nS))
            valid_actions = self.env.valid_actions
            if not valid_actions:
                continue
            action = random.choice(valid_actions)
            self.env.reset()
            self.env.current_state = state  # Pour compatibilité avec BaseEnvironment
            episode_states = [state]
            episode_actions = [action]
            episode_rewards = []
            done = False
            
            # Générer un épisode complet
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                episode_rewards.append(reward)
                if not done:
                    valid_actions = self.env.valid_actions
                    if not valid_actions:
                        break
                    action = self.get_action(next_state)
                    if action not in valid_actions:
                        action = random.choice(valid_actions)
                    episode_states.append(next_state)
                    episode_actions.append(action)
                state = next_state
            
            # Calculer G et mettre à jour Q et la politique
            G = 0
            visited = set()
            for t in reversed(range(len(episode_states))):
                s = episode_states[t]
                a = episode_actions[t]
                r = episode_rewards[t] if t < len(episode_rewards) else 0
                G = self.gamma * G + r
                if (s, a) not in visited:
                    self.returns[s][a].append(G)
                    self.Q[s, a] = np.mean(self.returns[s][a])
                    # Politique greedy par rapport à Q
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

    def save_model(self, filepath: str) -> bool:
        import pickle
        try:
            with open(filepath + '.pkl', 'wb') as f:
                pickle.dump(self.__dict__, f)
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return False 