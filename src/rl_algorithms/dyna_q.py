import numpy as np
import pickle
from .base_algorithm import BaseAlgorithm

class DynaQ(BaseAlgorithm):
    """
    Algorithme Dyna-Q : Q-Learning + planification par modèle (replay)
    """
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, gamma=0.99, epsilon=0.1, n_planning_steps=10):
        super().__init__("DynaQ", state_space_size, action_space_size)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning_steps = n_planning_steps
        self.q_function = np.zeros((state_space_size, action_space_size))
        self.model = dict()  # (state, action) -> (next_state, reward)

    def train(self, environment, num_episodes=1000, verbose=False):
        rewards_per_episode = []
        for episode in range(num_episodes):
            state = environment.reset()
            done = False
            total_reward = 0.0
            while not done:
                # Epsilon-greedy
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(environment.valid_actions)
                else:
                    q_vals = self.q_function[state, :]
                    valid_actions = environment.valid_actions
                    q_vals_valid = [q_vals[a] for a in valid_actions]
                    action = valid_actions[np.argmax(q_vals_valid)]
                next_state, reward, done, _ = environment.step(action)
                total_reward += reward
                # Q-learning update
                best_next = np.max(self.q_function[next_state, :])
                self.q_function[state, action] += self.learning_rate * (reward + self.gamma * best_next - self.q_function[state, action])
                # Model update
                self.model[(state, action)] = (next_state, reward)
                # Planning (replay)
                for _ in range(self.n_planning_steps):
                    if not self.model:
                        break
                    s, a = list(self.model.keys())[np.random.randint(len(self.model))]
                    s_next, r = self.model[(s, a)]
                    best_next = np.max(self.q_function[s_next, :])
                    self.q_function[s, a] += self.learning_rate * (r + self.gamma * best_next - self.q_function[s, a])
                state = next_state
            rewards_per_episode.append(total_reward)
            self.add_training_episode(episode, total_reward, 0)
            if verbose and episode % 100 == 0:
                print(f"Episode {episode}, reward: {total_reward}")
        self.is_trained = True
        return {
            "episodes": num_episodes,
            "final_avg_reward": float(np.mean(rewards_per_episode[-100:])),
            "episode_rewards": rewards_per_episode
        }

    def select_action(self, state: int, training: bool = False):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_size)
        return int(np.argmax(self.q_function[state, :]))

    def get_policy(self):
        return np.argmax(self.q_function, axis=1)

    def save_model(self, filepath: str) -> bool:
        try:
            with open(filepath + '.pkl', 'wb') as f:
                pickle.dump({
                    'q_function': self.q_function,
                    'model': self.model,
                    'params': {
                        'learning_rate': self.learning_rate,
                        'gamma': self.gamma,
                        'epsilon': self.epsilon,
                        'n_planning_steps': self.n_planning_steps
                    }
                }, f)
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return False

    def evaluate(self, environment, num_episodes=100, render=False):
        rewards = []
        for ep in range(num_episodes):
            state = environment.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.select_action(state, training=False)
                state, reward, done, _ = environment.step(action)
                total_reward += reward
                if render:
                    environment.render()
            rewards.append(total_reward)
        return {
            'mean_reward': float(np.mean(rewards)),
            'rewards': rewards
        }

    @classmethod
    def from_config(cls, config, environment):
        raise NotImplementedError("from_config n'est pas implémenté pour DynaQ")

    def load_model(self, filepath: str) -> bool:
        raise NotImplementedError("load_model n'est pas implémenté pour DynaQ") 