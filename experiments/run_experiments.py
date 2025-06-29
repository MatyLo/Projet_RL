import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.line_world import LineWorld
from algorithms.temporal_difference.q_learning import QLearning
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Création de l'environnement
    env = LineWorld(length=5, start_pos=0, goal_pos=4)
    
    # Création de l'algorithme
    agent = QLearning(
        environment=env,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1
    )
    
    # Entraînement
    n_episodes = 1000
    results = agent.train(n_episodes)
    
    # Affichage des résultats
    plt.figure(figsize=(10, 5))
    plt.plot(results["rewards"], label="Récompense par épisode")
    # Ajout de la moyenne glissante
    window = 50  # taille de la fenêtre de lissage
    if len(results["rewards"]) >= window:
        moving_avg = np.convolve(results["rewards"], np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(results["rewards"])), moving_avg, color='red', label='Moyenne glissante')
    plt.title("Récompenses par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Récompense")
    plt.legend()
    plt.show()
    
    # Test de l'agent entraîné
    print("\nTest de l'agent entraîné:")
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        print(f"Action: {action}, Récompense: {reward}")
    
    print(f"\nRécompense totale: {total_reward}")

if __name__ == "__main__":
    main() 