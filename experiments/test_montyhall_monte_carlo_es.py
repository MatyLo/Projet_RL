#!/usr/bin/env python3
"""
Test Monte Carlo Exploring Starts sur Monty Hall avec visualisation PyGame.
"""

import sys
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.monty_hall import MontyHall
from environments.visualization.monty_hall_visualizer import MontyHallVisualizer
from algorithms.monte_carlo.monte_carlo_es import MonteCarloES

def test_monte_carlo_es_monty_hall():
    print("=== Test Monte Carlo Exploring Starts sur Monty Hall ===")
    env = MontyHall()
    visualizer = MontyHallVisualizer()
    algo = MonteCarloES(environment=env, num_episodes=5000, gamma=0.99, epsilon=0.1)
    print("Entraînement...")
    results = algo.train()
    print("Entraînement terminé!")
    print("Politique apprise:")
    for state in range(len(env.S)):
        if state in env.T:
            print(f"État {state} (terminal): N/A")
        else:
            best_action = np.argmax(results['policy'][state])
            print(f"État {state}: Action {best_action} (proba: {results['policy'][state][best_action]:.3f})")
    print("\nVisualisation de quelques épisodes...")
    wins = 0
    episodes = 10
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        actions = []
        while not done:
            action = np.argmax(results['policy'][state])
            actions.append(action)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            episode_info = {
                "Épisode": ep+1,
                "Étapes": steps,
                "Récompense": total_reward,
                "Actions": actions
            }
            visualizer.render_monty_hall_state(env, episode_info)
            time.sleep(1.0)
        if info.get("result") == "win":
            wins += 1
    print(f"\nRésultats: {wins}/{episodes} épisodes gagnés ({wins/episodes*100:.1f}%)")
    print("Appuyez sur ÉCHAP ou fermez la fenêtre pour quitter...")
    while visualizer.running:
        event = visualizer.handle_events()
        if event is not None:
            break
        time.sleep(0.1)
    visualizer.quit()

if __name__ == "__main__":
    test_monte_carlo_es_monty_hall() 