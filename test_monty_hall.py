#!/usr/bin/env python3
"""
Script de test simple pour vérifier les environnements Monty Hall et les algorithmes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rl_environments import MontyHall, MontyHall2
from rl_algorithms import QLearning, SARSA, ValueIteration


def test_monty_hall_1():
    """Test de Monty Hall 1 avec Q-Learning."""
    print("=== Test Monty Hall 1 avec Q-Learning ===")
    
    # Créer l'environnement
    env = MontyHall()
    print(f"Environnement: {env.env_name}")
    print(f"États: {env.state_space_size}, Actions: {env.action_space_size}")
    
    # Créer l'algorithme
    algo = QLearning(
        state_space_size=env.state_space_size,
        action_space_size=env.action_space_size,
        learning_rate=0.1,
        gamma=0.9,
        epsilon=0.1
    )
    
    # Entraîner
    print("Entraînement...")
    results = algo.train(env, num_episodes=1000, verbose=True)
    print(f"Résultats: {results['final_avg_reward']:.3f}")
    
    # Tester la politique
    print("Test de la politique...")
    wins = 0
    for _ in range(100):
        state = env.reset()
        done = False
        while not done:
            action = algo.select_action(state, training=False)
            state, reward, done, _ = env.step(action)
        if reward > 0:
            wins += 1
    
    print(f"Taux de victoire: {wins/100:.2%}")
    print()


def test_monty_hall_2():
    """Test de Monty Hall 2 avec Value Iteration."""
    print("=== Test Monty Hall 2 avec Value Iteration ===")
    
    # Créer l'environnement
    env = MontyHall2()
    print(f"Environnement: {env.env_name}")
    print(f"États: {env.state_space_size}, Actions: {env.action_space_size}")
    
    # Créer l'algorithme
    algo = ValueIteration(
        state_space_size=env.state_space_size,
        action_space_size=env.action_space_size,
        gamma=0.9,
        theta=1e-6
    )
    
    # Entraîner
    print("Entraînement...")
    results = algo.train(env, verbose=True)
    print(f"Résultats: {results['max_value']:.3f}")
    
    # Tester la politique
    print("Test de la politique...")
    wins = 0
    for _ in range(100):
        state = env.reset()
        done = False
        while not done:
            action = algo.select_action(state, training=False)
            state, reward, done, _ = env.step(action)
        if reward > 0:
            wins += 1
    
    print(f"Taux de victoire: {wins/100:.2%}")
    print()


def test_monty_hall_2_sarsa():
    """Test de Monty Hall 2 avec SARSA."""
    print("=== Test Monty Hall 2 avec SARSA ===")
    
    # Créer l'environnement
    env = MontyHall2()
    print(f"Environnement: {env.env_name}")
    print(f"États: {env.state_space_size}, Actions: {env.action_space_size}")
    
    # Créer l'algorithme
    algo = SARSA(
        state_space_size=env.state_space_size,
        action_space_size=env.action_space_size,
        learning_rate=0.1,
        gamma=0.9,
        epsilon=0.1
    )
    
    # Entraîner
    print("Entraînement...")
    results = algo.train(env, num_episodes=1000, verbose=True)
    print(f"Résultats: {results['final_avg_reward']:.3f}")
    
    # Tester la politique
    print("Test de la politique...")
    wins = 0
    for _ in range(100):
        state = env.reset()
        done = False
        while not done:
            action = algo.select_action(state, training=False)
            state, reward, done, _ = env.step(action)
        if reward > 0:
            wins += 1
    
    print(f"Taux de victoire: {wins/100:.2%}")
    print()


if __name__ == "__main__":
    print("Tests des environnements Monty Hall et algorithmes")
    print("=" * 50)
    
    try:
        test_monty_hall_1()
        test_monty_hall_2()
        test_monty_hall_2_sarsa()
        print("Tous les tests terminés avec succès!")
    except Exception as e:
        print(f"Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc() 