import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.line_world import LineWorld
from algorithms.dynamic_programming.policy_iteration import PolicyIteration
from algorithms.dynamic_programming.value_iteration import ValueIteration
import matplotlib.pyplot as plt
import numpy as np

def plot_value_functions(pi_values, vi_values, title="Comparaison des fonctions de valeur"):
    """Affiche les fonctions de valeur des deux algorithmes."""
    plt.figure(figsize=(10, 5))
    x = range(len(pi_values))
    plt.plot(x, pi_values, 'b-', label='Policy Iteration')
    plt.plot(x, vi_values, 'r--', label='Value Iteration')
    plt.title(title)
    plt.xlabel('État')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_policy(algorithm, env, n_episodes=5):
    """Teste une politique sur plusieurs épisodes."""
    print(f"\nTest de la politique ({algorithm.__class__.__name__}):")
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nÉpisode {episode + 1}:")
        while not done:
            action = algorithm.get_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
            print(f"Action: {action}, Récompense: {reward}")
        
        print(f"Épisode terminé en {steps} pas avec une récompense totale de {total_reward}")

def main():
    # Création de l'environnement
    env = LineWorld(length=5, start_pos=0, goal_pos=4)
    
    # Paramètres communs
    discount_factor = 0.99
    theta = 0.0001
    max_iterations = 1000
    
    # Policy Iteration
    print("\nEntraînement avec Policy Iteration...")
    pi_agent = PolicyIteration(
        environment=env,
        discount_factor=discount_factor,
        theta=theta,
        max_iterations=max_iterations
    )
    pi_results = pi_agent.train()
    print(f"Policy Iteration a convergé en {pi_results['iterations']} itérations")
    
    # Value Iteration
    print("\nEntraînement avec Value Iteration...")
    vi_agent = ValueIteration(
        environment=env,
        discount_factor=discount_factor,
        theta=theta,
        max_iterations=max_iterations
    )
    vi_results = vi_agent.train()
    print(f"Value Iteration a convergé en {vi_results['iterations']} itérations")
    
    # Affichage des résultats
    plot_value_functions(
        pi_results['final_value_function'],
        vi_results['final_value_function'],
        "Comparaison des fonctions de valeur (Policy Iteration vs Value Iteration)"
    )
    
    # Test des politiques
    test_policy(pi_agent, env)
    test_policy(vi_agent, env)
    
    # Sauvegarde des modèles
    pi_agent.save("saved_models/policies/policy_iteration.npz")
    vi_agent.save("saved_models/policies/value_iteration.npz")

if __name__ == "__main__":
    main() 