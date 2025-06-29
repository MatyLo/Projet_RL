import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.line_world import LineWorld
from algorithms.dynamic_programming.policy_iteration import PolicyIteration
from algorithms.dynamic_programming.value_iteration import ValueIteration
import numpy as np
import matplotlib.pyplot as plt

def plot_value_functions(pi_values, vi_values, title="Comparaison des fonctions de valeur"):
    """Affiche les fonctions de valeur des deux algorithmes."""
    plt.figure(figsize=(10, 5))
    x = range(len(pi_values))
    plt.plot(x, pi_values, 'b-', label='Policy Iteration', marker='o')
    plt.plot(x, vi_values, 'r--', label='Value Iteration', marker='s')
    plt.title(title)
    plt.xlabel('État')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_policies(pi_policy, vi_policy, title="Comparaison des politiques"):
    """Affiche les politiques des deux algorithmes."""
    plt.figure(figsize=(10, 5))
    x = range(len(pi_policy))
    plt.plot(x, np.argmax(pi_policy, axis=1), 'b-', label='Policy Iteration', marker='o')
    plt.plot(x, np.argmax(vi_policy, axis=1), 'r--', label='Value Iteration', marker='s')
    plt.title(title)
    plt.xlabel('État')
    plt.ylabel('Action (0: Gauche, 1: Droite)')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_policy(algorithm, env, n_episodes=5):
    """Teste une politique sur plusieurs épisodes."""
    print(f"\nTest de la politique ({algorithm.__class__.__name__}):")
    total_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nÉpisode {episode + 1}:")
        print("État initial:", state)
        env.render()
        
        while not done:
            action = algorithm.get_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            print(f"\nAction: {'droite' if action == 1 else 'gauche'}")
            print(f"État: {state}, Récompense: {reward}")
            env.render()
        
        print(f"Épisode terminé en {steps} pas avec une récompense totale de {total_reward}")
        total_rewards.append(total_reward)
    
    print(f"\nRésumé des {n_episodes} épisodes:")
    print(f"Récompense moyenne: {np.mean(total_rewards):.2f}")
    print(f"Récompense min: {np.min(total_rewards):.2f}")
    print(f"Récompense max: {np.max(total_rewards):.2f}")

def main():
    # Création de l'environnement
    env = LineWorld(length=5, start_pos=0, goal_pos=4)
    
    # Paramètres communs
    discount_factor = 0.999999
    theta = 0.00001
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
    print("\nFonction de valeur finale (Policy Iteration):")
    print(pi_results['final_value_function'])
    print("\nFonction de valeur finale (Value Iteration):")
    print(vi_results['final_value_function'])
    
    # Visualisation des résultats
    plot_value_functions(
        pi_results['final_value_function'],
        vi_results['final_value_function'],
        "Comparaison des fonctions de valeur"
    )
    
    plot_policies(
        pi_agent.policy,
        vi_agent.policy,
        "Comparaison des politiques"
    )
    
    # Test des politiques
    print("\nTest de la politique Policy Iteration:")
    test_policy(pi_agent, env)
    
    print("\nTest de la politique Value Iteration:")
    test_policy(vi_agent, env)
    
    # Sauvegarde des modèles
    os.makedirs("saved_models/policies", exist_ok=True)
    pi_agent.save("saved_models/policies/policy_iteration.npz")
    vi_agent.save("saved_models/policies/value_iteration.npz")

if __name__ == "__main__":
    main() 