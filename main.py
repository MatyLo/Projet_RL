from environments.grid_world import GridWorldEnvDP
from environments.line_world import LineWorldEnvDP
from rl_algorithms.Dynamic_programming import compute_action_value_function, policy_iteration
from evaluation.evaluator import evaluate_policy, simulate_episode
from utils.io import save_model

if __name__ == "__main__":
    #env = LineWorldEnvDP()
    env = GridWorldEnvDP()
    policy, V, n_iterations = policy_iteration(env)
    Q = compute_action_value_function(env, V)
    
    print("Politique optimale :", policy)
    print("Valeurs optimales :", V)
    print("action value fonction :", Q)

    #save_model(policy, V, Q, path="models/lineworld_policy_iteration")
    save_model(policy, V, Q, path="models/gridworld_policy_iteration")
    
    print(f"Convergence atteinte en {n_iterations} itérations.")

    # Évaluer une politique
    evaluate_policy(env, policy)

    # Simuler un épisode
    states, actions, rewards, G = simulate_episode(env, policy)
    print("États visités :", states)
    print("Actions prises :", actions)
    print("Récompenses reçues :", rewards)
    print("Retour cumulé :", G)
