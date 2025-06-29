from envs.line_world import LineWorldEnvDP # type: ignore
from utils.io import load_model
from evaluation.evaluator import evaluate_policy, simulate_episode

def main():
    # Initialisation de l'environnement
    env = LineWorldEnvDP()

    # Chargement de la politique, des valeurs d'état et des Q-valeurs
    pi, V, Q = load_model("models/lineworld_policy_iteration")

    # Évaluation de la politique sur plusieurs épisodes
    evaluate_policy(env, pi, n_episodes=100)

    # Simulation d'un seul épisode avec la politique chargée
    states, actions, rewards, G = simulate_episode(env, pi)

    # Affichage des résultats
    print("États visités :", states)
    print("Actions prises :", [int(a) for a in actions])
    print("Récompenses reçues :", rewards)
    print("Retour cumulé :", G)

if __name__ == "__main__":
    main()
