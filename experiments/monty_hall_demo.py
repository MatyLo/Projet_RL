#!/usr/bin/env python3
"""
Démonstration interactive du problème de Monty Hall avec PyGame.
"""

import sys
import os
import time
import numpy as np

# Ajouter le répertoire parent au path pour importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.monty_hall import MontyHall
from environments.visualization.monty_hall_visualizer import MontyHallVisualizer
from algorithms.dynamic_programming.value_iteration import ValueIteration

def train_agent():
    """Entraîne un agent Value Iteration pour la démonstration."""
    print("Entraînement de l'agent...")
    
    env = MontyHall()
    agent = ValueIteration(
        environment=env,
        discount_factor=0.9,
        theta=0.001,
        max_iterations=1000
    )
    
    results = agent.train()
    print(f"Agent entraîné en {results['iterations']} itérations!")
    
    return agent

def play_game(env, visualizer, agent=None, mode="human"):
    """Joue une partie de Monty Hall."""
    state = env.reset()
    total_reward = 0
    steps = 0
    actions_taken = []
    
    while True:
        # Afficher l'état actuel
        episode_info = {
            "Mode": mode,
            "Étapes": steps,
            "Récompense": total_reward,
            "Actions": actions_taken
        }
        visualizer.render_monty_hall_state(env, episode_info)
        
        # Gérer les événements
        event = visualizer.handle_events()
        if event == "pause":
            continue
        elif not visualizer.running:
            return None
        
        # Choisir l'action
        if env.state == 0:
            # Choix initial de porte
            if mode == "human":
                print("Choisissez une porte (0, 1 ou 2):")
                action = np.random.choice([0, 1, 2])  # Pour la démo, choix aléatoire
            else:
                # Agent choisit aléatoirement pour le premier choix
                action = np.random.choice([0, 1, 2])
        else:
            # Action rester/changer
            if mode == "human":
                print("Voulez-vous changer de porte? (0: Garder, 1: Changer):")
                action = np.random.choice([0, 1])  # Pour la démo, choix aléatoire
            else:
                # Agent utilise sa politique
                action = np.argmax(agent.policy[state])
        
        actions_taken.append(action)
        
        # Exécuter l'action
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Pause pour voir le résultat
        time.sleep(1.0)
        
        if done:
            # Afficher le résultat final
            episode_info = {
                "Mode": mode,
                "Étapes": steps,
                "Récompense": total_reward,
                "Actions": actions_taken,
                "Résultat": "GAGNÉ!" if info.get("result") == "win" else "PERDU!"
            }
            visualizer.render_monty_hall_state(env, episode_info)
            time.sleep(3.0)
            break
    
    return info.get("result") == "win"

def main_demo():
    """Démonstration principale."""
    print("=== DÉMONSTRATION INTERACTIVE MONTY HALL ===")
    
    # Créer l'environnement et le visualiseur
    env = MontyHall()
    visualizer = MontyHallVisualizer()
    
    # Entraîner l'agent
    agent = train_agent()
    
    human_wins = 0
    agent_wins = 0
    human_games = 0
    agent_games = 0
    
    # Jouer quelques parties automatiquement
    print("Jouons quelques parties pour voir les performances!")
    
    # L'agent joue
    print("L'agent joue...")
    for i in range(5):
        result = play_game(env, visualizer, agent, "agent")
        if result is not None:
            agent_games += 1
            if result:
                agent_wins += 1
    
    print(f"Agent: {agent_wins}/{agent_games} ({agent_wins/agent_games*100:.1f}%)")
    
    # Attendre que l'utilisateur ferme la fenêtre
    print("Appuyez sur ÉCHAP ou fermez la fenêtre pour quitter...")
    while visualizer.running:
        event = visualizer.handle_events()
        if event is not None:
            break
        time.sleep(0.1)
    
    visualizer.quit()

if __name__ == "__main__":
    main_demo() 