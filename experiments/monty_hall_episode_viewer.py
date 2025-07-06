#!/usr/bin/env python3
"""
Visualiseur d'épisodes Monty Hall avec PyGame - Permet de voir le déroulement étape par étape.
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
    """Entraîne un agent Value Iteration."""
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

def view_episode_step_by_step(env, visualizer, agent=None, mode="agent", episode_num=1):
    """Visualise un épisode étape par étape."""
    print(f"\n=== Épisode {episode_num} - Mode: {mode} ===")
    
    state = env.reset()
    total_reward = 0
    steps = 0
    actions_taken = []
    
    print("Début de l'épisode - Appuyez sur ENTRÉE pour continuer...")
    
    while True:
        # Afficher l'état actuel
        episode_info = {
            "Épisode": episode_num,
            "Mode": mode,
            "Étapes": steps,
            "Récompense": total_reward,
            "Actions": actions_taken,
            "État": env.state,
            "Instruction": "Appuyez sur ENTRÉE pour continuer"
        }
        visualizer.render_monty_hall_state(env, episode_info)
        
        # Attendre l'action de l'utilisateur
        waiting_for_input = True
        while waiting_for_input and visualizer.running:
            event = visualizer.handle_events()
            if event == "step":
                waiting_for_input = False
            elif event == "pause":
                # Afficher l'état en pause
                episode_info["Instruction"] = "PAUSE - Appuyez sur ESPACE pour continuer"
                visualizer.render_monty_hall_state(env, episode_info)
            elif not visualizer.running:
                return None
        
        if not visualizer.running:
            return None
        
        # Choisir l'action
        if env.state == 0:
            # Choix initial de porte
            if mode == "agent":
                action = np.random.choice([0, 1, 2])
                action_name = f"Choisir porte {action}"
            else:
                # Mode aléatoire pour comparaison
                action = np.random.choice([0, 1, 2])
                action_name = f"Choisir porte {action}"
        else:
            # Action rester/changer
            if mode == "agent":
                action = np.argmax(agent.policy[state])
                action_name = "Changer" if action == 1 else "Garder"
            else:
                # Mode aléatoire pour comparaison
                action = np.random.choice([0, 1])
                action_name = "Changer" if action == 1 else "Garder"
        
        actions_taken.append(action)
        
        print(f"Étape {steps + 1}: {action_name}")
        
        # Exécuter l'action
        old_state = env.state
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Afficher le résultat de l'action
        episode_info = {
            "Épisode": episode_num,
            "Mode": mode,
            "Étapes": steps,
            "Récompense": total_reward,
            "Actions": actions_taken,
            "État": env.state,
            "Dernière action": action_name,
            "Récompense étape": reward
        }
        visualizer.render_monty_hall_state(env, episode_info)
        
        # Pause pour voir le résultat
        time.sleep(1.5)
        
        if done:
            # Afficher le résultat final
            result = "GAGNÉ! 🎉" if info.get("result") == "win" else "PERDU! ❌"
            episode_info = {
                "Épisode": episode_num,
                "Mode": mode,
                "Étapes": steps,
                "Récompense": total_reward,
                "Actions": actions_taken,
                "Résultat": result,
                "Temps total": f"{steps} étapes"
            }
            visualizer.render_monty_hall_state(env, episode_info)
            
            print(f"Épisode terminé: {result}")
            print(f"Récompense totale: {total_reward}")
            print(f"Nombre d'étapes: {steps}")
            
            # Pause pour voir le résultat final
            time.sleep(3.0)
            break
    
    return info.get("result") == "win"

def main_episode_viewer():
    """Visualiseur principal d'épisodes."""
    print("=== VISUALISEUR D'ÉPISODES MONTY HALL ===")
    print("Ce script permet de voir le déroulement des épisodes étape par étape.")
    print("Contrôles:")
    print("- ENTRÉE: Passer à l'étape suivante")
    print("- ESPACE: Pause/Reprendre")
    print("- ÉCHAP: Quitter")
    
    # Créer l'environnement et le visualiseur
    env = MontyHall()
    visualizer = MontyHallVisualizer()
    
    # Entraîner l'agent
    agent = train_agent()
    
    # Statistiques
    agent_wins = 0
    random_wins = 0
    agent_games = 0
    random_games = 0
    
    print("\nCommençons par voir quelques épisodes de l'agent entraîné...")
    
    # Voir quelques épisodes de l'agent
    for episode in range(1, 4):
        result = view_episode_step_by_step(env, visualizer, agent, "agent", episode)
        if result is not None:
            agent_games += 1
            if result:
                agent_wins += 1
        else:
            break
    
    if agent_games > 0:
        print(f"\nAgent: {agent_wins}/{agent_games} ({agent_wins/agent_games*100:.1f}%)")
    
    print("\nMaintenant, voyons quelques épisodes avec un agent aléatoire pour comparaison...")
    
    # Voir quelques épisodes aléatoires
    for episode in range(1, 4):
        result = view_episode_step_by_step(env, visualizer, None, "aléatoire", episode)
        if result is not None:
            random_games += 1
            if result:
                random_wins += 1
        else:
            break
    
    if random_games > 0:
        print(f"\nAléatoire: {random_wins}/{random_games} ({random_wins/random_games*100:.1f}%)")
    
    # Comparaison finale
    if agent_games > 0 and random_games > 0:
        print(f"\n=== COMPARAISON FINALE ===")
        print(f"Agent entraîné: {agent_wins}/{agent_games} ({agent_wins/agent_games*100:.1f}%)")
        print(f"Agent aléatoire: {random_wins}/{random_games} ({random_wins/random_games*100:.1f}%)")
        
        if agent_wins/agent_games > random_wins/random_games:
            print("L'agent entraîné fait mieux que l'agent aléatoire!")
        elif random_wins/random_games > agent_wins/agent_games:
            print("L'agent aléatoire fait mieux que l'agent entraîné!")
        else:
            print("Les deux agents ont les mêmes performances!")
    
    # Afficher la politique de l'agent
    print(f"\nPolitique de l'agent entraîné:")
    for state in range(len(env.S)):
        if state not in env.T:
            action = np.argmax(agent.policy[state])
            action_name = "Changer" if action == 1 else "Garder"
            value = agent.V[state]
            print(f"État {state}: {action_name} (V(s) = {value:.3f})")
    
    # Attendre que l'utilisateur ferme la fenêtre
    print("\nAppuyez sur ÉCHAP ou fermez la fenêtre pour quitter...")
    while visualizer.running:
        event = visualizer.handle_events()
        if event is not None:
            break
        time.sleep(0.1)
    
    visualizer.quit()

if __name__ == "__main__":
    main_episode_viewer() 