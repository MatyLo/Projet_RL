#!/usr/bin/env python3
"""
Visualiseur continu d'épisodes Monty Hall - Permet de voir plusieurs épisodes avec contrôles de vitesse.
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

def run_continuous_episodes(env, visualizer, agent, num_episodes=20):
    """Exécute plusieurs épisodes en continu avec contrôles de vitesse."""
    print(f"=== VISUALISATION CONTINUE - {num_episodes} ÉPISODES ===")
    print("Contrôles:")
    print("- ESPACE: Pause/Reprendre")
    print("- + : Augmenter la vitesse")
    print("- - : Diminuer la vitesse")
    print("- ÉCHAP: Quitter")
    
    # Variables de contrôle
    paused = False
    speed = 1.0  # Délai entre les étapes (secondes)
    min_speed = 0.1
    max_speed = 3.0
    
    # Statistiques
    wins = 0
    total_reward = 0
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        steps = 0
        actions_taken = []
        
        while True:
            # Afficher l'état actuel
            episode_info = {
                "Épisode": f"{episode}/{num_episodes}",
                "Mode": "Agent entraîné",
                "Étapes": steps,
                "Récompense": episode_reward,
                "Actions": actions_taken,
                "Vitesse": f"{speed:.1f}x",
                "Pause": "PAUSE" if paused else "EN COURS",
                "Statistiques": f"Gagnés: {wins}/{episode-1} ({wins/max(episode-1,1)*100:.1f}%)"
            }
            visualizer.render_monty_hall_state(env, episode_info)
            
            # Gérer les événements
            event = visualizer.handle_events()
            if event == "pause":
                paused = not paused
            elif not visualizer.running:
                return wins, episode - 1
            
            if paused:
                continue
            
            # Choisir l'action
            if env.state == 0:
                # Choix initial de porte
                action = np.random.choice([0, 1, 2])
            else:
                # Action rester/changer selon la politique
                action = np.argmax(agent.policy[state])
            
            actions_taken.append(action)
            
            # Exécuter l'action
            state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Délai selon la vitesse
            time.sleep(speed)
            
            if done:
                # Mettre à jour les statistiques
                if info.get("result") == "win":
                    wins += 1
                total_reward += episode_reward
                
                # Afficher le résultat final de l'épisode
                result = "GAGNÉ! 🎉" if info.get("result") == "win" else "PERDU! ❌"
                episode_info = {
                    "Épisode": f"{episode}/{num_episodes}",
                    "Mode": "Agent entraîné",
                    "Étapes": steps,
                    "Récompense": episode_reward,
                    "Actions": actions_taken,
                    "Résultat": result,
                    "Statistiques": f"Gagnés: {wins}/{episode} ({wins/episode*100:.1f}%)"
                }
                visualizer.render_monty_hall_state(env, episode_info)
                
                # Pause courte pour voir le résultat
                time.sleep(0.5)
                break
    
    return wins, num_episodes

def main_continuous_viewer():
    """Visualiseur continu principal."""
    print("=== VISUALISEUR CONTINU MONTY HALL ===")
    print("Ce script permet de voir plusieurs épisodes en continu.")
    
    # Créer l'environnement et le visualiseur
    env = MontyHall()
    visualizer = MontyHallVisualizer()
    
    # Entraîner l'agent
    agent = train_agent()
    
    # Afficher la politique de l'agent
    print(f"\nPolitique de l'agent:")
    for state in range(len(env.S)):
        if state not in env.T:
            action = np.argmax(agent.policy[state])
            action_name = "Changer" if action == 1 else "Garder"
            value = agent.V[state]
            print(f"État {state}: {action_name} (V(s) = {value:.3f})")
    
    # Exécuter les épisodes continus
    wins, total = run_continuous_episodes(env, visualizer, agent, num_episodes=20)
    
    # Résultats finaux
    if total > 0:
        win_rate = wins / total
        print(f"\n=== RÉSULTATS FINAUX ===")
        print(f"Épisodes gagnés: {wins}/{total}")
        print(f"Taux de réussite: {win_rate:.2%}")
        
        # Comparaison avec la théorie
        expected_rate = 2/3  # Théoriquement, changer donne 2/3 de chance de gagner
        print(f"Taux théorique (changer): {expected_rate:.2%}")
        
        if win_rate > expected_rate * 0.9:
            print("L'agent performe bien par rapport à la théorie!")
        else:
            print("L'agent pourrait mieux performer.")
    
    # Attendre que l'utilisateur ferme la fenêtre
    print("\nAppuyez sur ÉCHAP ou fermez la fenêtre pour quitter...")
    while visualizer.running:
        event = visualizer.handle_events()
        if event is not None:
            break
        time.sleep(0.1)
    
    visualizer.quit()

if __name__ == "__main__":
    main_continuous_viewer() 