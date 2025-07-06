#!/usr/bin/env python3
"""
Test de Policy Iteration sur l'environnement Monty Hall avec visualisation PyGame.
"""

import sys
import os
import time
import numpy as np

# Ajouter le répertoire parent au path pour importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.monty_hall import MontyHall
from environments.visualization.monty_hall_visualizer import MontyHallVisualizer
from algorithms.dynamic_programming.policy_iteration import PolicyIteration

def test_policy_iteration_monty_hall():
    """Test de Policy Iteration sur Monty Hall avec visualisation."""
    
    print("=== Test Policy Iteration sur Monty Hall ===")
    print("Initialisation de l'environnement...")
    
    # Créer l'environnement
    env = MontyHall()
    
    # Créer le visualiseur
    visualizer = MontyHallVisualizer()
    
    print("Création de l'algorithme Policy Iteration...")
    
    # Créer l'algorithme Policy Iteration
    policy_iteration = PolicyIteration(
        environment=env,
        discount_factor=0.9,
        theta=0.001,
        max_iterations=1000
    )
    
    print("Début de l'entraînement...")
    print("Appuyez sur ESPACE pour mettre en pause/reprendre")
    print("Appuyez sur ÉCHAP pour quitter")
    print("Appuyez sur ENTRÉE pour passer à l'étape suivante")
    
    # Variables pour le contrôle de la visualisation
    paused = False
    step_by_step = False
    
    # Entraînement avec visualisation
    print("Entraînement en cours...")
    
    # Effectuer l'entraînement complet
    training_results = policy_iteration.train()
    
    print(f"Entraînement terminé après {training_results['iterations']} itérations!")
    print(f"Convergence: {training_results['converged']}")
    
    # Afficher l'état final
    episode_info = {
        "Itérations": training_results['iterations'],
        "Convergence": training_results['converged'],
        "État": env.state
    }
    visualizer.render_monty_hall_state(env, episode_info)
    
    # Attendre un peu pour voir le résultat
    time.sleep(2.0)
    
    print("Entraînement terminé!")
    print("Test de la politique apprise...")
    
    # Test de la politique apprise
    test_episodes = 10
    wins = 0
    
    for episode in range(test_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Choisir l'action selon la politique
            if env.state == 0:
                # Choix initial - choisir une porte aléatoirement
                action = np.random.choice([0, 1, 2])
            else:
                # Action rester/changer selon la politique
                action = np.argmax(policy_iteration.policy[state])
            
            # Exécuter l'action
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Afficher l'état
            episode_info = {
                "Épisode": episode + 1,
                "Étapes": steps,
                "Récompense": total_reward,
                "Action": action,
                "État": env.state
            }
            visualizer.render_monty_hall_state(env, episode_info)
            
            # Gérer les événements
            event = visualizer.handle_events()
            if event == "pause":
                paused = not paused
            elif event == "step":
                step_by_step = True
            elif not visualizer.running:
                break
            
            if paused and not step_by_step:
                continue
            
            if step_by_step:
                step_by_step = False
                time.sleep(0.5)
            else:
                time.sleep(1.0)  # Plus lent pour voir le déroulement
            
            if done:
                if info.get("result") == "win":
                    wins += 1
                break
        
        if not visualizer.running:
            break
    
    # Afficher les résultats
    win_rate = wins / test_episodes if test_episodes > 0 else 0
    print(f"\nRésultats du test:")
    print(f"Épisodes gagnés: {wins}/{test_episodes}")
    print(f"Taux de réussite: {win_rate:.2%}")
    
    # Afficher la politique finale
    print(f"\nPolitique finale:")
    for state in range(len(env.S)):
        if state in env.T:
            print(f"État {state} (terminal): N/A")
        else:
            action_probs = policy_iteration.policy[state]
            best_action = np.argmax(action_probs)
            print(f"État {state}: Action {best_action} (proba: {action_probs[best_action]:.3f})")
    
    # Afficher la fonction de valeur
    print(f"\nFonction de valeur finale:")
    for state in range(len(env.S)):
        value = policy_iteration.V[state]
        print(f"État {state}: V(s) = {value:.3f}")
    
    # Attendre que l'utilisateur ferme la fenêtre
    print("\nAppuyez sur ÉCHAP ou fermez la fenêtre pour quitter...")
    while visualizer.running:
        event = visualizer.handle_events()
        if event is not None:
            break
        time.sleep(0.1)
    
    visualizer.quit()

if __name__ == "__main__":
    test_policy_iteration_monty_hall() 