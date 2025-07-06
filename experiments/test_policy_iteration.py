import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.grid_world import GridWorld
from environments.visualization.pygame_visualizer import GridWorldVisualizer
from algorithms.dynamic_programming.policy_iteration import PolicyIteration
import time

def test_policy_iteration():
    """Test avec Policy Iteration pour comparer avec Value Iteration."""
    print("=== TEST AVEC POLICY ITERATION ===")
    print("1. Entraînement de l'agent avec Policy Iteration...")
    print("2. Test de l'agent entraîné...")
    print()
    
    # Création de l'environnement
    env = GridWorld(width=4, height=4)
    visualizer = GridWorldVisualizer(env)
    
    try:
        # ÉTAPE 1 : ENTRAÎNEMENT
        print("Entraînement en cours...")
        agent = PolicyIteration(
            environment=env,
            discount_factor=0.99,
            theta=0.0001,
            max_iterations=1000
        )
        
        # Entraînement avec visualisation étape par étape
        iteration = 0
        policy_stable = False
        
        while not policy_stable and iteration < 50:  # Limiter pour la démo
            # Policy Evaluation
            agent.policy_evaluation()
            
            # Policy Improvement
            policy_stable = agent.policy_improvement()
            iteration += 1
            
            # Afficher l'évolution de l'apprentissage
            if iteration % 5 == 0:  # Tous les 5 pas
                state = env.reset()
                visualizer.render_state(
                    state=state,
                    value_function=agent.V,
                    episode_info={
                        "Phase": "Entraînement",
                        "Itération": iteration,
                        "Algorithme": "Policy Iteration",
                        "Politique stable": "Oui" if policy_stable else "Non"
                    }
                )
                time.sleep(0.8)
        
        print(f"Entraînement terminé en {iteration} itérations!")
        print(f"Politique stable: {policy_stable}")
        
        # ÉTAPE 2 : TEST DE L'AGENT ENTRÂINÉ
        print("\nTest de l'agent entraîné avec Policy Iteration...")
        print("L'agent va maintenant utiliser sa politique apprise!")
        
        for episode in range(3):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            print(f"\n--- ÉPISODE {episode + 1} (Policy Iteration) ---")
            
            while steps < 20:  # Maximum 20 pas
                # L'agent utilise sa politique apprise !
                action = agent.get_action(state)
                
                # Exécuter l'action
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Afficher l'état avec la politique apprise
                visualizer.render_state(
                    state=state,
                    value_function=agent.V,
                    policy={state: action},  # Montrer l'action choisie
                    episode_info={
                        "Phase": "Test",
                        "Algorithme": "Policy Iteration",
                        "Épisode": episode + 1,
                        "Pas": steps,
                        "Action": ["Haut", "Bas", "Gauche", "Droite"][action],
                        "Récompense": reward,
                        "Total": total_reward,
                        "Terminé": "Oui" if done else "Non"
                    }
                )
                
                print(f"Pas {steps}: Action {['Haut', 'Bas', 'Gauche', 'Droite'][action]} → Récompense {reward}")
                
                state = next_state
                time.sleep(1.0)  # Plus lent pour voir
                
                if done:
                    print(f"🎉 Épisode terminé en {steps} pas! Récompense totale: {total_reward}")
                    break
            
            print("Appuyez sur ESPACE pour l'épisode suivant...")
            while visualizer.running:
                event = visualizer.handle_events()
                if event == "pause":
                    break
                time.sleep(0.1)
        
        print("\nTest Policy Iteration terminé !")
        print("Comparaison avec Value Iteration :")
        print("- Policy Iteration : Alternance évaluation/amélioration")
        print("- Value Iteration : Calcul direct de la valeur optimale")
        print("Appuyez sur ESPACE pour fermer...")
        
        while visualizer.running:
            event = visualizer.handle_events()
            if event == "pause":
                break
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Erreur : {e}")
        import traceback
        traceback.print_exc()
    finally:
        visualizer.quit()

if __name__ == "__main__":
    test_policy_iteration() 