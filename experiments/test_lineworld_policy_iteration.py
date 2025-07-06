import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.line_world import LineWorld
from environments.visualization.pygame_visualizer import LineWorldVisualizer
from algorithms.dynamic_programming.policy_iteration import PolicyIteration
import time

def test_lineworld_policy_iteration():
    """Test de Policy Iteration sur LineWorld."""
    print("=== POLICY ITERATION SUR LINEWORLD ===")
    print("Test de l'algorithme Policy Iteration sur l'environnement linéaire")
    print()
    
    # Création de l'environnement LineWorld
    env = LineWorld(length=5, start_pos=0, goal_pos=4)
    visualizer = LineWorldVisualizer(env)
    
    try:
        # PARAMÈTRES
        discount_factor = 0.99
        theta = 0.0001
        max_iterations = 1000
        
        # ===== ENTRAÎNEMENT =====
        print("Entraînement Policy Iteration...")
        agent = PolicyIteration(
            environment=env,
            discount_factor=discount_factor,
            theta=theta,
            max_iterations=max_iterations
        )
        
        results = agent.train()
        print(f"Entraînement terminé!")
        print(f"- Itérations: {results['iterations']}")
        print(f"- Convergé: {results['converged']}")
        print(f"- Fonction de valeur finale: {[f'{v:.3f}' for v in agent.V]}")
        
        # ===== TEST =====
        print("\n=== TEST DE L'AGENT ENTRÂINÉ ===")
        
        for episode in range(3):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            print(f"\n--- ÉPISODE {episode + 1} ---")
            
            while steps < 20:  # Maximum 20 pas
                # L'agent utilise sa politique apprise
                action = agent.get_action(state)
                
                # Exécuter l'action
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Afficher l'état
                visualizer.render_state(
                    state=state,
                    value_function=agent.V,
                    policy={state: action},
                    episode_info={
                        "Algorithme": "Policy Iteration",
                        "Épisode": episode + 1,
                        "Pas": steps,
                        "Action": "Gauche" if action == 0 else "Droite",
                        "Récompense": reward,
                        "Total": total_reward,
                        "Terminé": "Oui" if done else "Non"
                    }
                )
                
                print(f"Pas {steps}: Action {'Gauche' if action == 0 else 'Droite'} → Récompense {reward}")
                
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
        
        print("\n=== RÉSUMÉ POLICY ITERATION ===")
        print("L'algorithme Policy Iteration a appris à:")
        print("- Alterner entre évaluation et amélioration de la politique")
        print("- Converger vers la politique optimale")
        print("- Maximiser la récompense totale")
        
        print("\nTest terminé ! Appuyez sur ESPACE pour fermer...")
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
    test_lineworld_policy_iteration() 