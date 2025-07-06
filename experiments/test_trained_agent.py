import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.grid_world import GridWorld
from environments.visualization.pygame_visualizer import GridWorldVisualizer
from algorithms.dynamic_programming.value_iteration import ValueIteration
import time

def test_trained_agent():
    """Test avec un agent entraîné qui sait résoudre le problème."""
    print("=== TEST AVEC AGENT ENTRÂINÉ ===")
    print("1. Entraînement de l'agent avec Value Iteration...")
    print("2. Test de l'agent entraîné...")
    print()
    
    # Création de l'environnement
    env = GridWorld(width=4, height=4)
    visualizer = GridWorldVisualizer(env)
    
    try:
        # ÉTAPE 1 : ENTRAÎNEMENT
        print("Entraînement en cours...")
        agent = ValueIteration(
            environment=env,
            discount_factor=0.99,
            theta=0.0001,
            max_iterations=1000
        )
        
        # Entraînement avec visualisation
        iteration = 0
        while iteration < 100:  # Limiter pour la démo
            agent.train()
            iteration += 1
            
            # Afficher l'évolution de l'apprentissage
            if iteration % 20 == 0:  # Tous les 20 pas
                state = env.reset()
                visualizer.render_state(
                    state=state,
                    value_function=agent.V,
                    episode_info={
                        "Phase": "Entraînement",
                        "Itération": iteration,
                        "Agent": "En cours d'apprentissage..."
                    }
                )
                time.sleep(0.5)
        
        print(f"Entraînement terminé en {iteration} itérations!")
        
        # ÉTAPE 2 : TEST DE L'AGENT ENTRÂINÉ
        print("\nTest de l'agent entraîné...")
        print("L'agent va maintenant utiliser sa politique apprise!")
        
        for episode in range(3):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            print(f"\n--- ÉPISODE {episode + 1} ---")
            
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
        
        print("\nTest terminé ! L'agent entraîné est beaucoup plus efficace!")
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
    test_trained_agent() 