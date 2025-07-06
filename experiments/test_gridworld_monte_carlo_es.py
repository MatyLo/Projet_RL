import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.grid_world import GridWorld
from environments.visualization.pygame_visualizer import GridWorldVisualizer
from algorithms.monte_carlo.monte_carlo_es import MonteCarloES
import time

def test_gridworld_monte_carlo_es():
    """Test de Monte Carlo ES sur GridWorld."""
    print("=== MONTE CARLO ES SUR GRIDWORLD ===")
    print("Test de l'algorithme Monte Carlo Exploring Starts sur GridWorld")
    print()
    
    # Création de l'environnement GridWorld
    env = GridWorld(width=4, height=4)
    visualizer = GridWorldVisualizer(env)
    
    try:
        # PARAMÈTRES
        discount_factor = 0.99
        epsilon = 0.1
        n_episodes = 1000
        
        # ===== ENTRAÎNEMENT =====
        print("Entraînement Monte Carlo ES...")
        agent = MonteCarloES(
            environment=env,
            gamma=discount_factor,
            epsilon=epsilon
        )
        
        print(f"Entraînement sur {n_episodes} épisodes...")
        results = agent.train()
        
        print(f"Entraînement terminé!")
        print(f"- Épisodes: {n_episodes}")
        print(f"- Récompense moyenne: {results.get('avg_reward', 'N/A')}")
        print(f"- Q-table shape: {agent.Q.shape}")
        
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
                    episode_info={
                        "Algorithme": "Monte Carlo ES",
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
        
        print("\n=== RÉSUMÉ MONTE CARLO ES ===")
        print("L'algorithme Monte Carlo ES a appris à:")
        print("- Explorer toutes les actions possibles")
        print("- Utiliser des épisodes complets pour l'apprentissage")
        print("- Converger vers une politique optimale")
        
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
    test_gridworld_monte_carlo_es() 