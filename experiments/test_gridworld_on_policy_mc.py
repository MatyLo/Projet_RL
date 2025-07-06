import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.grid_world import GridWorld
from environments.visualization.pygame_visualizer import GridWorldVisualizer
from algorithms.monte_carlo.on_policy_mc_control import OnPolicyFirstVisitMCControl
import time

def test_gridworld_on_policy_mc():
    """Test de On-Policy Monte Carlo Control sur GridWorld."""
    print("=== ON-POLICY MONTE CARLO CONTROL SUR GRIDWORLD ===")
    print("Test de l'algorithme On-Policy Monte Carlo Control sur GridWorld")
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
        print("Entraînement On-Policy Monte Carlo Control...")
        agent = OnPolicyFirstVisitMCControl(
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
                        "Algorithme": "On-Policy MC",
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
        
        print("\n=== RÉSUMÉ ON-POLICY MONTE CARLO ===")
        print("L'algorithme On-Policy Monte Carlo a appris à:")
        print("- Utiliser la même politique pour l'exploration et l'exploitation")
        print("- Améliorer la politique de manière incrémentale")
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
    test_gridworld_on_policy_mc() 