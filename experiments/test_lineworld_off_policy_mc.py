import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.line_world import LineWorld
from environments.visualization.pygame_visualizer import LineWorldVisualizer
from algorithms.monte_carlo.off_policy_mc_control import OffPolicyMCControl
import time

def test_lineworld_off_policy_mc():
    """Test de Off-Policy Monte Carlo Control sur LineWorld."""
    print("=== OFF-POLICY MONTE CARLO CONTROL SUR LINEWORLD ===")
    print("Test de l'algorithme Off-Policy Monte Carlo Control sur LineWorld")
    print()
    
    # Création de l'environnement LineWorld
    env = LineWorld(length=5, start_pos=0, goal_pos=4)
    visualizer = LineWorldVisualizer(env)
    
    try:
        # PARAMÈTRES
        discount_factor = 0.99
        n_episodes = 1000
        
        # ===== ENTRAÎNEMENT =====
        print("Entraînement Off-Policy Monte Carlo Control...")
        agent = OffPolicyMCControl(
            environment=env,
            gamma=discount_factor
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
                        "Algorithme": "Off-Policy MC",
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
        
        print("\n=== RÉSUMÉ OFF-POLICY MONTE CARLO ===")
        print("L'algorithme Off-Policy Monte Carlo a appris à:")
        print("- Utiliser une politique comportementale pour l'exploration")
        print("- Apprendre une politique cible optimale")
        print("- Converger vers la politique optimale")
        
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
    test_lineworld_off_policy_mc() 