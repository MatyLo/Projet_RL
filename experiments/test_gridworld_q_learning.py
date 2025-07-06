import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.grid_world import GridWorld
from environments.visualization.pygame_visualizer import GridWorldVisualizer
from algorithms.td_learning.q_learning import QLearning
import time

def test_gridworld_q_learning():
    """Test de Q-Learning sur GridWorld."""
    print("=== Q-LEARNING SUR GRIDWORLD ===")
    print("Test de l'algorithme Q-Learning (off-policy TD control) sur GridWorld")
    print()
    
    # Création de l'environnement GridWorld
    env = GridWorld(width=4, height=4)
    visualizer = GridWorldVisualizer(env)
    
    try:
        # PARAMÈTRES
        gamma = 0.99
        alpha = 0.1
        epsilon = 0.1
        n_episodes = 1000
        
        # ===== ENTRAÎNEMENT =====
        print("Entraînement Q-Learning...")
        agent = QLearning(
            environment=env,
            num_episodes=n_episodes,
            gamma=gamma,
            alpha=alpha,
            epsilon=epsilon
        )
        
        print(f"Entraînement sur {n_episodes} épisodes...")
        results = agent.train()
        
        print(f"Entraînement terminé!")
        print(f"- Épisodes: {n_episodes}")
        print(f"- Q-table shape: {agent.Q.shape}")
        print(f"- Alpha (taux d'apprentissage): {alpha}")
        print(f"- Epsilon (exploration): {epsilon}")
        
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
                        "Algorithme": "Q-Learning",
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
        
        print("\n=== RÉSUMÉ Q-LEARNING ===")
        print("L'algorithme Q-Learning a appris à:")
        print("- Utiliser l'apprentissage par différence temporelle")
        print("- Apprendre de manière off-policy (max Q-value)")
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
    test_gridworld_q_learning() 