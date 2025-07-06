import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.grid_world import GridWorld
import numpy as np

def test_environment_initialization():
    """Test l'initialisation de l'environnement GridWorld."""
    env = GridWorld(width=4, height=4, start_pos=(0,0), goal_pos=(3,3))
    assert len(env.S) == 16, "Le nombre d'états devrait être 16 (4x4)"
    assert len(env.A) == 4, "Le nombre d'actions devrait être 4 (haut, bas, gauche, droite)"
    assert len(env.R) == 2, "Le nombre de récompenses devrait être 2 (-1, 0)"
    assert env.T == [15], "L'état terminal devrait être [15] (case (3,3))"
    print("✓ Initialisation de l'environnement correcte")

def test_transition_matrix():
    """Test la matrice de transition de GridWorld."""
    env = GridWorld(width=4, height=4, start_pos=(0,0), goal_pos=(3,3))
    p = env.get_transition_matrix()
    assert p.shape == (16, 4, 16, 2), "Forme incorrecte de la matrice de transition"
    # Test : depuis (0,0), action bas (1) -> (1,0), reward -1
    assert p[0, 1, 4, 0] == 1.0, "Transition incorrecte pour (0,0) action bas"
    # Test : depuis (3,3), action droite (3) -> reste sur place, reward 0
    assert p[15, 3, 15, 1] == 1.0, "Transition incorrecte pour (3,3) action droite"
    print("✓ Matrice de transition correcte")

def test_step_function():
    """Test la fonction step de GridWorld."""
    env = GridWorld(width=4, height=4, start_pos=(0,0), goal_pos=(3,3))
    state = env.reset()
    assert state == 0, "État initial incorrect (devrait être 0)"
    # Action bas (1) depuis (0,0)
    next_state, reward, done, _ = env.step(1)
    assert next_state == 4, "Transition incorrecte pour action bas depuis (0,0)"
    assert reward == -1.0, "Récompense incorrecte pour action bas depuis (0,0)"
    assert not done, "L'épisode ne devrait pas être terminé"
    # Action droite (3) depuis (1,0)
    next_state, reward, done, _ = env.step(3)
    assert next_state == 5, "Transition incorrecte pour action droite depuis (1,0)"
    print("✓ Fonction step correcte")

def test_episode():
    """Test un épisode complet dans GridWorld."""
    env = GridWorld(width=4, height=4, start_pos=(0,0), goal_pos=(3,3))
    state = env.reset()
    total_reward = 0
    steps = 0
    print("\nTest d'un épisode complet:")
    print("État initial:", state)
    env.render()
    # Séquence d'actions pour aller en bas puis à droite jusqu'au but
    actions = [1, 1, 1, 3, 3, 3]
    for action in actions:
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        print(f"\nAction: {['haut','bas','gauche','droite'][action]}")
        print(f"État: {state}, Récompense: {reward}")
        env.render()
        if done:
            break
    print(f"\nÉpisode terminé en {steps} pas")
    print(f"Récompense totale: {total_reward}")
    assert state == 15, "L'épisode devrait se terminer à l'état 15 (3,3)"
    assert done, "L'épisode devrait être terminé"
    print("✓ Test d'épisode complet réussi")

def main():
    print("Début des tests de l'environnement GridWorld...\n")
    test_environment_initialization()
    test_transition_matrix()
    test_step_function()
    test_episode()
    print("\nTous les tests ont été passés avec succès!")

if __name__ == "__main__":
    main() 