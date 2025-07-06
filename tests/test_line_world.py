import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.line_world import LineWorld
import numpy as np

def test_environment_initialization():
    """Test l'initialisation de l'environnement."""
    env = LineWorld(length=5, start_pos=0, goal_pos=4)
    
    # Vérification des paramètres de base
    assert len(env.S) == 5, "Le nombre d'états devrait être 5"
    assert len(env.A) == 2, "Le nombre d'actions devrait être 2"
    assert len(env.R) == 3, "Le nombre de récompenses devrait être 3"
    assert env.T == [0, 4], "Les états terminaux devraient être [0, 4]"
    
    print("✓ Initialisation de l'environnement correcte")

def test_transition_matrix():
    """Test la matrice de transition."""
    env = LineWorld(length=5, start_pos=0, goal_pos=4)
    p = env.get_transition_matrix()
    
    # Vérification de la forme de la matrice
    assert p.shape == (5, 2, 5, 3), "Forme incorrecte de la matrice de transition"
    
    # Vérification de quelques transitions spécifiques
    # État 0, action gauche (reste à 0 avec récompense -1)
    assert p[0, 0, 0, 0] == 1.0, "Transition incorrecte pour état 0, action gauche"
    
    # État 0, action droite (va à 1 avec récompense 0)
    assert p[0, 1, 1, 1] == 1.0, "Transition incorrecte pour état 0, action droite"
    
    # État 4, action droite (reste à 4 avec récompense 1)
    assert p[4, 1, 4, 2] == 1.0, "Transition incorrecte pour état 4, action droite"
    
    print("✓ Matrice de transition correcte")

def test_step_function():
    """Test la fonction step."""
    env = LineWorld(length=5, start_pos=0, goal_pos=4)
    
    # Test de quelques transitions
    state = env.reset()
    assert state == 0, "État initial incorrect"
    
    # Test action droite depuis l'état 0
    next_state, reward, done, _ = env.step(1)  # Action droite
    assert next_state == 1, "Transition incorrecte pour action droite depuis état 0"
    assert reward == 0.0, "Récompense incorrecte pour action droite depuis état 0"
    assert not done, "L'épisode ne devrait pas être terminé"
    
    # Test action gauche depuis l'état 1
    next_state, reward, done, _ = env.step(0)  # Action gauche
    assert next_state == 0, "Transition incorrecte pour action gauche depuis état 1"
    assert reward == -1.0, "Récompense incorrecte pour action gauche depuis état 1"
    assert done, "L'épisode devrait être terminé (état terminal atteint)"
    
    print("✓ Fonction step correcte")

def test_episode():
    """Test un épisode complet."""
    env = LineWorld(length=5, start_pos=0, goal_pos=4)
    state = env.reset()
    total_reward = 0
    steps = 0
    
    print("\nTest d'un épisode complet:")
    print("État initial:", state)
    env.render()
    
    # Séquence d'actions: droite, droite, droite, droite
    actions = [1, 1, 1, 1]
    for action in actions:
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"\nAction: {'droite' if action == 1 else 'gauche'}")
        print(f"État: {state}, Récompense: {reward}")
        env.render()
        
        if done:
            break
    
    print(f"\nÉpisode terminé en {steps} pas")
    print(f"Récompense totale: {total_reward}")
    assert state == 4, "L'épisode devrait se terminer à l'état 4"
    assert done, "L'épisode devrait être terminé"
    
    print("✓ Test d'épisode complet réussi")

def main():
    """Exécute tous les tests."""
    print("Début des tests de l'environnement LineWorld...\n")
    
    test_environment_initialization()
    test_transition_matrix()
    test_step_function()
    test_episode()
    
    print("\nTous les tests ont été passés avec succès!")

if __name__ == "__main__":
    main() 