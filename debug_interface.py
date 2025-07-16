#!/usr/bin/env python3
"""
Script de debug pour l'interface Monty Hall.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def test_visualizer_creation():
    """Teste la création du visualizer."""
    print("Test 1: Création du visualizer...")
    try:
        from visualization.monty_hall_visualizer import MontyHallVisualizer
        vis = MontyHallVisualizer()
        print("✅ Visualizer créé avec succès")
        return vis
    except Exception as e:
        print(f"❌ Erreur lors de la création: {e}")
        return None

def test_main_menu(vis):
    """Teste l'affichage du menu principal."""
    print("Test 2: Affichage du menu principal...")
    try:
        mode, algorithm = vis.render_main_menu()
        print(f"✅ Menu affiché - Mode: {mode}, Algorithme: {algorithm}")
        return mode, algorithm
    except Exception as e:
        print(f"❌ Erreur lors de l'affichage du menu: {e}")
        return None, None

def test_environment_selection():
    """Teste la sélection d'environnement."""
    print("Test 3: Sélection d'environnement...")
    try:
        from visualization.run_monty_hall import run_with_environment_selection
        print("✅ Fonction de sélection d'environnement trouvée")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    print("=== Debug de l'interface Monty Hall ===\n")
    
    # Test 1: Création du visualizer
    vis = test_visualizer_creation()
    if vis is None:
        print("❌ Impossible de continuer sans visualizer")
        sys.exit(1)
    
    # Test 2: Menu principal (avec timeout)
    print("\nTest 2: Affichage du menu principal (5 secondes)...")
    import threading
    import time
    
    result = [None, None]
    
    def menu_test():
        try:
            result[0], result[1] = vis.render_main_menu()
        except Exception as e:
            print(f"❌ Erreur dans le menu: {e}")
    
    thread = threading.Thread(target=menu_test)
    thread.daemon = True
    thread.start()
    
    # Attendre 5 secondes
    time.sleep(5)
    print("⏰ Timeout atteint")
    
    # Test 3: Sélection d'environnement
    test_environment_selection()
    
    print("\n=== Debug terminé ===") 