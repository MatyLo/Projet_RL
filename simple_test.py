#!/usr/bin/env python3
"""
Test simple de l'interface Monty Hall.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def test_simple_interface():
    """Test simple de l'interface."""
    print("Test de l'interface Monty Hall...")
    
    try:
        from visualization.monty_hall_visualizer import MontyHallVisualizer
        import pygame
        
        print("1. Création du visualizer...")
        vis = MontyHallVisualizer()
        print("✅ Visualizer créé")
        
        print("2. Affichage d'un écran simple...")
        vis.clear_screen()
        vis.draw_text("TEST INTERFACE", 300, 250, vis.BLACK, vis.large_font)
        vis.draw_text("Appuyez sur Échap pour quitter", 250, 300, vis.BLACK, vis.font)
        vis.update_display()
        
        print("3. Attente d'une touche...")
        waiting = True
        while waiting and vis.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    vis.running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        vis.running = False
                        waiting = False
            
            vis.clock.tick(vis.fps)
        
        print("4. Fermeture...")
        vis.quit()
        print("✅ Test terminé avec succès")
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_interface() 