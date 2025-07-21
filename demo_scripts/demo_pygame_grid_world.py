"""
Démonstration PyGame pour GridWorld - Version Simplifiée.

Ce script permet de :
1. Jouer manuellement sur GridWorld (mode Human)
2. [FUTUR] Voir un agent entraîné jouer (mode Agent)
3. [FUTUR] Comparer performances humain vs agent (mode Compare)

Usage:
    python demo_scripts/demo_pygame_grid_world.py
"""

import sys
import os
import argparse
import pygame
import time
from pathlib import Path

# Ajout des chemins pour imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'utils'))


from src.rl_environments.grid_world import GridWorld
from utils.human_player import HumanPlayer

# Vérification de PyGame
try:
    import pygame
    pygame.init()
except ImportError:
    print("❌ PyGame non disponible. Installez-le avec: pip install pygame")
    sys.exit(1)


class GridWorldPyGameDemo:
    """Démonstration PyGame pour GridWorld avec modes multiples."""
    
    def __init__(self, model_dir: str = "outputs/grid_world"):
        """
        Initialise la démo.
        
        Args:
            model_dir: Répertoire contenant le modèle entraîné (pour usage futur)
        """
        self.model_dir = Path(model_dir)
        self.environment = GridWorld()
        
        # Interface PyGame
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.running = True
        
        # Couleurs
        self.COLORS = {
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'BLUE': (0, 100, 200),
            'GREEN': (0, 200, 0),
            'RED': (200, 0, 0),
            'GRAY': (128, 128, 128),
            'LIGHT_BLUE': (173, 216, 230),
            'YELLOW': (255, 255, 0),
            'DARK_GRAY': (64, 64, 64)
        }
        
        print(f"🎮 Démonstration GridWorld initialisée")
        print(f"📁 Répertoire modèle: {self.model_dir}")
    
    def _initialize_pygame(self):
        """Initialise PyGame pour la visualisation."""
        pygame.init()
        
        # Configuration de la fenêtre
        window_width = 700
        window_height = 700
        
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("GridWorld Demo - ESGI RL Project")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        print("✅ Interface PyGame initialisée")
    
    def _show_main_menu(self) -> str:
        """Affiche le menu principal et retourne le choix utilisateur."""
        if not self.screen:
            self._initialize_pygame()
        
        menu_options = [
            "1. Mode Human Player",
            "2. Mode Agent Trained", 
            "3. Mode Human vs Agent",
            "4. Quitter"
        ]
        
        selected_option = 0
        menu_running = True
        
        while menu_running:
            # Gestion des événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected_option = (selected_option - 1) % len(menu_options)
                    elif event.key == pygame.K_DOWN:
                        selected_option = (selected_option + 1) % len(menu_options)
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        if selected_option == 0:
                            return "human"
                        elif selected_option == 1:
                            return "agent"
                        elif selected_option == 2:
                            return "compare"
                        elif selected_option == 3:
                            return "quit"
            
            # Rendu du menu
            self.screen.fill(self.COLORS['WHITE'])
            
            # Titre
            title = self.font.render("GridWorld Demonstration", True, self.COLORS['BLACK'])
            title_rect = title.get_rect(center=(350, 150))
            self.screen.blit(title, title_rect)
            
            # Sous-titre
            subtitle = self.small_font.render("Projet ESGI - Apprentissage par Renforcement", True, self.COLORS['GRAY'])
            subtitle_rect = subtitle.get_rect(center=(350, 180))
            self.screen.blit(subtitle, subtitle_rect)
            
            # Options du menu
            for i, option in enumerate(menu_options):
                # Couleur selon disponibilité
                if i == 0:  # Mode humain disponible
                    color = self.COLORS['BLUE'] if i == selected_option else self.COLORS['BLACK']
                elif i in [1, 2]:  # Modes à implémenter
                    color = self.COLORS['DARK_GRAY'] if i == selected_option else self.COLORS['GRAY']
                else:  # Quitter
                    color = self.COLORS['RED'] if i == selected_option else self.COLORS['BLACK']
                
                text = self.small_font.render(option, True, color)
                text_rect = text.get_rect(center=(350, 250 + i * 40))
                self.screen.blit(text, text_rect)
                
                # Indicateur de sélection
                if i == selected_option:
                    border_color = self.COLORS['LIGHT_BLUE']
                    if i in [1, 2]:  # Options non disponibles
                        border_color = self.COLORS['GRAY']
                    pygame.draw.rect(self.screen, border_color, text_rect.inflate(20, 10), 2)
            
            # Instructions
            instructions = [
                "Utilisez les flèches ↑↓ pour naviguer",
                "Appuyez sur ENTRÉE ou ESPACE pour sélectionner",
                "ESC pour quitter à tout moment"
            ]
            
            for i, instruction in enumerate(instructions):
                text = self.small_font.render(instruction, True, self.COLORS['GRAY'])
                text_rect = text.get_rect(center=(350, 450 + i * 25))
                self.screen.blit(text, text_rect)
            
            # Note sur les fonctionnalités
            note = self.small_font.render("Note: Seul le mode Human est actuellement implémenté", True, self.COLORS['DARK_GRAY'])
            note_rect = note.get_rect(center=(350, 550))
            self.screen.blit(note, note_rect)
            
            pygame.display.flip()
            self.clock.tick(30)
        
        return "quit"
    
    def run_human_mode(self):
        """Lance le mode joueur humain - FONCTIONNEL."""
        print("\n🎮 MODE HUMAN PLAYER")
        print("=" * 40)
        print("🎯 Objectif: Atteindre la case verte (G) en évitant la rouge (L)")
        print("🕹️ Contrôles: WASD ou flèches directionnelles")
        print("💡 G = Goal (+1.0), L = Losing (-3.0), A = Agent")
        print("=" * 40)
        
        human_player = HumanPlayer(self.environment, "GridWorld_Human")
        
        try:
            result = human_player.play_episode(
                interface_mode="pygame",
                show_instructions=True,
                show_rewards=True
            )
            
            print(f"\n📊 RÉSULTATS DE LA PARTIE:")
            print(f"   Récompense totale: {result['total_reward']:.2f}")
            print(f"   Nombre d'étapes: {result['num_steps']}")
            print(f"   Succès: {'✅ OUI' if result['success'] else '❌ NON'}")
            
            # Attendre avant de retourner au menu
            input("\n▶️ Appuyez sur Entrée pour retourner au menu...")
            
        except Exception as e:
            print(f"❌ Erreur en mode humain: {e}")
            input("Appuyez sur Entrée pour retourner au menu...")
    
    def run_agent_mode(self):
        """Mode agent entraîné - À IMPLÉMENTER."""
        self._show_not_implemented_screen("Mode Agent Trained")
    
    def run_compare_mode(self):
        """Mode comparaison - À IMPLÉMENTER."""
        self._show_not_implemented_screen("Mode Human vs Agent")
    
    def _show_not_implemented_screen(self, mode_name: str):
        """Affiche un écran pour les fonctionnalités non implémentées."""
        if not self.screen:
            self._initialize_pygame()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                        return
            
            # Rendu de l'écran
            self.screen.fill(self.COLORS['WHITE'])
            
            # Titre
            title = self.font.render(f"{mode_name}", True, self.COLORS['BLACK'])
            title_rect = title.get_rect(center=(350, 200))
            self.screen.blit(title, title_rect)
            
            # Message principal
            messages = [
                "🚧 Cette fonctionnalité n'est pas encore implémentée",
                "",
                "À développer:",
                "• Chargement de modèles entraînés",
                "• Reconstruction des algorithmes",
                "• Démonstration pas-à-pas",
                "• Interface de comparaison",
                "",
                "Voir le code complet précédent pour l'implémentation"
            ]
            
            for i, message in enumerate(messages):
                if message:  # Ignorer les lignes vides
                    color = self.COLORS['GRAY'] if message.startswith("•") else self.COLORS['BLACK']
                    text = self.small_font.render(message, True, color)
                    text_rect = text.get_rect(center=(350, 280 + i * 25))
                    self.screen.blit(text, text_rect)
            
            # Instructions de retour
            back_text = self.small_font.render("Appuyez sur ESC ou ENTRÉE pour retourner au menu", True, self.COLORS['BLUE'])
            back_rect = back_text.get_rect(center=(350, 500))
            self.screen.blit(back_text, back_rect)
            
            pygame.display.flip()
            self.clock.tick(30)
    
    def run(self):
        """Lance la démonstration principale."""
        try:
            while self.running:
                choice = self._show_main_menu()
                
                if choice == "quit":
                    break
                elif choice == "human":
                    self.run_human_mode()
                elif choice == "agent":
                    self.run_agent_mode()
                elif choice == "compare":
                    self.run_compare_mode()
                
                # Petite pause entre les modes
                if self.running:
                    time.sleep(0.5)
                    
        except KeyboardInterrupt:
            print("\n⏸️ Démonstration interrompue par l'utilisateur")
        finally:
            if pygame.get_init():
                pygame.quit()
            print("👋 Démonstration terminée")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Démonstration PyGame pour GridWorld - Version Simplifiée",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
    python demo_scripts/demo_pygame_grid_world.py
    python demo_scripts/demo_pygame_grid_world.py --model-dir outputs/grid_world/custom_model

Fonctionnalités actuelles:
    ✅ Mode Human Player - Complètement fonctionnel
    🚧 Mode Agent Trained - À implémenter
    🚧 Mode Human vs Agent - À implémenter
        """
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='outputs/grid_world',
        help='Répertoire contenant le modèle entraîné (pour usage futur)'
    )
    
    args = parser.parse_args()
    
    print("🎮 Démonstration GridWorld PyGame - Version Simplifiée")
    print("=" * 60)
    print("✅ Mode Human Player: Fonctionnel")
    print("🚧 Mode Agent: À implémenter (structure prête)")
    print("🚧 Mode Compare: À implémenter (structure prête)")
    print("=" * 60)
    
    # Lancement de la démonstration
    demo = GridWorldPyGameDemo(args.model_dir)
    demo.run()


if __name__ == "__main__":
    main()