"""
Démonstration PyGame - Visualisation et démo pas-à-pas du modèle entraîné

Ce script charge un modèle sauvegardé et le démontre avec PyGame :
1. Charge le modèle depuis les fichiers sauvegardés
2. Interface PyGame interactive
3. Démonstration pas-à-pas avec Q-values
4. Mode humain vs IA

Usage:
    python demo_scripts/demo_pygame_v4.py
    python demo_scripts/demo_pygame_v4.py --model outputs/test_v4/model
"""

import sys
import os
import argparse
import pickle
import time

# Ajout des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'utils'))

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("❌ PyGame non disponible. Installez avec: pip install pygame")
    sys.exit(1)

try:
    from src.rl_environments.line_world import LineWorld, create_lineworld
    from src.rl_algorithms.temporal_difference.q_learning import QLearning
    from utils.agent import Agent
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)


class LineWorldPygameDemo:
    """
    Démonstration PyGame pour LineWorld avec modèle entraîné.
    
    Interface graphique interactive pour :
    - Démonstration automatique de l'IA
    - Mode humain interactif
    - Visualisation des Q-values
    - Comparaison humain vs IA
    """
    
    def __init__(self, agent, environment):
        """
        Initialise la démonstration PyGame.
        
        Args:
            agent: Agent entraîné
            environment: Environnement LineWorld
        """
        self.agent = agent
        self.environment = environment
        
        # Configuration PyGame
        self.width = 800
        self.height = 400
        self.cell_width = 120
        self.cell_height = 80
        
        # Couleurs
        self.COLORS = {
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'BLUE': (0, 100, 200),       # Agent
            'GREEN': (0, 200, 0),        # Cible (position 4)
            'RED': (200, 0, 0),          # Perte (position 0)
            'GRAY': (128, 128, 128),     # Bordures
            'LIGHT_BLUE': (173, 216, 230),  # Q-values panel
            'YELLOW': (255, 255, 0)      # Surbrillance
        }
        
        # État du jeu
        self.current_state = None
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.game_mode = "demo"  # "demo", "human", "step_by_step"
        self.paused = False
        self.show_q_values = True
        
        # PyGame
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
    
    def initialize_pygame(self):
        """Initialise PyGame."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"LineWorld - {self.agent.agent_name}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        print("✅ PyGame initialisé")
    
    def demonstrate_agent_auto(self, num_episodes=3, delay_ms=1000):
        """
        Démonstration automatique de l'agent.
        
        Args:
            num_episodes: Nombre d'épisodes à démontrer
            delay_ms: Délai entre les actions (millisecondes)
        """
        print(f"🎬 DÉMONSTRATION AUTOMATIQUE: {num_episodes} épisodes")
        
        for episode in range(num_episodes):
            print(f"\n--- Épisode {episode + 1}/{num_episodes} ---")
            
            # Reset environnement
            self.current_state = self.environment.reset()
            self.episode_reward = 0.0
            self.episode_steps = 0
            self.game_mode = "demo"
            
            running = True
            max_steps = getattr(self.environment, 'max_steps', 100)
            
            while running and self.episode_steps < max_steps:
                # Gestion événements
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return
                        elif event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                
                if not self.paused:
                    # Action de l'IA
                    action = self.agent.algorithm.select_action(self.current_state, training=False)
                    next_state, reward, done, info = self.environment.step(action)
                    
                    self.episode_reward += reward
                    self.episode_steps += 1
                    self.current_state = next_state
                    
                    print(f"Action: {action}, État: {next_state}, Reward: {reward:.1f}")
                    
                    if done:
                        success = "✅ SUCCÈS" if info.get("target_reached", False) else "❌ Échec"
                        print(f"Épisode terminé: {success} | Reward total: {self.episode_reward:.1f}")
                        pygame.time.wait(2000)  # Pause entre épisodes
                        break
                
                # Rendu
                self._render_game()
                pygame.display.flip()
                
                if not self.paused:
                    pygame.time.wait(delay_ms)
                
                self.clock.tick(60)
            
            if not running:
                break
    
    def demonstrate_step_by_step(self):
        """Démonstration pas-à-pas (pour soutenance)."""
        print("🎯 DÉMONSTRATION PAS-À-PAS (Appuyez sur ESPACE pour chaque étape)")
        
        # Reset
        self.current_state = self.environment.reset()
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.game_mode = "step_by_step"
        self.paused = True
        
        running = True
        max_steps = getattr(self.environment, 'max_steps', 100)
        waiting_for_step = True
        
        while running and self.episode_steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_SPACE and waiting_for_step:
                        # Exécuter une étape
                        action = self.agent.algorithm.select_action(self.current_state, training=False)
                        next_state, reward, done, info = self.environment.step(action)
                        
                        self.episode_reward += reward
                        self.episode_steps += 1
                        self.current_state = next_state
                        
                        print(f"Étape {self.episode_steps}: Action {action} → État {next_state} (Reward: {reward:.1f})")
                        
                        if done:
                            success = "SUCCÈS" if info.get("target_reached", False) else "Échec"
                            print(f"🏁 Épisode terminé: {success}")
                            waiting_for_step = False
                        
                        if done:
                            running = False
                    elif event.key == pygame.K_q:
                        self.show_q_values = not self.show_q_values
            
            # Rendu
            self._render_game()
            self._render_instructions()
            pygame.display.flip()
            self.clock.tick(60)
    
    def play_human_mode(self):
        """Mode humain interactif."""
        print("🎮 MODE HUMAIN")
        print("Utilisez les flèches ← → pour vous déplacer")
        
        # Reset
        self.current_state = self.environment.reset()
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.game_mode = "human"
        
        running = True
        max_steps = getattr(self.environment, 'max_steps', 100)
        
        while running and self.episode_steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    
                    action = None
                    if event.key == pygame.K_LEFT:
                        action = 0  # Gauche
                    elif event.key == pygame.K_RIGHT:
                        action = 1  # Droite
                    elif event.key == pygame.K_q:
                        self.show_q_values = not self.show_q_values
                    
                    if action is not None and action in self.environment.valid_actions:
                        next_state, reward, done, info = self.environment.step(action)
                        self.episode_reward += reward
                        self.episode_steps += 1
                        self.current_state = next_state
                        
                        print(f"Action humaine: {action} → État {next_state} (Reward: {reward:.1f})")
                        
                        if done:
                            success = "SUCCÈS" if info.get("target_reached", False) else "Échec"
                            print(f"🏁 Jeu terminé: {success} | Score: {self.episode_reward:.1f}")
                            pygame.time.wait(2000)
                            return {"reward": self.episode_reward, "success": info.get("target_reached", False)}
            
            # Rendu
            self._render_game()
            self._render_instructions()
            pygame.display.flip()
            self.clock.tick(60)
    
    def _render_game(self):
        """Rendu principal du jeu."""
        self.screen.fill(self.COLORS['WHITE'])
        
        # Titre
        title = f"LineWorld - {self.agent.agent_name} | Mode: {self.game_mode.upper()}"
        title_surface = self.small_font.render(title, True, self.COLORS['BLACK'])
        self.screen.blit(title_surface, (10, 10))
        
        # Stats
        stats = f"État: {self.current_state} | Reward: {self.episode_reward:.1f} | Étapes: {self.episode_steps}"
        stats_surface = self.small_font.render(stats, True, self.COLORS['BLACK'])
        self.screen.blit(stats_surface, (10, 35))
        
        # Ligne de jeu
        self._render_line_world()
        
        # Q-values si activé
        if self.show_q_values:
            self._render_q_values()
        
        # Instructions
        if self.game_mode in ["step_by_step", "human"]:
            self._render_instructions()
    
    def _render_line_world(self):
        """Rendu de la ligne de jeu."""
        start_x = 50
        start_y = 100
        
        for pos in range(self.environment.state_space_size):
            x = start_x + pos * self.cell_width
            y = start_y
            
            # Couleur de la cellule
            if pos == self.current_state:
                color = self.COLORS['BLUE']  # Agent
            elif pos == 0:
                color = self.COLORS['RED']   # Position perdante
            elif pos == 4:
                color = self.COLORS['GREEN'] # Position gagnante
            else:
                color = self.COLORS['WHITE'] # Position normale
            
            # Bordure spéciale pour la position actuelle
            border_color = self.COLORS['YELLOW'] if pos == self.current_state else self.COLORS['GRAY']
            border_width = 4 if pos == self.current_state else 2
            
            # Dessiner cellule
            pygame.draw.rect(self.screen, color, (x, y, self.cell_width, self.cell_height))
            pygame.draw.rect(self.screen, border_color, (x, y, self.cell_width, self.cell_height), border_width)
            
            # Numéro de position
            pos_text = self.font.render(str(pos), True, self.COLORS['BLACK'])
            text_rect = pos_text.get_rect(center=(x + self.cell_width//2, y + self.cell_height//2))
            self.screen.blit(pos_text, text_rect)
            
            # Labels
            if pos == 0:
                label = self.small_font.render("PERTE", True, self.COLORS['WHITE'])
                self.screen.blit(label, (x + 5, y + self.cell_height - 20))
            elif pos == 4:
                label = self.small_font.render("CIBLE", True, self.COLORS['WHITE'])
                self.screen.blit(label, (x + 5, y + self.cell_height - 20))
    
    def _render_q_values(self):
        """Rendu des Q-values."""
        if not hasattr(self.agent.algorithm, 'q_function'):
            return
        
        try:
            q_values = self.agent.algorithm.q_function[self.current_state]
            best_action = max(range(len(q_values)), key=lambda i: q_values[i])
            
            # Panneau Q-values
            panel_x = 600
            panel_y = 100
            panel_width = 180
            panel_height = 120
            
            # Fond
            pygame.draw.rect(self.screen, self.COLORS['LIGHT_BLUE'], 
                           (panel_x, panel_y, panel_width, panel_height))
            pygame.draw.rect(self.screen, self.COLORS['BLACK'], 
                           (panel_x, panel_y, panel_width, panel_height), 2)
            
            # Titre
            title = self.small_font.render(f"Q-Values État {self.current_state}:", True, self.COLORS['BLACK'])
            self.screen.blit(title, (panel_x + 5, panel_y + 5))
            
            # Q-values
            for action, q_val in enumerate(q_values):
                y_offset = panel_y + 30 + action * 25
                color = self.COLORS['RED'] if action == best_action else self.COLORS['BLACK']
                action_name = "Gauche" if action == 0 else "Droite"
                
                text = f"{action_name}: {q_val:.2f}"
                if action == best_action:
                    text += " ★"
                
                rendered = self.small_font.render(text, True, color)
                self.screen.blit(rendered, (panel_x + 10, y_offset))
                
        except Exception as e:
            error_text = self.small_font.render("Q-values non disponibles", True, self.COLORS['RED'])
            self.screen.blit(error_text, (600, 100))
    
    def _render_instructions(self):
        """Rendu des instructions."""
        instructions_y = 300
        
        if self.game_mode == "step_by_step":
            instructions = [
                "ESPACE: Étape suivante",
                "Q: Afficher/masquer Q-values",
                "ESC: Quitter"
            ]
        elif self.game_mode == "human":
            instructions = [
                "← →: Se déplacer",
                "Q: Afficher/masquer Q-values", 
                "ESC: Quitter"
            ]
        else:
            instructions = [
                "ESPACE: Pause/Resume",
                "ESC: Quitter"
            ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.COLORS['BLACK'])
            self.screen.blit(text, (50, instructions_y + i * 20))


def load_trained_model(model_path: str):
    """
    Charge un modèle entraîné sauvegardé.
    
    Args:
        model_path: Chemin vers le modèle (sans extension)
        
    Returns:
        Tuple (algorithm, environment)
    """
    print(f"📂 Chargement du modèle: {model_path}")
    
    # Créer l'environnement
    environment = create_lineworld()
    
    # Créer et charger l'algorithme
    algorithm = QLearning(
        state_space_size=environment.state_space_size,
        action_space_size=environment.action_space_size
    )
    
    if algorithm.load_model(model_path):
        print(f"✅ Modèle chargé avec succès")
        print(f"Algorithme: {algorithm.algo_name}")
        print(f"Entraîné: {algorithm.is_trained}")
        return algorithm, environment
    else:
        raise ValueError(f"Impossible de charger le modèle: {model_path}")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Démonstration PyGame du modèle entraîné")
    parser.add_argument('--model', '-m', default='outputs/test_v4/model',
                       help='Chemin vers le modèle sauvegardé')
    args = parser.parse_args()
    
    if not PYGAME_AVAILABLE:
        print("❌ PyGame requis pour cette démonstration")
        sys.exit(1)
    
    print("🎮 DÉMONSTRATION PYGAME - MODÈLE ENTRAÎNÉ")
    print("=" * 50)
    
    try:
        # Chargement du modèle
        algorithm, environment = load_trained_model(args.model)
        
        # Création de l'agent
        agent = Agent(algorithm, environment, "DemoAgent_Pygame")
        
        # Démonstration PyGame
        demo = LineWorldPygameDemo(agent, environment)
        demo.initialize_pygame()
        
        # Menu principal
        while True:
            print(f"\n🎮 MENU DÉMONSTRATION PYGAME")
            print("1. Démonstration automatique de l'IA")
            print("2. Démonstration pas-à-pas (soutenance)")
            print("3. Mode humain interactif")
            print("4. Quitter")
            
            choice = input("Votre choix (1-4): ").strip()
            
            if choice == "1":
                demo.demonstrate_agent_auto(num_episodes=3, delay_ms=1500)
            elif choice == "2":
                demo.demonstrate_step_by_step()
            elif choice == "3":
                result = demo.play_human_mode()
                if result:
                    print(f"Résultat humain: {result}")
            elif choice == "4":
                break
            else:
                print("❌ Choix invalide")
        
        pygame.quit()
        print("👋 Démonstration terminée")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()