#!/usr/bin/env python3
"""
Démonstration PyGame - Agent Entraîné + Mode Humain

Ce script permet de :
1. Voir l'agent entraîné jouer en PyGame
2. Jouer soi-même en mode humain
3. Comparer les performances

Usage:
    python demo_scripts/demo_pygame.py
"""

import sys
import os
import numpy as np

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

from src.rl_environments.line_world import LineWorld
from src.rl_algorithms.temporal_difference.q_learning import QLearning
from utils.agent import Agent
from utils.human_player import HumanPlayer


class LineWorldPygameDemo:
    """Démonstration PyGame pour LineWorld."""
    
    def __init__(self, environment, agent=None):
        """
        Initialise la démonstration.
        
        Args:
            environment: Environnement LineWorld
            agent: Agent entraîné (optionnel)
        """
        self.env = environment
        self.agent = agent
        
        # Configuration PyGame
        self.width = 800
        self.height = 300
        self.cell_width = 120
        self.cell_height = 80
        
        # Couleurs
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 100, 200)
        self.GREEN = (0, 200, 0)
        self.RED = (200, 0, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_BLUE = (173, 216, 230)
        
        # État du jeu
        self.state = None
        self.episode_reward = 0.0
        self.episode_steps = 0
        
    def demonstrate_agent(self, num_episodes=3):
        """Démonstration de l'agent entraîné."""
        if not self.agent or not PYGAME_AVAILABLE:
            print("❌ Agent ou PyGame non disponible")
            return
        
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Démonstration Agent Entraîné - LineWorld")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)
        
        print(f"🎬 DÉMONSTRATION AGENT: {self.agent.agent_name}")
        print("Fermez la fenêtre ou appuyez sur ESC pour arrêter")
        
        for episode in range(num_episodes):
            print(f"\n--- Épisode {episode + 1}/{num_episodes} ---")
            
            # Reset environnement
            self.state = self.env.reset()
            self.episode_reward = 0.0
            self.episode_steps = 0
            
            running = True
            max_steps = getattr(self.env, 'max_steps', 100)
            
            while running and self.episode_steps < max_steps:
                # Gestion événements
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                
                if not running:
                    break
                
                # Action de l'agent
                action = self.agent.algorithm.select_action(self.state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                self.episode_reward += reward
                self.episode_steps += 1
                
                # Rendu
                self._render_game(screen, font, small_font, action)
                pygame.display.flip()
                
                self.state = next_state
                
                if done:
                    success = "✅ SUCCÈS!" if info.get("target_reached", False) else "❌ Échec"
                    print(f"Épisode terminé: {success} | Reward: {self.episode_reward:.2f} | Steps: {self.episode_steps}")
                    
                    # Pause entre épisodes
                    pygame.time.wait(2000)
                    break
                
                # Délai pour visualisation
                pygame.time.wait(1000)
                clock.tick(60)
            
            if not running:
                break
        
        pygame.quit()
        print("🎬 Démonstration terminée")
    
    def human_mode(self):
        """Mode humain avec PyGame."""
        if not PYGAME_AVAILABLE:
            print("❌ PyGame non disponible, utilisation du mode console")
            human = HumanPlayer(self.env, "Joueur")
            return human.play_episode(interface_mode="console")
        
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Mode Humain - LineWorld")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)
        
        print(f"🎮 MODE HUMAIN")
        print("Utilisez les flèches ← → ou A/D pour vous déplacer")
        print("Fermez la fenêtre ou ESC pour quitter")
        
        # Reset environnement
        self.state = self.env.reset()
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        running = True
        max_steps = getattr(self.env, 'max_steps', 100)
        
        while running and self.episode_steps < max_steps:
            action = None
            
            # Gestion événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    elif event.key in [pygame.K_LEFT, pygame.K_a]:
                        action = 0  # Gauche
                    elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                        action = 1  # Droite
            
            if not running:
                break
            
            # Exécuter action si valide
            if action is not None:
                next_state, reward, done, info = self.env.step(action)
                self.episode_reward += reward
                self.episode_steps += 1
                
                print(f"Action: {action} | État: {self.state} → {next_state} | Reward: {reward:.2f}")
                
                self.state = next_state
                
                if done:
                    success = "✅ SUCCÈS!" if info.get("target_reached", False) else "❌ Échec"
                    print(f"Partie terminée: {success}")
                    print(f"Reward total: {self.episode_reward:.2f} | Steps: {self.episode_steps}")
                    running = False
            
            # Rendu
            self._render_game(screen, font, small_font)
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        
        return {
            "total_reward": self.episode_reward,
            "steps": self.episode_steps,
            "success": info.get("target_reached", False) if 'info' in locals() else False
        }
    
    def _render_game(self, screen, font, small_font, last_action=None):
        """Rendu du jeu."""
        screen.fill(self.WHITE)
        
        # Titre
        title_text = f"LineWorld - État: {self.state} | Reward: {self.episode_reward:.1f} | Steps: {self.episode_steps}"
        title_surface = small_font.render(title_text, True, self.BLACK)
        screen.blit(title_surface, (10, 10))
        
        # Positions
        start_x = 50
        start_y = 80
        
        for pos in range(self.env.state_space_size):
            x = start_x + pos * self.cell_width
            y = start_y
            
            # Couleur de la cellule
            if pos == self.state:
                color = self.BLUE  # Position actuelle (agent)
            elif hasattr(self.env, 'target_position') and pos == self.env.target_position:
                color = self.GREEN  # Cible
            else:
                color = self.WHITE  # Position vide
            
            # Dessiner cellule
            pygame.draw.rect(screen, color, (x, y, self.cell_width, self.cell_height))
            pygame.draw.rect(screen, self.BLACK, (x, y, self.cell_width, self.cell_height), 2)
            
            # Numéro de position
            pos_text = font.render(str(pos), True, self.BLACK)
            text_rect = pos_text.get_rect(center=(x + self.cell_width//2, y + self.cell_height//2))
            screen.blit(pos_text, text_rect)
        
        # Instructions
        instructions = [
            "Flèches ← → ou A/D pour se déplacer",
            "Objectif: Atteindre la case verte",
            "ESC ou fermer pour quitter"
        ]
        
        for i, instruction in enumerate(instructions):
            text = small_font.render(instruction, True, self.BLACK)
            screen.blit(text, (50, 200 + i * 25))
        
        # Afficher dernière action si disponible
        if last_action is not None:
            action_text = f"Dernière action: {'← Gauche' if last_action == 0 else '→ Droite'}"
            text = small_font.render(action_text, True, self.RED)
            screen.blit(text, (400, 200))
        
        # Q-values si agent disponible
        if self.agent and hasattr(self.agent.algorithm, 'q_function'):
            self._render_q_values(screen, small_font)
    
    def _render_q_values(self, screen, small_font):
        """Affiche les Q-values de l'agent."""
        try:
            q_values = self.agent.algorithm.q_function[self.state]
            best_action = np.argmax(q_values)
            
            # Panneau Q-values
            panel_x = 600
            panel_y = 80
            
            # Fond
            pygame.draw.rect(screen, self.LIGHT_BLUE, (panel_x, panel_y, 150, 80))
            pygame.draw.rect(screen, self.BLACK, (panel_x, panel_y, 150, 80), 2)
            
            # Titre
            title = small_font.render("Q-Values:", True, self.BLACK)
            screen.blit(title, (panel_x + 5, panel_y + 5))
            
            # Q-values
            for action, q_val in enumerate(q_values):
                color = self.RED if action == best_action else self.BLACK
                action_name = "Gauche" if action == 0 else "Droite"
                text = f"{action_name}: {q_val:.2f}"
                if action == best_action:
                    text += " ★"
                
                rendered = small_font.render(text, True, color)
                screen.blit(rendered, (panel_x + 5, panel_y + 25 + action * 15))
                
        except Exception as e:
            pass  # Ignore les erreurs d'affichage Q-values


def main():
    """Fonction principale."""
    print("🎮 DÉMONSTRATION PYGAME - LINEWORLD")
    print("=" * 50)
    
    # 1. Créer et entraîner un agent
    print("1️⃣ Création et entraînement de l'agent...")
    env = LineWorld(line_length=5, start_position=0, target_position=4)
    
    config = {
        'learning_rate': 0.1,
        'gamma': 0.9,
        'epsilon': 0.1
    }
    
    algorithm = QLearning.from_config(config, env)
    algorithm.train(env, num_episodes=200, verbose=False)
    
    agent = Agent(algorithm, env, "DemoAgent")
    print(f"✅ Agent entraîné: {agent.agent_name}")
    
    # 2. Créer la démonstration
    demo = LineWorldPygameDemo(env, agent)
    
    # 3. Menu interactif
    while True:
        print(f"\n🎮 MENU DÉMONSTRATION")
        print("1. Démonstration agent entraîné (PyGame)")
        print("2. Mode humain (PyGame)")
        print("3. Mode humain (Console)")
        print("4. Comparer humain vs agent")
        print("5. Quitter")
        
        choice = input("Votre choix (1-5): ").strip()
        
        if choice == "1":
            demo.demonstrate_agent(num_episodes=3)
            
        elif choice == "2":
            result = demo.human_mode()
            print(f"Résultat: {result}")
            
        elif choice == "3":
            human = HumanPlayer(env, "Joueur")
            result = human.play_episode(interface_mode="console")
            
        elif choice == "4":
            # Comparaison simple
            print("\n🏆 COMPARAISON HUMAIN VS AGENT")
            print("Jouez d'abord...")
            human = HumanPlayer(env, "Joueur")
            human_result = human.play_episode(interface_mode="console")
            
            print("\nÉvaluation de l'agent...")
            agent_results = agent.evaluate_performance(num_episodes=10, verbose=False)
            
            print(f"\n📊 RÉSULTATS:")
            print(f"Humain: {human_result['total_reward']:.2f} points")
            print(f"Agent (moyenne): {agent_results['avg_reward']:.2f} points")
            
            if human_result['total_reward'] > agent_results['avg_reward']:
                print("🏆 L'humain gagne!")
            elif human_result['total_reward'] == agent_results['avg_reward']:
                print("🤝 Égalité!")
            else:
                print("🤖 L'agent gagne!")
                
        elif choice == "5":
            print("👋 Au revoir!")
            break
            
        else:
            print("❌ Choix invalide")


if __name__ == "__main__":
    if not PYGAME_AVAILABLE:
        print("❌ PyGame requis. Installez avec: pip install pygame")
        sys.exit(1)
    
    main()