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

from rl_environments.TwoRoundRPS import TwoRoundRPSEnvironment
from rl_algorithms.temporal_difference.q_learning import QLearning
from agent import Agent
from human_player import HumanPlayer


class RPSPygameDemo:
    def __init__(self, environment, agent=None):
        self.env = environment
        self.agent = agent

        self.width, self.height = 600, 350
        self.buttons = self._create_buttons()

        """self.state = self.env.reset()
        self.episode_reward = 0
        self.render_data = self.env.render(mode='pygame')"""
        self.state = None
        self.episode_reward = 0.0
        self.episode_steps = 0

        # Pygame colors
        self.COLORS = {
            "WHITE": (255, 255, 255),
            "BLACK": (0, 0, 0),
            "GRAY": (200, 200, 200),
            "GREEN": (0, 200, 0),
            "RED": (200, 0, 0),
            "YELLOW": (200, 200, 0)
        }

    def _create_buttons(self):
        # Boutons Rock (0), Paper (1), Scissors (2)
        button_w = 150
        spacing = 20
        start_x = (600 - (3 * button_w + 2 * spacing)) // 2
        y = 250
        buttons = []
        for i in range(3):
            rect = pygame.Rect(start_x + i * (button_w + spacing), y, button_w, 50)
            buttons.append((rect, i))
        return buttons

    def demonstrate_agent(self, num_episodes=3):
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        font = pygame.font.SysFont("Arial", 24)
        clock = pygame.time.Clock()
        
        # Couleurs pour l'animation du bouton
        button_colors = [
            (255, 100, 100),  # Rouge clair
            (100, 255, 100),  # Vert clair
            (100, 100, 255),  # Bleu clair
            (255, 255, 100),  # Jaune
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
            (255, 150, 100),  # Orange
            (200, 100, 255),  # Violet
        ]
        current_color_index = 0
        
        for ep in range(num_episodes):
            self.state = self.env.reset()
            self.render_data = self.env.render(mode="pygame")
            done = False
            
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        break
                
                # Sélection de l'action
                action = self.agent.algorithm.select_action(self.state, training=False)
                
                # Changement de couleur du bouton
                current_color = button_colors[current_color_index % len(button_colors)]
                current_color_index += 1
                
                # Affichage avec la nouvelle couleur
                self._render_with_colored_action(screen, font, action, current_color)
                pygame.display.flip()
                
                # Pause pour voir l'action sélectionnée
                pygame.time.wait(800)  # 800ms pour voir le bouton coloré
                
                # Exécution de l'action
                self.state, reward, done, info = self.env.step(action)
                self.render_data = self.env.render(mode="pygame")
                
                # Affichage du résultat
                self._render(screen, font)
                pygame.display.flip()
                
                # Pause plus longue entre les actions
                pygame.time.wait(2500)  # 1.5 secondes entre chaque action
                
            # Pause entre les épisodes
            pygame.time.wait(2000)
        
        pygame.quit()

    def _render_with_colored_action(self, screen, font, action, color):
        """Fonction auxiliaire pour afficher l'interface avec le bouton d'action coloré"""
        # Appel du rendu normal
        self._render(screen, font)
        
        # Trouver le bouton correspondant à l'action sélectionnée
        selected_button_rect = None
        for rect, button_action in self.buttons:
            if button_action == action:
                selected_button_rect = rect
                break
        
        if selected_button_rect:
            # Dessiner un contour coloré autour du bouton sélectionné
            pygame.draw.rect(screen, color, selected_button_rect, 4)
            
            # Optionnel : remplir légèrement le bouton avec la couleur
            overlay_color = (*color, 50)  # Couleur semi-transparente
            overlay_surface = pygame.Surface((selected_button_rect.width, selected_button_rect.height))
            overlay_surface.set_alpha(50)
            overlay_surface.fill(color)
            screen.blit(overlay_surface, (selected_button_rect.x, selected_button_rect.y))
            
            # Redessiner le texte du bouton par-dessus
            labels = ["Rock", "Paper", "Scissors"]
            label = font.render(labels[action], True, self.COLORS["BLACK"])
            screen.blit(label, (selected_button_rect.x + 30, selected_button_rect.y + 10))
        
        # Afficher quelle action est sélectionnée
        labels = ["Rock", "Paper", "Scissors"]
        action_text = font.render(f"Selected: {labels[action]}", True, color)
        screen.blit(action_text, (20, self.height - 40))

    def human_mode(self):
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        font = pygame.font.SysFont("Arial", 24)
        clock = pygame.time.Clock()

        self.state = self.env.reset()
        self.render_data = self.env.render(mode="pygame")
        done = False

        while not done:
            self._render(screen, font)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for rect, action in self.buttons:
                        if rect.collidepoint(event.pos):
                            self.state, reward, done, info = self.env.step(action)
                            self.render_data = self.env.render(mode="pygame")
                            break

            clock.tick(30)

        pygame.time.wait(3000)
        pygame.quit()

    def _render(self, screen, font):
        screen.fill(self.COLORS["WHITE"])

        # Infos générales
        y = 20
        screen.blit(font.render(self.render_data["round_info"], True, self.COLORS["BLACK"]), (20, y))
        y += 30

        if self.render_data["current_round"] >= 2:
            result = self.render_data["round1_result"]
            if result:
                color = self.COLORS[self.env.get_result_color(result)]
                screen.blit(font.render(f"Round 1 result: {result}", True, color), (20, y))
                y += 30

        if self.render_data["game_finished"]:
            result = self.render_data["game_summary"]["game_result"]
            screen.blit(font.render(f"Final Result: {result}", True, self.COLORS["BLACK"]), (20, y))

        # Boutons
        labels = ["Rock", "Paper", "Scissors"]
        for rect, action in self.buttons:
            pygame.draw.rect(screen, self.COLORS["GRAY"], rect)
            label = font.render(labels[action], True, self.COLORS["BLACK"])
            screen.blit(label, (rect.x + 30, rect.y + 10))


def main():
    #Fonction principale adaptée à RPSPygameDemo.
    print("🎮 DÉMONSTRATION PYGAME - Two Round RPS")
    print("=" * 50)

    # 1. Créer et entraîner un agent
    print("1️⃣ Création et entraînement de l'agent...")
    env = TwoRoundRPSEnvironment()

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
    demo = RPSPygameDemo(environment=env, agent=agent)

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
            demo.demonstrate_agent(num_episodes=1)

        elif choice == "2":
            result = demo.human_mode()
            print(f"Résultat: {result}")

        elif choice == "3":
            human = HumanPlayer(env, "Joueur")
            result = human.play_episode(interface_mode="console")

        elif choice == "4":
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
    try:
        import pygame
        PYGAME_AVAILABLE = True
    except ImportError:
        PYGAME_AVAILABLE = False

    if not PYGAME_AVAILABLE:
        print("❌ PyGame requis. Installez avec: pip install pygame")
        sys.exit(1)

    main()
