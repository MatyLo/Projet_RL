"""
Démonstration PyGame - Two Round Rock Paper Scissors

Ce script permet de :
1. Voir l'agent entraîné jouer en PyGame
2. Jouer soi-même en mode humain
3. Quitter

Usage:
    python demo_scripts/demo_rps_pygame.py
"""

import sys
import os
import numpy as np
import time

# Configuration des chemins (adaptez selon votre structure)
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

# Imports des modules
from rl_environments.TwoRoundRPS import TwoRoundRPSEnvironment
from rl_algorithms.q_learning import QLearning
from rl_algorithms.policy_iteration import PolicyIteration
from rl_algorithms.value_iteration import ValueIteration
from rl_algorithms.monte_carlo_es import MonteCarloES
from rl_algorithms.on_policy_first_visit_mc_control import OnPolicyFirstVisitMCControl
from rl_algorithms.off_policy_mc_control import OffPolicyMCControl
from rl_algorithms.sarsa import SARSA
from rl_algorithms.dyna_q import DynaQ
from agent import Agent
from human_player import HumanPlayer


class TwoRoundRPSPygameDemo:
    """Démonstration PyGame pour Two Round Rock Paper Scissors."""
    
    def __init__(self, environment, agent=None):
        """
        Initialise la démonstration.
        
        Args:
            environment: Environnement TwoRoundRPSEnvironment
            agent: Agent entraîné (optionnel)
        """
        self.env = environment
        self.agent = agent
        
        # Configuration PyGame
        self.width = 1000
        self.height = 700
        
        # Couleurs
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 100, 200)
        self.GREEN = (0, 200, 0)
        self.RED = (200, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_BLUE = (173, 216, 230)
        self.LIGHT_GREEN = (144, 238, 144)
        self.LIGHT_RED = (255, 182, 193)
        self.LIGHT_YELLOW = (255, 255, 224)
        
        # Symboles pour les actions
        self.ACTION_SYMBOLS = {
            0: "🪨",  # Rock
            1: "📄",  # Paper
            2: "✂️"   # Scissors
        }
        
        # État du jeu
        self.state = None
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.game_history = []
        

    def demonstrate_agent(self, num_episodes=5):
        """Démonstration de l'agent entraîné."""
        if not self.agent or not PYGAME_AVAILABLE:
            print("❌ Agent ou PyGame non disponible")
            return
        
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Démonstration Agent Entraîné - Two Round RPS")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 48)
        small_font = pygame.font.Font(None, 32)
        
        print(f"🎬 DÉMONSTRATION AGENT: {self.agent.agent_name}")
        #print("Fermez la fenêtre ou appuyez sur ESC pour arrêter")
        
        for episode in range(num_episodes):
            print(f"\n--- Épisode {episode + 1}/{num_episodes} ---")
            # DÉBOGAGE : Avant reset
            print(f"Q-table shape: {self.agent.algorithm.q_function.shape}")
            print(f"Q-table non-zéro: {np.count_nonzero(self.agent.algorithm.q_function)}")
        
            # Reset environnement
            self.state = self.env.reset()
            # DÉBOGAGE : Après reset
            print(f"État initial: {self.state}")
            print(f"Q-values pour cet état: {self.agent.algorithm.q_function[self.state]}")
        
            self.episode_reward = 0.0
            self.episode_steps = 0
            self.game_history = []
            
            running = True
            round_num = 1
            
            while running and round_num <= 2:
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
                
                # Enregistrer l'historique
                self.game_history.append({
                    'round': info['round'],
                    'agent_choice': action,
                    'opponent_choice': info['opponent_choice'],
                    'result': info['result'],
                    'description': info['round_description']
                })
                
                print(f"Round {info['round']}: {info['round_description']}")
                
                # Rendu
                self._render_game(screen, font, small_font, info, action)
                pygame.display.flip()
                
                self.state = next_state
                
                if done:
                    game_summary = info.get('game_summary', {})
                    print(f"Jeu terminé: {game_summary.get('game_result', 'Inconnu')}")
                    print(f"Score final: {game_summary.get('total_score', 0)}")
                    
                    # Rendu final
                    self._render_game(screen, font, small_font, info, action, game_finished=True)
                    pygame.display.flip()
                    
                    # Pause entre épisodes
                    pygame.time.wait(3000)
                    break
                
                round_num += 1
                # Délai entre les rounds
                pygame.time.wait(2000)
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
        pygame.display.set_caption("Mode Humain - Two Round RPS")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 48)
        small_font = pygame.font.Font(None, 32)
        
        print("🎮 MODE HUMAIN")
        print("Utilisez les touches 1, 2, 3 pour Rock, Paper, Scissors")
        print("Fermez la fenêtre ou ESC pour quitter")
        
        # Reset environnement
        self.state = self.env.reset()
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.game_history = []
        
        running = True
        waiting_for_action = True
        current_info = None
        
        while running and self.env.current_round <= 2:
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
                    elif event.key == pygame.K_1:
                        action = 0  # Rock
                    elif event.key == pygame.K_2:
                        action = 1  # Paper
                    elif event.key == pygame.K_3:
                        action = 2  # Scissors
            
            if not running:
                break
            
            # Exécuter action si valide
            if action is not None and waiting_for_action:
                next_state, reward, done, info = self.env.step(action)
                self.episode_reward += reward
                self.episode_steps += 1
                
                # Enregistrer l'historique
                self.game_history.append({
                    'round': info['round'],
                    'agent_choice': action,
                    'opponent_choice': info['opponent_choice'],
                    'result': info['result'],
                    'description': info['round_description']
                })
                
                print(f"Round {info['round']}: {info['round_description']}")
                
                self.state = next_state
                current_info = info
                waiting_for_action = False
                
                if done:
                    game_summary = info.get('game_summary', {})
                    print(f"Jeu terminé: {game_summary.get('game_result', 'Inconnu')}")
                    print(f"Score final: {game_summary.get('total_score', 0)}")
                    
                    # Rendu final et attendre
                    self._render_game(screen, font, small_font, info, action, game_finished=True)
                    pygame.display.flip()
                    
                    # Attendre avant de fermer
                    start_time = time.time()
                    while time.time() - start_time < 3:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                                break
                    break
                else:
                    # Attendre un peu avant le prochain round
                    start_time = time.time()
                    while time.time() - start_time < 2:
                        self._render_game(screen, font, small_font, info, action)
                        pygame.display.flip()
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                                break
                        if not running:
                            break
                    
                    waiting_for_action = True
            
            # Rendu
            self._render_game(screen, font, small_font, current_info, 
                            None if waiting_for_action else action)
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        
        return {
            "total_reward": self.episode_reward,
            "steps": self.episode_steps,
            "game_history": self.game_history
        }
    
    def _render_game(self, screen, font, small_font, info=None, last_action=None, game_finished=False):
        """Rendu du jeu."""
        screen.fill(self.WHITE)
        
        # Titre
        title = "Two Round Rock Paper Scissors"
        title_surface = font.render(title, True, self.BLACK)
        title_rect = title_surface.get_rect(center=(self.width//2, 50))
        screen.blit(title_surface, title_rect)
        
        # Informations de l'épisode
        episode_info = f"Reward: {self.episode_reward:.1f} | Steps: {self.episode_steps}"
        episode_surface = small_font.render(episode_info, True, self.BLACK)
        screen.blit(episode_surface, (20, 100))
        
        # État actuel du jeu
        if self.env.current_round == 1:
            round_info = "Round 1 - L'adversaire joue aléatoirement"
        elif self.env.current_round == 2:
            round_info = f"Round 2 - L'adversaire jouera: {self.env.ACTION_NAMES[self.env.agent_round1_choice]}"
        else:
            round_info = "Jeu terminé"
        
        round_surface = small_font.render(round_info, True, self.BLUE)
        screen.blit(round_surface, (20, 140))
        
        # Choix des actions (boutons visuels)
        self._render_action_buttons(screen, font, small_font)
        
        # Historique des rounds
        self._render_game_history(screen, small_font)
        
        # Instructions
        if not game_finished:
            instructions = [
                "Appuyez sur:",
                "1 - Rock (Pierre) 🪨",
                "2 - Paper (Feuille) 📄", 
                "3 - Scissors (Ciseaux) ✂️",
                "ESC - Quitter"
            ]
            
            for i, instruction in enumerate(instructions):
                color = self.RED if i == 0 else self.BLACK
                text = small_font.render(instruction, True, color)
                screen.blit(text, (20, 500 + i * 30))
        
        # Résultat final si terminé
        if game_finished and info and 'game_summary' in info:
            self._render_final_result(screen, font, small_font, info['game_summary'])
        
        # Q-values si agent disponible
        if self.agent and hasattr(self.agent.algorithm, 'q_function'):
            self._render_q_values(screen, small_font)
    
    def _render_action_buttons(self, screen, font, small_font):
        """Affiche les boutons d'action."""
        button_width = 150
        button_height = 100
        button_spacing = 200
        start_x = (self.width - 3 * button_width - 2 * button_spacing) // 2
        button_y = 200
        
        actions = [
            (0, "Rock", "🪨", self.LIGHT_RED),
            (1, "Paper", "📄", self.LIGHT_BLUE),
            (2, "Scissors", "✂️", self.LIGHT_YELLOW)
        ]
        
        for i, (action, name, symbol, color) in enumerate(actions):
            x = start_x + i * (button_width + button_spacing)
            
            # Bouton
            pygame.draw.rect(screen, color, (x, button_y, button_width, button_height))
            pygame.draw.rect(screen, self.BLACK, (x, button_y, button_width, button_height), 3)
            
            # Numéro
            num_text = font.render(str(action + 1), True, self.BLACK)
            num_rect = num_text.get_rect(center=(x + button_width//2, button_y + 20))
            screen.blit(num_text, num_rect)
            
            # Nom
            name_text = small_font.render(name, True, self.BLACK)
            name_rect = name_text.get_rect(center=(x + button_width//2, button_y + 50))
            screen.blit(name_text, name_rect)
            
            # Symbole
            symbol_text = font.render(symbol, True, self.BLACK)
            symbol_rect = symbol_text.get_rect(center=(x + button_width//2, button_y + 80))
            screen.blit(symbol_text, symbol_rect)
    
    def _render_game_history(self, screen, small_font):
        """Affiche l'historique du jeu."""
        if not self.game_history:
            return
        
        # Titre
        history_title = small_font.render("Historique des rounds:", True, self.BLACK)
        screen.blit(history_title, (600, 200))
        
        # Rounds
        for i, round_data in enumerate(self.game_history):
            y_pos = 240 + i * 80
            
            # Fond coloré selon le résultat
            result_color = self.LIGHT_GREEN if round_data['result'] > 0 else \
                          self.LIGHT_RED if round_data['result'] < 0 else \
                          self.LIGHT_YELLOW
            
            pygame.draw.rect(screen, result_color, (600, y_pos, 350, 70))
            pygame.draw.rect(screen, self.BLACK, (600, y_pos, 350, 70), 2)
            
            # Titre du round
            round_title = small_font.render(f"Round {round_data['round']}", True, self.BLACK)
            screen.blit(round_title, (610, y_pos + 5))
            
            # Choix
            agent_choice_name = self.env.ACTION_NAMES[round_data['agent_choice']]
            opponent_choice_name = round_data['opponent_choice']  # Déjà une string
            
            choices_text = f"Vous: {agent_choice_name} vs Adversaire: {opponent_choice_name}"
            choices_surface = small_font.render(choices_text, True, self.BLACK)
            screen.blit(choices_surface, (610, y_pos + 25))
            
            # Résultat
            result_text = "Victoire!" if round_data['result'] > 0 else \
                         "Défaite!" if round_data['result'] < 0 else \
                         "Égalité!"
            result_surface = small_font.render(result_text, True, self.BLACK)
            screen.blit(result_surface, (610, y_pos + 45))
    
    def _render_final_result(self, screen, font, small_font, game_summary):
        """Affiche le résultat final."""
        # Fond
        result_rect = pygame.Rect(200, 400, 600, 200)
        color = self.LIGHT_GREEN if game_summary['total_score'] > 0 else \
                self.LIGHT_RED if game_summary['total_score'] < 0 else \
                self.LIGHT_YELLOW
        
        pygame.draw.rect(screen, color, result_rect)
        pygame.draw.rect(screen, self.BLACK, result_rect, 3)
        
        # Titre
        title = font.render("RÉSULTAT FINAL", True, self.BLACK)
        title_rect = title.get_rect(center=(result_rect.centerx, result_rect.y + 40))
        screen.blit(title, title_rect)
        
        # Score
        score_text = f"Score total: {game_summary['total_score']}"
        score_surface = small_font.render(score_text, True, self.BLACK)
        score_rect = score_surface.get_rect(center=(result_rect.centerx, result_rect.y + 80))
        screen.blit(score_surface, score_rect)
        
        # Résultat
        result_text = game_summary['game_result']
        result_surface = small_font.render(result_text, True, self.BLACK)
        result_rect_pos = result_surface.get_rect(center=(result_rect.centerx, result_rect.y + 110))
        screen.blit(result_surface, result_rect_pos)
        
        # Détails
        details = [
            f"Round 1: {game_summary['round1_score']}",
            f"Round 2: {game_summary['round2_score']}"
        ]
        
        for i, detail in enumerate(details):
            detail_surface = small_font.render(detail, True, self.BLACK)
            detail_rect = detail_surface.get_rect(center=(result_rect.centerx, result_rect.y + 140 + i * 25))
            screen.blit(detail_surface, detail_rect)
    
    def _render_q_values(self, screen, small_font):
        """Affiche les Q-values de l'agent."""
        if self.state is None:
            return
            
        try:
            q_values = self.agent.algorithm.q_function[self.state]
            best_action = np.argmax(q_values)
            
            # Panneau Q-values
            panel_x = 20
            panel_y = 350
            panel_width = 200
            panel_height = 120
            
            # Fond
            pygame.draw.rect(screen, self.LIGHT_BLUE, (panel_x, panel_y, panel_width, panel_height))
            pygame.draw.rect(screen, self.BLACK, (panel_x, panel_y, panel_width, panel_height), 2)
            
            # Titre
            title = small_font.render("Q-Values (Agent):", True, self.BLACK)
            screen.blit(title, (panel_x + 5, panel_y + 5))
            
            # Q-values
            for action, q_val in enumerate(q_values):
                color = self.RED if action == best_action else self.BLACK
                action_name = self.env.ACTION_NAMES[action]
                text = f"{action_name}: {q_val:.2f}"
                if action == best_action:
                    text += " ★"
                
                rendered = small_font.render(text, True, color)
                screen.blit(rendered, (panel_x + 5, panel_y + 30 + action * 20))
                
        except Exception as e:
            pass  # Ignore les erreurs d'affichage Q-values


def main():
    """Fonction principale."""
    print("🎮 DÉMONSTRATION PYGAME - TWO ROUND ROCK PAPER SCISSORS")
    print("=" * 60)
    
    # 1. Créer et entraîner un agent
    print("1️⃣ Création et entraînement de l'agent...")
    env = TwoRoundRPSEnvironment()

    #Chargement du modèle sauvegardé et création agent entraîné
    """### Policy iteration
    algo_sauv = PolicyIteration(
        state_space_size=env.state_space_size,
        action_space_size=env.action_space_size)
    algo_sauv.load_model(filepath = "outputs/rps/modèle/policy_iteration.pkl")
    agent = Agent(algo_sauv, env, "Agent_pi_rps_entraine")"""

    """### Value iteration
    algo_sauv = ValueIteration(
        state_space_size=env.state_space_size,
        action_space_size=env.action_space_size)
    algo_sauv.load_model(filepath = "outputs/rps/modèle/value_iteration.pkl")
    agent = Agent(algo_sauv, env, "Agent_vi_rps_entraine")"""

    """### Monte Carlo ES
    algo_sauv = MonteCarloES(
        state_space_size=env.state_space_size,
        action_space_size=env.action_space_size)
    algo_sauv.load_model(filepath = "outputs/rps/modèle/monte_carlo_es.pkl")
    agent = Agent(algo_sauv, env, "Agent_mce_rps_entraine")"""

    """### On policy first visit monte carlo control
    algo_sauv = OnPolicyFirstVisitMCControl(
        state_space_size=env.state_space_size,
        action_space_size=env.action_space_size)
    algo_sauv.load_model(filepath = "outputs/rps/modèle/on_monte_carlo.pkl")
    agent = Agent(algo_sauv, env, "Agent_on_mc_rps_entraine")"""

    """### Off policy monte carlo control
    algo_sauv = OffPolicyMCControl(
        state_space_size=env.state_space_size,
        action_space_size=env.action_space_size)
    algo_sauv.load_model(filepath = "outputs/rps/modèle/off_monte_carlo.pkl")
    agent = Agent(algo_sauv, env, "Agent_off_mc_rps_entraine")"""

    """### Sarsa
    algo_sauv = SARSA(
        state_space_size=env.state_space_size,
        action_space_size=env.action_space_size)
    algo_sauv.load_model(filepath = "outputs/rps/modèle/sarsa.pkl")
    agent = Agent(algo_sauv, env, "Agent_sarsa_rps_entraine")"""

    """### Q learning
    algo_sauv = QLearning(
        state_space_size=env.state_space_size,
        action_space_size=env.action_space_size)
    algo_sauv.load_model(filepath = "outputs/rps/modèle/q_learning.pkl")
    agent = Agent(algo_sauv, env, "Agent_ql_rps_entraine")"""

    ### DynaQ
    algo_sauv = DynaQ(
            state_space_size=env.state_space_size,
            action_space_size=env.action_space_size)
    algo_sauv.load_model(filepath = "outputs/rps/modèle/dyna_q.pkl")
    agent = Agent(algo_sauv, env, "Agent_dynaq_rps_entraine")
    
    print(f"✅ Agent entraîné: {agent.agent_name}")
    
    # 2. Créer la démonstration
    demo = TwoRoundRPSPygameDemo(env, agent)
    
    # 3. Menu interactif
    while True:
        print(f"\n🎮 MENU DÉMONSTRATION")
        print("1. Démonstration agent entraîné (PyGame)")
        print("2. Mode humain (PyGame)")
        #print("3. Mode humain (Console)")
        #print("4. Comparer humain vs agent")
        #print("5. Statistiques de l'agent")
        print("3. Quitter")
        
        choice = input("Votre choix (1-3): ").strip()
        
        if choice == "1":
            demo.demonstrate_agent(num_episodes=5)
            
        elif choice == "2":
            result = demo.human_mode()
            print(f"\n📊 Résultat de votre partie:")
            print(f"Score total: {result['total_reward']}")
            print(f"Nombre de steps: {result['steps']}")

        elif choice == "3":
            print("👋 Au revoir!")
            break
            
        else:
            print("❌ Choix invalide")
            
        """elif choice == "3":
            human = HumanPlayer(env, "Joueur")
            result = human.play_episode(interface_mode="console")
            
        elif choice == "4":
            # Comparaison simple
            print("\n🏆 COMPARAISON HUMAIN VS AGENT")
            print("Jouez d'abord...")
            human = HumanPlayer(env, "Joueur")
            human_result = human.play_episode(interface_mode="console")
            
            print("\nÉvaluation de l'agent...")
            agent_results = agent.evaluate_performance(num_episodes=100, verbose=False)
            
            print(f"\n📊 RÉSULTATS:")
            print(f"Humain: {human_result['total_reward']:.2f} points")
            print(f"Agent (moyenne): {agent_results['avg_reward']:.2f} points")
            print(f"Agent (taux de victoire): {agent_results.get('win_rate', 0):.1%}")
            
            if human_result['total_reward'] > agent_results['avg_reward']:
                print("🏆 L'humain gagne!")
            elif human_result['total_reward'] == agent_results['avg_reward']:
                print("🤝 Égalité!")
            else:
                print("🤖 L'agent gagne!")
        
        elif choice == "5":
            print("\n📈 STATISTIQUES DE L'AGENT")
            stats = agent.evaluate_performance(num_episodes=500, verbose=False)
            print(f"Performance moyenne: {stats['avg_reward']:.3f}")
            print(f"Écart-type: {stats['std_reward']:.3f}")
            print(f"Récompense min: {stats['min_reward']:.3f}")
            print(f"Récompense max: {stats['max_reward']:.3f}")"""
            
        


if __name__ == "__main__":
    if not PYGAME_AVAILABLE:
        print("❌ PyGame requis. Installez avec: pip install pygame")
        sys.exit(1)
    
    main()