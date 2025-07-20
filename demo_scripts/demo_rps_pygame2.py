"""
Démonstration PyGame - Two Round Rock Paper Scissors - Multi-Algorithmes

Ce script permet de :
1. Voir différents agents entraînés jouer en PyGame
2. Jouer soi-même en mode humain
3. Comparer les performances entre algorithmes

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
from rl_algorithms.sarsa import SARSA
from rl_algorithms.dyna_q import DynaQ
from rl_algorithms.monte_carlo_es import MonteCarloES
from rl_algorithms.on_policy_first_visit_mc_control import OnPolicyFirstVisitMCControl
from rl_algorithms.off_policy_mc_control import OffPolicyMCControl
from rl_algorithms.policy_iteration import PolicyIteration
from rl_algorithms.value_iteration import ValueIteration
from agent import Agent
from human_player import HumanPlayer


class TwoRoundRPSPygameDemo:
    """Démonstration PyGame pour Two Round Rock Paper Scissors avec multi-algorithmes."""
    
    def __init__(self, environment):
        """
        Initialise la démonstration.
        
        Args:
            environment: Environnement TwoRoundRPSEnvironment
        """
        self.env = environment
        self.trained_agents = {}  # Dictionnaire pour stocker les agents entraînés
        
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
        
        # Configuration des algorithmes
        self.algorithms_config = {
            'policy_iteration': {
                'class': PolicyIteration,
                'name': 'Policy Iteration',
                'config': {'gamma': 0.9, 'theta': 1e-8},
                'episodes': 1000
            },
            'value_iteration': {
                'class': ValueIteration,
                'name': 'Value Iteration',
                'config': {'gamma': 0.9, 'theta': 1e-8},
                'episodes': 1000
            },
            'monte_carlo_es': {
                'class': MonteCarloES,
                'name': 'Monte Carlo ES',
                'config': {'gamma': 0.9, 'epsilon': 0.1},
                'episodes': 5000
            },
            'on_policy_mc': {
                'class': OnPolicyFirstVisitMonteCarlo,
                'name': 'On-Policy First Visit MC',
                'config': {'gamma': 0.9, 'epsilon': 0.1},
                'episodes': 5000
            },
            'off_policy_mc': {
                'class': OffPolicyMonteCarlo,
                'name': 'Off-Policy Monte Carlo',
                'config': {'gamma': 0.9, 'epsilon': 0.1},
                'episodes': 5000
            },
            'sarsa': {
                'class': SARSA,
                'name': 'SARSA',
                'config': {'learning_rate': 0.1, 'gamma': 0.9, 'epsilon': 0.1},
                'episodes': 2000
            },
            'q_learning': {
                'class': QLearning,
                'name': 'Q-Learning',
                'config': {'learning_rate': 0.1, 'gamma': 0.9, 'epsilon': 0.1},
                'episodes': 2000
            },
            'dyna_q': {
                'class': DynaQ,
                'name': 'Dyna-Q',
                'config': {'learning_rate': 0.1, 'gamma': 0.9, 'epsilon': 0.1, 'planning_steps': 10},
                'episodes': 1500
            }
        }

    def train_agent(self, algorithm_key):
        """
        Entraîne un agent avec l'algorithme spécifié.
        
        Args:
            algorithm_key: Clé de l'algorithme dans algorithms_config
            
        Returns:
            Agent entraîné
        """
        if algorithm_key in self.trained_agents:
            return self.trained_agents[algorithm_key]
        
        print(f"🔄 Entraînement de l'agent {self.algorithms_config[algorithm_key]['name']}...")
        
        config = self.algorithms_config[algorithm_key]
        algorithm_class = config['class']
        algorithm_config = config['config']
        num_episodes = config['episodes']
        
        # Créer l'algorithme
        algorithm = algorithm_class.from_config(algorithm_config, self.env)
        
        # Entraîner
        start_time = time.time()
        algorithm.train(self.env, num_episodes=num_episodes, verbose=False)
        training_time = time.time() - start_time
        
        # Créer l'agent
        agent = Agent(algorithm, self.env, f"{config['name']} Agent")
        self.trained_agents[algorithm_key] = agent
        
        print(f"✅ Agent {config['name']} entraîné en {training_time:.2f}s")
        
        return agent

    def demonstrate_agent(self, algorithm_key, num_episodes=3):
        """Démonstration de l'agent entraîné."""
        if not PYGAME_AVAILABLE:
            print("❌ PyGame non disponible")
            return
        
        agent = self.train_agent(algorithm_key)
        
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Démonstration {agent.agent_name} - Two Round RPS")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 48)
        small_font = pygame.font.Font(None, 32)
        
        print(f"🎬 DÉMONSTRATION AGENT: {agent.agent_name}")
        
        total_rewards = []
        
        for episode in range(num_episodes):
            print(f"\n--- Épisode {episode + 1}/{num_episodes} ---")
            
            # Reset environnement
            self.state = self.env.reset()
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
                        elif event.key == pygame.K_SPACE:
                            # Passer à l'épisode suivant
                            round_num = 3
                            break
                
                if not running:
                    break
                
                # Action de l'agent
                action = agent.algorithm.select_action(self.state, training=False)
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
                self._render_game(screen, font, small_font, info, action, agent)
                pygame.display.flip()
                
                self.state = next_state
                
                if done:
                    game_summary = info.get('game_summary', {})
                    print(f"Jeu terminé: {game_summary.get('game_result', 'Inconnu')}")
                    print(f"Score final: {game_summary.get('total_score', 0)}")
                    total_rewards.append(self.episode_reward)
                    
                    # Rendu final
                    self._render_game(screen, font, small_font, info, action, agent, game_finished=True)
                    pygame.display.flip()
                    
                    # Pause entre épisodes avec possibilité de passer
                    start_time = time.time()
                    while time.time() - start_time < 3:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                                break
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE or event.key == pygame.K_ESCAPE:
                                    running = False
                                    break
                        if not running:
                            break
                    break
                
                round_num += 1
                # Délai entre les rounds
                pygame.time.wait(1500)
                clock.tick(60)
            
            if not running:
                break
        
        pygame.quit()
        
        # Afficher les statistiques
        if total_rewards:
            print(f"\n📊 STATISTIQUES - {agent.agent_name}")
            print(f"Nombre d'épisodes: {len(total_rewards)}")
            print(f"Récompense moyenne: {np.mean(total_rewards):.3f}")
            print(f"Récompense totale: {sum(total_rewards):.1f}")
            if len(total_rewards) > 1:
                print(f"Écart-type: {np.std(total_rewards):.3f}")
        
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
    
    def _render_game(self, screen, font, small_font, info=None, last_action=None, agent=None, game_finished=False):
        """Rendu du jeu."""
        screen.fill(self.WHITE)
        
        # Titre avec nom de l'agent
        if agent:
            title = f"Two Round RPS - {agent.agent_name}"
        else:
            title = "Two Round Rock Paper Scissors - Mode Humain"
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
        if not agent:  # Mode humain
            self._render_action_buttons(screen, font, small_font)
        
        # Historique des rounds
        self._render_game_history(screen, small_font)
        
        # Instructions
        if not game_finished:
            if agent:
                instructions = [
                    "Démonstration automatique",
                    "ESPACE - Épisode suivant",
                    "ESC - Quitter"
                ]
            else:
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
        
        # Q-values/Policy si agent disponible
        if agent and hasattr(agent.algorithm, 'q_function'):
            self._render_q_values(screen, small_font, agent)
        elif agent and hasattr(agent.algorithm, 'policy'):
            self._render_policy(screen, small_font, agent)
    
    def _render_action_buttons(self, screen, font, small_font):
        """Affiche les boutons d'action pour le mode humain."""
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
            opponent_choice_name = round_data['opponent_choice']
            
            choices_text = f"Agent: {agent_choice_name} vs Adversaire: {opponent_choice_name}"
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
    
    def _render_q_values(self, screen, small_font, agent):
        """Affiche les Q-values de l'agent."""
        if self.state is None:
            return
            
        try:
            q_values = agent.algorithm.q_function[self.state]
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
            title = small_font.render("Q-Values:", True, self.BLACK)
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
            pass
    
    def _render_policy(self, screen, small_font, agent):
        """Affiche la politique de l'agent (pour les algorithmes basés sur les politiques)."""
        if self.state is None:
            return
            
        try:
            if hasattr(agent.algorithm, 'policy'):
                policy = agent.algorithm.policy[self.state]
                
                # Panneau Policy
                panel_x = 20
                panel_y = 350
                panel_width = 200
                panel_height = 120
                
                # Fond
                pygame.draw.rect(screen, self.LIGHT_GREEN, (panel_x, panel_y, panel_width, panel_height))
                pygame.draw.rect(screen, self.BLACK, (panel_x, panel_y, panel_width, panel_height), 2)
                
                # Titre
                title = small_font.render("Policy:", True, self.BLACK)
                screen.blit(title, (panel_x + 5, panel_y + 5))
                
                # Policy probabilities ou action déterministe
                if isinstance(policy, (int, np.integer)):
                    # Politique déterministe
                    action_name = self.env.ACTION_NAMES[policy]
                    text = f"Action: {action_name} ★"
                    rendered = small_font.render(text, True, self.RED)
                    screen.blit(rendered, (panel_x + 5, panel_y + 30))
                elif hasattr(policy, '__len__'):
                    # Politique stochastique
                    for action, prob in enumerate(policy):
                        if prob > 0:
                            action_name = self.env.ACTION_NAMES[action]
                            text = f"{action_name}: {prob:.2f}"
                            color = self.RED if prob == max(policy) else self.BLACK
                            rendered = small_font.render(text, True, color)
                            screen.blit(rendered, (panel_x + 5, panel_y + 30 + action * 20))
                
        except Exception as e:
            pass

    def compare_algorithms(self, algorithm_keys, num_episodes=100):
        """Compare les performances de plusieurs algorithmes."""
        print(f"\n🏆 COMPARAISON DES ALGORITHMES")
        print("=" * 60)
        
        results = {}
        
        for algo_key in algorithm_keys:
            print(f"\n🔄 Évaluation de {self.algorithms_config[algo_key]['name']}...")
            agent = self.train_agent(algo_key)
            
            # Évaluer les performances
            episode_rewards = []
            for episode in range(num_episodes):
                state = self.env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action = agent.algorithm.select_action(state, training=False)
                    state, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
            
            # Calculer les statistiques
            results[algo_key] = {
                'name': self.algorithms_config[algo_key]['name'],
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'win_rate': np.sum(np.array(episode_rewards) > 0) / len(episode_rewards),
                'episode_rewards': episode_rewards
            }
        
        # Afficher les résultats
        print(f"\n📊 RÉSULTATS DE COMPARAISON ({num_episodes} épisodes)")
        print("-" * 80)
        print(f"{'Algorithme':<25} {'Moyenne':<10} {'Écart-type':<12} {'Min':<8} {'Max':<8} {'Taux victoire'}")
        print("-" * 80)
        
        # Trier par performance moyenne
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
        
        for i, (algo_key, stats) in enumerate(sorted_results):
            rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            print(f"{rank_symbol:<3} {stats['name']:<22} "
                  f"{stats['mean_reward']:<10.3f} {stats['std_reward']:<12.3f} "
                  f"{stats['min_reward']:<8.1f} {stats['max_reward']:<8.1f} "
                  f"{stats['win_rate']:<12.1%}")
        
        return results


def main():
    """Fonction principale."""
    print("🎮 DÉMONSTRATION PYGAME - TWO ROUND ROCK PAPER SCISSORS")
    print("🤖 Multi-Algorithmes d'Apprentissage par Renforcement")
    print("=" * 80)
    
    # Créer l'environnement
    env = TwoRoundRPSEnvironment()
    demo = TwoRoundRPSPygameDemo(env)
    
    # Menu interactif
    while True:
        print(f"\n🎮 MENU DÉMONSTRATION")
        print("=" * 50)
        print("🤖 AGENTS ENTRAÎNÉS:")
        print("1. Policy Iteration")
        print("2. Value Iteration")
        print("3. Monte Carlo ES")
        print("4. On-Policy First Visit Monte Carlo")
        print("5. Off-Policy Monte Carlo")
        print("6. SARSA")
        print("7. Q-Learning")
        print("8. Dyna-Q")
        print("\n🎮 MODES DE JEU:")
        print("9. Mode Humain (PyGame)")
        print("10. Mode Humain (Console)")
        print("\n📊 COMPARAISONS:")
        print("11. Comparer tous les algorithmes")
        print("12. Comparer algorithmes sélectionnés")
        print("13. Évaluation rapide d'un algorithme")
        print("\n0. Quitter")
        
        choice = input("Votre choix (0-13): ").strip()
        
        # Algorithmes individuels
        algorithm_map = {
            "1": "policy_iteration",
            "2": "value_iteration", 
            "3": "monte_carlo_es",
            "4": "on_policy_mc",
            "5": "off_policy_mc",
            "6": "sarsa",
            "7": "q_learning",
            "8": "dyna_q"
        }
        
        if choice in algorithm_map:
            algo_key = algorithm_map[choice]
            algo_name = demo.algorithms_config[algo_key]['name']
            print(f"\n🎬 Lancement de la démonstration: {algo_name}")
            
            # Demander le nombre d'épisodes
            try:
                episodes = input(f"Nombre d'épisodes de démonstration (3-10, défaut=3): ").strip()
                episodes = int(episodes) if episodes else 3
                episodes = max(1, min(10, episodes))
            except ValueError:
                episodes = 3
            
            demo.demonstrate_agent(algo_key, num_episodes=episodes)
            
        elif choice == "9":
            print("\n🎮 Mode Humain (PyGame)")
            result = demo.human_mode()
            print(f"\n📊 Résultat de votre partie:")
            print(f"Score total: {result['total_reward']}")
            print(f"Nombre de steps: {result['steps']}")
            
        elif choice == "10":
            print("\n🎮 Mode Humain (Console)")
            human = HumanPlayer(env, "Joueur")
            result = human.play_episode(interface_mode="console")
            
        elif choice == "11":
            print("\n🏆 Comparaison de tous les algorithmes")
            try:
                episodes = input("Nombre d'épisodes d'évaluation (50-500, défaut=100): ").strip()
                episodes = int(episodes) if episodes else 100
                episodes = max(50, min(500, episodes))
            except ValueError:
                episodes = 100
            
            all_algorithms = list(demo.algorithms_config.keys())
            demo.compare_algorithms(all_algorithms, num_episodes=episodes)
            
        elif choice == "12":
            print("\n🏆 Comparaison d'algorithmes sélectionnés")
            print("Choisissez les algorithmes à comparer (ex: 1,3,7 pour Policy Iteration, Monte Carlo ES, Q-Learning):")
            
            for key, algo_key in algorithm_map.items():
                print(f"{key}. {demo.algorithms_config[algo_key]['name']}")
            
            selection = input("Votre sélection (numéros séparés par des virgules): ").strip()
            
            try:
                selected_nums = [s.strip() for s in selection.split(",")]
                selected_algorithms = []
                
                for num in selected_nums:
                    if num in algorithm_map:
                        selected_algorithms.append(algorithm_map[num])
                
                if len(selected_algorithms) < 2:
                    print("❌ Veuillez sélectionner au moins 2 algorithmes")
                    continue
                
                episodes = input("Nombre d'épisodes d'évaluation (50-500, défaut=100): ").strip()
                episodes = int(episodes) if episodes else 100
                episodes = max(50, min(500, episodes))
                
                demo.compare_algorithms(selected_algorithms, num_episodes=episodes)
                
            except Exception as e:
                print(f"❌ Erreur dans la sélection: {e}")
                
        elif choice == "13":
            print("\n📊 Évaluation rapide d'un algorithme")
            print("Choisissez l'algorithme à évaluer:")
            
            for key, algo_key in algorithm_map.items():
                print(f"{key}. {demo.algorithms_config[algo_key]['name']}")
            
            algo_choice = input("Votre choix: ").strip()
            
            if algo_choice in algorithm_map:
                algo_key = algorithm_map[algo_choice]
                
                try:
                    episodes = input("Nombre d'épisodes d'évaluation (100-1000, défaut=200): ").strip()
                    episodes = int(episodes) if episodes else 200
                    episodes = max(100, min(1000, episodes))
                except ValueError:
                    episodes = 200
                
                print(f"\n🔄 Évaluation de {demo.algorithms_config[algo_key]['name']}...")
                agent = demo.train_agent(algo_key)
                
                # Évaluer les performances
                episode_rewards = []
                wins = 0
                losses = 0
                draws = 0
                
                for episode in range(episodes):
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    
                    while not done:
                        action = agent.algorithm.select_action(state, training=False)
                        state, reward, done, info = env.step(action)
                        episode_reward += reward
                    
                    episode_rewards.append(episode_reward)
                    
                    if episode_reward > 0:
                        wins += 1
                    elif episode_reward < 0:
                        losses += 1
                    else:
                        draws += 1
                
                # Afficher les statistiques détaillées
                print(f"\n📊 ÉVALUATION DÉTAILLÉE - {demo.algorithms_config[algo_key]['name']}")
                print("-" * 60)
                print(f"Épisodes évalués: {episodes}")
                print(f"Récompense moyenne: {np.mean(episode_rewards):.4f}")
                print(f"Écart-type: {np.std(episode_rewards):.4f}")
                print(f"Récompense min: {np.min(episode_rewards):.1f}")
                print(f"Récompense max: {np.max(episode_rewards):.1f}")
                print(f"Médiane: {np.median(episode_rewards):.3f}")
                print(f"\nRésultats de jeu:")
                print(f"🏆 Victoires: {wins} ({wins/episodes:.1%})")
                print(f"❌ Défaites: {losses} ({losses/episodes:.1%})")
                print(f"🤝 Égalités: {draws} ({draws/episodes:.1%})")
                
                # Distribution des récompenses
                reward_counts = {}
                for reward in episode_rewards:
                    reward_counts[reward] = reward_counts.get(reward, 0) + 1
                
                print(f"\nDistribution des scores:")
                for reward in sorted(reward_counts.keys()):
                    count = reward_counts[reward]
                    percentage = count / episodes * 100
                    print(f"Score {reward:+.1f}: {count:4d} épisodes ({percentage:5.1f}%)")
            else:
                print("❌ Choix invalide")
                
        elif choice == "0":
            print("👋 Au revoir!")
            break
            
        else:
            print("❌ Choix invalide")
            
        # Pause avant de revenir au menu
        input("\nAppuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    if not PYGAME_AVAILABLE:
        print("❌ PyGame requis pour les démonstrations graphiques.")
        print("Installez avec: pip install pygame")
        print("Le mode console reste disponible.")
    
    main()