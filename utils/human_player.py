"""
Mode Agent Humain - Interface pour qu'un humain puisse jouer sur les environnements.

Cette classe permet à un utilisateur humain de jouer manuellement sur les environnements
pour tester les règles et comprendre la dynamique du jeu.

Support PyGame pour interface graphique interactive.

Placement: utils/human_player.py
"""

import sys
from typing import Dict, List, Any, Optional
import time

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️ PyGame non disponible. Mode console uniquement.")

from src.rl_environments.base_environment import BaseEnvironment


class HumanPlayer:
    """
    Agent humain pour jouer manuellement sur les environnements RL.
    
    Cette classe fournit une interface pour qu'un utilisateur puisse
    interagir manuellement avec les environnements d'apprentissage par renforcement.
    
    Supporte les interfaces console et PyGame.
    """
    
    def __init__(self, environment: BaseEnvironment, player_name: str = "Human"):
        """
        Initialise le joueur humain.
        
        Args:
            environment (BaseEnvironment): Environnement de jeu
            player_name (str): Nom du joueur humain
        """
        self.environment = environment
        self.player_name = player_name
        self.game_history = []
        self.total_games = 0
        self.total_wins = 0
        
        # Configuration PyGame
        self.pygame_initialized = False
        self.screen = None
        self.clock = None
        self.font = None
        
        print(f"🎮 Joueur humain initialisé: {player_name}")
        print(f"Environnement: {environment.env_name}")
    
    def play_episode(self, 
                    interface_mode: str = "pygame",
                    show_instructions: bool = True,
                    show_rewards: bool = True,
                    show_q_values: bool = False,
                    trained_algorithm=None) -> Dict[str, Any]:
        """
        Lance un épisode de jeu humain.
        
        Args:
            interface_mode (str): "pygame" ou "console"
            show_instructions (bool): Afficher les instructions
            show_rewards (bool): Afficher les récompenses
            show_q_values (bool): Afficher les Q-values d'un algorithme entraîné
            trained_algorithm: Algorithme entraîné pour comparaison (optionnel)
            
        Returns:
            Dict[str, Any]: Résultats de l'épisode
        """
        print(f"\n🎯 NOUVEAU JEU - {self.player_name}")
        print(f"Environnement: {self.environment.env_name}")
        print(f"Interface: {interface_mode}")
        print("=" * 50)
        
        if show_instructions:
            self._show_instructions()
        
        # Choix de l'interface
        if interface_mode == "pygame" and PYGAME_AVAILABLE:
            return self._play_episode_pygame(show_rewards, show_q_values, trained_algorithm)
        else:
            return self._play_episode_console(show_rewards, show_q_values, trained_algorithm)
    
    def _play_episode_console(self, show_rewards: bool, show_q_values: bool, trained_algorithm) -> Dict[str, Any]:
        """Lance un épisode en mode console."""
        # Initialisation de l'épisode
        state = self.environment.reset()
        episode_data = {
            "player": self.player_name,
            "environment": self.environment.env_name,
            "interface": "console",
            "steps": [],
            "total_reward": 0.0,
            "success": False,
            "num_steps": 0
        }
        
        print(f"\n🏁 ÉTAT INITIAL:")
        print(f"Position: {state} - {self.environment.get_state_description(state)}")
        self.environment.render(mode='console')
        
        if show_q_values and trained_algorithm and hasattr(trained_algorithm, 'q_function'):
            self._show_algorithm_suggestion(state, trained_algorithm)
        
        # Boucle de jeu
        max_steps = getattr(self.environment, 'max_steps', 1000)
        for step in range(max_steps):
            # Obtenir l'action du joueur humain
            try:
                action = self._get_human_action_console(state)
            except KeyboardInterrupt:
                print("\n❌ Jeu interrompu par le joueur")
                break
            except EOFError:
                print("\n❌ Fin d'entrée détectée")
                break
            
            if action is None:
                print("❌ Action invalide ou abandon du jeu")
                break
            
            # Exécuter l'action
            prev_state = state
            next_state, reward, done, info = self.environment.step(action)
            
            # Enregistrer l'étape
            step_info = {
                "step": step + 1,
                "state": prev_state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "info": info
            }
            episode_data["steps"].append(step_info)
            episode_data["total_reward"] += reward
            episode_data["num_steps"] += 1
            
            # Affichage du résultat
            print(f"\n⏯️ ÉTAPE {step + 1}:")
            print(f"Action choisie: {action}")
            print(f"Nouvel état: {next_state} - {self.environment.get_state_description(next_state)}")
            
            if show_rewards:
                print(f"Récompense: {reward:.2f} | Total: {episode_data['total_reward']:.2f}")
            
            if info:
                for key, value in info.items():
                    if key in ['target_reached', 'boundary_hit', 'max_steps_reached']:
                        status = "✅" if value else "❌"
                        print(f"{key}: {status}")
            
            self.environment.render(mode='console')
            
            if show_q_values and trained_algorithm and hasattr(trained_algorithm, 'q_function'):
                self._show_algorithm_suggestion(next_state, trained_algorithm)
            
            state = next_state
            
            if done:
                if info.get("target_reached", False):
                    print("🎉 FÉLICITATIONS! Vous avez atteint la cible!")
                    episode_data["success"] = True
                    self.total_wins += 1
                elif info.get("max_steps_reached", False):
                    print("⏰ Temps écoulé! Nombre maximum d'étapes atteint.")
                else:
                    print("🏁 Épisode terminé.")
                break
        
        return self._finalize_episode(episode_data)
    
    def _play_episode_pygame(self, show_rewards: bool, show_q_values: bool, trained_algorithm) -> Dict[str, Any]:
        """Lance un épisode en mode PyGame."""
        if not self._initialize_pygame():
            print("❌ Impossible d'initialiser PyGame, utilisation du mode console")
            return self._play_episode_console(show_rewards, show_q_values, trained_algorithm)
        
        # Initialisation de l'épisode
        state = self.environment.reset()
        episode_data = {
            "player": self.player_name,
            "environment": self.environment.env_name,
            "interface": "pygame",
            "steps": [],
            "total_reward": 0.0,
            "success": False,
            "num_steps": 0
        }
        
        running = True
        max_steps = getattr(self.environment, 'max_steps', 1000)
        
        while running and episode_data["num_steps"] < max_steps:
            # Gestion des événements PyGame
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    action = self._get_action_from_key(event.key, state)
            
            if action is not None:
                # Exécuter l'action
                prev_state = state
                next_state, reward, done, info = self.environment.step(action)
                
                # Enregistrer l'étape
                step_info = {
                    "step": episode_data["num_steps"] + 1,
                    "state": prev_state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                    "info": info
                }
                episode_data["steps"].append(step_info)
                episode_data["total_reward"] += reward
                episode_data["num_steps"] += 1
                
                state = next_state
                
                if done:
                    if info.get("target_reached", False):
                        episode_data["success"] = True
                        self.total_wins += 1
                    running = False
            
            # Rendu de l'interface PyGame
            self._render_pygame(state, episode_data, show_rewards, show_q_values, trained_algorithm)
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
        self.pygame_initialized = False
        
        return self._finalize_episode(episode_data)
    
    def _initialize_pygame(self) -> bool:
        """Initialise PyGame pour l'interface graphique."""
        if not PYGAME_AVAILABLE:
            return False
        
        try:
            pygame.init()
            
            # Configuration de la fenêtre selon l'environnement
            if "lineworld" in self.environment.env_name.lower():
                width = 800
                height = 200
            elif "gridworld" in self.environment.env_name.lower():
                width = 600
                height = 600
            else:
                width = 800
                height = 600
            
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(f"Human Player - {self.environment.env_name}")
            
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
            
            self.pygame_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation PyGame: {e}")
            return False
    
    def _get_action_from_key(self, key: int, state: int) -> Optional[int]:
        """
        Convertit une touche PyGame en action pour l'environnement.
        
        Args:
            key (int): Code de la touche pressée
            state (int): État actuel (pour validation)
            
        Returns:
            Optional[int]: Action correspondante ou None
        """
        if "lineworld" in self.environment.env_name.lower():
            # LineWorld: Flèches gauche/droite ou A/D
            if key == pygame.K_LEFT or key == pygame.K_a:
                return 0  # Gauche
            elif key == pygame.K_RIGHT or key == pygame.K_d:
                return 1  # Droite
                
        elif "gridworld" in self.environment.env_name.lower():
            # GridWorld: Flèches directionnelles ou WASD
            if key == pygame.K_UP or key == pygame.K_w:
                return 0  # Haut
            elif key == pygame.K_RIGHT or key == pygame.K_d:
                return 1  # Droite
            elif key == pygame.K_DOWN or key == pygame.K_s:
                return 2  # Bas
            elif key == pygame.K_LEFT or key == pygame.K_a:
                return 3  # Gauche
                
        elif "monty" in self.environment.env_name.lower():
            # Monty Hall: Chiffres 1-3 ou 1-5
            if pygame.K_1 <= key <= pygame.K_9:
                door_number = key - pygame.K_1
                if door_number < self.environment.action_space_size:
                    return door_number
                    
        elif "rock" in self.environment.env_name.lower():
            # Rock Paper Scissors: R/P/S
            if key == pygame.K_r:
                return 0  # Rock
            elif key == pygame.K_p:
                return 1  # Paper
            elif key == pygame.K_s:
                return 2  # Scissors
        
        return None
    
    def _render_pygame(self, state: int, episode_data: Dict[str, Any], 
                      show_rewards: bool, show_q_values: bool, trained_algorithm):
        """Rendu de l'interface PyGame."""
        # Couleurs
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        BLUE = (0, 100, 200)
        GREEN = (0, 200, 0)
        RED = (200, 0, 0)
        GRAY = (128, 128, 128)
        LIGHT_BLUE = (173, 216, 230)
        
        # Effacer l'écran
        self.screen.fill(WHITE)
        
        # Rendu spécifique selon l'environnement
        if "lineworld" in self.environment.env_name.lower():
            self._render_lineworld_pygame(state, episode_data, show_rewards, WHITE, BLACK, BLUE, GREEN, RED)
        elif "gridworld" in self.environment.env_name.lower():
            self._render_gridworld_pygame(state, episode_data, show_rewards, WHITE, BLACK, BLUE, GREEN, RED, GRAY)
        else:
            self._render_generic_pygame(state, episode_data, show_rewards, WHITE, BLACK, BLUE)
        
        # Affichage des informations générales
        self._render_info_panel(episode_data, show_rewards, show_q_values, trained_algorithm, state, BLACK)
        
        # Affichage des Q-values si demandé
        if show_q_values and trained_algorithm and hasattr(trained_algorithm, 'q_function'):
            self._render_q_values_pygame(state, trained_algorithm, BLACK, LIGHT_BLUE)
        
        # Mise à jour de l'affichage
        pygame.display.flip()
    
    def _render_lineworld_pygame(self, state: int, episode_data: Dict[str, Any], 
                               show_rewards: bool, WHITE, BLACK, BLUE, GREEN, RED):
        """Rendu spécifique pour LineWorld."""
        # Dimensions
        cell_width = 120
        cell_height = 80
        start_x = 50
        start_y = 80
        
        # Dessiner les positions
        for pos in range(self.environment.state_space_size):
            x = start_x + pos * cell_width
            y = start_y
            
            # Couleur de la cellule
            if pos == state:
                color = BLUE  # Position actuelle
            elif hasattr(self.environment, 'target_position') and pos == self.environment.target_position:
                color = GREEN  # Cible
            else:
                color = WHITE  # Position vide
            
            # Dessiner la cellule
            pygame.draw.rect(self.screen, color, (x, y, cell_width, cell_height))
            pygame.draw.rect(self.screen, BLACK, (x, y, cell_width, cell_height), 2)
            
            # Numéro de position
            text = self.font.render(str(pos), True, BLACK)
            text_rect = text.get_rect(center=(x + cell_width//2, y + cell_height//2))
            self.screen.blit(text, text_rect)
        
        # Instructions
        instructions = [
            "Utilisez les flèches ← → ou A/D pour vous déplacer",
            "Objectif: Atteindre la position verte"
        ]
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, BLACK)
            self.screen.blit(text, (50, 20 + i * 25))
    
    def _render_gridworld_pygame(self, state: int, episode_data: Dict[str, Any], 
                               show_rewards: bool, WHITE, BLACK, BLUE, GREEN, RED, GRAY):
        """Rendu spécifique pour GridWorld."""
        # TODO: Implémentation GridWorld PyGame
        # Pour l'instant, rendu basique
        text = self.font.render("GridWorld PyGame - En cours d'implémentation", True, BLACK)
        self.screen.blit(text, (50, 50))
        
        text2 = self.font.render(f"État actuel: {state}", True, BLACK)
        self.screen.blit(text2, (50, 100))
        
        instructions = [
            "Utilisez les flèches ↑↓←→ ou WASD pour vous déplacer"
        ]
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, BLACK)
            self.screen.blit(text, (50, 150 + i * 25))
    
    def _render_generic_pygame(self, state: int, episode_data: Dict[str, Any], 
                             show_rewards: bool, WHITE, BLACK, BLUE):
        """Rendu générique pour autres environnements."""
        text = self.font.render(f"Environnement: {self.environment.env_name}", True, BLACK)
        self.screen.blit(text, (50, 50))
        
        text2 = self.font.render(f"État actuel: {state}", True, BLACK)
        self.screen.blit(text2, (50, 100))
        
        text3 = self.font.render("Interface PyGame générique", True, BLACK)
        self.screen.blit(text3, (50, 150))
    
    def _render_info_panel(self, episode_data: Dict[str, Any], show_rewards: bool, 
                         show_q_values: bool, trained_algorithm, state: int, BLACK):
        """Affiche le panneau d'informations."""
        info_y = self.screen.get_height() - 120
        
        if show_rewards:
            reward_text = f"Récompense totale: {episode_data['total_reward']:.2f}"
            text = self.small_font.render(reward_text, True, BLACK)
            self.screen.blit(text, (50, info_y))
        
        steps_text = f"Étapes: {episode_data['num_steps']}"
        text = self.small_font.render(steps_text, True, BLACK)
        self.screen.blit(text, (50, info_y + 25))
        
        # Instructions de sortie
        exit_text = "Fermez la fenêtre ou ESC pour quitter"
        text = self.small_font.render(exit_text, True, BLACK)
        self.screen.blit(text, (50, info_y + 50))
    
    def _render_q_values_pygame(self, state: int, trained_algorithm, BLACK, LIGHT_BLUE):
        """Affiche les Q-values dans l'interface PyGame."""
        try:
            q_values = trained_algorithm.q_function[state]
            best_action = np.argmax(q_values)
            
            # Panneau Q-values
            panel_x = self.screen.get_width() - 200
            panel_y = 50
            
            # Fond du panneau
            pygame.draw.rect(self.screen, LIGHT_BLUE, (panel_x, panel_y, 150, 100))
            pygame.draw.rect(self.screen, BLACK, (panel_x, panel_y, 150, 100), 2)
            
            # Titre
            title = self.small_font.render("Q-Values:", True, BLACK)
            self.screen.blit(title, (panel_x + 5, panel_y + 5))
            
            # Q-values
            for action, q_val in enumerate(q_values):
                color = BLACK if action != best_action else (200, 0, 0)  # Rouge pour meilleure action
                text = f"A{action}: {q_val:.2f}"
                if action == best_action:
                    text += " ★"
                
                rendered = self.small_font.render(text, True, color)
                self.screen.blit(rendered, (panel_x + 5, panel_y + 25 + action * 20))
                
        except Exception as e:
            print(f"Erreur affichage Q-values: {e}")
    
    def _get_human_action_console(self, state: int) -> Optional[int]:
        """
        Obtient l'action du joueur humain via l'interface console.
        
        Args:
            state (int): État actuel
            
        Returns:
            Optional[int]: Action choisie par le joueur (None si abandon)
        """
        valid_actions = self.environment.valid_actions
        
        print(f"\n🎮 À VOTRE TOUR!")
        print(f"État actuel: {state}")
        print(f"Actions disponibles: {valid_actions}")
        
        # Instructions spécifiques selon le type d'environnement
        if "lineworld" in self.environment.env_name.lower():
            print("0 = Gauche ← | 1 = Droite →")
        elif "gridworld" in self.environment.env_name.lower():
            print("0 = Haut ↑ | 1 = Droite → | 2 = Bas ↓ | 3 = Gauche ←")
        elif "monty" in self.environment.env_name.lower():
            print("Choisissez le numéro de la porte")
        elif "rock" in self.environment.env_name.lower():
            print("0 = Pierre | 1 = Feuille | 2 = Ciseaux")
        
        while True:
            try:
                user_input = input("Votre action (ou 'q' pour quitter): ").strip().lower()
                
                if user_input in ['q', 'quit', 'exit']:
                    return None
                
                action = int(user_input)
                
                if action in valid_actions:
                    return action
                else:
                    print(f"❌ Action invalide! Choisissez parmi: {valid_actions}")
                    
            except ValueError:
                print("❌ Veuillez entrer un nombre valide ou 'q' pour quitter")
            except (KeyboardInterrupt, EOFError):
                return None
    
    def _show_instructions(self):
        """Affiche les instructions de jeu selon l'environnement."""
        print(f"\n📖 INSTRUCTIONS DE JEU:")
        print(f"Environnement: {self.environment.env_name}")
        
        if "lineworld" in self.environment.env_name.lower():
            print("🎯 Objectif: Atteindre la position cible")
            print("🕹️ Actions: 0 (Gauche) ou 1 (Droite)")
            print("🎮 PyGame: Flèches ←→ ou touches A/D")
            print("⚠️ Attention: Sortir des limites donne une pénalité")
            
        elif "gridworld" in self.environment.env_name.lower():
            print("🎯 Objectif: Atteindre la case cible")
            print("🕹️ Actions: 0 (Haut), 1 (Droite), 2 (Bas), 3 (Gauche)")
            print("🎮 PyGame: Flèches ↑↓←→ ou touches WASD")
            print("⚠️ Attention: Évitez les obstacles")
            
        elif "monty" in self.environment.env_name.lower():
            print("🎯 Objectif: Choisir la porte gagnante")
            print("🕹️ Actions: Numéro de la porte à choisir")
            print("🎮 PyGame: Touches numériques 1, 2, 3...")
            print("💡 Astuce: Changez ou gardez votre choix selon la stratégie")
            
        elif "rock" in self.environment.env_name.lower():
            print("🎯 Objectif: Gagner le maximum de rounds")
            print("🕹️ Actions: 0 (Pierre), 1 (Feuille), 2 (Ciseaux)")
            print("🎮 PyGame: Touches R (Rock), P (Paper), S (Scissors)")
            print("💡 Astuce: Analysez le comportement de l'adversaire")
        
        print("💾 Console: Tapez 'q' à tout moment pour quitter")
        print("💾 PyGame: Fermez la fenêtre pour quitter")
        print("-" * 50)
    
    def _show_algorithm_suggestion(self, state: int, algorithm):
        """
        Affiche la suggestion d'un algorithme entraîné.
        
        Args:
            state (int): État actuel
            algorithm: Algorithme entraîné
        """
        try:
            if hasattr(algorithm, 'q_function'):
                q_values = algorithm.q_function[state]
                best_action = algorithm.select_action(state, training=False)
                
                print(f"\n🤖 SUGGESTION DE L'ALGORITHME {algorithm.algo_name}:")
                print(f"Q-values: {q_values}")
                print(f"Action recommandée: {best_action} (Q={q_values[best_action]:.3f})")
                
        except Exception as e:
            print(f"⚠️ Erreur lors de l'affichage des suggestions: {e}")
    
    def _finalize_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Finalise et affiche le résumé de l'épisode."""
        # Résumé de l'épisode
        self.total_games += 1
        self.game_history.append(episode_data)
        
        print(f"\n📊 RÉSUMÉ DE L'ÉPISODE:")
        print(f"Interface utilisée: {episode_data.get('interface', 'inconnue')}")
        print(f"Nombre d'étapes: {episode_data['num_steps']}")
        print(f"Récompense totale: {episode_data['total_reward']:.2f}")
        print(f"Succès: {'✅' if episode_data['success'] else '❌'}")
        print(f"Statistiques globales: {self.total_wins}/{self.total_games} victoires ({self.total_wins/self.total_games*100:.1f}%)")
        
        return episode_data
    
    def play_multiple_episodes(self, 
                             num_episodes: int = 5,
                             interface_mode: str = "pygame",
                             show_instructions: bool = True,
                             **kwargs) -> List[Dict[str, Any]]:
        """
        Lance plusieurs épisodes consécutifs.
        
        Args:
            num_episodes (int): Nombre d'épisodes à jouer
            interface_mode (str): Interface à utiliser
            show_instructions (bool): Afficher les instructions au début
            **kwargs: Arguments pour play_episode
            
        Returns:
            List[Dict[str, Any]]: Résultats de tous les épisodes
        """
        print(f"\n🎮 SESSION DE JEU MULTIPLE")
        print(f"Joueur: {self.player_name}")
        print(f"Nombre d'épisodes: {num_episodes}")
        print(f"Interface: {interface_mode}")
        print("=" * 50)
        
        episodes_results = []
        
        for episode_num in range(num_episodes):
            print(f"\n🎯 ÉPISODE {episode_num + 1}/{num_episodes}")
            
            # Afficher les instructions seulement au premier épisode
            show_instr = show_instructions and episode_num == 0
            
            try:
                episode_result = self.play_episode(
                    interface_mode=interface_mode,
                    show_instructions=show_instr,
                    **kwargs
                )
                episodes_results.append(episode_result)
                
                # Demander si continuer (sauf pour PyGame qui gère ça différemment)
                if episode_num < num_episodes - 1 and interface_mode == "console":
                    continue_input = input(f"\nContinuer vers l'épisode {episode_num + 2}? (y/n): ").strip().lower()
                    if continue_input in ['n', 'no', 'non']:
                        print("Session de jeu interrompue par le joueur")
                        break
                        
            except KeyboardInterrupt:
                print("\nSession de jeu interrompue")
                break
        
        # Affichage du résumé final
        self._show_session_summary(episodes_results)
        
        return episodes_results
    
    def _show_session_summary(self, episodes_results: List[Dict[str, Any]]):
        """
        Affiche un résumé de la session de jeu.
        
        Args:
            episodes_results (List): Résultats des épisodes
        """
        if not episodes_results:
            return
        
        print(f"\n📊 RÉSUMÉ DE LA SESSION")
        print("=" * 50)
        
        total_episodes = len(episodes_results)
        total_wins = sum(1 for episode in episodes_results if episode['success'])
        total_reward = sum(episode['total_reward'] for episode in episodes_results)
        avg_reward = total_reward / total_episodes
        avg_steps = sum(episode['num_steps'] for episode in episodes_results) / total_episodes
        
        interfaces_used = set(ep.get('interface', 'unknown') for ep in episodes_results)
        
        print(f"Épisodes joués: {total_episodes}")
        print(f"Interfaces utilisées: {', '.join(interfaces_used)}")
        print(f"Victoires: {total_wins}/{total_episodes} ({total_wins/total_episodes*100:.1f}%)")
        print(f"Récompense moyenne: {avg_reward:.2f}")
        print(f"Nombre d'étapes moyen: {avg_steps:.1f}")
        
        # Détail par épisode
        print(f"\n📋 DÉTAIL PAR ÉPISODE:")
        print(f"{'Épisode':<8}{'Interface':<10}{'Succès':<8}{'Récompense':<12}{'Étapes':<8}")
        print("-" * 46)
        
        for i, episode in enumerate(episodes_results, 1):
            success_icon = "✅" if episode['success'] else "❌"
            interface = episode.get('interface', 'unknown')[:9]
            print(f"{i:<8}{interface:<10}{success_icon:<8}{episode['total_reward']:<12.2f}{episode['num_steps']:<8}")
    
    def compare_with_algorithm(self, 
                             algorithm,
                             num_episodes: int = 5,
                             interface_mode: str = "pygame",
                             show_comparison: bool = True) -> Dict[str, Any]:
        """
        Compare les performances humaines avec un algorithme.
        
        Args:
            algorithm: Algorithme entraîné à comparer
            num_episodes (int): Nombre d'épisodes pour la comparaison
            interface_mode (str): Interface à utiliser
            show_comparison (bool): Afficher la comparaison détaillée
            
        Returns:
            Dict[str, Any]: Résultats de la comparaison
        """
        print(f"\n🏆 DÉFI HUMAIN VS ALGORITHME")
        print(f"Humain: {self.player_name}")
        print(f"Algorithme: {algorithm.algo_name}")
        print(f"Interface: {interface_mode}")
        print("=" * 50)
        
        # Jeu humain
        print(f"\n👤 TOUR DU JOUEUR HUMAIN")
        human_results = self.play_multiple_episodes(
            num_episodes=num_episodes,
            interface_mode=interface_mode,
            show_instructions=True,
            show_q_values=True,
            trained_algorithm=algorithm
        )
        
        # Évaluation de l'algorithme
        print(f"\n🤖 TOUR DE L'ALGORITHME")
        from utils.agent import Agent
        
        agent = Agent(algorithm, self.environment)
        algo_results = agent.evaluate_performance(
            num_episodes=num_episodes,
            verbose=True
        )
        
        # Comparaison
        if show_comparison:
            self._show_human_vs_algo_comparison(human_results, algo_results)
        
        # Résultats de la comparaison
        human_avg_reward = sum(ep['total_reward'] for ep in human_results) / len(human_results)
        human_win_rate = sum(1 for ep in human_results if ep['success']) / len(human_results)
        
        comparison = {
            "human_performance": {
                "avg_reward": human_avg_reward,
                "win_rate": human_win_rate,
                "episodes": len(human_results),
                "interface_used": interface_mode
            },
            "algorithm_performance": {
                "avg_reward": algo_results["avg_reward"],
                "win_rate": algo_results["success_rate"],
                "episodes": algo_results["num_episodes"]
            },
            "winner": "Human" if human_avg_reward > algo_results["avg_reward"] else "Algorithm"
        }
        
        return comparison
    
    def _show_human_vs_algo_comparison(self, human_results: List[Dict], algo_results: Dict):
        """Affiche la comparaison détaillée humain vs algorithme."""
        print(f"\n🏆 RÉSULTATS DU DÉFI")
        print("=" * 50)
        
        human_avg_reward = sum(ep['total_reward'] for ep in human_results) / len(human_results)
        human_win_rate = sum(1 for ep in human_results if ep['success']) / len(human_results)
        
        print(f"{'Métrique':<20}{'Humain':<15}{'Algorithme':<15}{'Gagnant':<10}")
        print("-" * 60)
        
        # Récompense moyenne
        winner_reward = "Humain" if human_avg_reward > algo_results["avg_reward"] else "Algorithme"
        print(f"{'Récompense moy.':<20}{human_avg_reward:<15.2f}{algo_results['avg_reward']:<15.2f}{winner_reward:<10}")
        
        # Taux de succès
        winner_success = "Humain" if human_win_rate > algo_results["success_rate"] else "Algorithme"
        print(f"{'Taux de succès':<20}{human_win_rate:<15.2%}{algo_results['success_rate']:<15.2%}{winner_success:<10}")
        
        # Gagnant global
        overall_winner = "HUMAIN" if human_avg_reward > algo_results["avg_reward"] else "ALGORITHME"
        print(f"\n🏆 GAGNANT GLOBAL: {overall_winner}")
    
    def get_player_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques complètes du joueur.
        
        Returns:
            Dict[str, Any]: Statistiques du joueur
        """
        if not self.game_history:
            return {"message": "Aucune partie jouée"}
        
        rewards = [game['total_reward'] for game in self.game_history]
        steps = [game['num_steps'] for game in self.game_history]
        interfaces = [game.get('interface', 'unknown') for game in self.game_history]
        
        return {
            "player_name": self.player_name,
            "environment": self.environment.env_name,
            "total_games": self.total_games,
            "total_wins": self.total_wins,
            "win_rate": self.total_wins / self.total_games if self.total_games > 0 else 0,
            "avg_reward": sum(rewards) / len(rewards),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "avg_steps": sum(steps) / len(steps),
            "interfaces_used": list(set(interfaces)),
            "game_history": self.game_history
        }
    
    def save_player_data(self, filepath: str) -> bool:
        """
        Sauvegarde les données du joueur.
        
        Args:
            filepath (str): Chemin de sauvegarde
            
        Returns:
            bool: True si succès
        """
        try:
            import json
            
            player_data = self.get_player_statistics()
            
            with open(f"{filepath}_human_player.json", 'w') as f:
                json.dump(player_data, f, indent=2)
            
            print(f"✅ Données du joueur sauvegardées: {filepath}_human_player.json")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return False
    
    def __str__(self) -> str:
        """Représentation textuelle du joueur."""
        return f"HumanPlayer({self.player_name}, {self.total_wins}/{self.total_games} victoires)"


# Fonction utilitaire pour lancer rapidement le mode humain
def quick_human_game(environment, player_name: str = "Player1", 
                    interface_mode: str = "pygame", num_episodes: int = 1):
    """
    Lance rapidement un jeu humain sur un environnement.
    
    Args:
        environment: Environnement de jeu
        player_name (str): Nom du joueur
        interface_mode (str): Interface à utiliser ("pygame" ou "console")
        num_episodes (int): Nombre d'épisodes
    """
    human = HumanPlayer(environment, player_name)
    
    if num_episodes == 1:
        return human.play_episode(interface_mode=interface_mode)
    else:
        return human.play_multiple_episodes(num_episodes, interface_mode=interface_mode)


if __name__ == "__main__":
    print("🎮 Mode Agent Humain - Prêt pour les tests!")
    print("Utilise PyGame pour interface graphique interactive")
    print("Utilise quick_human_game(environment) pour tester rapidement")