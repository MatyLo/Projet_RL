import pygame
import sys
import time
import importlib
import inspect
from typing import Dict, Any, List, Tuple
import numpy as np

class MontyHallVisualizer:
    """Visualiseur PyGame pour l'environnement Monty Hall."""
    
    def __init__(self, width=800, height=600, fps=60):
        pygame.init()
        self.width = width
        self.height = height
        self.fps = fps
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Monty Hall Problem Visualizer")
        self.clock = pygame.time.Clock()
        # Couleurs
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_BLUE = (173, 216, 230)
        self.BROWN = (139, 69, 19)
        self.PURPLE = (128, 0, 128)
        self.ORANGE = (255, 165, 0)
        self.DARK_GREEN = (0, 100, 0)
        # Police
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 48)
        self.running = True
        # Dimensions des portes
        self.door_width = 120
        self.door_height = 200
        self.door_spacing = 50

    def get_available_algorithms(self) -> List[str]:
        """Détecte automatiquement les algorithmes disponibles."""
        try:
            from rl_algorithms import QLearning, SARSA, ValueIteration
            algorithms = []
            algorithms.append('Q-Learning')
            algorithms.append('SARSA')
            algorithms.append('Value Iteration')
            return algorithms
        except ImportError:
            # Fallback si l'import échoue
            return ['Q-Learning', 'SARSA', 'Value Iteration']

    def render_main_menu(self) -> Tuple[str, str]:
        """Affiche le menu principal et retourne le mode et l'algorithme choisis."""
        mode = None
        algorithm = None
        
        while mode is None and self.running:
            self.clear_screen()
            
            # Fond dégradé
            for i in range(self.height):
                color = (
                    int(230 - i * 80 / self.height),
                    int(245 - i * 60 / self.height),
                    int(255 - i * 40 / self.height)
                )
                pygame.draw.line(self.screen, color, (0, i), (self.width, i))
            
            # Titre
            pygame.draw.rect(self.screen, (33, 150, 243), (0, 0, self.width, 100), border_radius=0)
            self.draw_text("MONTY HALL PROBLEM", self.width//2 - 200, 20, self.WHITE, self.large_font)
            self.draw_text("Choisissez votre mode de jeu", self.width//2 - 180, 60, self.WHITE, self.font)
            
            # Boutons de mode
            btn_width = 200
            btn_height = 80
            btn_y = 200
            
            # Bouton Humain
            human_rect = pygame.Rect(self.width//2 - btn_width - 50, btn_y, btn_width, btn_height)
            human_color = (76, 175, 80) if pygame.mouse.get_pos()[1] >= btn_y and pygame.mouse.get_pos()[1] <= btn_y + btn_height and pygame.mouse.get_pos()[0] >= human_rect.x and pygame.mouse.get_pos()[0] <= human_rect.x + btn_width else (56, 142, 60)
            pygame.draw.rect(self.screen, human_color, human_rect, border_radius=20)
            self.draw_text("MODE HUMAIN", human_rect.x + 20, human_rect.y + 25, self.WHITE, self.font)
            self.draw_text("Jouez vous-même", human_rect.x + 30, human_rect.y + 50, self.WHITE, self.small_font)
            
            # Bouton Agent
            agent_rect = pygame.Rect(self.width//2 + 50, btn_y, btn_width, btn_height)
            agent_color = (255, 152, 0) if pygame.mouse.get_pos()[1] >= btn_y and pygame.mouse.get_pos()[1] <= btn_y + btn_height and pygame.mouse.get_pos()[0] >= agent_rect.x and pygame.mouse.get_pos()[0] <= agent_rect.x + btn_width else (245, 124, 0)
            pygame.draw.rect(self.screen, agent_color, agent_rect, border_radius=20)
            self.draw_text("MODE AGENT", agent_rect.x + 30, agent_rect.y + 25, self.WHITE, self.font)
            self.draw_text("IA joue pour vous", agent_rect.x + 25, agent_rect.y + 50, self.WHITE, self.small_font)
            
            # Instructions
            self.draw_text("Cliquez sur un mode pour commencer", self.width//2 - 200, 350, self.BLACK, self.font)
            
            self.update_display()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None, None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        return None, None
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_x, mouse_y = event.pos
                    if human_rect.collidepoint(mouse_x, mouse_y):
                        mode = "human"
                        algorithm = None
                    elif agent_rect.collidepoint(mouse_x, mouse_y):
                        mode = "agent"
                        # Afficher le menu de sélection d'algorithme
                        algorithm = self.render_algorithm_menu()
            
            self.clock.tick(self.fps)
        
        return mode, algorithm

    def render_algorithm_menu(self) -> str:
        """Affiche le menu de sélection d'algorithme et retourne l'algorithme choisi."""
        algorithms = self.get_available_algorithms()
        selected_algorithm = None
        
        while selected_algorithm is None and self.running:
            self.clear_screen()
            
            # Fond dégradé
            for i in range(self.height):
                color = (
                    int(230 - i * 80 / self.height),
                    int(245 - i * 60 / self.height),
                    int(255 - i * 40 / self.height)
                )
                pygame.draw.line(self.screen, color, (0, i), (self.width, i))
            
            # Titre
            pygame.draw.rect(self.screen, (33, 150, 243), (0, 0, self.width, 100), border_radius=0)
            self.draw_text("SÉLECTION DE L'ALGORITHME", self.width//2 - 250, 20, self.WHITE, self.large_font)
            self.draw_text("Choisissez l'algorithme d'apprentissage", self.width//2 - 200, 60, self.WHITE, self.font)
            
            # Boutons d'algorithmes
            btn_width = 250
            btn_height = 70
            btn_y = 150
            btn_spacing = 20
            
            mouse_x, mouse_y = pygame.mouse.get_pos()
            
            for i, algorithm in enumerate(algorithms):
                btn_x = self.width//2 - btn_width//2
                btn_y_pos = btn_y + i * (btn_height + btn_spacing)
                btn_rect = pygame.Rect(btn_x, btn_y_pos, btn_width, btn_height)
                
                # Couleur selon le survol
                if btn_rect.collidepoint(mouse_x, mouse_y):
                    btn_color = (76, 175, 80)
                else:
                    btn_color = (56, 142, 60)
                
                pygame.draw.rect(self.screen, btn_color, btn_rect, border_radius=15)
                self.draw_text(algorithm, btn_x + 20, btn_y_pos + 20, self.WHITE, self.font)
            
            # Bouton retour
            back_rect = pygame.Rect(50, self.height - 80, 120, 50)
            back_color = (244, 67, 54) if back_rect.collidepoint(mouse_x, mouse_y) else (211, 47, 47)
            pygame.draw.rect(self.screen, back_color, back_rect, border_radius=10)
            self.draw_text("Retour", back_rect.x + 20, back_rect.y + 15, self.WHITE, self.font)
            
            # Instructions
            self.draw_text("Cliquez sur un algorithme pour le sélectionner", self.width//2 - 250, self.height - 120, self.BLACK, self.font)
            
            self.update_display()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_x, mouse_y = event.pos
                    
                    # Vérifier les boutons d'algorithmes
                    for i, algorithm in enumerate(algorithms):
                        btn_x = self.width//2 - btn_width//2
                        btn_y_pos = btn_y + i * (btn_height + btn_spacing)
                        btn_rect = pygame.Rect(btn_x, btn_y_pos, btn_width, btn_height)
                        if btn_rect.collidepoint(mouse_x, mouse_y):
                            selected_algorithm = algorithm
                            break
                    
                    # Vérifier le bouton retour
                    if back_rect.collidepoint(mouse_x, mouse_y):
                        return None
            
            self.clock.tick(self.fps)
        
        return selected_algorithm

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    return "pause"
                elif event.key == pygame.K_RETURN:
                    return "step"
        return None

    def draw_text(self, text: str, x: int, y: int, color: Tuple[int, int, int] = None, font=None):
        if color is None:
            color = self.BLACK
        if font is None:
            font = self.font
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def clear_screen(self):
        self.screen.fill(self.WHITE)

    def update_display(self):
        pygame.display.flip()
        self.clock.tick(self.fps)

    def quit(self):
        pygame.quit()
        sys.exit()

    def render_monty_hall_state(self, env, episode_info: Dict[str, Any] = None, hover_door: int = None, hover_btn: str = None):
        self.clear_screen()
        for i in range(self.height):
            color = (
                int(230 - i * 80 / self.height),
                int(245 - i * 60 / self.height),
                int(255 - i * 40 / self.height)
            )
            pygame.draw.line(self.screen, color, (0, i), (self.width, i))
        pygame.draw.rect(self.screen, (33, 150, 243), (0, 0, self.width, 70), border_radius=0)
        self.draw_text("PROBLÈME DE MONTY HALL", self.width//2 - 200, 15, self.WHITE, self.large_font)
        door_start_x = (self.width - (3 * self.door_width + 2 * self.door_spacing)) // 2
        door_y = 150
        shadow_offset = 10
        radius = 25
        for i in range(3):
            door_x = door_start_x + i * (self.door_width + self.door_spacing)
            shadow_rect = pygame.Rect(door_x + shadow_offset, door_y + shadow_offset, self.door_width, self.door_height)
            pygame.draw.rect(self.screen, (180, 180, 180), shadow_rect, border_radius=radius)
            if env.state == 0:
                door_color = (255, 255, 255) if hover_door == i else self.LIGHT_BLUE
            elif env.state == 1:
                if i == env.eliminated_door:
                    door_color = self.RED
                elif i == env.chosen_door:
                    door_color = (66, 165, 245) if hover_door == i else self.BLUE
                else:
                    door_color = (255, 255, 255) if hover_door == i else self.LIGHT_BLUE
            elif env.state == 2:
                if i == env.final_choice:
                    door_color = self.GREEN if env.final_choice == env.winning_door else self.RED
                elif i == env.winning_door:
                    door_color = self.GREEN
                elif i == env.eliminated_door:
                    door_color = self.BROWN
                else:
                    door_color = self.LIGHT_BLUE
            else:
                door_color = self.LIGHT_BLUE
            door_rect = pygame.Rect(door_x, door_y, self.door_width, self.door_height)
            pygame.draw.rect(self.screen, door_color, door_rect, border_radius=radius)
            pygame.draw.rect(self.screen, self.BLACK, door_rect, 4, border_radius=radius)
            self.draw_text(f"Porte {i}", door_x + 25, door_y + 15, self.PURPLE, self.font)
            if env.state == 1 and i == env.eliminated_door:
                self.draw_text("CHÈVRE", door_x + 20, door_y + 80, self.BLACK, self.font)
            elif env.state == 2:
                if i == env.winning_door:
                    self.draw_text("VOITURE", door_x + 20, door_y + 80, self.BLACK, self.font)
                elif i == env.eliminated_door:
                    self.draw_text("CHÈVRE", door_x + 20, door_y + 80, self.BLACK, self.font)
        info_y = 400
        pygame.draw.rect(self.screen, (255, 255, 255), (0, info_y - 10, self.width, 50), border_radius=0)
        if env.state == 0:
            self.draw_text("Cliquez sur une porte pour la sélectionner", self.width//2 - 200, info_y, (33, 150, 243), self.font)
        elif env.state == 1:
            self.draw_text(f"Porte {env.chosen_door} choisie. Cliquez sur un bouton pour décider.", self.width//2 - 250, info_y, (33, 150, 243), self.font)
        elif env.state == 2:
            if env.final_choice == env.winning_door:
                self.draw_text("🎉 GAGNÉ ! La voiture était derrière la porte choisie.", self.width//2 - 250, info_y, self.GREEN, self.font)
            else:
                self.draw_text("❌ PERDU. La voiture était ailleurs.", self.width//2 - 200, info_y, self.RED, self.font)
        if env.state == 1:
            btn_y = info_y + 60
            btn_width = 180
            btn_height = 60
            garder_rect = pygame.Rect(self.width//2 - btn_width - 30, btn_y, btn_width, btn_height)
            changer_rect = pygame.Rect(self.width//2 + 30, btn_y, btn_width, btn_height)
            garder_color = (66, 165, 245) if hover_btn == "garder" else (33, 150, 243)
            changer_color = (76, 175, 80) if hover_btn == "changer" else (56, 142, 60)
            pygame.draw.rect(self.screen, garder_color, garder_rect, border_radius=30)
            pygame.draw.rect(self.screen, changer_color, changer_rect, border_radius=30)
            self.draw_text("Garder", garder_rect.x + 40, garder_rect.y + 15, self.WHITE, self.font)
            self.draw_text("Changer", changer_rect.x + 35, changer_rect.y + 15, self.WHITE, self.font)
            pygame.draw.circle(self.screen, self.WHITE, (garder_rect.x + 25, garder_rect.y + 30), 12)
            pygame.draw.circle(self.screen, self.WHITE, (changer_rect.x + btn_width - 25, changer_rect.y + 30), 12)
        if episode_info:
            episode_y = self.height - 80
            pygame.draw.rect(self.screen, (245, 245, 245), (0, episode_y, self.width, 80), border_radius=0)
            for idx, (key, value) in enumerate(episode_info.items()):
                self.draw_text(f"{key}: {value}", 30 + idx*220, episode_y + 25, self.PURPLE, self.small_font)
        self.update_display()

    def get_human_action(self, env) -> int:
        action = None
        waiting = True
        hover_door = None
        hover_btn = None
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        return None
                elif event.type == pygame.MOUSEMOTION:
                    mouse_x, mouse_y = event.pos
                    hover_door = None
                    hover_btn = None
                    if env.state == 0:
                        door_start_x = (self.width - (3 * self.door_width + 2 * self.door_spacing)) // 2
                        door_y = 150
                        for i in range(3):
                            door_x = door_start_x + i * (self.door_width + self.door_spacing)
                            door_rect = pygame.Rect(door_x, door_y, self.door_width, self.door_height)
                            if door_rect.collidepoint(mouse_x, mouse_y):
                                hover_door = i
                    elif env.state == 1:
                        btn_y = 400 + 60
                        btn_width = 180
                        btn_height = 60
                        garder_rect = pygame.Rect(self.width//2 - btn_width - 30, btn_y, btn_width, btn_height)
                        changer_rect = pygame.Rect(self.width//2 + 30, btn_y, btn_width, btn_height)
                        if garder_rect.collidepoint(mouse_x, mouse_y):
                            hover_btn = "garder"
                        elif changer_rect.collidepoint(mouse_x, mouse_y):
                            hover_btn = "changer"
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_x, mouse_y = event.pos
                    if env.state == 0:
                        door_start_x = (self.width - (3 * self.door_width + 2 * self.door_spacing)) // 2
                        door_y = 150
                        for i in range(3):
                            door_x = door_start_x + i * (self.door_width + self.door_spacing)
                            door_rect = pygame.Rect(door_x, door_y, self.door_width, self.door_height)
                            if door_rect.collidepoint(mouse_x, mouse_y):
                                action = i
                                waiting = False
                                break
                    elif env.state == 1:
                        btn_y = 400 + 60
                        btn_width = 180
                        btn_height = 60
                        garder_rect = pygame.Rect(self.width//2 - btn_width - 30, btn_y, btn_width, btn_height)
                        changer_rect = pygame.Rect(self.width//2 + 30, btn_y, btn_width, btn_height)
                        if garder_rect.collidepoint(mouse_x, mouse_y):
                            action = 0
                            waiting = False
                        elif changer_rect.collidepoint(mouse_x, mouse_y):
                            action = 1
                            waiting = False
            self.render_monty_hall_state(env, hover_door=hover_door, hover_btn=hover_btn)
            self.clock.tick(self.fps)
        return action

    def render_monty_hall2_state(self, env, episode_info: Dict[str, Any] = None, hover_door: int = None, hover_btn: str = None):
        self.clear_screen()
        for i in range(self.height):
            color = (
                int(230 - i * 80 / self.height),
                int(245 - i * 60 / self.height),
                int(255 - i * 40 / self.height)
            )
            pygame.draw.line(self.screen, color, (0, i), (self.width, i))
        pygame.draw.rect(self.screen, (33, 150, 243), (0, 0, self.width, 70), border_radius=0)
        self.draw_text("MONTY HALL 2 (5 portes)", self.width//2 - 200, 15, self.WHITE, self.large_font)
        n_doors = 5
        door_start_x = (self.width - (n_doors * self.door_width + (n_doors-1) * self.door_spacing)) // 2
        door_y = 150
        shadow_offset = 10
        radius = 25
        for i in range(n_doors):
            door_x = door_start_x + i * (self.door_width + self.door_spacing)
            shadow_rect = pygame.Rect(door_x + shadow_offset, door_y + shadow_offset, self.door_width, self.door_height)
            pygame.draw.rect(self.screen, (180, 180, 180), shadow_rect, border_radius=radius)
            if i not in env.remaining_doors:
                door_color = self.BROWN
            elif env.state == 3 and i == env.choice_at_3:
                door_color = (66, 165, 245) if hover_door == i else self.BLUE
            else:
                door_color = (255, 255, 255) if hover_door == i else self.LIGHT_BLUE
            door_rect = pygame.Rect(door_x, door_y, self.door_width, self.door_height)
            pygame.draw.rect(self.screen, door_color, door_rect, border_radius=radius)
            pygame.draw.rect(self.screen, self.BLACK, door_rect, 4, border_radius=radius)
            self.draw_text(f"Porte {i}", door_x + 25, door_y + 15, self.PURPLE, self.font)
            if i not in env.remaining_doors:
                self.draw_text("ÉLIMINÉE", door_x + 10, door_y + 80, self.BLACK, self.font)
            elif env.state == 4 and i == env.winning_door:
                self.draw_text("VOITURE", door_x + 20, door_y + 80, self.BLACK, self.font)
        info_y = 400
        pygame.draw.rect(self.screen, (255, 255, 255), (0, info_y - 10, self.width, 50), border_radius=0)
        if env.state == 0:
            self.draw_text("Cliquez sur une porte pour la sélectionner", self.width//2 - 220, info_y, (33, 150, 243), self.font)
        elif env.state == 1:
            self.draw_text("Cliquez sur une porte restante pour la sélectionner", self.width//2 - 280, info_y, (33, 150, 243), self.font)
        elif env.state == 2:
            self.draw_text("Choisissez votre porte finale parmi les 3 restantes", self.width//2 - 280, info_y, (33, 150, 243), self.font)
        elif env.state == 3:
            self.draw_text("Choisissez : Garder ou Changer", self.width//2 - 180, info_y, (33, 150, 243), self.font)
        elif env.state == 4:
            if env.final_choice == env.winning_door:
                self.draw_text("🎉 GAGNÉ ! La voiture était derrière la porte choisie.", self.width//2 - 250, info_y, self.GREEN, self.font)
            else:
                self.draw_text("❌ PERDU. La voiture était ailleurs.", self.width//2 - 200, info_y, self.RED, self.font)
        if env.state == 3:
            btn_y = info_y + 60
            btn_width = 180
            btn_height = 60
            garder_rect = pygame.Rect(self.width//2 - btn_width - 30, btn_y, btn_width, btn_height)
            changer_rect = pygame.Rect(self.width//2 + 30, btn_y, btn_width, btn_height)
            garder_color = (66, 165, 245) if hover_btn == "garder" else (33, 150, 243)
            changer_color = (76, 175, 80) if hover_btn == "changer" else (56, 142, 60)
            pygame.draw.rect(self.screen, garder_color, garder_rect, border_radius=30)
            pygame.draw.rect(self.screen, changer_color, changer_rect, border_radius=30)
            self.draw_text("Garder", garder_rect.x + 40, garder_rect.y + 15, self.WHITE, self.font)
            self.draw_text("Changer", changer_rect.x + 35, changer_rect.y + 15, self.WHITE, self.font)
            pygame.draw.circle(self.screen, self.WHITE, (garder_rect.x + 25, garder_rect.y + 30), 12)
            pygame.draw.circle(self.screen, self.WHITE, (changer_rect.x + btn_width - 25, changer_rect.y + 30), 12)
        if episode_info:
            episode_y = self.height - 80
            pygame.draw.rect(self.screen, (245, 245, 245), (0, episode_y, self.width, 80), border_radius=0)
            for idx, (key, value) in enumerate(episode_info.items()):
                self.draw_text(f"{key}: {value}", 30 + idx*220, episode_y + 25, self.PURPLE, self.small_font)
        self.update_display()

    def get_human_action_mh2(self, env) -> int:
        action = None
        waiting = True
        hover_door = None
        hover_btn = None
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    hover_door = None
                    hover_btn = None
                    if env.state in [0, 1, 2]:
                        door_start_x = (self.width - (5 * self.door_width + 4 * self.door_spacing)) // 2
                        door_y = 150
                        for i in env.remaining_doors:
                            door_x = door_start_x + i * (self.door_width + self.door_spacing)
                            door_rect = pygame.Rect(door_x, door_y, self.door_width, self.door_height)
                            if door_rect.collidepoint(mx, my):
                                hover_door = i
                    elif env.state == 3:
                        btn_y = 400 + 60
                        btn_width = 180
                        btn_height = 60
                        garder_rect = pygame.Rect(self.width//2 - btn_width - 30, btn_y, btn_width, btn_height)
                        changer_rect = pygame.Rect(self.width//2 + 30, btn_y, btn_width, btn_height)
                        if garder_rect.collidepoint(mx, my):
                            hover_btn = "garder"
                        elif changer_rect.collidepoint(mx, my):
                            hover_btn = "changer"
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if env.state in [0, 1, 2]:
                        door_start_x = (self.width - (5 * self.door_width + 4 * self.door_spacing)) // 2
                        door_y = 150
                        for i in env.remaining_doors:
                            door_x = door_start_x + i * (self.door_width + self.door_spacing)
                            door_rect = pygame.Rect(door_x, door_y, self.door_width, self.door_height)
                            if door_rect.collidepoint(mx, my):
                                action = i
                                waiting = False
                                break
                    elif env.state == 3:
                        btn_y = 400 + 60
                        btn_width = 180
                        btn_height = 60
                        garder_rect = pygame.Rect(self.width//2 - btn_width - 30, btn_y, btn_width, btn_height)
                        changer_rect = pygame.Rect(self.width//2 + 30, btn_y, btn_width, btn_height)
                        if garder_rect.collidepoint(mx, my):
                            action = 0
                            waiting = False
                        elif changer_rect.collidepoint(mx, my):
                            action = 1
                            waiting = False
            self.render_monty_hall2_state(env, hover_door=hover_door, hover_btn=hover_btn)
            self.clock.tick(self.fps)
        return action 