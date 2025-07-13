import pygame
import sys
import time
from typing import Dict, Any, List, Tuple
import numpy as np

class MontyHallVisualizer:
    """Visualiseur PyGame pour l'environnement Monty Hall."""
    
    def __init__(self, width=800, height=600, fps=60):
        """
        Initialise le visualiseur PyGame pour Monty Hall.
        
        Args:
            width: Largeur de la fenêtre
            height: Hauteur de la fenêtre
            fps: Images par seconde
        """
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
        
        # Police
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 48)
        
        self.running = True
        
        # Dimensions des portes
        self.door_width = 120
        self.door_height = 200
        self.door_spacing = 50
        
    def handle_events(self):
        """Gère les événements PyGame."""
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
        """Dessine du texte à l'écran."""
        if color is None:
            color = self.BLACK
        if font is None:
            font = self.font
        
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))
    
    def clear_screen(self):
        """Efface l'écran."""
        self.screen.fill(self.WHITE)
    
    def update_display(self):
        """Met à jour l'affichage."""
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def quit(self):
        """Ferme PyGame."""
        pygame.quit()
        sys.exit()
    
    def render_monty_hall_state(self, env, episode_info: Dict[str, Any] = None, hover_door: int = None, hover_btn: str = None):
        """Affiche l'état actuel du problème de Monty Hall avec une interface moderne et stylée."""
        self.clear_screen()

        # Dégradé de fond
        for i in range(self.height):
            color = (
                int(230 - i * 80 / self.height),
                int(245 - i * 60 / self.height),
                int(255 - i * 40 / self.height)
            )
            pygame.draw.line(self.screen, color, (0, i), (self.width, i))

        # Titre avec bandeau
        pygame.draw.rect(self.screen, (33, 150, 243), (0, 0, self.width, 70), border_radius=0)
        self.draw_text("PROBLÈME DE MONTY HALL", self.width//2 - 200, 15, self.WHITE, self.large_font)

        # Position des portes
        door_start_x = (self.width - (3 * self.door_width + 2 * self.door_spacing)) // 2
        door_y = 150
        shadow_offset = 10
        radius = 25

        # Dessiner les 3 portes stylisées
        for i in range(3):
            door_x = door_start_x + i * (self.door_width + self.door_spacing)
            # Ombre portée
            shadow_rect = pygame.Rect(door_x + shadow_offset, door_y + shadow_offset, self.door_width, self.door_height)
            pygame.draw.rect(self.screen, (180, 180, 180), shadow_rect, border_radius=radius)
            # Couleur de la porte
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
            # Porte avec coins arrondis
            door_rect = pygame.Rect(door_x, door_y, self.door_width, self.door_height)
            pygame.draw.rect(self.screen, door_color, door_rect, border_radius=radius)
            pygame.draw.rect(self.screen, self.BLACK, door_rect, 4, border_radius=radius)
            # Numéro de la porte
            self.draw_text(f"Porte {i}", door_x + 25, door_y + 15, self.PURPLE, self.font)
            # Contenu de la porte
            if env.state == 1 and i == env.eliminated_door:
                self.draw_text("CHÈVRE", door_x + 20, door_y + 80, self.BLACK, self.font)
            elif env.state == 2:
                if i == env.winning_door:
                    self.draw_text("VOITURE", door_x + 20, door_y + 80, self.BLACK, self.font)
                elif i == env.eliminated_door:
                    self.draw_text("CHÈVRE", door_x + 20, door_y + 80, self.BLACK, self.font)

        # Instructions dynamiques
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

        # Boutons modernes pour garder/changer
        if env.state == 1:
            btn_y = info_y + 60
            btn_width = 180
            btn_height = 60
            garder_rect = pygame.Rect(self.width//2 - btn_width - 30, btn_y, btn_width, btn_height)
            changer_rect = pygame.Rect(self.width//2 + 30, btn_y, btn_width, btn_height)
            # Effet hover
            garder_color = (66, 165, 245) if hover_btn == "garder" else (33, 150, 243)
            changer_color = (76, 175, 80) if hover_btn == "changer" else (56, 142, 60)
            pygame.draw.rect(self.screen, garder_color, garder_rect, border_radius=30)
            pygame.draw.rect(self.screen, changer_color, changer_rect, border_radius=30)
            self.draw_text("Garder", garder_rect.x + 40, garder_rect.y + 15, self.WHITE, self.font)
            self.draw_text("Changer", changer_rect.x + 35, changer_rect.y + 15, self.WHITE, self.font)
            # Icônes (optionnel, simple cercle)
            pygame.draw.circle(self.screen, self.WHITE, (garder_rect.x + 25, garder_rect.y + 30), 12)
            pygame.draw.circle(self.screen, self.WHITE, (changer_rect.x + btn_width - 25, changer_rect.y + 30), 12)

        # Score/infos épisode
        if episode_info:
            episode_y = self.height - 80
            pygame.draw.rect(self.screen, (245, 245, 245), (0, episode_y, self.width, 80), border_radius=0)
            for idx, (key, value) in enumerate(episode_info.items()):
                self.draw_text(f"{key}: {value}", 30 + idx*220, episode_y + 25, self.PURPLE, self.small_font)

        self.update_display()

    def get_human_action(self, env) -> int:
        """
        Attend un clic de l'utilisateur sur une porte (état 0) ou sur un bouton (état 1), et retourne l'action choisie.
        Retourne :
            int: l'action choisie (0, 1 ou 2 pour les portes, 0=garder, 1=changer pour la décision finale)
        """
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
        """Affiche l'état actuel du problème Monty Hall 2 (5 portes) avec la vraie logique du paradoxe."""
        self.clear_screen()

        # Dégradé de fond
        for i in range(self.height):
            color = (
                int(230 - i * 80 / self.height),
                int(245 - i * 60 / self.height),
                int(255 - i * 40 / self.height)
            )
            pygame.draw.line(self.screen, color, (0, i), (self.width, i))

        # Titre
        pygame.draw.rect(self.screen, (33, 150, 243), (0, 0, self.width, 70), border_radius=0)
        self.draw_text("MONTY HALL 2 (5 portes)", self.width//2 - 200, 15, self.WHITE, self.large_font)

        # Position des portes
        n_doors = 5
        door_start_x = (self.width - (n_doors * self.door_width + (n_doors-1) * self.door_spacing)) // 2
        door_y = 150
        shadow_offset = 10
        radius = 25

        # Dessiner les 5 portes stylisées
        for i in range(n_doors):
            door_x = door_start_x + i * (self.door_width + self.door_spacing)
            # Ombre portée
            shadow_rect = pygame.Rect(door_x + shadow_offset, door_y + shadow_offset, self.door_width, self.door_height)
            pygame.draw.rect(self.screen, (180, 180, 180), shadow_rect, border_radius=radius)
            # Couleur de la porte
            if i not in env.remaining_doors:
                door_color = self.BROWN
            elif env.state == 3 and i == env.choice_at_3:
                door_color = (66, 165, 245) if hover_door == i else self.BLUE
            else:
                door_color = (255, 255, 255) if hover_door == i else self.LIGHT_BLUE
            # Porte avec coins arrondis
            door_rect = pygame.Rect(door_x, door_y, self.door_width, self.door_height)
            pygame.draw.rect(self.screen, door_color, door_rect, border_radius=radius)
            pygame.draw.rect(self.screen, self.BLACK, door_rect, 4, border_radius=radius)
            # Numéro de la porte
            self.draw_text(f"Porte {i}", door_x + 25, door_y + 15, self.PURPLE, self.font)
            # Contenu de la porte
            if i not in env.remaining_doors:
                self.draw_text("ÉLIMINÉE", door_x + 10, door_y + 80, self.BLACK, self.font)
            elif env.state == 4 and i == env.winning_door:
                self.draw_text("VOITURE", door_x + 20, door_y + 80, self.BLACK, self.font)

        # Instructions dynamiques
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

        # Boutons modernes pour garder/changer uniquement à l'étape finale
        if env.state == 3:
            btn_y = info_y + 60
            btn_width = 180
            btn_height = 60
            garder_rect = pygame.Rect(self.width//2 - btn_width - 30, btn_y, btn_width, btn_height)
            changer_rect = pygame.Rect(self.width//2 + 30, btn_y, btn_width, btn_height)
            # Effet hover
            garder_color = (66, 165, 245) if hover_btn == "garder" else (33, 150, 243)
            changer_color = (76, 175, 80) if hover_btn == "changer" else (56, 142, 60)
            pygame.draw.rect(self.screen, garder_color, garder_rect, border_radius=30)
            pygame.draw.rect(self.screen, changer_color, changer_rect, border_radius=30)
            self.draw_text("Garder", garder_rect.x + 40, garder_rect.y + 15, self.WHITE, self.font)
            self.draw_text("Changer", changer_rect.x + 35, changer_rect.y + 15, self.WHITE, self.font)
            pygame.draw.circle(self.screen, self.WHITE, (garder_rect.x + 25, garder_rect.y + 30), 12)
            pygame.draw.circle(self.screen, self.WHITE, (changer_rect.x + btn_width - 25, changer_rect.y + 30), 12)

        # Score/infos épisode
        if episode_info:
            episode_y = self.height - 80
            pygame.draw.rect(self.screen, (245, 245, 245), (0, episode_y, self.width, 80), border_radius=0)
            for idx, (key, value) in enumerate(episode_info.items()):
                self.draw_text(f"{key}: {value}", 30 + idx*220, episode_y + 25, self.PURPLE, self.small_font)

        self.update_display()

    def get_human_action_mh2(self, env) -> int:
        """
        Attend un clic de l'utilisateur sur une porte (à chaque étape tant qu'il reste plus de 2 portes),
        ou sur un bouton (garder/changer) à l'étape finale.
        Retourne :
            int: l'action choisie (0-4 pour les portes restantes, 0=garder, 1=changer pour la décision finale)
        """
        import pygame
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