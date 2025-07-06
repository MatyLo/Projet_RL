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
    
    def render_monty_hall_state(self, env, episode_info: Dict[str, Any] = None):
        """Affiche l'état actuel du problème de Monty Hall."""
        self.clear_screen()
        
        # Titre
        self.draw_text("PROBLÈME DE MONTY HALL", self.width//2 - 150, 20, self.BLACK, self.large_font)
        
        # Position des portes
        door_start_x = (self.width - (3 * self.door_width + 2 * self.door_spacing)) // 2
        door_y = 150
        
        # Dessiner les 3 portes
        for i in range(3):
            door_x = door_start_x + i * (self.door_width + self.door_spacing)
            
            # Couleur de la porte selon l'état
            door_color = self.GRAY
            
            if env.state == 0:
                # État initial - toutes les portes fermées
                door_color = self.LIGHT_BLUE
            elif env.state == 1:
                # Après révélation
                if i == env.opened_door:
                    door_color = self.RED  # Porte ouverte (chèvre)
                elif i == env.agent_first_choice:
                    door_color = self.BLUE  # Porte choisie
                else:
                    door_color = self.LIGHT_BLUE  # Porte restante
            elif env.state in [2, 3]:
                # État terminal
                if i == env.agent_final_choice:
                    if env.state == 2:
                        door_color = self.GREEN  # Gagné (voiture)
                    else:
                        door_color = self.RED  # Perdu (chèvre)
                elif i == env.winning_door:
                    door_color = self.GREEN  # Voiture (même si pas choisie)
                elif i == env.opened_door:
                    door_color = self.BROWN  # Chèvre révélée
            
            # Dessiner la porte
            door_rect = pygame.Rect(door_x, door_y, self.door_width, self.door_height)
            pygame.draw.rect(self.screen, door_color, door_rect)
            pygame.draw.rect(self.screen, self.BLACK, door_rect, 3)
            
            # Numéro de la porte
            door_label = f"Porte {i}"
            self.draw_text(door_label, door_x + 10, door_y + 10, self.BLACK, self.small_font)
            
            # Contenu de la porte
            if env.state == 1 and i == env.opened_door:
                self.draw_text("CHÈVRE", door_x + 20, door_y + 80, self.BLACK, self.font)
            elif env.state in [2, 3]:
                if i == env.winning_door:
                    self.draw_text("VOITURE", door_x + 20, door_y + 80, self.BLACK, self.font)
                elif i == env.opened_door:
                    self.draw_text("CHÈVRE", door_x + 20, door_y + 80, self.BLACK, self.font)
        
        # Informations sur l'état
        info_y = 400
        if env.state == 0:
            self.draw_text("Choisissez une porte (0, 1 ou 2)", 20, info_y, self.BLACK, self.font)
        elif env.state == 1:
            self.draw_text(f"Porte {env.agent_first_choice} choisie", 20, info_y, self.BLACK, self.font)
            self.draw_text(f"Porte {env.opened_door} révélée (chèvre)", 20, info_y + 30, self.BLACK, self.font)
            self.draw_text("Action: 0 = Garder, 1 = Changer", 20, info_y + 60, self.BLACK, self.font)
        elif env.state == 2:
            self.draw_text("🎉 GAGNÉ ! 🎉", 20, info_y, self.GREEN, self.large_font)
            self.draw_text(f"Porte {env.agent_final_choice} contenait la voiture!", 20, info_y + 40, self.BLACK, self.font)
        elif env.state == 3:
            self.draw_text("❌ PERDU ❌", 20, info_y, self.RED, self.large_font)
            self.draw_text(f"Porte {env.agent_final_choice} contenait une chèvre", 20, info_y + 40, self.BLACK, self.font)
            self.draw_text(f"La voiture était derrière la porte {env.winning_door}", 20, info_y + 70, self.BLACK, self.font)
        
        # Informations sur l'épisode
        if episode_info:
            episode_y = 500
            for key, value in episode_info.items():
                self.draw_text(f"{key}: {value}", 20, episode_y, self.BLACK, self.small_font)
                episode_y += 25
        
        # Légende
        legend_y = 550
        self.draw_text("Légende:", self.width - 200, legend_y, self.BLACK, self.small_font)
        pygame.draw.rect(self.screen, self.GREEN, (self.width - 200, legend_y + 20, 20, 20))
        self.draw_text("Voiture", self.width - 170, legend_y + 20, self.BLACK, self.small_font)
        pygame.draw.rect(self.screen, self.BROWN, (self.width - 200, legend_y + 45, 20, 20))
        self.draw_text("Chèvre", self.width - 170, legend_y + 45, self.BLACK, self.small_font)
        pygame.draw.rect(self.screen, self.BLUE, (self.width - 200, legend_y + 70, 20, 20))
        self.draw_text("Choisie", self.width - 170, legend_y + 70, self.BLACK, self.small_font)
        
        self.update_display() 