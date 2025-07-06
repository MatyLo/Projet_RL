import pygame
import sys
import time
from typing import Dict, Any, List, Tuple
import numpy as np

class PyGameVisualizer:
    """Visualiseur PyGame pour les environnements de reinforcement learning."""
    
    def __init__(self, width=800, height=600, fps=60):
        """
        Initialise le visualiseur PyGame.
        
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
        pygame.display.set_caption("Reinforcement Learning Visualizer")
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
        
        # Police
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        self.running = True
    
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

class LineWorldVisualizer(PyGameVisualizer):
    """Visualiseur spécifique pour LineWorld."""
    
    def __init__(self, env, width=800, height=400):
        super().__init__(width, height)
        self.env = env
        self.cell_width = width // (env.length + 2)  # +2 pour les marges
        self.cell_height = height // 3
        
    def render_state(self, state: int, value_function: List[float] = None, 
                    policy: Dict[int, int] = None, episode_info: Dict[str, Any] = None):
        """Affiche l'état actuel de LineWorld."""
        self.clear_screen()
        
        # Dessiner la ligne
        start_x = self.cell_width
        line_y = self.height // 2
        
        for i in range(self.env.length):
            x = start_x + i * self.cell_width
            rect = pygame.Rect(x, line_y - self.cell_height//2, 
                             self.cell_width - 10, self.cell_height)
            
            # Couleur de fond selon la valeur
            if value_function is not None and i < len(value_function):
                # Normaliser la valeur entre 0 et 1 pour la couleur
                normalized_value = (value_function[i] - min(value_function)) / (max(value_function) - min(value_function) + 1e-8)
                color_intensity = int(255 * normalized_value)
                bg_color = (255 - color_intensity, 255, 255 - color_intensity)
            else:
                bg_color = self.LIGHT_BLUE
            
            pygame.draw.rect(self.screen, bg_color, rect)
            pygame.draw.rect(self.screen, self.BLACK, rect, 2)
            
            # Afficher la valeur si disponible
            if value_function is not None and i < len(value_function):
                value_text = f"{value_function[i]:.2f}"
                self.draw_text(value_text, x + 5, line_y - 10, self.BLACK, self.small_font)
            
            # Afficher l'agent
            if i == state:
                agent_rect = pygame.Rect(x + 5, line_y - self.cell_height//2 + 5, 
                                       self.cell_width - 20, self.cell_height - 10)
                pygame.draw.rect(self.screen, self.RED, agent_rect)
                self.draw_text("A", x + self.cell_width//2 - 5, line_y - 5, self.WHITE, self.small_font)
            
            # Afficher l'objectif
            if i == self.env.goal_pos:
                goal_rect = pygame.Rect(x + 5, line_y - self.cell_height//2 + 5, 
                                      self.cell_width - 20, self.cell_height - 10)
                pygame.draw.rect(self.screen, self.GREEN, goal_rect)
                self.draw_text("G", x + self.cell_width//2 - 5, line_y - 5, self.WHITE, self.small_font)
            
            # Afficher la politique si disponible
            if policy is not None and i in policy:
                action = policy[i]
                action_text = "←" if action == 0 else "→"
                self.draw_text(action_text, x + self.cell_width//2 - 5, line_y + self.cell_height//2 + 5, 
                             self.BLUE, self.small_font)
        
        # Informations sur l'épisode
        if episode_info:
            info_y = 20
            for key, value in episode_info.items():
                self.draw_text(f"{key}: {value}", 10, info_y, self.BLACK, self.small_font)
                info_y += 25
        
        self.update_display()

class GridWorldVisualizer(PyGameVisualizer):
    """Visualiseur spécifique pour GridWorld."""
    
    def __init__(self, env, width=600, height=600):
        super().__init__(width, height)
        self.env = env
        self.cell_width = width // env.width
        self.cell_height = height // env.height
        
    def render_state(self, state: int, value_function: List[float] = None, 
                    policy: Dict[int, int] = None, episode_info: Dict[str, Any] = None):
        """Affiche l'état actuel de GridWorld."""
        self.clear_screen()
        
        # Dessiner la grille
        for row in range(self.env.height):
            for col in range(self.env.width):
                x = col * self.cell_width
                y = row * self.cell_height
                rect = pygame.Rect(x, y, self.cell_width, self.cell_height)
                
                current_state = row * self.env.width + col
                
                # Couleur de fond selon la valeur
                if value_function is not None and current_state < len(value_function):
                    normalized_value = (value_function[current_state] - min(value_function)) / (max(value_function) - min(value_function) + 1e-8)
                    color_intensity = int(255 * normalized_value)
                    bg_color = (255 - color_intensity, 255, 255 - color_intensity)
                else:
                    bg_color = self.LIGHT_BLUE
                
                pygame.draw.rect(self.screen, bg_color, rect)
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)
                
                # Afficher la valeur si disponible
                if value_function is not None and current_state < len(value_function):
                    value_text = f"{value_function[current_state]:.1f}"
                    self.draw_text(value_text, x + 5, y + 5, self.BLACK, self.small_font)
                
                # Afficher l'agent
                if current_state == state:
                    agent_rect = pygame.Rect(x + 5, y + 5, self.cell_width - 10, self.cell_height - 10)
                    pygame.draw.rect(self.screen, self.RED, agent_rect)
                    self.draw_text("A", x + self.cell_width//2 - 5, y + self.cell_height//2 - 5, 
                                 self.WHITE, self.small_font)
                
                # Afficher l'objectif
                if current_state == self.env.T[0]:
                    goal_rect = pygame.Rect(x + 5, y + 5, self.cell_width - 10, self.cell_height - 10)
                    pygame.draw.rect(self.screen, self.GREEN, goal_rect)
                    self.draw_text("G", x + self.cell_width//2 - 5, y + self.cell_height//2 - 5, 
                                 self.WHITE, self.small_font)
                
                # Afficher la politique si disponible
                if policy is not None and current_state in policy:
                    action = policy[current_state]
                    action_symbols = ["↑", "↓", "←", "→"]
                    if action < len(action_symbols):
                        self.draw_text(action_symbols[action], x + self.cell_width//2 - 5, 
                                     y + self.cell_height - 20, self.BLUE, self.small_font)
        
        # Informations sur l'épisode
        if episode_info:
            info_y = 10
            for key, value in episode_info.items():
                self.draw_text(f"{key}: {value}", 10, info_y, self.BLACK, self.small_font)
                info_y += 20
        
        self.update_display() 