#!/usr/bin/env python3
"""
Démonstration interactive autonome du problème de Monty Hall avec PyGame.
Version autonome qui hérite directement de BaseEnvironment.
"""

import sys
import os
import time
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional

# Ajouter le répertoire parent au path pour importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.base_environment import BaseEnvironment
from environments.visualization.monty_hall_visualizer import MontyHallVisualizer
from algorithms.dynamic_programming.value_iteration import ValueIteration

class MontyHallInteractive(BaseEnvironment):
    """
    Environnement Monty Hall autonome pour démonstration interactive.
    Version simplifiée qui hérite directement de BaseEnvironment.
    """
    
    def __init__(self):
        super().__init__("MontyHallInteractive")
        self.num_doors = 3
        self.winning_door = None
        self.chosen_door = None
        self.eliminated_door = None
        self.final_choice = None
        
        # États: 0=choix initial, 1=choix final, 2=terminé
        self.state = 0
        
        # Actions: 0,1,2 pour les portes, 0=rester, 1=changer pour l'état 1
        self.valid_actions_list = [0, 1, 2]
        
        # Attributs pour compatibilité avec le visualiseur
        self.opened_door = None
        self.doors = [True] * 3  # True = fermée, False = ouverte
        # Attributs pour compatibilité visualiseur
        self.agent_first_choice = None
        self.agent_final_choice = None
        
    @property
    def state_space_size(self) -> int:
        return 3  # 3 états possibles
    
    @property
    def action_space_size(self) -> int:
        return 3  # 3 actions possibles (portes 0,1,2)
    
    @property
    def valid_actions(self) -> List[int]:
        if self.state == 0:
            return [0, 1, 2]  # Choix de porte
        elif self.state == 1:
            return [0, 1]  # Rester (0) ou changer (1)
        else:
            return []
    
    def reset(self) -> int:
        """Remet l'environnement à l'état initial."""
        self.winning_door = random.randint(0, 2)
        self.chosen_door = None
        self.eliminated_door = None
        self.final_choice = None
        self.state = 0
        self.opened_door = None
        self.doors = [True] * 3
        self.agent_first_choice = None
        self.agent_final_choice = None
        self._reset_episode_stats()
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Exécute une action dans l'environnement."""
        if not self.is_valid_action(action):
            raise ValueError(f"Action invalide {action} dans l'état {self.state}")
        
        reward = 0.0
        done = False
        info = {}
        
        if self.state == 0:
            # Choix initial de porte
            self.chosen_door = action
            self.agent_first_choice = action
            # Le présentateur élimine une porte non gagnante
            available_doors = [i for i in range(3) if i != self.chosen_door and i != self.winning_door]
            self.eliminated_door = random.choice(available_doors)
            
            # Mettre à jour les attributs pour le visualiseur
            self.opened_door = self.eliminated_door
            self.doors[self.eliminated_door] = False
            
            self.state = 1
            info['eliminated_door'] = self.eliminated_door
            
        elif self.state == 1:
            # Choix final: rester ou changer
            if action == 0:  # Rester
                self.final_choice = self.chosen_door
            else:  # Changer
                remaining_doors = [i for i in range(3) if i != self.chosen_door and i != self.eliminated_door]
                self.final_choice = remaining_doors[0]
            self.agent_final_choice = self.final_choice
            # Vérifier si gagné
            won = (self.final_choice == self.winning_door)
            reward = 1.0 if won else 0.0
            done = True
            
            self.state = 2
            info['result'] = 'win' if won else 'lose'
            info['winning_door'] = self.winning_door
            info['final_choice'] = self.final_choice
        
        self._update_episode_stats(action, reward, self.state, done)
        return self.state, reward, done, info
    
    def render(self, mode: str = 'console') -> Optional[Any]:
        """Affiche l'état actuel de l'environnement."""
        if mode == 'console':
            print(f"État: {self.state}")
            print(f"Porte choisie: {self.chosen_door}")
            print(f"Porte éliminée: {self.eliminated_door}")
            print(f"Choix final: {self.final_choice}")
            if self.state == 2:
                print(f"Porte gagnante: {self.winning_door}")
        return None
    
    def get_state_description(self, state: int) -> str:
        """Retourne une description textuelle d'un état."""
        descriptions = {
            0: "Choix initial de porte",
            1: "Choix final: rester ou changer",
            2: "Partie terminée"
        }
        return descriptions.get(state, "État inconnu")
    
    def get_game_state(self):
        """Retourne l'état du jeu pour le visualiseur."""
        return {
            'state': self.state,
            'chosen_door': self.chosen_door,
            'eliminated_door': self.eliminated_door,
            'winning_door': self.winning_door if self.state == 2 else None,
            'final_choice': self.final_choice
        }
    
    # Méthodes de compatibilité avec les algorithmes RL
    def get_state_space(self):
        """Retourne l'espace des états."""
        return list(range(self.state_space_size))
    
    def get_action_space(self):
        """Retourne l'espace des actions."""
        return list(range(self.action_space_size))
    
    def get_rewards(self):
        """Retourne la liste des récompenses possibles."""
        return [0.0, 1.0]
    
    def get_terminal_states(self):
        """Retourne la liste des états terminaux."""
        return [2]
    
    def get_transition_matrix(self):
        """Retourne la matrice de transition pour les algorithmes de programmation dynamique."""
        # Matrice simplifiée pour Monty Hall
        p = np.zeros((3, 3, 3, 2))  # (states, actions, next_states, rewards)
        
        # État 0: choix initial
        for a in range(3):
            p[0, a, 1, 0] = 1.0  # Toujours aller à l'état 1
        
        # État 1: choix final
        for a in range(2):
            p[1, a, 2, 0] = 0.5  # 50% chance de gagner
            p[1, a, 2, 1] = 0.5  # 50% chance de perdre
        
        # État 2: terminal
        p[2, 0, 2, 0] = 1.0  # Rester sur place
        
        return p

def train_agent():
    """Entraîne un agent Value Iteration pour la démonstration."""
    print("Entraînement de l'agent...")
    
    env = MontyHallInteractive()
    agent = ValueIteration(
        environment=env,
        discount_factor=0.9,
        theta=0.001,
        max_iterations=1000
    )
    
    results = agent.train()
    print(f"Agent entraîné en {results['iterations']} itérations!")
    
    return agent

def play_game(env, visualizer, agent=None, mode="human"):
    """Joue une partie de Monty Hall."""
    state = env.reset()
    total_reward = 0
    steps = 0
    actions_taken = []
    
    while True:
        # Afficher l'état actuel
        episode_info = {
            "Mode": mode,
            "Étapes": steps,
            "Récompense": total_reward,
            "Actions": actions_taken
        }
        visualizer.render_monty_hall_state(env, episode_info)
        
        # Gérer les événements
        event = visualizer.handle_events()
        if event == "pause":
            continue
        elif not visualizer.running:
            return None
        
        # Choisir l'action
        if env.state == 0:
            # Choix initial de porte
            if mode == "human":
                action = visualizer.get_human_action(env)
                if action is None:
                    break
            else:
                action = np.random.choice([0, 1, 2])
        else:
            # Action rester/changer
            if mode == "human":
                action = visualizer.get_human_action(env)
                if action is None:
                    break
            else:
                action = np.argmax(agent.policy[state])
        
        actions_taken.append(action)
        
        # Exécuter l'action
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Pause pour voir le résultat
        time.sleep(1.0)
        
        if done:
            # Afficher le résultat final
            episode_info = {
                "Mode": mode,
                "Étapes": steps,
                "Récompense": total_reward,
                "Actions": actions_taken,
                "Résultat": "GAGNÉ!" if info.get("result") == "win" else "PERDU!"
            }
            visualizer.render_monty_hall_state(env, episode_info)
            time.sleep(3.0)
            break
    
    return info.get("result") == "win"

def main_demo():
    """Démonstration principale."""
    print("=== DÉMONSTRATION INTERACTIVE MONTY HALL (AUTONOME) ===")
    
    # Créer l'environnement et le visualiseur
    env = MontyHallInteractive()
    visualizer = MontyHallVisualizer()
    
    # Entraîner l'agent
    agent = train_agent()
    
    # Partie humaine
    print("\n--- Partie humaine ---")
    human_result = play_game(env, visualizer, agent, mode="human")
    if human_result is not None:
        print(f"Résultat humain : {'GAGNÉ' if human_result else 'PERDU'}")
    time.sleep(2)
    
    # Partie agent
    print("\n--- Partie agent ---")
    agent_result = play_game(env, visualizer, agent, mode="agent")
    if agent_result is not None:
        print(f"Résultat agent : {'GAGNÉ' if agent_result else 'PERDU'}")
    time.sleep(2)
    
    # Attendre que l'utilisateur ferme la fenêtre
    print("Appuyez sur ÉCHAP ou fermez la fenêtre pour quitter...")
    while visualizer.running:
        event = visualizer.handle_events()
        if event is not None:
            break
        time.sleep(0.1)
    
    visualizer.quit()

if __name__ == "__main__":
    main_demo() 