#!/usr/bin/env python3
"""
Démonstration interactive autonome du problème de Monty Hall 2 avec PyGame.
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

class MontyHall2StepByStep(BaseEnvironment):
    """
    Environnement Monty Hall 2 (5 portes) pas à pas, hybride :
    - Jusqu'à 3 portes restantes : l'utilisateur choisit à chaque étape, une porte perdante est éliminée.
    - Quand il reste 3 portes :
        * L'utilisateur choisit une porte (ou garde la sienne)
        * Le présentateur élimine une des deux autres perdantes
        * L'utilisateur choisit 'garder' ou 'changer' (boutons)
        * On révèle le résultat
    """
    def __init__(self, n_doors: int = 5):
        self.n_doors = n_doors
        self.state = 0  # 0: choix initial, 1: choix/garde, 2: choix final (garder/changer), 3: terminé
        self.doors = list(range(n_doors))
        self.winning_door = None
        self.agent_choice = None
        self.agent_first_choice = None
        self.eliminated_doors = []
        self.remaining_doors = list(range(n_doors))
        self.steps = 0
        self.final_choice = None
        self.done = False
        self.last_eliminated = None
        self.choice_at_3 = None  # Le choix de l'utilisateur quand il reste 3 portes

    def reset(self) -> int:
        self.state = 0
        self.winning_door = random.randint(0, self.n_doors - 1)
        self.agent_choice = None
        self.agent_first_choice = None
        self.eliminated_doors = []
        self.remaining_doors = list(range(self.n_doors))
        self.steps = 0
        self.final_choice = None
        self.done = False
        self.last_eliminated = None
        self.choice_at_3 = None
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        info = {}
        reward = 0.0
        # Jusqu'à 3 portes restantes : mode pas à pas
        if self.state == 0:
            self.agent_choice = action
            self.agent_first_choice = action
            self.state = 1
            self.steps = 1
            elim = self._eliminate_one_door()
            self.eliminated_doors.append(elim)
            self.remaining_doors.remove(elim)
            self.last_eliminated = elim
            info['eliminated'] = elim
            info['remaining'] = self.remaining_doors.copy()
            return self.state, reward, False, info
        elif self.state == 1:
            # L'utilisateur choisit une porte parmi les restantes
            if action in self.remaining_doors:
                self.agent_choice = action
            # Si on arrive à 3 portes après élimination, passer à l'étape spéciale Monty Hall 1
            if len(self.remaining_doors) > 3:
                elim = self._eliminate_one_door()
                self.eliminated_doors.append(elim)
                self.remaining_doors.remove(elim)
                self.last_eliminated = elim
                info['eliminated'] = elim
                info['remaining'] = self.remaining_doors.copy()
                return self.state, reward, False, info
            elif len(self.remaining_doors) == 3:
                # On passe directement à l'étape spéciale Monty Hall 1 (choix final)
                self.choice_at_3 = self.agent_choice  # Utiliser le choix de l'état 1
                # Le présentateur élimine automatiquement une des deux autres perdantes
                candidates = [d for d in self.remaining_doors if d != self.choice_at_3 and d != self.winning_door]
                elim = random.choice(candidates)
                self.eliminated_doors.append(elim)
                self.remaining_doors.remove(elim)
                self.last_eliminated = elim
                # On passe directement au choix final (garder/changer)
                self.state = 3
                return self.state, reward, False, {'eliminated': elim, 'remaining': self.remaining_doors.copy()}
        elif self.state == 3:
            # Boutons garder/changer : action 0 = garder, 1 = changer
            if action == 0:
                # Garder la porte choisie à l'étape des 3 portes
                self.final_choice = self.choice_at_3
            else:
                # Prendre l'autre porte restante
                other = [d for d in self.remaining_doors if d != self.choice_at_3][0]
                self.final_choice = other
            self.done = True
            self.state = 4
            reward = 1.0 if self.final_choice == self.winning_door else 0.0
            info['result'] = 'win' if reward == 1.0 else 'lose'
            info['winning_door'] = self.winning_door
            info['final_choice'] = self.final_choice
            info['remaining'] = self.remaining_doors.copy()
            return self.state, reward, True, info
        else:
            return self.state, reward, True, info

    def _eliminate_one_door(self) -> int:
        candidates = [d for d in self.remaining_doors if d != self.agent_choice and d != self.winning_door]
        return random.choice(candidates)

    @property
    def valid_actions(self) -> List[int]:
        if self.state in [0, 1]:
            return self.remaining_doors.copy()
        elif self.state == 3:
            return [0, 1]  # garder/changer
        else:
            return []

    @property
    def action_space_size(self) -> int:
        return self.n_doors

    @property
    def state_space_size(self) -> int:
        return 5

    def render(self, mode: str = 'console'):
        if mode == 'console':
            print(f"Portes restantes: {self.remaining_doors}")
            print(f"Portes éliminées: {self.eliminated_doors}")
            print(f"Votre choix: {self.agent_choice}")
            if self.state == 4:
                print(f"Porte gagnante: {self.winning_door}")
                print(f"Gagné ? {self.final_choice == self.winning_door}")

    def get_state_description(self, state: int) -> str:
        descriptions = {
            0: "Choix initial de porte",
            1: "Choix/garde à chaque étape",
            2: "Choix spécial Monty Hall 1 (3 portes)",
            3: "Choix final garder/changer",
            4: "Partie terminée"
        }
        return descriptions.get(state, "État inconnu")

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
        return [4]
    
    def get_transition_matrix(self):
        """Retourne la matrice de transition pour les algorithmes de programmation dynamique."""
        # Matrice simplifiée pour Monty Hall 2
        p = np.zeros((5, 5, 5, 2))  # (states, actions, next_states, rewards)
        
        # État 0: choix initial
        for a in range(5):
            p[0, a, 1, 0] = 1.0  # Toujours aller à l'état 1
        
        # État 1: choix/garde
        for a in range(5):
            p[1, a, 3, 0] = 1.0  # Aller à l'état 3 (choix final)
        
        # État 3: choix final garder/changer
        for a in range(2):
            p[3, a, 4, 0] = 0.5  # 50% chance de gagner
            p[3, a, 4, 1] = 0.5  # 50% chance de perdre
        
        # État 4: terminal
        p[4, 0, 4, 0] = 1.0  # Rester sur place
        
        return p

def train_agent():
    """Entraîne un agent Value Iteration pour la démonstration."""
    print("Entraînement de l'agent...")
    
    env = MontyHall2StepByStep()
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
    """Joue une partie de Monty Hall 2."""
    state = env.reset()
    total_reward = 0
    steps = 0
    actions_taken = []
    
    while not env.done:
        # Afficher l'état actuel
        episode_info = {
            "Mode": mode,
            "Étapes": steps,
            "Récompense": total_reward,
            "Actions": actions_taken
        }
        visualizer.render_monty_hall2_state(env, episode_info)
        
        # Gérer les événements
        event = visualizer.handle_events()
        if event == "pause":
            continue
        elif not visualizer.running:
            return None
        
        # Choisir l'action
        if env.state in [0, 1]:
            # Choix de porte
            if mode == "human":
                action = visualizer.get_human_action_mh2(env)
                if action is None:
                    break
            else:
                action = np.random.choice(env.remaining_doors)
        elif env.state == 3:
            # Action garder/changer
            if mode == "human":
                action = visualizer.get_human_action_mh2(env)
                if action is None:
                    break
            else:
                action = np.argmax(agent.policy[state]) if agent.policy[state] is not None else np.random.choice([0, 1])
        
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
            visualizer.render_monty_hall2_state(env, episode_info)
            time.sleep(3.0)
            break
    
    return info.get("result") == "win"

def main_demo():
    """Démonstration principale."""
    print("=== DÉMONSTRATION INTERACTIVE MONTY HALL 2 (AUTONOME) ===")
    
    # Créer l'environnement et le visualiseur
    env = MontyHall2StepByStep()
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

# Boucle interactive simple (pour compatibilité)
if __name__ == "__main__":
    main_demo() 