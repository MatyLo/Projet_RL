"""
Environnement Monty Hall Level 2
Étend le problème classique à 5 portes avec 4 actions successives.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from src.rl_environments.base_environment import BaseEnvironment


class MontyHallLevel2(BaseEnvironment):
    """
    Environnement Monty Hall avec 5 portes et 4 actions successives.
    
    États:
    - Les états encodent: phase du jeu, porte choisie, portes éliminées, porte gagnante
    - Phase: 0 (choix initial), 1-3 (éliminations successives), 4 (choix final)
    
    Actions:
    - Phase 0: Choisir une porte parmi 5 (actions 0-4)
    - Phases 1-3: Garder (0) ou changer vers une porte spécifique (1-4, selon portes disponibles)
    - Phase 4: Choix final entre les 2 dernières portes
    """
    
    def __init__(self):
        super().__init__("MontyHall_Level2")
        self.num_doors = 5
        self.num_actions_before_final = 4
        
        # État du jeu
        self.phase = 0  # 0: choix initial, 1-3: éliminations, 4: choix final
        self.winning_door = None  # Porte gagnante (0-4)
        self.chosen_door = None   # Porte actuellement choisie
        self.eliminated_doors = set()  # Portes éliminées
        self.available_doors = set(range(5))  # Portes disponibles
        
        # Pour le state encoding
        self.reset()
    
    @property
    def state_space_size(self) -> int:
        """
        Espace d'états basé sur:
        - Phase (5 phases: 0-4)
        - Porte choisie (5 possibilités)
        - Portes éliminées (combinaisons possibles)
        - Porte gagnante (5 possibilités)
        
        Approximation: 5 * 5 * (2^5) * 5 = 4000 états maximum
        En pratique, beaucoup moins car toutes les combinaisons ne sont pas valides.
        """
        return 5000  # Approximation sécurisée
    
    @property
    def action_space_size(self) -> int:
        """5 actions maximum (une pour chaque porte)"""
        return 5
    
    @property
    def valid_actions(self) -> List[int]:
        """Retourne les actions valides selon la phase actuelle"""
        if self.phase == 0:
            # Phase initiale: choisir parmi toutes les portes
            return list(range(5))
        elif self.phase < 4:
            # Phases intermédiaires: garder (0) ou changer vers portes disponibles
            actions = [0]  # Garder la porte actuelle
            for door in self.available_doors:
                if door != self.chosen_door:
                    actions.append(door + 1)  # +1 car 0 = garder
            return actions
        else:
            # Phase finale: choisir entre les 2 dernières portes
            remaining_doors = list(self.available_doors)
            return remaining_doors
    
    def reset(self) -> int:
        """Remet l'environnement à l'état initial"""
        self.phase = 0
        self.winning_door = np.random.randint(0, 5)
        self.chosen_door = None
        self.eliminated_doors = set()
        self.available_doors = set(range(5))
        
        self.current_state = self._encode_state()
        self._reset_episode_stats()
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Exécute une action dans l'environnement"""
        if not self.is_valid_action(action):
            raise ValueError(f"Action {action} not valid in phase {self.phase}")
        
        reward = 0.0
        done = False
        info = {}
        
        if self.phase == 0:
            # Choix initial de la porte
            self.chosen_door = action
            self.phase = 1
            info['action_type'] = 'initial_choice'
            info['chosen_door'] = self.chosen_door
            
        elif self.phase < 4:
            # Phases d'élimination
            if action == 0:
                # Garder la porte actuelle
                info['action_type'] = 'keep'
            else:
                # Changer vers une autre porte
                new_door = action - 1
                if new_door in self.available_doors and new_door != self.chosen_door:
                    self.chosen_door = new_door
                    info['action_type'] = 'switch'
                    info['new_chosen_door'] = self.chosen_door
                else:
                    raise ValueError(f"Invalid door change to {new_door}")
            
            # Éliminer une porte (pas la choisie, pas la gagnante)
            eliminable_doors = self.available_doors - {self.chosen_door, self.winning_door}
            if eliminable_doors:
                eliminated = np.random.choice(list(eliminable_doors))
                self.eliminated_doors.add(eliminated)
                self.available_doors.remove(eliminated)
                info['eliminated_door'] = eliminated
            
            self.phase += 1
            
        else:  # phase == 4
            # Choix final
            self.chosen_door = action
            done = True
            reward = 1.0 if self.chosen_door == self.winning_door else 0.0
            info['action_type'] = 'final_choice'
            info['won'] = reward == 1.0
            info['winning_door'] = self.winning_door
        
        next_state = self._encode_state()
        self._update_episode_stats(action, reward, next_state, done)
        self.current_state = next_state
        
        return next_state, reward, done, info
    
    def _encode_state(self) -> int:
        """Encode l'état actuel en un entier"""
        # Simple encoding: phase * 1000 + chosen_door * 100 + hash des portes éliminées
        state = self.phase * 1000
        
        if self.chosen_door is not None:
            state += self.chosen_door * 100
        
        # Ajouter information sur les portes éliminées
        eliminated_hash = sum([2**door for door in self.eliminated_doors]) % 100
        state += eliminated_hash
        
        return state
    
    def _decode_state(self, state: int) -> Tuple[int, Optional[int], set]:
        """Décode un état (utile pour debug)"""
        phase = state // 1000
        chosen_door = (state % 1000) // 100 if (state % 1000) // 100 < 5 else None
        eliminated_hash = state % 100
        
        # Reconstruction approximative des portes éliminées
        eliminated_doors = set()
        for door in range(5):
            if eliminated_hash & (2**door):
                eliminated_doors.add(door)
        
        return phase, chosen_door, eliminated_doors
    
    def render(self, mode: str = 'console') -> Optional[Any]:
        """Affiche l'état actuel"""
        if mode == 'console':
            print(f"\n=== Monty Hall Level 2 - Phase {self.phase} ===")
            print(f"Portes disponibles: {sorted(self.available_doors)}")
            print(f"Porte choisie: {self.chosen_door}")
            print(f"Portes éliminées: {sorted(self.eliminated_doors)}")
            
            if self.phase == 4:
                print(f"Porte gagnante: {self.winning_door}")
                print(f"Résultat: {'GAGNÉ' if self.chosen_door == self.winning_door else 'PERDU'}")
            
            print(f"Actions valides: {self.valid_actions}")
            print("=" * 40)
    
    def get_state_description(self, state: int) -> str:
        """Description textuelle d'un état"""
        phase, chosen_door, eliminated_doors = self._decode_state(state)
        return (f"Phase {phase}, Porte choisie: {chosen_door}, "
                f"Éliminées: {sorted(eliminated_doors)}")

    def get_transition_probabilities(self, state: int, action: int) -> Dict[int, float]:
        """
        Version simplifiée et efficace des probabilités de transition.
        """
        phase, chosen_door, eliminated_doors = self._decode_state(state)
        
        # Pour Policy Iteration, on peut simplifier en utilisant des transitions déterministes
        # L'aléatoire de l'élimination des portes peut être abstrait
        
        if phase == 0:
            # Choix initial → transition vers phase 1
            next_state = 1000 + action * 100
            return {next_state: 1.0}
            
        elif phase < 4:
            # Phases intermédiaires → transition déterministe vers phase suivante
            if action == 0:
                # Garder la porte
                next_state = (phase + 1) * 1000 + chosen_door * 100 + (state % 100)
            else:
                # Changer vers une autre porte
                new_door = action - 1
                next_state = (phase + 1) * 1000 + new_door * 100 + (state % 100)
            
            return {next_state: 1.0}
            
        else:
            # Phase finale → état terminal
            return {state: 1.0}
    
    
    def get_transition_probabilities2(self, state: int, action: int) -> Dict[int, float]:
        """
        Retourne les probabilités de transition pour la programmation dynamique.
        
        Dans Monty Hall, la seule source d'aléatoire est:
        1. Le choix de la porte gagnante (au reset)
        2. Le choix de la porte à éliminer parmi les possibles
        """
        # Sauvegarder l'état actuel
        current_state_backup = self.current_state
        phase_backup = self.phase
        chosen_door_backup = self.chosen_door
        eliminated_doors_backup = self.eliminated_doors.copy()
        available_doors_backup = self.available_doors.copy()
        
        # Simuler la transition
        self.current_state = state
        phase, chosen_door, eliminated_doors = self._decode_state(state)
        self.phase = phase
        self.chosen_door = chosen_door
        self.eliminated_doors = eliminated_doors
        self.available_doors = set(range(5)) - eliminated_doors
        
        transitions = {}
        
        if self.phase < 4 and self.phase > 0:
            # Phase avec élimination aléatoire
            eliminable_doors = self.available_doors - {self.chosen_door, self.winning_door}
            if eliminable_doors:
                prob_per_elimination = 1.0 / len(eliminable_doors)
                for eliminated_door in eliminable_doors:
                    # Calculer l'état résultant
                    temp_eliminated = self.eliminated_doors.copy()
                    temp_eliminated.add(eliminated_door)
                    temp_available = self.available_doors - {eliminated_door}
                    
                    # Encoder le nouvel état
                    next_state = (self.phase + 1) * 1000
                    if self.chosen_door is not None:
                        next_state += self.chosen_door * 100
                    eliminated_hash = sum([2**door for door in temp_eliminated]) % 100
                    next_state += eliminated_hash
                    
                    transitions[next_state] = prob_per_elimination
            else:
                # Pas d'élimination possible, transition déterministe
                next_state = (self.phase + 1) * 1000
                if self.chosen_door is not None:
                    next_state += self.chosen_door * 100
                transitions[next_state] = 1.0
        else:
            # Transition déterministe
            if action in self.valid_actions:
                # Simuler l'action
                if self.phase == 0:
                    next_state = 1000 + action * 100
                elif self.phase == 4:
                    next_state = state  # État final
                else:
                    next_state = (self.phase + 1) * 1000
                    if action == 0:
                        next_state += self.chosen_door * 100
                    else:
                        next_state += (action - 1) * 100
                
                transitions[next_state] = 1.0
        
        # Restaurer l'état
        self.current_state = current_state_backup
        self.phase = phase_backup
        self.chosen_door = chosen_door_backup
        self.eliminated_doors = eliminated_doors_backup
        self.available_doors = available_doors_backup
        
        return transitions if transitions else {state: 1.0}
    
    def get_reward_function(self, state: int, action: int, next_state: int) -> float:
        """Retourne la récompense pour une transition donnée"""
        phase, chosen_door, eliminated_doors = self._decode_state(state)
        
        # Récompense seulement à la fin
        if phase == 4:
            # Action finale: récompense basée sur si la porte choisie est gagnante
            return 1.0 if action == self.winning_door else 0.0
        else:
            return 0.0
    
    def get_terminal_states(self) -> List[int]:
        """
        Retourne la liste des états terminaux.
        Dans Monty Hall Level 2, les états terminaux sont ceux de la phase 4
        après avoir fait le choix final.
        """
        terminal_states = []
        
        # Les états terminaux correspondent à la phase 4 avec toutes les
        # combinaisons possibles de portes choisies et éliminées
        for chosen_door in range(5):
            for eliminated_pattern in range(32):  # 2^5 patterns possibles
                # Vérifier que le pattern est valide (3 portes éliminées exactement)
                eliminated_doors = []
                for door in range(5):
                    if eliminated_pattern & (2**door):
                        eliminated_doors.append(door)
                
                # Il doit y avoir exactement 3 portes éliminées
                # et la porte choisie ne doit pas être éliminée
                if len(eliminated_doors) == 3 and chosen_door not in eliminated_doors:
                    # État terminal : phase 4 (4000) + chosen_door * 100 + pattern
                    terminal_state = 4000 + chosen_door * 100 + (eliminated_pattern % 100)
                    terminal_states.append(terminal_state)
        
        return terminal_states
    
    def is_terminal_state(self, state: int) -> bool:
        """Vérifie si un état est terminal"""
        phase, _, _ = self._decode_state(state)
        return phase >= 4
    
    def get_terminal_states_infos(self) -> List[int]:
        """
        Retourne des informations sur la stratégie optimale.
        Utile pour l'évaluation des algorithmes.
        """
        # Dans Monty Hall Level 2, la stratégie optimale théorique
        # est de garder la même porte jusqu'au choix final, puis changer
        return {
            'optimal_win_probability': 4/5,  # Théorique pour 5 portes
            'strategy': 'keep_then_switch',
            'description': 'Garder la porte initiale pendant les éliminations, puis changer au choix final'
        }

