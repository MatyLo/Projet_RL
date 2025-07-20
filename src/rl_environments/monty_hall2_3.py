"""
Implémentation de l'environnement Monty Hall Level 2 avec 5 portes.

L'agent doit effectuer 4 actions successives avant l'ouverture finale.
Compatible avec les algorithmes de programmation dynamique.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from src.rl_environments.base_environment import BaseEnvironment


class MontyHallLevel2Environment(BaseEnvironment):
    """
    Environnement Monty Hall Level 2 avec 5 portes.
    
    États:
    - 0: État initial (5 portes disponibles)
    - 1: Après premier choix (4 portes restantes, 1 retirée)
    - 2: Après deuxième choix (3 portes restantes, 1 retirée)
    - 3: Après troisième choix (2 portes restantes, 1 retirée)
    - 4: Après quatrième choix (état final, résultat connu)
    
    Actions dépendent de l'état actuel et des portes disponibles.
    """
    
    def __init__(self):
        super().__init__("MontyHall_Level2")
        
        # Configuration du jeu
        self.num_doors = 5
        self.num_actions = 4
        
        # État du jeu
        self.winning_door = None
        self.agent_door = None
        self.available_doors = None
        self.removed_doors = None
        self.game_step = 0
        
        # Pour la programmation dynamique
        self._precompute_transitions()
        
    @property
    def state_space_size(self) -> int:
        """5 états possibles (0 à 4)."""
        return 5
    
    @property
    def action_space_size(self) -> int:
        """Maximum 5 actions possibles (choix parmi les 5 portes)."""
        return self.num_doors
    
    @property
    def valid_actions(self) -> List[int]:
        """Retourne les actions valides selon l'état actuel."""
        if self.current_state == 0:
            # Premier choix : toutes les portes sont disponibles
            return list(range(self.num_doors))
        elif self.current_state in [1, 2, 3]:
            # Choix suivants : garder sa porte (action = agent_door) ou changer
            actions = [self.agent_door]  # Garder la porte actuelle
            actions.extend(self.available_doors)  # Changer pour une porte disponible
            return sorted(list(set(actions)))
        else:
            # État final : aucune action valide
            return []
    
    def reset(self) -> int:
        """Remet l'environnement à l'état initial."""
        self._reset_episode_stats()
        
        # Initialisation du jeu
        self.winning_door = np.random.randint(0, self.num_doors)
        self.agent_door = None
        self.available_doors = list(range(self.num_doors))
        self.removed_doors = []
        self.game_step = 0
        self.current_state = 0
        
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Exécute une action dans l'environnement."""
        if not self.is_valid_action(action):
            raise ValueError(f"Action {action} invalide dans l'état {self.current_state}")
        
        reward = 0.0
        done = False
        info = {}
        
        if self.current_state == 0:
            # Premier choix de l'agent
            self.agent_door = action
            self.available_doors.remove(action)
            self._remove_losing_door()
            self.current_state = 1
            self.game_step = 1
            
        elif self.current_state in [1, 2, 3]:
            # Choix suivants : garder ou changer
            if action != self.agent_door:
                # L'agent change de porte
                self.agent_door = action
                if action in self.available_doors:
                    self.available_doors.remove(action)
            
            # Passer à l'état suivant
            if self.current_state < 3:
                self._remove_losing_door()
                self.current_state += 1
                self.game_step += 1
            else:
                # Dernier choix, passage à l'état final
                self.current_state = 4
                self.game_step += 1
                
                # Calcul de la récompense finale
                if self.agent_door == self.winning_door:
                    reward = 1.0
                else:
                    reward = 0.0
                done = True
        
        info = {
            'winning_door': self.winning_door,
            'agent_door': self.agent_door,
            'available_doors': self.available_doors.copy(),
            'removed_doors': self.removed_doors.copy(),
            'game_step': self.game_step
        }
        
        # Mise à jour des statistiques
        next_state = self.current_state
        self._update_episode_stats(action, reward, next_state, done)
        
        return next_state, reward, done, info
    
    def _remove_losing_door(self):
        """Retire une porte perdante parmi les portes disponibles."""
        # Portes disponibles qui ne sont pas la porte gagnante
        losing_doors = [door for door in self.available_doors 
                       if door != self.winning_door]
        
        if losing_doors:
            door_to_remove = np.random.choice(losing_doors)
            self.available_doors.remove(door_to_remove)
            self.removed_doors.append(door_to_remove)
    
    def render(self, mode: str = 'console') -> Optional[Any]:
        """Affiche l'état actuel de l'environnement."""
        if mode != 'console':
            return None
            
        print(f"\n=== Monty Hall Level 2 - État {self.current_state} ===")
        print(f"Étape du jeu: {self.game_step}/4")
        
        if self.agent_door is not None:
            print(f"Porte de l'agent: {self.agent_door}")
        
        print(f"Portes disponibles: {self.available_doors}")
        print(f"Portes retirées: {self.removed_doors}")
        print(f"Actions valides: {self.valid_actions}")
        
        if self.current_state == 4:
            print(f"RÉSULTAT: {'GAGNÉ!' if self.agent_door == self.winning_door else 'PERDU!'}")
            print(f"Porte gagnante était: {self.winning_door}")
    
    def get_state_description(self, state: int) -> str:
        """Retourne une description textuelle d'un état."""
        descriptions = {
            0: "État initial - Choix de la première porte",
            1: "Après premier choix - Une porte retirée, 3 choix restants",
            2: "Après deuxième choix - Deux portes retirées, 2 choix restants", 
            3: "Après troisième choix - Trois portes retirées, 1 choix restant",
            4: "État final - Résultat révélé"
        }
        return descriptions.get(state, f"État {state} inconnu")
    
    def get_transition_probabilities(self, state: int, action: int) -> Dict[int, float]:
        """
        Retourne les probabilités de transition pour la programmation dynamique.
        
        Dans Monty Hall Level 2, les transitions sont déterministes:
        - État 0 → État 1 avec probabilité 1.0
        - État 1 → État 2 avec probabilité 1.0
        - État 2 → État 3 avec probabilité 1.0
        - État 3 → État 4 avec probabilité 1.0
        - État 4 → État 4 avec probabilité 1.0 (état absorbant)
        """
        if state == 4:
            # État final absorbant
            return {4: 1.0}
        elif state in [0, 1, 2, 3]:
            # Transition déterministe vers l'état suivant
            return {state + 1: 1.0}
        else:
            return {}
    
    def get_reward_function(self, state: int, action: int, next_state: int) -> float:
        """
        Retourne la récompense pour une transition donnée.
        
        Seule la transition vers l'état final (4) peut donner une récompense.
        La récompense dépend de si l'action finale mène à la porte gagnante.
        """
        if state == 3 and next_state == 4:
            # Transition finale - calculer si l'action mène à la victoire
            # Cette méthode est appelée pour tous les scénarios possibles
            # Pour la programmation dynamique, on retourne la probabilité de gagner
            
            # Si on garde la porte initiale, probabilité = 1/5
            # Si on change, probabilité = 4/5
            # Mais ici on doit calculer pour une action spécifique
            
            # Simplification pour DP : 
            # - Si action correspond à une stratégie de changement → 0.8
            # - Si action correspond à garder la porte initiale → 0.2
            return 0.8  # Valeur moyenne, à ajuster selon la stratégie
        else:
            return 0.0
    
    def get_terminal_states(self) -> List[int]:
        """
        Retourne la liste des états terminaux.
        Essentiel pour les algorithmes de programmation dynamique.
        
        Returns:
            List[int]: Liste des états terminaux
        """
        return [4]  # Seul l'état 4 est terminal
    
    def is_terminal_state(self, state: int) -> bool:
        """
        Vérifie si un état est terminal.
        
        Args:
            state (int): État à vérifier
            
        Returns:
            bool: True si l'état est terminal
        """
        return state == 4
    
    def get_all_states(self) -> List[int]:
        """
        Retourne tous les états possibles de l'environnement.
        Utile pour les algorithmes de programmation dynamique.
        
        Returns:
            List[int]: Liste de tous les états (0 à 4)
        """
        return list(range(self.state_space_size))
    
    def _precompute_transitions(self):
        """Précalcule les transitions pour optimiser la programmation dynamique."""
        # Cette méthode peut être utilisée pour optimiser les algorithmes DP
        # En précalculant toutes les transitions possibles
        pass
    
    def get_optimal_policy_hint(self) -> Dict[int, int]:
        """
        Retourne un indice sur la politique optimale.
        Utile pour vérifier les algorithmes de DP.
        """
        # La stratégie optimale est de toujours changer de porte
        # quand c'est possible (sauf au premier choix)
        return {
            0: 0,  # Premier choix arbitraire
            1: -1, # Changer (action différente de la porte actuelle)
            2: -1, # Changer
            3: -1, # Changer
            4: -1  # Aucune action
        }
    
    def compute_win_probability(self, always_switch: bool = True) -> float:
        """
        Calcule la probabilité théorique de gagner selon la stratégie.
        
        Args:
            always_switch (bool): Si True, toujours changer de porte
            
        Returns:
            float: Probabilité de gagner
        """
        if always_switch:
            return 4.0 / 5.0  # 80%
        else:
            return 1.0 / 5.0  # 20%
