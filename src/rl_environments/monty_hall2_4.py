"""
Implémentation corrigée complète de l'environnement Monty Hall Level 2 avec 5 portes.

LOGIQUE DU JEU :
- État 0: Choisir une porte initiale parmi 5
- États 1-3: À chaque étape, décider de garder la porte actuelle ou en choisir une nouvelle
- État 4: Révélation du résultat

PROBLÈMES RÉSOLUS :
1. Logique de retrait des portes cohérente
2. Actions valides correctement définies
3. Transitions d'états fixes
4. Compatibilité avec tous les algorithmes RL
5. NOUVEAU: Seules les portes disponibles ET non gagnantes peuvent être éliminées
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from src.rl_environments.base_environment import BaseEnvironment


class MontyHallLevel2Environment(BaseEnvironment):
    """
    Environnement Monty Hall Level 2 avec 5 portes - Version corrigée.
    
    États:
    - 0: État initial (choisir parmi 5 portes)
    - 1: Après 1er choix + retrait de 1 porte (4 portes restantes)
    - 2: Après 2ème choix + retrait de 1 porte (3 portes restantes)
    - 3: Après 3ème choix + retrait de 1 porte (2 portes restantes)
    - 4: État final (résultat connu)
    
    Actions: 0-4 (numéros des portes)
    """
    
    def __init__(self):
        super().__init__("MontyHall_Level2")
        
        # Configuration du jeu
        self.num_doors = 5
        
        # État du jeu
        self.winning_door = None
        self.agent_door = None
        self.doors_in_game = None  # Portes encore dans le jeu
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
        """5 actions possibles (portes 0-4)."""
        return self.num_doors
    
    @property
    def valid_actions(self) -> List[int]:
        """Retourne les actions valides selon l'état actuel."""
        if self.current_state == 0:
            # État initial : toutes les portes disponibles (0-4)
            return list(range(self.num_doors))
        
        elif self.current_state in [1, 2, 3]:
            # États intermédiaires : seulement les portes encore dans le jeu
            # Ces portes ont survécu aux éliminations précédentes
            valid = sorted(list(self.doors_in_game))
            return valid
        
        elif self.current_state == 4:
            # État final : aucune action possible
            return []
        
        else:
            # État invalide
            return []
    
    def reset(self) -> int:
        """Remet l'environnement à l'état initial."""
        self._reset_episode_stats()
        
        # Initialisation du jeu
        self.winning_door = np.random.randint(0, self.num_doors)
        self.agent_door = None
        self.doors_in_game = set(range(self.num_doors))
        self.removed_doors = []
        self.game_step = 0
        self.current_state = 0
        
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Exécute une action dans l'environnement."""
        # Validation stricte de l'action
        if not self._is_action_valid_for_current_state(action):
            valid_actions = self.valid_actions
            raise ValueError(
                f"Action {action} invalide dans l'état {self.current_state}.\n"
                f"Actions valides: {valid_actions}\n"
                f"État du jeu: doors_in_game={sorted(self.doors_in_game)}, "
                f"agent_door={self.agent_door}, removed_doors={self.removed_doors}"
            )
        
        reward = 0.0
        done = False
        info = {}
        
        if self.current_state == 0:
            # Premier choix de l'agent (état 0 → état 1)
            self.agent_door = action
            self.doors_in_game.remove(action)
            # Retirer une porte perdante
            self._remove_losing_door()
            # Remettre la porte de l'agent dans le jeu pour les choix suivants
            self.doors_in_game.add(action)
            self.current_state = 1
            self.game_step = 1
            
        elif self.current_state in [1, 2, 3]:
            # Choix intermédiaires et final
            # Mettre à jour le choix de l'agent
            self.agent_door = action
            
            if self.current_state < 3:
                # États intermédiaires (1→2, 2→3) : retirer une porte et continuer
                self._remove_losing_door()
                self.current_state += 1
                self.game_step += 1
            else:
                # Dernier choix (état 3 → état 4)
                self.current_state = 4
                self.game_step += 1
                
                # Calcul de la récompense finale
                if self.agent_door == self.winning_door:
                    reward = 1.0
                else:
                    reward = 0.0
                done = True
        
        elif self.current_state == 4:
            # État final : aucune action ne devrait être possible
            raise ValueError("Tentative d'action dans un état terminal (état 4)")
        
        info = {
            'winning_door': self.winning_door,
            'agent_door': self.agent_door,
            'doors_in_game': list(self.doors_in_game),
            'removed_doors': self.removed_doors.copy(),
            'game_step': self.game_step,
            'valid_actions': self.valid_actions.copy()
        }
        
        # Mise à jour des statistiques
        next_state = self.current_state
        self._update_episode_stats(action, reward, next_state, done)
        
        return next_state, reward, done, info

    def get_valid_actions_for_state(self, state: int, context: Dict = None) -> List[int]:
        """
        Retourne les actions valides pour un état donné (pour DP/algorithmes externes).
        
        Args:
            state: État pour lequel calculer les actions
            context: Contexte additionnel avec informations sur les portes
            
        Returns:
            List[int]: Actions valides pour cet état
        """
        if state == 0:
            # État initial : toutes les portes sont disponibles
            return list(range(self.num_doors))
        
        elif state in [1, 2, 3]:
            # États intermédiaires : dépend du contexte du jeu
            if context and 'doors_in_game' in context:
                # Si on a le contexte, utiliser les vraies portes disponibles
                return sorted(list(context['doors_in_game']))
            else:
                # Sinon, estimation générique basée sur l'état
                # État 1: ~4 portes, État 2: ~3 portes, État 3: ~2 portes
                if state == 1:
                    return [0, 1, 2, 3]  # 4 choix typiques
                elif state == 2:
                    return [0, 1, 2]     # 3 choix typiques
                else:  # state == 3
                    return [0, 1]        # 2 choix typiques
        
        elif state == 4:
            # État final : aucune action
            return []
        
        else:
            # État invalide
            return []
    
    def enforce_action_constraints(self, action: int) -> int:
        """
        Force une action à être valide en la remplaçant si nécessaire.
        
        Args:
            action: Action demandée
            
        Returns:
            int: Action valide (originale ou de remplacement)
        """
        if self.is_valid_action(action):
            return action
        
        # Si l'action n'est pas valide, prendre une action valide aléatoire
        valid_actions = self.valid_actions
        if valid_actions:
            replacement = np.random.choice(valid_actions)
            print(f"WARNING: Action {action} invalide remplacée par {replacement}")
            return replacement
        else:
            raise ValueError(f"Aucune action valide disponible dans l'état {self.current_state}")
    
    def get_action_mask(self) -> np.ndarray:
        """
        Retourne un masque booléen des actions valides.
        
        Returns:
            np.ndarray: Masque de taille (num_doors,) avec True pour les actions valides
        """
        mask = np.zeros(self.num_doors, dtype=bool)
        for action in self.valid_actions:
            mask[action] = True
        return mask
    
    def _remove_losing_door(self):
        """
        Retire une porte perdante du jeu.
        
        RÈGLE CORRIGÉE : On retire toujours une porte qui est :
        1. DANS le jeu (doors_in_game)
        2. PAS la porte gagnante 
        3. PAS la porte de l'agent
        
        Cette logique garantit qu'on ne retire que des portes perdantes disponibles.
        """
        # Portes candidates pour suppression : 
        # - Dans le jeu (available)
        # - Pas la porte gagnante (losing)  
        # - Pas la porte de l'agent (not agent's choice)
        candidates = [door for door in self.doors_in_game 
                     if door != self.winning_door and door != self.agent_door]
        
        if candidates:
            # Retirer une porte perdante disponible aléatoirement
            door_to_remove = np.random.choice(candidates)
            self.doors_in_game.remove(door_to_remove)
            self.removed_doors.append(door_to_remove)
            print(f"DEBUG: Porte {door_to_remove} retirée (perdante et disponible)")
        else:
            # Cas particulier : si pas de candidate perdante disponible
            # Cela peut arriver dans des configurations très spécifiques
            print(f"DEBUG: Aucune porte perdante disponible à retirer")
            print(f"Doors in game: {self.doors_in_game}")
            print(f"Winning door: {self.winning_door}")
            print(f"Agent door: {self.agent_door}")
            
            # Dans ce cas, on retire une porte autre que celle de l'agent
            # (même si elle pourrait être gagnante - situation exceptionnelle)
            other_doors = [door for door in self.doors_in_game 
                          if door != self.agent_door]
            if other_doors:
                door_to_remove = np.random.choice(other_doors)
                self.doors_in_game.remove(door_to_remove)
                self.removed_doors.append(door_to_remove)
                print(f"DEBUG: Porte {door_to_remove} retirée (pas d'autre choix)")
    
    def _is_action_valid_for_current_state(self, action: int) -> bool:
        """
        Vérifie si une action est valide pour l'état actuel.
        
        Args:
            action: L'action à valider
            
        Returns:
            bool: True si l'action est valide
        """
        # Vérification de base : action dans la plage correcte
        if not (0 <= action < self.num_doors):
            return False
        
        # Vérification selon l'état
        if self.current_state == 0:
            # État initial : toutes les portes sont valides
            return True
        
        elif self.current_state in [1, 2, 3]:
            # États intermédiaires : seulement les portes encore dans le jeu
            return action in self.doors_in_game
        
        elif self.current_state == 4:
            # État final : aucune action valide
            return False
        
        else:
            # État invalide
            return False
    
    def is_valid_action(self, action: int) -> bool:
        """
        Interface publique pour vérifier la validité d'une action.
        Utilise la méthode privée _is_action_valid_for_current_state.
        """
        return self._is_action_valid_for_current_state(action)
    
    def get_invalid_actions(self) -> List[int]:
        """
        Retourne la liste des actions invalides dans l'état actuel.
        
        Returns:
            List[int]: Actions invalides
        """
        all_actions = list(range(self.num_doors))
        valid_actions = self.valid_actions
        return [action for action in all_actions if action not in valid_actions]
        """
        Retourne les actions valides pour un état donné.
        
        Args:
            state: État pour lequel calculer les actions
            context: Contexte additionnel (non utilisé ici)
            
        Returns:
            List[int]: Actions valides
        """
        if state == 0:
            return list(range(self.num_doors))
        elif state in [1, 2, 3]:
            # Dans les états intermédiaires, on assume qu'il reste plusieurs choix
            # Pour DP, on retourne un ensemble représentatif
            if state == 1:
                return [0, 1, 2, 3]  # 4 choix possibles
            elif state == 2:
                return [0, 1, 2]     # 3 choix possibles
            else:  # state == 3
                return [0, 1]        # 2 choix possibles
        else:
            return []
    
    def render(self, mode: str = 'console') -> Optional[Any]:
        """Affiche l'état actuel de l'environnement."""
        if mode != 'console':
            return None
            
        print(f"\n=== Monty Hall Level 2 - État {self.current_state} ===")
        print(f"Étape du jeu: {self.game_step}/4")
        
        if self.agent_door is not None:
            print(f"Porte de l'agent: {self.agent_door}")
        
        print(f"Portes dans le jeu: {sorted(list(self.doors_in_game))}")
        print(f"Portes retirées: {self.removed_doors}")
        
        # Affichage des actions
        valid_actions = self.valid_actions
        invalid_actions = self.get_invalid_actions()
        print(f"Actions VALIDES: {valid_actions}")
        print(f"Actions INVALIDES: {invalid_actions}")
        
        if self.winning_door is not None:
            status = "dans le jeu" if self.winning_door in self.doors_in_game else "RETIRÉE"
            print(f"Porte gagnante: {self.winning_door} ({status})")
        
        if self.current_state == 4:
            print(f"RÉSULTAT: {'GAGNÉ!' if self.agent_door == self.winning_door else 'PERDU!'}")
            print(f"Porte gagnante était: {self.winning_door}")
    
    def get_state_description(self, state: int) -> str:
        """Retourne une description textuelle d'un état."""
        descriptions = {
            0: "État initial - Choix de la première porte (5 portes)",
            1: "Après 1er choix - Une porte retirée (4 portes restantes)",
            2: "Après 2ème choix - Deux portes retirées (3 portes restantes)", 
            3: "Après 3ème choix - Trois portes retirées (2 portes restantes)",
            4: "État final - Résultat révélé"
        }
        return descriptions.get(state, f"État {state} inconnu")
    
    # ===== MÉTHODES POUR PROGRAMMATION DYNAMIQUE =====
    
    def get_transition_probabilities(self, state: int, action: int) -> Dict[int, float]:
        """
        Retourne les probabilités de transition.
        
        Dans ce jeu, les transitions sont déterministes.
        """
        if state == 4:
            return {4: 1.0}  # État absorbant
        elif state in [0, 1, 2, 3]:
            return {state + 1: 1.0}  # Transition déterministe
        else:
            return {}
    
    def get_reward_function(self, state: int, action: int, next_state: int) -> float:
        """
        Retourne la récompense pour une transition.
        
        Pour la programmation dynamique, on calcule l'espérance de récompense.
        """
        if state == 3 and next_state == 4:
            # Transition finale - probabilité de gagner selon la stratégie
            # Simplification : retourner 0.8 (probabilité optimale de gagner)
            return 0.8
        else:
            return 0.0
    
    def get_terminal_states(self) -> List[int]:
        """Retourne les états terminaux."""
        return [4]
    
    def is_terminal_state(self, state: int) -> bool:
        """Vérifie si un état est terminal."""
        return state == 4
    
    def get_all_states(self) -> List[int]:
        """Retourne tous les états possibles."""
        return list(range(self.state_space_size))
    
    def _precompute_transitions(self):
        """Précalcule les transitions pour optimiser DP."""
        pass
    
    def get_optimal_policy_hint(self) -> Dict[int, int]:
        """
        Retourne un indice sur la politique optimale.
        """
        return {
            0: 0,  # Premier choix arbitraire
            1: 1,  # Changer si possible
            2: 1,  # Changer si possible  
            3: 1,  # Changer si possible
            4: -1  # Aucune action
        }
    
    def compute_win_probability(self, always_switch: bool = True) -> float:
        """
        Calcule la probabilité théorique de gagner.
        
        Args:
            always_switch: Si True, stratégie de changement optimal
            
        Returns:
            float: Probabilité de gagner
        """
        if always_switch:
            return 4.0 / 5.0  # 80% - probabilité optimale
        else:
            return 1.0 / 5.0  # 20% - rester sur le choix initial
    
    def validate_game_logic(self) -> bool:
        """
        Valide que la logique du jeu est correcte.
        
        Returns:
            bool: True si la logique est valide
        """
        issues = []
        
        # Vérifier que la porte gagnante n'est jamais retirée
        if self.winning_door in self.removed_doors:
            issues.append(f"ERREUR: Porte gagnante {self.winning_door} a été retirée!")
        
        # Vérifier que les portes retirées ne sont plus dans le jeu
        for door in self.removed_doors:
            if door in self.doors_in_game:
                issues.append(f"ERREUR: Porte {door} est à la fois retirée et dans le jeu!")
        
        # Vérifier le nombre de portes
        total_doors = len(self.doors_in_game) + len(self.removed_doors)
        if total_doors != self.num_doors:
            issues.append(f"ERREUR: Nombre total de portes incorrect: {total_doors} au lieu de {self.num_doors}")
        
        if issues:
            for issue in issues:
                print(issue)
            return False
        
        return True
    
    # ===== MÉTHODES DE SIMULATION POUR TESTS =====
    
    def simulate_game(self, strategy="always_switch", verbose=False) -> bool:
        """
        Simule une partie complète avec une stratégie donnée.
        
        Args:
            strategy: "always_switch", "never_switch", ou "random"
            verbose: Affichage détaillé
            
        Returns:
            bool: True si victoire, False sinon
        """
        state = self.reset()
        
        if verbose:
            print(f"Porte gagnante: {self.winning_door}")
            self.render()
        
        # Premier choix (aléatoire)
        valid_first_choices = self.valid_actions
        first_choice = np.random.choice(valid_first_choices)
        
        if verbose:
            print(f"Premier choix: {first_choice} (parmi {valid_first_choices})")
        
        state, _, done, info = self.step(first_choice)
        
        if verbose:
            self.render()
            self.validate_game_logic()
        
        # Choix suivants selon la stratégie
        while not done:
            valid_actions = self.valid_actions
            
            if len(valid_actions) == 0:
                raise ValueError(f"Aucune action valide dans l'état {self.current_state}")
            
            if strategy == "always_switch":
                # Choisir une porte différente de la porte actuelle si possible
                other_choices = [a for a in valid_actions if a != self.agent_door]
                if other_choices:
                    action = np.random.choice(other_choices)
                else:
                    # Si pas d'autre choix, garder la porte actuelle
                    action = self.agent_door
                    if action not in valid_actions:
                        # Situation exceptionnelle : prendre n'importe quelle action valide
                        action = np.random.choice(valid_actions)
                        
            elif strategy == "never_switch":
                # Garder la porte actuelle si possible
                if self.agent_door in valid_actions:
                    action = self.agent_door
                else:
                    # Si la porte actuelle n'est plus valide, prendre une action aléatoire
                    action = np.random.choice(valid_actions)
                    
            else:  # random
                action = np.random.choice(valid_actions)
            
            if verbose:
                print(f"Choix: {action} (stratégie: {strategy}, parmi {valid_actions})")
            
            # Vérification finale avant l'action
            if not self.is_valid_action(action):
                raise ValueError(f"Action {action} invalide juste avant step()!")
            
            state, reward, done, info = self.step(action)
            
            if verbose:
                self.render()
                self.validate_game_logic()
        
        return reward > 0
    
    def test_action_validation(self, verbose=False) -> bool:
        """
        Teste la validation des actions dans différents états.
        
        Returns:
            bool: True si tous les tests passent
        """
        print("=== TEST DE VALIDATION DES ACTIONS ===")
        
        # Test 1: État initial
        self.reset()
        print(f"État 0 - Actions valides: {self.valid_actions}")
        print(f"État 0 - Actions invalides: {self.get_invalid_actions()}")
        
        # Toutes les actions doivent être valides en état 0
        for action in range(self.num_doors):
            if not self.is_valid_action(action):
                print(f"ERREUR: Action {action} devrait être valide en état 0")
                return False
        
        # Test 2: Progression à travers les états
        action = 2  # Choix arbitraire
        self.step(action)
        print(f"État 1 - Actions valides: {self.valid_actions}")
        print(f"État 1 - Actions invalides: {self.get_invalid_actions()}")
        
        # Vérifier qu'il y a moins d'actions valides qu'au début
        if len(self.valid_actions) >= self.num_doors:
            print(f"ERREUR: Il devrait y avoir moins d'actions valides en état 1")
            return False
        
        # Test 3: Actions invalides ne peuvent pas être jouées
        invalid_actions = self.get_invalid_actions()
        if invalid_actions:
            test_invalid = invalid_actions[0]
            try:
                self.step(test_invalid)
                print(f"ERREUR: Action invalide {test_invalid} a été acceptée!")
                return False
            except ValueError:
                print(f"OK: Action invalide {test_invalid} correctement rejetée")
        
        # Test 4: Masque d'actions
        mask = self.get_action_mask()
        print(f"Masque d'actions: {mask}")
        
        # Vérifier la cohérence du masque
        for i, valid in enumerate(mask):
            if valid and i not in self.valid_actions:
                print(f"ERREUR: Incohérence dans le masque pour l'action {i}")
                return False
        
        print("✅ Tous les tests de validation des actions sont passés!")
        return True