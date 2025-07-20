import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from src.rl_environments.base_environment import BaseEnvironment

class MontyHall2(BaseEnvironment):
    """
    Environnement Monty Hall Level 2 - Version corrigée avec élimination progressive
    
    5 portes disponibles, 4 actions successives de l'agent:
    1. Choix initial (5 portes) → Élimination de 1 porte → 4 portes restantes
    2. Maintenir ou changer après 1ère élimination (4→3 portes)
    3. Maintenir ou changer après 2ème élimination (3→2 portes)
    4. Choix final entre les 2 portes restantes
    
    États:
    0: Choix initial (5 portes)
    1: Après 1ère élimination (4 portes restantes)
    2: Après 2ème élimination (3 portes restantes)
    3: Après 3ème élimination (2 portes restantes)
    4: Partie terminée
    """
    
    def __init__(self, n_doors: int = 5):
        super().__init__("MontyHall2")
        self.n_doors = n_doors
        self.reset()
    
    def reset(self) -> int:
        """Réinitialise l'environnement"""
        self.state = 0
        self.winning_door = random.randint(0, self.n_doors - 1)
        self.agent_choice = None
        self.eliminated_doors = []
        self.remaining_doors = list(range(self.n_doors))
        self.done = False
        self.action_count = 0
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Exécute une action dans l'environnement
        
        État 0: action = numéro de porte (0 à 4)
        États 1,2,3: action = 0 (maintenir) ou 1+ (changer)
        """
        info = {}
        reward = 0.0
        
        if self.done:
            return self.state, 0.0, True, info
        
        # Mapping des actions pour compatibilité avec BaseAlgorithm
        mapped_action = self._map_action(action)
        
        if self.state == 0:  # Action 1: Choix initial
            if mapped_action not in self.remaining_doors:
                # Action invalide même après mapping
                return self.state, -0.1, False, {'error': 'invalid_action'}
            
            self.agent_choice = mapped_action
            self.action_count += 1
            
            # Éliminer 1 seule porte (pas celle de l'agent ni la gagnante)
            eliminated = self._eliminate_doors(1)
            self.eliminated_doors.extend(eliminated)
            for door in eliminated:
                self.remaining_doors.remove(door)
            
            self.state = 1
            info.update({
                'agent_choice': self.agent_choice,
                'eliminated': eliminated,
                'remaining': self.remaining_doors.copy(),
                'remaining_count': len(self.remaining_doors),
                'action_count': self.action_count,
                'original_action': action,
                'mapped_action': mapped_action
            })
            
        elif self.state == 1:  # Action 2: Après 1ère élimination (4 portes)
            mapped_action = action % 2  # Forcer mapping 0/1
            
            if mapped_action == 1:  # Changer
                other_doors = [d for d in self.remaining_doors if d != self.agent_choice]
                if other_doors:
                    self.agent_choice = random.choice(other_doors)
            
            self.action_count += 1
            
            # Éliminer 1 porte supplémentaire
            eliminated = self._eliminate_doors(1)
            self.eliminated_doors.extend(eliminated)
            for door in eliminated:
                self.remaining_doors.remove(door)
            
            self.state = 2
            info.update({
                'agent_choice': self.agent_choice,
                'eliminated': eliminated,
                'remaining': self.remaining_doors.copy(),
                'remaining_count': len(self.remaining_doors),
                'action_count': self.action_count,
                'switched': mapped_action == 1,
                'original_action': action,
                'mapped_action': mapped_action
            })
            
        elif self.state == 2:  # Action 3: Après 2ème élimination (3 portes)
            mapped_action = action % 2  # Forcer mapping 0/1
            
            if mapped_action == 1:  # Changer
                other_doors = [d for d in self.remaining_doors if d != self.agent_choice]
                if other_doors:
                    self.agent_choice = random.choice(other_doors)
            
            self.action_count += 1
            
            # Éliminer 1 porte supplémentaire
            eliminated = self._eliminate_doors(1)
            self.eliminated_doors.extend(eliminated)
            for door in eliminated:
                self.remaining_doors.remove(door)
            
            self.state = 3
            info.update({
                'agent_choice': self.agent_choice,
                'eliminated': eliminated,
                'remaining': self.remaining_doors.copy(),
                'remaining_count': len(self.remaining_doors),
                'action_count': self.action_count,
                'switched': mapped_action == 1,
                'original_action': action,
                'mapped_action': mapped_action
            })
            
        elif self.state == 3:  # Action 4: Choix final (2 portes)
            mapped_action = action % 2  # Forcer mapping 0/1
            
            if mapped_action == 1:  # Changer vers l'autre porte
                other_doors = [d for d in self.remaining_doors if d != self.agent_choice]
                if other_doors:
                    self.agent_choice = other_doors[0]  # Il ne reste qu'une autre porte
            
            self.action_count += 1
            
            # Partie terminée, calculer la récompense
            self.done = True
            self.state = 4
            reward = 1.0 if self.agent_choice == self.winning_door else 0.0
            
            info.update({
                'agent_choice': self.agent_choice,
                'remaining': self.remaining_doors.copy(),
                'remaining_count': len(self.remaining_doors),
                'action_count': self.action_count,
                'switched': mapped_action == 1,
                'result': 'win' if reward == 1.0 else 'lose',
                'winning_door': self.winning_door,
                'final_choice': self.agent_choice,
                'original_action': action,
                'mapped_action': mapped_action
            })
            
        return self.state, reward, self.done, info
    
    def _map_action(self, action: int) -> int:
        """
        Mappe les actions pour compatibilité avec BaseAlgorithm
        
        État 0: actions 0-4 valides (choix de porte)
        États 1,2,3: actions 0-4 mappées sur 0,1 (maintenir/changer)
        """
        if self.state == 0:
            # État initial : actions 0-4 correspondent aux portes
            return action
        elif self.state in [1, 2, 3]:
            # États d'élimination : mapper actions 0-4 sur 0,1
            return action % 2
        else:
            # État terminal : pas d'action
            return 0
    
    def _eliminate_doors(self, count: int) -> List[int]:
        """Élimine 'count' portes qui ne sont ni celle de l'agent ni la gagnante"""
        candidates = [d for d in self.remaining_doors 
                     if d != self.agent_choice and d != self.winning_door]
        
        # S'assurer qu'on ne peut pas éliminer plus de portes que disponible
        count = min(count, len(candidates))
        
        if count <= 0:
            return []
        
        return random.sample(candidates, count)
    
    @property
    def valid_actions(self) -> List[int]:
        """
        Retourne toutes les actions possibles (0-4) pour compatibilité BaseAlgorithm
        Le mapping interne se charge de la conversion
        """
        return list(range(self.action_space_size))
    
    @property
    def action_space_size(self) -> int:
        """Taille de l'espace d'action (maximum entre n_doors et 2)"""
        return max(self.n_doors, 2)
    
    @property
    def state_space_size(self) -> int:
        """Nombre d'états possibles"""
        return 5  # États 0, 1, 2, 3, 4
    
    def render(self, mode: str = 'console'):
        """Affiche l'état actuel de l'environnement"""
        if mode == 'console':
            print(f"=== Monty Hall Level 2 ===")
            print(f"État: {self.state} - {self.get_state_description(self.state)}")
            print(f"Action: {self.action_count}/4")
            print(f"Portes restantes: {self.remaining_doors} (total: {len(self.remaining_doors)})")
            print(f"Portes éliminées: {self.eliminated_doors}")
            print(f"Choix actuel de l'agent: {self.agent_choice}")
            
            if self.done:
                print(f"Porte gagnante: {self.winning_door}")
                print(f"Choix final: {self.agent_choice}")
                result = "GAGNÉ" if self.agent_choice == self.winning_door else "PERDU"
                print(f"Résultat: {result}")
                print("="*30)
    
    def get_state_description(self, state: int) -> str:
        """Description textuelle de l'état"""
        descriptions = {
            0: "Choix initial (5 portes)",
            1: "Après 1ère élimination (4 portes)",
            2: "Après 2ème élimination (3 portes)",
            3: "Après 3ème élimination (2 portes)",
            4: "Partie terminée"
        }
        return descriptions.get(state, "État inconnu")
    
    def get_state_space(self) -> List[int]:
        """Retourne la liste des états possibles"""
        return list(range(self.state_space_size))
    
    def get_action_space(self) -> List[int]:
        """Retourne la liste des actions possibles"""
        return list(range(self.action_space_size))
    
    def get_rewards(self) -> List[float]:
        """Retourne les récompenses possibles"""
        return [-0.1, 0.0, 1.0]  # Pénalité pour action invalide, neutre, victoire
    
    def get_terminal_states(self) -> List[int]:
        """Retourne les états terminaux"""
        return [4]
    
    def get_optimal_policy_hint(self) -> Dict[str, str]:
        """Indices sur la politique optimale"""
        return {
            'state_0': 'Choisir n\'importe quelle porte (actions 0-4, probabilité 1/5)',
            'state_1': 'CHANGER recommandé (4 portes restantes)',
            'state_2': 'CHANGER recommandé (3 portes restantes)',
            'state_3': 'CHANGER recommandé (2 portes restantes)',
            'general': 'La stratégie optimale est de changer à chaque étape',
            'mapping': 'Actions 0,2,4 → maintenir (0), Actions 1,3 → changer (1)',
            'probabilité': 'Probabilité de victoire en changeant toujours ≈ 80%'
        }
    
    def get_transition_matrix(self) -> np.ndarray:
        """Matrice de transition P[s,a,s'] (simplifiée)"""
        states = self.state_space_size
        actions = self.action_space_size
        
        P = np.zeros((states, actions, states))
        
        # État 0 -> État 1 (choix initial)
        for a in range(min(actions, self.n_doors)):
            P[0, a, 1] = 1.0
        
        # État 1 -> État 2 (après 1ère élimination)
        for a in range(actions):
            P[1, a, 2] = 1.0
        
        # État 2 -> État 3 (après 2ème élimination)
        for a in range(actions):
            P[2, a, 3] = 1.0
        
        # État 3 -> État 4 (décision finale)
        for a in range(actions):
            P[3, a, 4] = 1.0
        
        # État 4 -> État 4 (terminal)
        for a in range(actions):
            P[4, a, 4] = 1.0
        
        return P


# Exemple d'utilisation et de test
"""if __name__ == "__main__":
    env = MontyHall2()
    
    # Test d'une partie complète
    print("=== Test d'une partie complète ===")
    state = env.reset()
    env.render()
    
    # Action 1: Choix initial (choisir la porte 0)
    print("\nAction 1: Choix initial - porte 0")
    state, reward, done, info = env.step(0)
    env.render()
    print(f"Info: {info}")
    
    # Action 2: Maintenir ou changer (changer)
    print("\nAction 2: Changer de porte")
    state, reward, done, info = env.step(1)
    env.render()
    print(f"Info: {info}")
    
    # Action 3: Maintenir ou changer (changer)
    print("\nAction 3: Changer de porte")
    state, reward, done, info = env.step(1)
    env.render()
    print(f"Info: {info}")
    
    # Action 4: Choix final (changer)
    print("\nAction 4: Choix final - changer")
    state, reward, done, info = env.step(1)
    env.render()
    print(f"Info: {info}")
    
    print(f"\nRécompense finale: {reward}")
    print(f"Partie terminée: {done}")
    
    # Test de la stratégie optimale sur plusieurs parties
    print("\n=== Test de stratégie optimale (toujours changer) ===")
    wins = 0
    num_games = 1000
    
    for i in range(num_games):
        env.reset()
        # Choix initial aléatoire
        env.step(random.randint(0, 4))
        # Toujours changer aux 3 décisions suivantes
        env.step(1)  # Changer
        env.step(1)  # Changer
        _, reward, _, _ = env.step(1)  # Changer
        
        if reward == 1.0:
            wins += 1
    
    win_rate = wins / num_games
    print(f"Taux de victoire avec stratégie 'toujours changer': {win_rate:.2%}")
    print(f"Attendu théorique: ~80% (4/5)")"""