import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from src.rl_environments.base_environment import BaseEnvironment

class MontyHall2(BaseEnvironment):
    """
    Environnement Monty Hall Level 2 - Version déterministe pour Dynamic Programming
    
    5 portes disponibles, 4 actions successives de l'agent:
    1. Choix initial (5 portes) → Élimination de 1 porte → 4 portes restantes
    2. Maintenir ou changer après 1ère élimination (4→3 portes)  
    3. Maintenir ou changer après 2ème élimination (3→2 portes)
    4. Choix final entre les 2 portes restantes
    
    États encodés simplement: step_count (0-4)
    Actions: 
    - État 0: actions 0-4 (choix de porte)
    - États 1,2,3: actions 0-1 (0=maintenir, 1=changer)
    """
    
    def __init__(self, n_doors: int = 5, deterministic: bool = True):
        super().__init__("MontyHall2")
        self.n_doors = n_doors
        self.deterministic = deterministic
        
        # Configuration pour BaseAlgorithm
        self._state_space_size = 5  # États 0,1,2,3,4
        self._action_space_size = max(n_doors, 2)  # Max entre n_doors et 2 pour compatibilité
        
        # Précalcul des matrices pour DP
        self._transition_probs = None
        self._reward_matrix = None
        self._precompute_dp_matrices()
        
        self.reset()
    
    def reset(self) -> int:
        """Réinitialise l'environnement"""
        self.step_count = 0
        if self.deterministic:
            # Pour DP: porte gagnante fixe (symétrie des résultats)
            self.winning_door = 0
        else:
            self.winning_door = random.randint(0, self.n_doors - 1)
        
        self.agent_choice = None
        self.eliminated_doors = set()
        self.remaining_doors = set(range(self.n_doors))
        self.done = False
        
        return self.step_count
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Exécute une action dans l'environnement"""
        info = {}
        reward = 0.0
        
        if self.done:
            return self.step_count, 0.0, True, info
        
        if self.step_count == 0:  # Choix initial des portes
            if action >= self.n_doors or action < 0:
                return self.step_count, -0.1, False, {'error': 'invalid_action'}
            
            self.agent_choice = action
            # Élimination déterministe d'une porte
            eliminated = self._eliminate_doors_deterministic(1)
            self.eliminated_doors.update(eliminated)
            self.remaining_doors -= eliminated
            
            info.update({
                'agent_choice': self.agent_choice,
                'eliminated': list(eliminated),
                'remaining': list(self.remaining_doors),
                'step': self.step_count
            })
            
        elif self.step_count in [1, 2, 3]:  # Décisions maintenir/changer
            # Mapper toutes les actions sur 0/1
            mapped_action = action % 2
            
            if mapped_action == 1:  # Changer
                other_doors = list(self.remaining_doors - {self.agent_choice})
                if other_doors:
                    if self.deterministic:
                        # Choix déterministe: plus petite porte disponible
                        self.agent_choice = min(other_doors)
                    else:
                        self.agent_choice = random.choice(other_doors)
            
            # Actions spécifiques selon l'étape
            if self.step_count < 3:  # Étapes 1 et 2: éliminer une porte
                eliminated = self._eliminate_doors_deterministic(1)
                self.eliminated_doors.update(eliminated)
                self.remaining_doors -= eliminated
            else:  # Étape 3: décision finale
                self.done = True
                reward = 1.0 if self.agent_choice == self.winning_door else 0.0
            
            info.update({
                'agent_choice': self.agent_choice,
                'eliminated': list(eliminated) if self.step_count < 3 else [],
                'remaining': list(self.remaining_doors),
                'step': self.step_count,
                'switched': mapped_action == 1,
                'result': 'win' if reward == 1.0 else ('lose' if self.done else 'continue'),
                'mapped_action': mapped_action,
                'original_action': action
            })
        
        self.step_count += 1
        return self.step_count, reward, self.done, info
    
    def _eliminate_doors_deterministic(self, count: int) -> set:
        """Élimination déterministe des portes (pour DP)"""
        candidates = [d for d in self.remaining_doors 
                     if d != self.agent_choice and d != self.winning_door]
        
        count = min(count, len(candidates))
        if count <= 0:
            return set()
        
        if self.deterministic:
            # Élimination déterministe: plus petits indices
            return set(sorted(candidates)[:count])
        else:
            return set(random.sample(candidates, count))
    
    def _precompute_dp_matrices(self):
        """Précalcule les matrices pour Dynamic Programming"""
        states = self._state_space_size
        actions = self._action_space_size
        
        # Matrice de transition P[s,a,s']
        self._transition_probs = np.zeros((states, actions, states))
        # Matrice de récompenses R[s,a]
        self._reward_matrix = np.zeros((states, actions))
        
        # État 0 -> État 1 (choix initial)
        for action in range(min(actions, self.n_doors)):
            self._transition_probs[0, action, 1] = 1.0
            self._reward_matrix[0, action] = 0.0
        
        # Actions invalides à l'état 0
        for action in range(self.n_doors, actions):
            self._transition_probs[0, action, 0] = 1.0  # Reste dans l'état
            self._reward_matrix[0, action] = -0.1
        
        # État 1 -> État 2 (première décision)
        for action in range(actions):
            self._transition_probs[1, action, 2] = 1.0
            self._reward_matrix[1, action] = 0.0
        
        # État 2 -> État 3 (deuxième décision)
        for action in range(actions):
            self._transition_probs[2, action, 3] = 1.0
            self._reward_matrix[2, action] = 0.0
        
        # État 3 -> État 4 (décision finale avec récompense)
        for action in range(actions):
            self._transition_probs[3, action, 4] = 1.0
            # Récompense selon la stratégie optimale
            # Si on change toujours, probabilité de victoire ≈ 4/5 = 0.8
            mapped_action = action % 2
            if mapped_action == 1:  # Changer
                self._reward_matrix[3, action] = 0.8  # Probabilité théorique
            else:  # Maintenir
                self._reward_matrix[3, action] = 0.2  # Probabilité théorique
        
        # État 4 -> État 4 (terminal)
        for action in range(actions):
            self._transition_probs[4, action, 4] = 1.0
            self._reward_matrix[4, action] = 0.0
    
    # ============ PROPRIÉTÉS pour compatibilité BaseAlgorithm ============
    
    @property
    def state_space_size(self) -> int:
        """Nombre d'états possibles"""
        return self._state_space_size
    
    @property
    def action_space_size(self) -> int:
        """Nombre d'actions possibles"""
        return self._action_space_size
    
    @property
    def valid_actions(self) -> List[int]:
        """Actions valides selon l'état actuel"""
        if self.step_count == 0:
            return list(range(self.n_doors))
        else:
            return [0, 1]  # maintenir, changer
    
    def get_state_space(self) -> List[int]:
        """Retourne la liste des états possibles"""
        return list(range(self.state_space_size))
    
    def get_action_space(self) -> List[int]:
        """Retourne la liste des actions possibles"""
        return list(range(self.action_space_size))
    
    def get_rewards(self) -> List[float]:
        """Retourne les récompenses possibles"""
        return [-0.1, 0.0, 0.2, 0.8, 1.0]
    
    def get_terminal_states(self) -> List[int]:
        """Retourne les états terminaux"""
        return [4]
    
    # ============ MÉTHODES pour Dynamic Programming ============
    
    def get_transition_probabilities(self, state: int, action: int) -> Dict[int, float]:
        """
        Retourne les probabilités de transition P(s'|s,a)
        
        Args:
            state: État actuel
            action: Action prise
            
        Returns:
            Dict {next_state: probability}
        """
        if (self._transition_probs is None or 
            state >= self.state_space_size or 
            action >= self.action_space_size):
            return {}
        
        transitions = {}
        for next_state in range(self.state_space_size):
            prob = self._transition_probs[state, action, next_state]
            if prob > 0:
                transitions[next_state] = prob
        
        return transitions
    
    def get_reward(self, state: int, action: int) -> float:
        """
        Retourne la récompense R(s,a)
        
        Args:
            state: État actuel
            action: Action prise
            
        Returns:
            Récompense attendue
        """
        if (self._reward_matrix is None or 
            state >= self.state_space_size or 
            action >= self.action_space_size):
            return 0.0
        
        return self._reward_matrix[state, action]
    
    def get_transition_matrix(self) -> np.ndarray:
        """Retourne la matrice de transition complète P[s,a,s']"""
        return self._transition_probs.copy() if self._transition_probs is not None else None
    
    def get_reward_matrix(self) -> np.ndarray:
        """Retourne la matrice de récompenses R[s,a]"""
        return self._reward_matrix.copy() if self._reward_matrix is not None else None
    
    def is_terminal(self, state: int) -> bool:
        """Vérifie si un état est terminal"""
        return state in self.get_terminal_states()
    
    # ============ MÉTHODES d'affichage et debug ============
    
    def render(self, mode: str = 'console'):
        """Affiche l'état actuel de l'environnement"""
        if mode == 'console':
            print(f"=== Monty Hall Level 2 ===")
            print(f"État: {self.step_count} - {self.get_state_description(self.step_count)}")
            print(f"Portes restantes: {sorted(self.remaining_doors)} (total: {len(self.remaining_doors)})")
            print(f"Portes éliminées: {sorted(self.eliminated_doors)}")
            print(f"Choix actuel de l'agent: {self.agent_choice}")
            
            if self.done:
                print(f"Porte gagnante: {self.winning_door}")
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
    
    def get_optimal_policy_hint(self) -> Dict[str, str]:
        """Indices sur la politique optimale"""
        return {
            'state_0': 'Choisir n\'importe quelle porte (actions 0-4, probabilité identique)',
            'state_1': 'CHANGER recommandé (action 1)',
            'state_2': 'CHANGER recommandé (action 1)',
            'state_3': 'CHANGER recommandé (action 1)',
            'general': 'La stratégie optimale est de changer à chaque décision',
            'probabilité': 'Probabilité de victoire optimale: 80% (4/5)',
            'mapping': 'Actions paires → maintenir (0), Actions impaires → changer (1)'
        }
    
    def test_environment(self, num_games: int = 1000, verbose: bool = False) -> Dict[str, float]:
        """
        Test de l'environnement avec différentes stratégies
        
        Args:
            num_games: Nombre de parties à simuler
            verbose: Affichage détaillé
            
        Returns:
            Statistiques des stratégies testées
        """
        strategies = {
            'always_keep': [0, 0, 0, 0],      # Toujours maintenir
            'always_switch': [0, 1, 1, 1],   # Toujours changer après choix initial
            'random': None                     # Choix aléatoires
        }
        
        results = {}
        
        for strategy_name, actions in strategies.items():
            wins = 0
            
            for _ in range(num_games):
                state = self.reset()
                done = False
                
                step = 0
                while not done and step < 4:
                    if strategy_name == 'random':
                        if step == 0:
                            action = random.randint(0, 4)
                        else:
                            action = random.randint(0, 1)
                    else:
                        if step == 0:
                            action = 0  # Choix initial arbitraire
                        else:
                            action = actions[step]
                    
                    state, reward, done, _ = self.step(action)
                    if done and reward == 1.0:
                        wins += 1
                    step += 1
            
            win_rate = wins / num_games
            results[strategy_name] = win_rate
            
            if verbose:
                print(f"Stratégie '{strategy_name}': {win_rate:.1%} de victoires")
        
        return results


# Tests et validation
if __name__ == "__main__":
    print("=== Test de l'environnement MontyHall2 ===")
    
    # Test basique
    env = MontyHall2(deterministic=True)
    print(f"Espace d'états: {env.state_space_size}")
    print(f"Espace d'actions: {env.action_space_size}")
    
    # Test d'une partie complète avec stratégie optimale
    print("\n=== Partie avec stratégie optimale ===")
    state = env.reset()
    env.render()
    
    # Choix initial
    state, reward, done, info = env.step(0)
    print(f"Action 1 (choix porte 0): reward={reward}, info={info}")
    env.render()
    
    # Toujours changer
    for i in [1, 2, 3]:
        state, reward, done, info = env.step(1)  # Changer
        print(f"Action {i+1} (changer): reward={reward}, info={info}")
        env.render()
        if done:
            break
    
    # Test des stratégies
    print("\n=== Test des stratégies sur 1000 parties ===")
    results = env.test_environment(1000, verbose=True)
    
    # Vérification des matrices DP
    print(f"\n=== Vérification matrices DP ===")
    P = env.get_transition_matrix()
    R = env.get_reward_matrix()
    print(f"Matrice de transition: {P.shape}")
    print(f"Matrice de récompenses: {R.shape}")
    
    # Test transitions spécifiques
    print(f"\nTransitions depuis état 0:")
    for action in range(min(5, env.action_space_size)):
        transitions = env.get_transition_probabilities(0, action)
        reward = env.get_reward(0, action)
        print(f"  Action {action}: {transitions}, reward={reward}")