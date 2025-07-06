import numpy as np

class MontyHall:
    """
    Environnement Monty Hall 'paradox' level 1 :
    - 3 portes (A, B, C)
    - L'agent choisit une porte
    - Une porte non gagnante est retirée
    - L'agent choisit de rester ou changer
    - Récompense 1.0 si la porte finale est gagnante, 0.0 sinon
    """
    def __init__(self):
        self.S = [0, 1, 2, 3, 4]  # 0: choix initial, 1: après retrait, 2: terminal win, 3: terminal lose, 4: terminal (abandon)
        self.A = [0, 1, 2]  # 0,1,2: choix de porte, 0: rester, 1: changer (après retrait)
        self.R = [0.0, 1.0]
        self.T = [2, 3, 4]
        self.state = 0
        self.p = None  # Non utilisé ici, transitions déterministes
        self._init_episode_vars()

    def _init_episode_vars(self):
        self.doors = [0, 1, 2]  # 0:A, 1:B, 2:C
        self.winning_door = np.random.choice(self.doors)
        self.agent_first_choice = None
        self.opened_door = None
        self.agent_final_choice = None
        self.state = 0

    def reset(self):
        self._init_episode_vars()
        return self.state

    def step(self, action):
        # État 0 : choix initial
        if self.state == 0:
            self.agent_first_choice = action  # action = 0, 1, 2 (choix de porte)
            # Retirer une porte non gagnante et non choisie
            possible_doors = [d for d in self.doors if d != self.agent_first_choice and d != self.winning_door]
            self.opened_door = np.random.choice(possible_doors)
            self.state = 1
            return self.state, 0.0, False, {}
        # État 1 : choix rester/changer
        elif self.state == 1:
            if action == 0:  # rester
                self.agent_final_choice = self.agent_first_choice
            elif action == 1:  # changer
                # Prendre la porte restante (ni choisie, ni ouverte)
                self.agent_final_choice = [d for d in self.doors if d != self.agent_first_choice and d != self.opened_door][0]
            # Terminal : win ou lose
            if self.agent_final_choice == self.winning_door:
                self.state = 2
                return self.state, 1.0, True, {"result": "win"}
            else:
                self.state = 3
                return self.state, 0.0, True, {"result": "lose"}
        else:
            # Déjà terminal
            return self.state, 0.0, True, {"result": "terminal"}

    def render(self):
        if self.state == 0:
            print(f"Portes: [A, B, C] (0, 1, 2)")
            print("Choisissez une porte (action = 0, 1 ou 2)")
        elif self.state == 1:
            print(f"Porte ouverte: {self.opened_door} (non gagnante)")
            print("Action: 0 = rester, 1 = changer")
        elif self.state == 2:
            print("Gagné ! La porte était la bonne.")
        elif self.state == 3:
            print("Perdu. La porte n'était pas la bonne.")
        else:
            print("État terminal.")

    def get_state_space(self):
        return self.S

    def get_action_space(self):
        return self.A

    def get_terminal_states(self):
        return self.T
    
    def get_rewards(self):
        """Retourne l'espace des récompenses."""
        return self.R
    
    def get_transition_matrix(self):
        """Retourne la matrice de transition P(s,a,s',r)."""
        # Créer une matrice de transition 4D: P(s,a,s',r)
        n_states = len(self.S)
        n_actions = len(self.A)
        n_rewards = len(self.R)
        
        P = np.zeros((n_states, n_actions, n_states, n_rewards))
        
        # Remplir la matrice de transition selon les règles du jeu
        for s in self.S:
            for a in self.A:
                if s == 0:  # État initial - choix de porte
                    if a in [0, 1, 2]:  # Choix valide de porte
                        # Toujours passer à l'état 1 après choix
                        P[s, a, 1, 0] = 1.0  # Récompense 0.0
                elif s == 1:  # État après révélation
                    if a == 0:  # Rester (action 0)
                        # Déterminer si on gagne ou perd
                        # Pour simplifier, on suppose une probabilité 1/3 de gagner
                        P[s, a, 2, 1] = 1/3  # Gagner (récompense 1.0)
                        P[s, a, 3, 0] = 2/3  # Perdre (récompense 0.0)
                    elif a == 1:  # Changer (action 1)
                        # Avec changement, probabilité 2/3 de gagner
                        P[s, a, 2, 1] = 2/3  # Gagner (récompense 1.0)
                        P[s, a, 3, 0] = 1/3  # Perdre (récompense 0.0)
                    elif a == 2:  # Action 2 (non utilisée dans cet état)
                        # Rester dans l'état actuel
                        P[s, a, s, 0] = 1.0
                else:  # États terminaux
                    P[s, a, s, 0] = 1.0  # Rester dans l'état terminal
        
        return P 