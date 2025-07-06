import numpy as np

class MontyHall2:
    """
    Environnement Monty Hall 'paradox' level 2 :
    - 5 portes (0 à 4)
    - L'agent effectue 4 choix successifs (chaque fois il choisit une porte à éliminer)
    - À la fin, il reste 2 portes : l'agent choisit de rester sur sa porte initiale ou de changer
    - Récompense 1.0 si la porte finale est gagnante, 0.0 sinon
    """
    def __init__(self):
        self.S = list(range(7))  # 0: choix init, 1-4: éliminations, 5: choix final, 6: terminal
        self.A = [0, 1, 2, 3, 4]  # Choix de porte à chaque étape
        self.R = [0.0, 1.0]
        self.T = [6]
        self.state = 0
        self.p = None
        self._init_episode_vars()

    def _init_episode_vars(self):
        self.doors = [0, 1, 2, 3, 4]
        self.winning_door = np.random.choice(self.doors)
        self.agent_first_choice = None
        self.eliminated_doors = []
        self.remaining_doors = self.doors.copy()
        self.final_choice = None
        self.state = 0
        self.elimination_step = 0

    def reset(self):
        self._init_episode_vars()
        return self.state

    def step(self, action):
        # État 0 : choix initial
        if self.state == 0:
            self.agent_first_choice = action
            self.state = 1
            return self.state, 0.0, False, {}
        # États 1 à 4 : éliminations successives
        elif 1 <= self.state <= 4:
            if action in self.remaining_doors and action != self.agent_first_choice and action != self.winning_door:
                self.eliminated_doors.append(action)
                self.remaining_doors.remove(action)
                self.elimination_step += 1
                if self.elimination_step < 4:
                    self.state += 1
                    return self.state, 0.0, False, {}
                else:
                    self.state = 5
                    return self.state, 0.0, False, {}
            else:
                # Action invalide (porte déjà éliminée ou porte initiale/gagnante)
                return self.state, 0.0, False, {"error": "Action invalide"}
        # État 5 : choix final (0 = rester, 1 = changer)
        elif self.state == 5:
            if action == 0:
                self.final_choice = self.agent_first_choice
            elif action == 1:
                # Prendre la porte restante qui n'est pas celle de l'agent
                self.final_choice = [d for d in self.remaining_doors if d != self.agent_first_choice][0]
            self.state = 6
            reward = 1.0 if self.final_choice == self.winning_door else 0.0
            return self.state, reward, True, {"result": "win" if reward == 1.0 else "lose"}
        else:
            # Terminal
            return self.state, 0.0, True, {"result": "terminal"}

    def render(self):
        if self.state == 0:
            print(f"Portes: {self.doors}")
            print("Choisissez votre porte initiale (action = 0 à 4)")
        elif 1 <= self.state <= 4:
            print(f"Portes restantes: {self.remaining_doors}")
            print(f"Éliminez une porte non gagnante et non choisie (action = porte à éliminer)")
        elif self.state == 5:
            print(f"Deux portes restantes: {self.remaining_doors}")
            print(f"Votre choix initial: {self.agent_first_choice}")
            print("Action: 0 = rester, 1 = changer")
        elif self.state == 6:
            if self.final_choice == self.winning_door:
                print("Gagné ! La porte était la bonne.")
            else:
                print("Perdu. La porte n'était pas la bonne.")
        else:
            print("État terminal.")

    def get_state_space(self):
        return self.S

    def get_action_space(self):
        return self.A

    def get_terminal_states(self):
        return self.T 