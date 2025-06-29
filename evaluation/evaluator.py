import numpy as np

def evaluate_policy(env, policy, n_episodes=100, gamma=0.999999, start_state=2):
    returns = []
    for _ in range(n_episodes):
        s = start_state
        G = 0.0
        t = 0
        while s not in env.get_terminal_states():
            a = policy[s]
            found = False
            for s_p in env.S:
                for r_idx, r in enumerate(env.R):
                    if env.p[s, a, s_p, r_idx] > 0:
                        G += (gamma ** t) * r
                        s = s_p
                        t += 1
                        found = True
                        break
                if found:
                    break
        returns.append(G)
    mean_return = np.mean(returns)
    print(f"Reward moyen sur {n_episodes} épisodes : {mean_return:.2f}")
    return mean_return


def simulate_episode(env, policy, start_state=2, gamma=0.999999):
    """
    Simule un épisode complet avec une politique donnée.
    Retourne :
    - états visités
    - actions prises
    - récompenses reçues
    - retour cumulé G
    """
    s = start_state
    states = [s]
    actions = []
    rewards = []
    G = 0.0
    t = 0

    while s not in env.get_terminal_states():
        a = policy[s]
        actions.append(a)
        found = False
        for s_p in env.S:
            for r_idx, r in enumerate(env.R):
                if env.p[s, a, s_p, r_idx] > 0:
                    rewards.append(r)
                    G += (gamma ** t) * r
                    s = s_p
                    states.append(s)
                    t += 1
                    found = True
                    break
            if found:
                break

    return states, actions, rewards, G
