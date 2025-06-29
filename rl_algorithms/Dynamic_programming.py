import numpy as np
from environments.line_world import LineWorldEnvDP

def policy_iteration(env: LineWorldEnvDP, theta=1e-5, gamma=0.999999):
    S = env.get_states()
    A = env.get_actions()
    R = env.get_rewards()
    T = env.get_terminal_states()
    p = env.get_transition_probabilities()

    V = np.random.random(len(S))
    V[T] = 0.0

    pi = np.array([np.random.choice(A) for _ in S])
    pi[T] = 0  # Politique triviale pour états terminaux

    iteration_count = 0  # compteur

    while True:
        iteration_count += 1  # nouvelle itération de policy improvement
        # Policy Evaluation
        while True:
            delta = 0.0
            for s in S:
                v = V[s]
                total = 0.0
                for s_p in S:
                    for r_idx in range(len(R)):
                        r = R[r_idx]
                        total += p[s, pi[s], s_p, r_idx] * (r + gamma * V[s_p])
                V[s] = total
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in S:
            old_action = pi[s]
            best_action = None
            best_score = float('-inf')

            for a in A:
                score = 0.0
                for s_p in S:
                    for r_idx in range(len(R)):
                        r = R[r_idx]
                        score += p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
                if best_action is None or score > best_score:
                    best_action = a
                    best_score = score

            if best_action != old_action:
                policy_stable = False
            pi[s] = best_action

        if policy_stable:
            break

    return pi, V, iteration_count


def compute_action_value_function(env, V, gamma=0.999999):
    S = env.get_states()
    A = env.get_actions()
    R = env.get_rewards()
    p = env.get_transition_probabilities()

    Q = np.zeros((len(S), len(A)))

    for s in S:
        for a in A:
            total = 0.0
            for s_p in S:
                for r_idx, r in enumerate(R):
                    total += p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
            Q[s, a] = total
    return Q
