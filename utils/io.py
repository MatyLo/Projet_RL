import numpy as np

def save_model(pi, V, Q, path="models/lineworld_policy_iteration"):
    np.save(f"{path}_pi.npy", pi)
    np.save(f"{path}_V.npy", V)
    np.save(f"{path}_Q.npy", Q)

def load_model(path="models/lineworld_policy_iteration"):
    pi = np.load(f"{path}_pi.npy")
    V = np.load(f"{path}_V.npy")
    Q = np.load(f"{path}_Q.npy")
    return pi, V, Q
