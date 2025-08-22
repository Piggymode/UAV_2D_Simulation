import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def NLPF_find_target(position, path, L1, is_loop: bool):
    P   = np.asarray(path, dtype=float)   # shape (N,2) [n,e]
    pos = np.asarray(position, dtype=float)
    N   = P.shape[0]

    d  = np.hypot(P[:,0] - pos[0], P[:,1] - pos[1])
    i0 = int(np.argmin(d))

    cand = np.flatnonzero(d <= float(L1))
    if cand.size == 0:
        return P[i0, :]

    if not is_loop:
        fwd = cand[cand >= i0]
        target_idx = int(fwd.max()) if fwd.size else int(cand.max())
        return P[target_idx, :]

    rel  = (cand - i0) % N
    half = N // 2

    mask = rel <= half
    if np.any(mask):
        cand2, rel2 = cand[mask], rel[mask]
    else:
        cand2, rel2 = cand, rel

    target_idx = int(cand2[np.argmax(rel2)])
    return P[target_idx, :]
 
def NLPF_Guidance(state, target_wp):
    n,e,psi,v = state
    
    L1 = np.linalg.norm(target_wp - np.array([n, e]))
    
    eta = np.arctan2(target_wp[1] - e, target_wp[0] - n) -psi
    #eta = np.arctan2(np.sin(eta), np.cos(eta))
    #eta = np.clip(np.arctan2(np.sin(eta), np.cos(eta)), -np.pi/2, np.pi/2)
    
    omega = (2 * v / L1) * np.sin(eta)
    return omega
