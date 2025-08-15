import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def find_target(position, path, L1, is_loop):
    P   = np.asarray(path, dtype=float)
    pos = np.asarray(position, dtype=float)
    N   = P.shape[0]

    d = np.sqrt((P[:,0] - pos[0])**2 + (P[:,1] - pos[1])**2)
    i0 = int(np.argmin(d))

    cand = np.where(d <= L1)[0]  
    if cand.size == 0:
        target_Idx = i0
        return P[target_Idx, :]
    else:
        half = N // 2
        if 0 in cand: 
            cand_wrapped = cand[cand <= half]
            if cand_wrapped.size > 0:
                target_Idx = int(np.max(cand_wrapped))
            else:
                target_Idx = int(np.max(cand))
        else:
            target_Idx = int(np.max(cand))
    return P[target_Idx, :]
 
def NLPF_Guidance(state, target_wp):
    n,e,psi,v = state
    L1 = np.linalg.norm(target_wp - np.array([n, e]))
    #print(L1)
    eta = np.arctan2(target_wp[1] - e, target_wp[0] - n) -psi
    eta = np.arctan2(np.sin(eta), np.cos(eta))
    omega = (2 * v / L1) * np.sin(eta)
    return omega
