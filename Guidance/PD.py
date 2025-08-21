import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def find_target(position, path, L1, is_loop):
    P   = np.asarray(path, dtype=float)
    pos = np.asarray(position, dtype=float)
    N   = P.shape[0]

    d = np.sqrt((P[:,0] - pos[0])**2 + (P[:,1] - pos[1])**2)
    i0 = int(np.argmin(d))
    
    if is_loop:
        if i0 == 0:
            i_prev = N-1;  i_cur  = i0; i_next = i0+1 
        elif i0 == N-1:
            i_prev = i0-1; i_cur  = i0; i_next = 0 
        else:
            i_prev = i0-1; i_cur  = i0; i_next = i0+1 
    else:
        if i0 == 0:
            i_prev = i0;  i_cur  = i0; i_next = i0+1 
        elif i0 == N-1:
            i_prev = i0-1; i_cur  = i0; i_next = i0
        else:
            i_prev = i0-1; i_cur  = i0; i_next = i0+1 

    return P[i_prev, :], P[i_cur, :], P[i_next, :]
 
def PD_Guidance(state, wp_prev, wp_cur, wp_next):
    n,e,psi,v = state
    dN = wp_next(1) - wp_prev(1)
    dE = wp_next(2) - wp_prev(2)
    psi_des = np.arctan2(dE, dN)
    L1 = np.linalg.norm(target_wp - np.array([n, e]))
    #print(L1)
    eta = np.arctan2(target_wp[1] - e, target_wp[0] - n) -psi
    eta = np.arctan2(np.sin(eta), np.cos(eta))
    omega = (2 * v / L1) * np.sin(eta)
    return omega
