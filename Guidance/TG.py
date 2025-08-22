import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def TG_find_target(position, path, is_loop):
    P   = np.asarray(path, dtype=float)
    pos = np.asarray(position, dtype=float)
    N   = P.shape[0]

    d = np.sqrt((P[:,0] - pos[0])**2 + (P[:,1] - pos[1])**2)
    i0 = int(np.argmin(d))
    
    if is_loop:
        i_prev = (i0 - 1) % N
        i_next = (i0 + 1) % N
    else:
        i_prev = max(i0 - 1, 0)
        i_next = min(i0 + 1, N - 1)
            
    heading_path = np.atan2(P[i_next, 1] - P[i_prev, 1], P[i_next, 0] - P[i_prev, 0])
    return P[i0, :], heading_path
 
def TG_Guidance(state, target_wp, heading_path, Kp, K, kappa_ff=0):
    n,e,psi,v = state
    nt, et = float(target_wp[0]), float(target_wp[1])
    psi_p   = float(heading_path)
    
    c, s = np.cos(psi_p), np.sin(psi_p)
    Rm   = np.array([[ c,  s],
                     [-s,  c]])
    LOS   = np.array([n - nt, e - et])    # Line of Sight
    r_path = Rm @ LOS
    
    psi_e = psi - psi_p
    psi_e = np.arctan2(np.sin(psi_e), np.cos(psi_e))
    
    ey   = r_path[1] 
    dy = v*np.sin(psi_e)
    dx = v*np.cos(psi_e)
    omega = -Kp*(K*dy - ey*dx)

    #ex   = r_path[1]
    #ey   = r_path[1] 
    #ey_d = v*psi_e

    ##omega = v * kappa_ff - (Kp * ey + Kd * ey_d)/ex
    #omega = psi_e + (Kp * ey + Kd * ey_d)/ex
    return omega
    
