import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def TG_find_target(position, path, is_loop):
    P   = np.asarray(path, dtype=float)
    pos = np.asarray(position, dtype=float)
    N   = P.shape[0]

    # 가장 가까운 점
    d  = np.sqrt((P[:,0] - pos[0])**2 + (P[:,1] - pos[1])**2)
    i0 = int(np.argmin(d))

    # ★ 진행 방향을 강제
    if is_loop:
        i_fwd = (i0 + 1) % N
        seg_a, seg_b = P[i0], P[i_fwd]
    else:
        if i0 < N - 1:
            seg_a, seg_b = P[i0], P[i0 + 1]   # 앞으로
        else:
            seg_a, seg_b = P[i0 - 1], P[i0]   # 마지막 구간만 뒤에서 앞으로 접근

    target, _ = proj_point(seg_a, seg_b, pos)

    heading_path = np.atan2(seg_b[1] - seg_a[1], seg_b[0] - seg_a[0])
    heading_path = np.atan2(np.sin(heading_path), np.cos(heading_path))  # wrap
    # print(target, heading_path)

    return target, heading_path
 
def proj_point(a, b, p):
    ab = b - a
    n2 = float(np.dot(ab, ab))
    if n2 < 1e-12:  # degenerate segment
        return a, 0.0
    t = np.dot(p - a, ab) / n2
    t = np.clip(t, 0.0, 1.0)
    q = a + t * ab
    return q, t

def TG_Guidance(state, target_wp, heading_path, Kp, K, kappa_ff=0):
    n,e,psi,v = state
    nt, et = float(target_wp[0]), float(target_wp[1])
    psi_p   = float(heading_path)
    
    c, s = np.cos(psi_p), np.sin(psi_p)
    Rm   = np.array([[ c,  s],
                     [-s,  c]])
    LOS   = np.array([nt - n, et - e])    # Line of Sight
    r_path = Rm @ LOS
    
    psi_e = psi - psi_p
    psi_e = np.atan2(np.sin(psi_e), np.cos(psi_e))
    
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
    
