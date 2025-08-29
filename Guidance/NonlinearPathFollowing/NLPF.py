import numpy as np

from Guidance.NonlinearPathFollowing.Path_Hold import generate_hold
from Guidance.NonlinearPathFollowing.Path_Racetrack import generate_racetrack
from Guidance.NonlinearPathFollowing.Path_Straight import generate_straight_p2p
 
def NLPF_Guidance(state, triplet, L1, num_points=600):
    n,e,psi,v = state
    target_wp, _, _, _ = NLPF_target_from_triplet(state, triplet, L1=200.0, num_points=600)
    L1 = np.clip(np.linalg.norm(target_wp - np.array([n, e])), 50, 200)
    eta = np.atan2(target_wp[1] - e, target_wp[0] - n) -psi
    omega = (2 * v / L1) * np.sin(eta)
    return omega

def NLPF_target_from_triplet(state, triplet, L1, num_points=600):
    path, is_loop, meta = build_local_path_from_triplet(state, triplet, num_points=num_points)
    tgt = NLPF_find_target(state[:2], path, L1, is_loop)
    return tgt, path, is_loop, meta

def build_local_path_from_triplet(state, triplet, num_points=600):
    prev, cur, _ = triplet
    typ = cur[0]

    if typ == "HOLD":
        _, cN, cE, R, d, _dur = cur
        WP_gen, is_loop = generate_hold((float(cN), float(cE)), float(R), int(num_points), int(d))
        meta = {"scenario":"HOLD","params":{"center":(float(cN),float(cE)),"radius":float(R),"num_points":int(num_points),"direction":int(d)}}

    elif typ == "LINE":
        if prev is not None: 
           sN, sE,= prev[1], prev[2]
        else:
            sN, sE = float(state[0]), float(state[1])
        eN, eE = cur[1], cur[2]
        WP_gen, is_loop = generate_straight_p2p((float(sN),float(sE)), (float(eN),float(eE)), int(num_points))
        meta = {"scenario":"LINE","params":{"start":(float(sN),float(sE)),"end":(float(eN),float(eE)),"num_points":int(num_points)}}

    elif typ == "RACETRACK":
        _, cN, cE, R, Ls, chi_deg, d, _lod = cur
        WP_gen, is_loop = generate_racetrack((float(cN),float(cE)), float(R), float(Ls), int(num_points), float(chi_deg), int(d))
        meta = {"scenario":"RACETRACK","params":{"center":(float(cN),float(cE)),"radius":float(R),"length":float(Ls),"num_points":int(num_points),"bearing_deg":float(chi_deg),"direction":int(d)}}
    else:
        raise ValueError(f"Unknown leg type: {typ}")

    return WP_gen, is_loop, meta

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
