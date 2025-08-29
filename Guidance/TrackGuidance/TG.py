# TG.py
import numpy as np

def TG_Guidance(state, triplet, Kp, K):
    n,e,psi,v = map(float, state)
    psi_p, ey, q = TG_target_from_triplet(state, triplet)
    psi_e = wrap_pi(psi - psi_p)
    dy = v*np.sin(psi_e)
    dx = v*np.cos(psi_e)
    omega = -Kp*(K*dy - float(ey)*dx)
    return float(omega)

def TG_target_from_triplet(state, triplet):
    n, e, psi, v = map(float, state)
    prev, cur, _ = triplet
    typ = cur[0]

    if typ == "LINE":
        if prev is not None: 
           sN, sE,= prev[1], prev[2]
        else:
            sN, sE = float(state[0]), float(state[1])
        eN, eE = cur[1], cur[2]
        a = np.array([sN, sE], float); b = np.array([eN, eE], float); p = np.array([n, e], float)
        ab = b - a
        n2 = float(np.dot(ab, ab))
        if n2 < 1e-12:
            q = a
        t = np.clip(np.dot(p - a, ab) / n2, 0.0, 1.0)
        q = a + t * ab
    
        psi_p = np.arctan2(b[1]-a[1], b[0]-a[0])

    elif typ == "HOLD":
        _, cN, cE, R, d, _dur = cur
        q, psi_p = nearest_on_hold([n, e], [float(cN), float(cE)], float(R), int(d))

    elif typ == "RACETRACK":
        _, cN, cE, R, Ls, chi_deg, d, _lod = cur
        q, psi_p = nearest_on_racetrack(
            p_NE=[n, e],
            center_NE=[float(cN), float(cE)],
            R=float(R),
            Ls=float(Ls),
            chi=np.deg2rad(float(chi_deg)),
            d=int(d),
        )
    else:
        raise ValueError(f"Unknown leg type: {typ}")

    dqN, dqE = (q[0]-n), (q[1]-e)
    ey = (-np.sin(psi_p))*dqN + (np.cos(psi_p))*dqE
    return float(psi_p), float(ey), np.asarray(q, float)

def nearest_on_hold(p_NE, center_NE, R, d):
    p = np.asarray(p_NE, float)
    C = np.asarray(center_NE, float)
    v = p - C
    r = np.linalg.norm(v)
    if r < 1e-9:
        theta = 0.0
        q = C + np.array([R, 0.0])
    else:
        theta = np.arctan2(v[1], v[0])
        q = C + (R / r) * v
    psi_p = wrap_pi(theta + (np.pi/2 if int(d) == +1 else -np.pi/2))
    return q, psi_p

def nearest_on_racetrack(p_NE, center_NE, R, Ls, chi, d):
    C = np.asarray(center_NE, float)
    p = np.asarray(p_NE, float)
    c, s = np.cos(chi), np.sin(chi)

    # 글로벌→트랙(x:+chi, y:오른쪽+)
    x =  c*(p[0]-C[0]) + s*(p[1]-C[1])
    y = -s*(p[0]-C[0]) + c*(p[1]-C[1])

    # 직선 후보 (y=±R, x∈[-Ls/2, Ls/2])
    xcl = np.clip(x, -Ls/2.0, Ls/2.0)
    q_top = np.array([xcl, +R])
    q_bot = np.array([xcl, -R])
    # 진행방향: d=+1(CW) → 위:+x(0), 아래:-x(π); d=-1(CCW) → 위:-x(π), 아래:+x(0)
    psi_top_local = np.pi if int(d) > 0 else 0.0
    psi_bot_local = 0.0 if int(d) > 0 else np.pi

    # 각도 클램프(안정)
    def clamp_theta(theta, lo, hi):
        width = (hi - lo) % (2*np.pi)
        if width == 0: width = 2*np.pi
        a = (theta - lo) % (2*np.pi)
        if a <= width:
            return (lo + a) % (2*np.pi)
        # 밖이면 가까운 경계로 스냅
        dist_to_lo = (2*np.pi - a) % (2*np.pi)
        dist_to_hi = (a - width) % (2*np.pi)
        return lo if dist_to_lo <= dist_to_hi else hi

    # 반원 후보
    thR = clamp_theta(np.arctan2(y, x - (+Ls/2.0)), -np.pi/2, +np.pi/2)     # 오른쪽
    thL = clamp_theta(np.arctan2(y, x - (-Ls/2.0)), +np.pi/2, +3*np.pi/2)    # 왼쪽

    qR = np.array([+Ls/2.0 + R*np.cos(thR), 0.0 + R*np.sin(thR)])
    qL = np.array([-Ls/2.0 + R*np.cos(thL), 0.0 + R*np.sin(thL)])

    # 접선 헤딩(연속): ψ_local = θ − d·π/2
    psiR_local = thR + int(d)*(np.pi/2)
    psiL_local = thL + int(d)*(np.pi/2)

    # 최근접 기본 선택
    cand  = [("top", q_top, psi_top_local), ("bot", q_bot, psi_bot_local),
             ("arcR", qR, psiR_local),      ("arcL", qL, psiL_local)]
    dists = [np.hypot(x - q[0], y - q[1]) for _, q, _ in cand]
    name, q_loc, psi_local = cand[int(np.argmin(dists))]

    # 접점 히스테리시스: 접점에선 '직선' 강제 선택
    epsJ = max(2.0, 0.01*float(R))  # 2 m or 1% of R
    J = [np.array([+Ls/2.0, +R]), np.array([+Ls/2.0, -R]),
         np.array([-Ls/2.0, +R]), np.array([-Ls/2.0, -R])]
    dj = [np.hypot(x - Jk[0], y - Jk[1]) for Jk in J]
    jmin = int(np.argmin(dj))
    if (dj[jmin] <= epsJ) and name.startswith("arc"):
        if jmin in (0, 2):   # 위쪽(y=+R) 직선
            q_loc, psi_local = np.array([xcl, +R]), psi_top_local
        else:                # 아래쪽(y=-R) 직선
            q_loc, psi_local = np.array([xcl, -R]), psi_bot_local

    psi_p = wrap_pi(chi + psi_local)

    # 로컬→글로벌
    qN = C[0] + c*q_loc[0] - s*q_loc[1]
    qE = C[1] + s*q_loc[0] + c*q_loc[1]
    return np.array([qN, qE]), float(psi_p)


def wrap_pi(a: float) -> float:
    return np.arctan2(np.sin(a), np.cos(a))
