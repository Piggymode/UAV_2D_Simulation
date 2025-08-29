import numpy as np

from Mission_Planning.Path_Hold import generate_hold
from Mission_Planning.Path_Racetrack import generate_racetrack
from Mission_Planning.Path_Straight import generate_straight_p2p

def mission_planner_step(t, pos_NE, psi, mission_state, mission_spec):
    """
    역할:
      - (prev, cur, next) 3-튜플만 산출한다.
      - 경로 점찍기/shape discretization은 하지 않는다. (NLPF/TG에서 처리)

    Inputs
      t            : [s] guidance time (monotonic)
      pos_NE       : np.array([n, e])
      psi          : [rad] heading (북=0, 시계방향 +)
      mission_state: [leg_idx, enter_time, aux]  (lazy init = None)
      mission_spec : 리스트 of mission legs
                     ("HOLD",     cN, cE, R, dir(+1/-1), duration[s])
                     ("LINE",     sN, sE, eN, eE, mode("FLY_OVER"/"FLY_BY"))
                     ("RACETRACK",cN, cE, R, Ls, bearing_deg, dir(+1/-1), laps[int]/duration[float])
    Returns
      ((prev, cur, next), meta), new_state
        - prev/cur/next는 mission_spec의 원소(튜플) 또는 None
        - meta = {"leg_idx": idx, "typ": str}
    """
    # ---- Lazy init ----
    if mission_state is None:
        mission_state = [0, 0.0, None]  # [leg_idx, enter_time, aux]

    leg_idx, enter_time, aux = mission_state
    num_legs = len(mission_spec)
    leg = mission_spec[leg_idx]
    typ = leg[0]

    # ---- 완료 조건 정의 ----
    done = False
    if typ == "HOLD":
        done = (t - enter_time) >= float(leg[5])

    elif typ == "LINE":
        eN, eE = float(leg[1]), float(leg[2])
        done = np.hypot(pos_NE[0] - eN, pos_NE[1] - eE) < 50.0

    elif typ == "RACETRACK":
        laps_or_dur = leg[7]
        if isinstance(laps_or_dur, int):
            if aux is None:
                aux = {'laps': 0, 'prev_near': False}
            cN, cE = float(leg[1]), float(leg[2])
            near = np.hypot(pos_NE[0]-cN, pos_NE[1]-cE) < (float(leg[3]) + 5.0)
            if near and not aux['prev_near']:
                aux['laps'] += 1
            aux['prev_near'] = near
            done = aux['laps'] >= laps_or_dur
        else:
            done = (t - enter_time) >= float(laps_or_dur)
    else:
        raise ValueError(f"Unknown leg type: {typ}")

    # ---- 완료 시 전환 및 Auto-HOLD ----
    if done:
        last_leg = (leg_idx == num_legs - 1)
        if last_leg and (typ == "LINE"):
            # 마지막이 LINE이면 자동 HOLD 추가
            AUTO_HOLD_RADIUS   = 300.0
            AUTO_HOLD_DIR      = +1
            AUTO_HOLD_DURATION = 999.0
            eN, eE = float(leg[3]), float(leg[4])
            mission_spec.append(("HOLD", eN, eE, AUTO_HOLD_RADIUS, AUTO_HOLD_DIR, AUTO_HOLD_DURATION))
            leg_idx += 1
            enter_time, aux = t, None
        elif leg_idx < len(mission_spec) - 1:
            leg_idx += 1
            enter_time, aux = t, None

    # ---- prev, cur, next 산출 ----
    cur_idx = leg_idx
    prev_idx = max(cur_idx - 1, 0)
    next_idx = min(cur_idx + 1, len(mission_spec) - 1)

    prev_leg = mission_spec[prev_idx] if cur_idx > 0 else None
    cur_leg  = mission_spec[cur_idx]
    next_leg = mission_spec[next_idx] if next_idx != cur_idx else None

    meta = {"leg_idx": cur_idx, "typ": cur_leg[0]}
    return ((prev_leg, cur_leg, next_leg), meta), [leg_idx, enter_time, aux]


def append_path(prev, triplet, state, num_points):
    new, _ = build_viz_path_from_triplet(triplet, state=state, num_points=600)
    
    if new is None:
        return prev
    if isinstance(new, (list, tuple)) and len(new) == 2:
        new = np.column_stack((np.asarray(new[0]).ravel(), np.asarray(new[1]).ravel()))
    new = np.asarray(new)
    if new.ndim == 2 and new.shape[0] == 2:  # (2,N)
        new = new.T
    if new.ndim != 2 or new.shape[1] != 2:
        raise ValueError("path must be (N,2)/(2,N)/(wN,wE)")
    if prev is None or prev.size == 0:
        return new.copy()
    if np.allclose(prev[-1], new[0]):
        new = new[1:]
    return prev if new.size == 0 else np.vstack([prev, new])

def build_viz_path_from_triplet(triplet, state=None, num_points=600):
    prev, cur, _ = triplet
    typ = cur[0]
    if typ == "HOLD":
        _, cN, cE, R, d, _dur = cur
        P, is_loop = generate_hold((float(cN), float(cE)), float(R), int(num_points), int(d))
        return P, bool(is_loop)
    elif typ == "LINE":
        if prev is not None: 
           sN, sE,= prev[1], prev[2]
        else:
            sN, sE = float(state[0]), float(state[1])
        eN, eE = cur[1], cur[2]
        P, is_loop = generate_straight_p2p((sN, sE), (float(eN), float(eE)), int(num_points))
        return P, bool(is_loop)
    elif typ == "RACETRACK":
        _, cN, cE, R, Ls, chi_deg, d, _lod = cur
        P, is_loop = generate_racetrack((float(cN), float(cE)), float(R), float(Ls), int(num_points), float(chi_deg), int(d))
        return P, bool(is_loop)
    else:
        raise ValueError(f"Unknown leg type: {typ}")
