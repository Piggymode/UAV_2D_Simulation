import numpy as np
from Mission_Planning.Path_Hold import generate_hold
from Mission_Planning.Path_Racetrack import generate_racetrack
from Mission_Planning.Path_Straight import generate_straight_p2p, generate_straight_bearing

PRESETS = {
    "hold": {
        "center": (1000.0, 1500.0),
        "radius": 600.0,
        "num_points": 1000,
        "direction": 1, 
    },
    "racetrack": {
        "center": (1000.0, 3000.0),
        "radius": 600.0,
        "length": 2000.0,
        "num_points": 1000,
        "bearing_deg": 30.0,
        "direction": 1,
    },
    "straight_p2p": {
        "start": (0.0, 0.0),
        "end": (3000.0, 4500.0),
        "num_points": 1000,
    },
    "straight_bearing": {
        "start": (0.0, 0.0),
        "bearing_deg": 60.0,
        "length": 7000.0,
        "num_points": 1000,
    },
}

def make_waypoints(scenario, **overrides):
    if scenario not in PRESETS:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Copy PRESET
    params = PRESETS[scenario].copy()
    params.update(overrides)

    if scenario == "hold":
        WP, is_loop = generate_hold(
            params["center"], params["radius"], params["num_points"], params["direction"]
        )

    elif scenario == "racetrack":
        WP, is_loop = generate_racetrack(
            params["center"], params["radius"], params["length"],
            params["num_points"], params["bearing_deg"], params["direction"]
        )

    elif scenario == "straight_p2p":
        WP, is_loop = generate_straight_p2p(
            params["start"], params["end"], params["num_points"]
        )

    else:  # "straight_bearing"
        WP, is_loop = generate_straight_bearing(
            params["start"], params["bearing_deg"], params["length"], params["num_points"]
        )

    meta = {"scenario": scenario, "params": params}
    return WP, is_loop, meta

def mission_planner_step(t, pos_NE, psi, mission_state, mission_spec):
    # ---- Lazy init ----
    if mission_state is None:
        mission_state = [0, 0.0, None]  # [leg_idx, enter_time, aux]

    leg_idx, enter_time, aux = mission_state
    leg = mission_spec[leg_idx]
    typ = leg[0]

    # 1) 현재 leg → 경로 생성
    if typ == "HOLD":
        _, cN, cE, R, d, _ = leg
        WP_gen, is_loop = generate_hold((cN, cE), R, 600, d)
        meta = {"scenario": "HOLD",
                "params": {"center": (cN, cE), "radius": R, "num_points": 600, "direction": d}}
    elif typ == "LINE":
        _, sN, sE, eN, eE, _mode = leg
        WP_gen, is_loop = generate_straight_p2p((sN, sE), (eN, eE), 600)
        meta = {"scenario": "LINE",
                "params": {"start": (sN, sE), "end": (eN, eE), "num_points": 600}}
    else:  # "RACETRACK"
        _, cN, cE, R, Ls, chi_deg, d, _laps_or_dur = leg
        WP_gen, is_loop = generate_racetrack((cN, cE), R, Ls, 600, chi_deg, d)
        meta = {"scenario": "RACETRACK",
                "params": {"center": (cN, cE), "radius": R, "length": Ls,
                           "num_points": 600, "bearing_deg": chi_deg, "direction": d}}

    # 2) 완료 조건
    if typ == "HOLD":
        done = (t - enter_time) >= float(leg[5])
    elif typ == "LINE":
        eN, eE = float(leg[3]), float(leg[4])
        done = np.hypot(pos_NE[0]-eN, pos_NE[1]-eE) < 50.0
    else:  # RACETRACK
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

    # 3) 완료 시 전환 (마지막이 LINE이면 자동 HOLD 추가)
    if done:
        last_leg = (leg_idx == len(mission_spec) - 1)
        if last_leg and (typ == "LINE"):
            # --- Auto-HOLD defaults (local to this function) ---
            AUTO_HOLD_RADIUS   = 300.0
            AUTO_HOLD_DIR      = +1    # +1=CW, -1=CCW (네 규칙 유지)
            AUTO_HOLD_DURATION = 999.0 # 사실상 "계속 대기"

            eN, eE = float(leg[3]), float(leg[4])
            mission_spec.append(("HOLD", eN, eE, AUTO_HOLD_RADIUS, AUTO_HOLD_DIR, AUTO_HOLD_DURATION))
            leg_idx += 1
            enter_time, aux = t, None

            # 새로 붙인 HOLD 경로/메타 즉시 반영(도착과 동시에 원 선회 시작)
            WP_gen, is_loop = generate_hold((eN, eE), AUTO_HOLD_RADIUS, 600, AUTO_HOLD_DIR)
            meta = {"scenario": "HOLD",
                    "params": {"center": (eN, eE), "radius": AUTO_HOLD_RADIUS,
                               "num_points": 600, "direction": AUTO_HOLD_DIR}}
        elif leg_idx < len(mission_spec) - 1:
            leg_idx += 1
            enter_time, aux = t, None
            # 다음 틱에 새 leg가 적용됨

    return (WP_gen, is_loop, meta), [leg_idx, enter_time, aux]

def append_path(prev, new):
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
