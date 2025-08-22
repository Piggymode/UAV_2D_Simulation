from Mission_Planning.Path_Hold import generate_hold
from Mission_Planning.Path_Racetrack import generate_racetrack
from Mission_Planning.Path_Straight import generate_straight_p2p
from Mission_Planning.Path_Straight import generate_straight_bearing

PRESETS = {
    "hold": {
        "center": (1000.0, 1500.0),
        "radius": 600.0,
        "num_points": 200,
        "direction": 1, 
    },
    "racetrack": {
        "center": (1000.0, 3000.0),
        "radius": 600.0,
        "length": 2000.0,
        "num_points": 200,
        "bearing_deg": 30.0,
        "direction": 1,
    },
    "straight_p2p": {
        "start": (0.0, 0.0),
        "end": (3000.0, 4500.0),
        "num_points": 600,
    },
    "straight_bearing": {
        "start": (0.0, 0.0),
        "bearing_deg": 60.0,
        "length": 7000.0,
        "num_points": 600,
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
