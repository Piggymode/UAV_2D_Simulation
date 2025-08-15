import numpy as np

def generate_racetrack(WP_center, radius, straight_length, num_points, bearing_deg, direction):
    waypoints = []
    is_loop=True

    # Bottom straight
    for i in range(num_points):
        x = -straight_length / 2 + i * (straight_length / num_points); y = -radius
        waypoints.append([x, y])

    # Right semicircle (bottom to top)
    for i in range(num_points):
        angle = np.pi * (i / num_points)
        x = radius * np.sin(angle) + straight_length / 2; y = -radius * np.cos(angle)
        waypoints.append([x, y])

    # Top straight
    for i in range(num_points):
        x = straight_length / 2 - i * (straight_length / num_points); y = radius
        waypoints.append([x, y])

    # Left semicircle (top to bottom)
    for i in range(num_points):
        angle = np.pi * (i / num_points) + np.pi
        x = radius * np.sin(angle) - straight_length / 2; y = -radius * np.cos(angle)
        waypoints.append([x, y])

    waypoints = np.array(waypoints)

    # CW/CCW
    if direction == -1:
        waypoints = waypoints[::-1]

    # Rotate by bearing
    bearing_rad = np.deg2rad(bearing_deg)
    rotation_matrix = np.array([
        [np.cos(bearing_rad), -np.sin(bearing_rad)],
        [np.sin(bearing_rad),  np.cos(bearing_rad)]
    ])
    Path_Racetrack = waypoints @ rotation_matrix.T + WP_center

    return Path_Racetrack, is_loop
