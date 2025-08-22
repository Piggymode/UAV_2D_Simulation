import numpy as np
import matplotlib.pyplot as plt

def generate_hold(WP_center, radius, num_points, direction):
    waypoints = []
    is_loop=True

    for i in range(num_points):
        angle = 2 * np.pi * (i / num_points)
        n = radius * np.cos(angle)
        e = radius * np.sin(angle)
        waypoints.append([n, e])

    waypoints = np.array(waypoints)

    # CW/CCW
    if direction == -1:
        waypoints = waypoints[::-1]

    Path_Hold = waypoints + WP_center

    return Path_Hold, is_loop

if __name__ == "__main__":
    WP_center = [2000, -500]
    WP_radius = 600
    WP_num_points = 400
    WP_direction = -1
    WP_gen, is_loop = generate_hold(WP_center, WP_radius, WP_num_points, WP_direction)
    
    
    fig, ax = plt.subplots()
    ax.set_xlim(-1500, 500)
    ax.set_ylim(1000, 3000)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.plot(WP_gen[:, 1], WP_gen[:, 0], 'k:', lw=1.2, label="Path")
    
    plt.show()