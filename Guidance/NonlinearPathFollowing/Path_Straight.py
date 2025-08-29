import numpy as np
import matplotlib.pyplot as plt

def generate_straight_bearing(start, bearing, length, num_points):
    start = np.array(start)
    psi = np.deg2rad(bearing)
    
    end = start + length * np.array([np.cos(psi), np.sin(psi)])
    Path_Straight, is_loop = generate_straight_p2p(start, end, num_points)
    
    return Path_Straight, is_loop

def generate_straight_p2p(start, end, num_points):
    is_loop=False

    start = np.array(start)
    end = np.array(end)
    t = np.linspace(0, 1, num_points)
    
    # Interpolate start to end
    Path_Straight = np.outer(1 - t, start) + np.outer(t, end)
    
    return Path_Straight, is_loop

if __name__ == "__main__":
    #P_start = [0, 0]
    #P_end = [2000, 1000]
    #WP_num_points = 200
    #
    #WP_gen, is_loop = generate_straight_p2p(P_start, P_end, WP_num_points)
    
    #or
    
    P_start = [0, 0]
    bearing = 240
    length = 3000
    WP_num_points = 200
    
    WP_gen, is_loop = generate_straight_bearing(P_start, bearing, length, WP_num_points)
    
    # 시각화
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.scatter([WP_gen[0,1], WP_gen[-1,1]], [WP_gen[0,0], WP_gen[-1,0]], c='r', marker='o', label="Start/End")
    ax.plot(WP_gen[:,1], WP_gen[:,0], 'k:', linewidth=2, label="Path")
    
    plt.show()
