import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import patches

from Mission_Planning.Path_Racetrack import generate_racetrack
from Guidance.NLPF import find_target, NLPF_Guidance

# =========================
# Generic Numerical Integrator
# =========================
def integrate_step(f, state, control, dt, method="rk4", t=0.0, params=None):
    m = method.lower()
    if m == "rk4":
        k1 = f(t, state, control, params)
        k2 = f(t + 0.5*dt, state + 0.5*dt*k1, control, params)
        k3 = f(t + 0.5*dt, state + 0.5*dt*k2, control, params)
        k4 = f(t + dt, state + dt*k3, control, params)
        return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    elif m == "euler":
        k1 = f(t, state, control, params)
        return state + dt * k1
    else:
        raise ValueError(f"Unknown method: {method}")

# =========================
# Unicycle Dynamics (x, y, theta, v), control=[omega, a]
# =========================
def unicycle_dynamics(t, state, control, params=None):
    n, e, psi, v = state
    omega, acc = control  # control = [yaw rate, acceleration]

    n_dot     = v * np.cos(psi)
    e_dot     = v * np.sin(psi)
    psi_dot = omega
    v_dot     = acc

    return np.array([n_dot, e_dot, psi_dot, v_dot], dtype=float)

# =========================
# Execute Simulation
# Control Input u_seq: shape (N, 3) , [time, omega, acc]
# =========================
def simulate_unicycle(Guidance_Method, x0, T_tot, sim_dt, ctrl_dt, method="rk4"):
    total_steps = int(T_tot/sim_dt)

    T = np.linspace(0.0, T_tot, total_steps+1)
    X = np.zeros((total_steps+1, 4), dtype=float)
    X[0] = np.asarray(x0, dtype=float)

    i = 0
    u = [Guidance_Method(X[0]), 0.0]
    ctrl_chk= ctrl_dt
    
    t = 0.0
    for k in range(total_steps):
        if t >= ctrl_chk - 1e-12: 
            omega = Guidance_Method(X[k])
            u = [omega, 0.0] # [omega, acc]
            ctrl_chk += ctrl_dt
        
        X[k+1] = integrate_step(unicycle_dynamics, X[k], u, sim_dt, method, t)
        t += sim_dt

    return T, X

# =========================
# Animation & plot
# =========================
def animate_simulation(hist, interval_ms, path=None, stride=1, tri_len=80.0, tri_w=40.0, path_style = 'k:', trail_style='b--'):
    ns, es, psis = hist[:,0], hist[:,1], hist[:,2]
    # ---- path 입력 파싱 (선택) ----
    has_path = path is not None
    if has_path:
        if isinstance(path, (tuple, list)) and len(path) == 2:
            wN = np.asarray(path[:, 0]).ravel()
            wE = np.asarray(path[:, 1]).ravel()
        else:
            P = np.asarray(path)
            if P.ndim != 2 or (2 not in P.shape):
                raise ValueError("path must be (N,2)[n,e] or (wN,wE) Form.")
            if P.shape[1] == 2:   # (N,2): [n,e]
                wN, wE = P[:, 0], P[:, 1]
            else:                 # (2,N): [ [n...],[e...] ]
                wN, wE = P[0, :], P[1, :]

    # ---- Figure/Axis ----
    fig, ax = plt.subplots()

    n_min = np.min(ns)
    n_max = np.max(ns)
    e_min = np.min(es)
    e_max = np.max(es)
    if has_path:
        n_min = min(n_min, np.min(wN))
        n_max = max(n_max, np.max(wN))
        e_min = min(e_min, np.min(wE))
        e_max = max(e_max, np.max(wE))
        
    pad = 200.0
    ax.set_xlim(np.min(es)-pad, np.max(es)+pad)
    ax.set_ylim(np.min(ns)-pad, np.max(ns)+pad)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    if has_path:
        path_line, = ax.plot(wE, wN, path_style, lw=1.2, label="Path")
    else:
        path_line = None
        
    trail_line, = ax.plot([], [], trail_style, lw=1.8, alpha=0.8, label="Trail")

    tri_pts = _triangle_points(es[0], ns[0], psis[0], L=tri_len, W=tri_w)
    aircraft = patches.Polygon(tri_pts, closed=True, ec='b', fc='g', alpha=0.9, zorder=5, label="UAV")

    ax.add_patch(aircraft)
    ax.legend(loc='best')

    def init():
        trail_line.set_data([], [])
        aircraft.set_xy(_triangle_points(es[0], ns[0], psis[0], L=tri_len, W=tri_w))
        # path_line은 정적이라 초기화 불필요
        return (trail_line, aircraft) if path_line is None else (trail_line, aircraft, path_line)

    def update(frame):
        # 궤적 업데이트 (x=E, y=N)
        trail_line.set_data(es[:frame+1], ns[:frame+1])
        tri = _triangle_points(es[frame], ns[frame], psis[frame], L=tri_len, W=tri_w)
        aircraft.set_xy(tri)
        return (trail_line, aircraft) if path_line is None else (trail_line, aircraft, path_line)


    frames = range(0, len(hist), max(200, int(stride)))
    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  init_func=init, interval=interval_ms,
                                  blit=False, repeat=False)
    return ani, fig


def _triangle_points(x, y, theta, L=80.0, W=40.0):
    base = np.array([
        [ L/2,   0.0   ],
        [-L/2,  W/2   ],
        [-L/2, -W/2   ] 
    ])

    c, s = np.cos(np.pi/2-theta), np.sin(np.pi/2-theta)
    R = np.array([[c, -s],[s, c]])
    rot = (R @ base.T).T
    rot[:, 0] += x
    rot[:, 1] += y
    return rot

# =========================
# Main
# =========================
if __name__ == "__main__":
    total_time = 300.0
    sim_dt  = 0.002
    ctrl_dt = 0.05
    save_video = False
    
    WP_center = [0000, 0000]
    WP_radius = 600
    WP_length = 2000
    WP_num_points = 200
    WP_bearing_deg = 30
    WP_direction = -1
    WP_gen, is_loop = generate_racetrack(WP_center, WP_radius, WP_length, WP_num_points, WP_bearing_deg, WP_direction)
    
    
    def Guidance_Method(state, L1=200, cmd_range=[-1, 1]):
        pos = state[:2]
        tgt_wp = find_target(pos, WP_gen, L1, is_loop)
        cmd = np.clip(NLPF_Guidance(state, tgt_wp), cmd_range[0], cmd_range[1])
        return cmd
    
    # Simulation
    x0 = [-100.0, -900.0, 0.0, 50.0] # [n, e, psi(rad), v(m/s)]
    T, X = simulate_unicycle(Guidance_Method, x0, total_time, sim_dt, ctrl_dt, method="rk4")
    
    # Visualize
    interval_ms=0.001
    ani, fig = animate_simulation(X, interval_ms, path=WP_gen)
    if save_video == True:
        ani.save("uav_simulation.mp4", writer=animation.FFMpegWriter(fps=30))
    
    plt.show(block=False)
    plt.close(fig) 

