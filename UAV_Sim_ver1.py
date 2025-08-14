import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import patches

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
def simulate_unicycle(u_seq, x0, T_tot=100, sim_dt=0.01, ctrl_dt=0.1, method="rk4"):
    u_seq = np.asarray(u_seq, dtype=float)
    total_steps = int(T_tot/sim_dt)

    T = np.linspace(0.0, T_tot, total_steps+1)
    X = np.zeros((total_steps+1, 4), dtype=float)
    X[0] = np.asarray(x0, dtype=float)

    i = 0
    u = u_seq[i, 1:3]
    ctrl_chk= ctrl_dt
    
    t = 0.0
    for k in range(total_steps):
        if t >= ctrl_chk - 1e-12: 
            while (i + 1) < len(u_seq) and t >= u_seq[i+1, 0] - 1e-12:
                i += 1
            u=u_seq[i, 1:3]
            ctrl_chk += ctrl_dt
        
        X[k+1] = integrate_step(unicycle_dynamics, X[k], u, sim_dt, method, t)
        t += sim_dt

    return T, X

# =========================
# Animation & plot
# =========================
def animate_simulation(hist, interval_ms, stride=1, tri_len=80.0, tri_w=40.0, trail_style='k--'):
    ns, es, psis = hist[:,0], hist[:,1], hist[:,2]

    fig, ax = plt.subplots()
    ax.set_xlim(np.min(es)-200, np.max(es)+200)
    ax.set_ylim(np.min(ns)-200, np.max(ns)+200)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.3)

    trail_line, = ax.plot([], [], trail_style, lw=1.8, alpha=0.8, label="Trail")

    tri_pts = _triangle_points(es[0], ns[0], psis[0], L=tri_len, W=tri_w)
    aircraft = patches.Polygon(tri_pts, closed=True, ec='k', fc='g', alpha=0.9, zorder=5, label="UAV")

    ax.add_patch(aircraft)
    ax.legend(loc='best')

    def init():
        trail_line.set_data([], [])
        aircraft.set_xy(_triangle_points(es[0], ns[0], psis[0], L=tri_len, W=tri_w))
        return trail_line, aircraft

    def update(frame):
        trail_line.set_data(es[:frame+1], ns[:frame+1])

        tri = _triangle_points(es[frame], ns[frame], psis[frame], L=tri_len, W=tri_w)
        aircraft.set_xy(tri)
        return trail_line, aircraft

    frames = range(0, len(hist), max(20, int(stride)))
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

    c, s = np.cos(np.pi/2-theta), np.sin(np.pi/2-theta)  # 북쪽=0 정렬
    R = np.array([[c, -s],[s, c]])
    rot = (R @ base.T).T
    rot[:, 0] += x
    rot[:, 1] += y
    return rot

# =========================
# Main
# =========================
if __name__ == "__main__":
    total_time = 50.0
    sim_dt  = 0.01
    ctrl_dt = 0.1
    interval_ms=1
    
    # Input Setting, time, omega, acc
    uav1_u = np.zeros((6, 3), dtype=float)
    uav1_u[0, :] = [0.0 ,   0.0,  0.0]
    uav1_u[1, :] = [10.0,   0.2,  0.0]
    uav1_u[2, :] = [20.0,   0.0,  0.0]
    uav1_u[3, :] = [30.0,  -0.2,  0.0]
    uav1_u[4, :] = [40.0,   0.0,  0.0]
    uav1_u[5, :] = [101.0,  0.0,  0.0]

    # Initialize [n, e, psi(rad), v(m/s)]
    x0 = [-100.0, -900.0, 0.0, 50.0]

    # Simulation
    T, X = simulate_unicycle(uav1_u, x0, total_time, sim_dt, ctrl_dt, method="rk4")

    # Visualize
    ani, fig = animate_simulation(X, interval_ms)
    ani.save("uav_simulation.mp4", writer=animation.FFMpegWriter(fps=30))
    
    plt.show(block=False)   # 창을 띄우되 코드 실행은 계속 진행
    plt.pause(5)            # 3초 동안 표시
    plt.close(fig) 

