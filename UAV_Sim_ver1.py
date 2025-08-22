import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation

from Mission_Planner import make_waypoints
from Guidance.TG import TG_find_target, TG_Guidance
from Guidance.NLPF import NLPF_find_target, NLPF_Guidance
import animation_plot
#from animation_plot import animate_simulation
#from animation_plot import animate_simulation
#from animation_plot import animate_simulation

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
    U = np.zeros((total_steps+1, 2), dtype=float)
    X[0] = np.asarray(x0, dtype=float)
    U[0] = np.asarray([0.0, 0.0], dtype=float)

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
        U[k+1] = u
        t += sim_dt

    return T, X, U

# =========================
# Main
# =========================
if __name__ == "__main__":
    total_time = 150.0
    sim_dt  = 0.002
    ctrl_dt = 0.05
    save_video = True
    
    """
    Scenarios and Options:
    ----------------------
    1. "hold" (circular holding pattern)
        center      : (float, float), center point (default (1000, 1500))
        radius      : float, circle radius (default 600.0)
        num_points  : int, number of sampled waypoints (default 200)
        direction   : int, rotation (+1 CCW / -1 CW) (default -1)
    
    2. "racetrack" (oval/racetrack pattern)
        center      : (float, float), center of the racetrack (default (1000, 3000))
        radius      : float, semicircle radius (default 600.0)
        length      : float, straight leg length (default 2000.0)
        num_points  : int, number of waypoints per segment (default 200)
        bearing_deg : float, orientation angle of the racetrack in degrees (default 30.0)
        direction   : int, rotation (+1 CCW / -1 CW) (default -1)
    
    3. "straight_p2p" (straight line between two points)
        start       : (float, float), starting point (default (0,0))
        end         : (float, float), ending point (default (3000, 4500))
        num_points  : int, number of waypoints (default 600)
    
    4. "straight_bearing" (straight line with given bearing and length)
        start       : (float, float), starting point (default (0,0))
        bearing_deg : float, bearing angle in degrees (default 60.0)
        length      : float, line length (default 7000.0)
        num_points  : int, number of waypoints (default 600)
    """
    WP_gen, is_loop, meta = make_waypoints("hold", center=[0, 2000], radius=800, num_points=6000, direction=1)
    print(f"[WP] {meta['scenario']} -> {meta['params']}")
    
    
    def Guidance_Method1(state, L1=840, cmd_range=[-1, 1]):
        pos = state[:2]
        tgt_wp = NLPF_find_target(pos, WP_gen, L1, is_loop)
        cmd = np.clip(NLPF_Guidance(state, tgt_wp), cmd_range[0], cmd_range[1])
        return cmd
    def Guidance_Method2(state, L1=970, cmd_range=[-1, 1]):
        pos = state[:2]
        tgt_wp = NLPF_find_target(pos, WP_gen, L1, is_loop)
        cmd = np.clip(NLPF_Guidance(state, tgt_wp), cmd_range[0], cmd_range[1])
        return cmd
    def Guidance_Method3(state, L1=1100, cmd_range=[-1, 1]):
        pos = state[:2]
        tgt_wp = NLPF_find_target(pos, WP_gen, L1, is_loop)
        cmd = np.clip(NLPF_Guidance(state, tgt_wp), cmd_range[0], cmd_range[1])
        return cmd
    def Guidance_Method4(state, L1=1230, cmd_range=[-1, 1]):
        pos = state[:2]
        tgt_wp = NLPF_find_target(pos, WP_gen, L1, is_loop)
        cmd = np.clip(NLPF_Guidance(state, tgt_wp), cmd_range[0], cmd_range[1])
        return cmd
    # Simulation
    x0 = [000.0, 0000.0, np.pi/2, 50.0] # [n, e, psi(rad), v(m/s)]
    T, X1, U1 = simulate_unicycle(Guidance_Method1, x0, total_time, sim_dt, ctrl_dt, method="rk4")
    _, X2, U2 = simulate_unicycle(Guidance_Method2, x0, total_time, sim_dt, ctrl_dt, method="rk4")
    _, X3, U3 = simulate_unicycle(Guidance_Method3, x0, total_time, sim_dt, ctrl_dt, method="rk4")
    _, X4, U4 = simulate_unicycle(Guidance_Method4, x0, total_time, sim_dt, ctrl_dt, method="rk4")
    
    # Visualize
    ani, fig = animation_plot.animate_simulation([X1, X2, X3, X4], interval_ms=1, stride=120, path=WP_gen)
    if save_video == True:
        ani.save("uav_simulation.mp4", writer=animation.FFMpegWriter(fps=30))
    
    plt.show(block=False)
    plt.close(fig) 


    animation_plot.plot_uav_paths([X1, X2, X3, X4], path=WP_gen)
    animation_plot.plot_turn_rates([U1, U2, U3, U4], times=T)
    plt.show()
