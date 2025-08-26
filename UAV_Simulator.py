import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation

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

    u = [Guidance_Method(X[0]), 0.0]
    ctrl_chk = ctrl_dt
    t = 0.0
    for k in range(total_steps):
        if t >= ctrl_chk - 1e-12:
            omega = Guidance_Method(X[k])
            u = [omega, 0.0]
            ctrl_chk += ctrl_dt
        X[k+1] = integrate_step(unicycle_dynamics, X[k], u, sim_dt, method, t)
        U[k+1] = u
        t += sim_dt
    return T, X, U
