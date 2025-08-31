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

