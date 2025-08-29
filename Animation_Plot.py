import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon

def _normalize_path_like(P):
    if P is None:
        return None
    a = np.asarray(P, dtype=float)
    if a.ndim == 2 and a.shape[1] == 2:     # (N,2) [N,E]
        return a
    if a.ndim == 2 and a.shape[0] == 2:     # (2,N) -> (N,2)
        return a.T
    if isinstance(P, (list, tuple)) and len(P) == 2:
        wN, wE = np.asarray(P[0], float).ravel(), np.asarray(P[1], float).ravel()
        return np.column_stack([wN, wE])
    raise ValueError("path must be (N,2) or (2,N) or (wN,wE).")

def _triangle_vertices_EN(e, n, psi, L=18.0, W=10.0):
    ue, un = np.sin(psi), np.cos(psi)   # heading (E,N)
    le, ln = -un, ue
    re, rn = +un, -ue
    nose_e, nose_n = e + L*ue, n + L*un
    tail_e, tail_n = e - 0.5*L*ue, n - 0.5*L*un
    rl_e, rl_n = tail_e + 0.5*W*le, tail_n + 0.5*W*ln
    rr_e, rr_n = tail_e + 0.5*W*re, tail_n + 0.5*W*rn
    return np.array([[nose_e, nose_n], [rl_e, rl_n], [rr_e, rr_n]])

def _cumulative_dist_EN(E, N):
    dE = np.diff(E, prepend=E[0])
    dN = np.diff(N, prepend=N[0])
    step = np.hypot(dE, dN)
    return np.cumsum(step)

def animate_simulation(
    histories,
    path=None,               
    interval_ms=50,
    stride=1,
    tail_length_m=200.0,
    figsize=(8, 8),
    colors=None,
    labels=None,
    uav_scale=1.0,
    path_linestyle=":",
    path_alpha=0.20,
    blit=False,
    show=True,

    leg_index_histories=None,    
    path_by_leg_list=None,    
):
    # ---- normalize histories ----
    assert isinstance(histories, (list, tuple)) and len(histories) > 0
    Hs = [np.asarray(H, float) for H in histories]
    n_uav = len(Hs)
    Tlen  = min(H.shape[0] for H in Hs)

    # ---- colors/labels ----
    if colors is None:
        base = plt.rcParams.get("axes.prop_cycle").by_key().get("color", None) or [f"C{i}" for i in range(10)]
        colors = [base[i % len(base)] for i in range(n_uav)]
    if labels is None:
        labels = [f"UAV{i+1}" for i in range(n_uav)]

    # ---- static paths (optional) ----
    paths = None
    if path is not None:
        if isinstance(path, (list, tuple)):
            assert len(path) == n_uav, "path list length must match histories"
            paths = [_normalize_path_like(p) if p is not None else None for p in path]
        else:
            P = _normalize_path_like(path)
            paths = [P for _ in range(n_uav)]

    # ---- figure/axes ----
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("UAV Simulation")

    # ---- bounds from histories (+ paths + leg paths) ----
    Ns = np.concatenate([H[:Tlen, 0] for H in Hs])
    Es = np.concatenate([H[:Tlen, 1] for H in Hs])
    Nmin, Nmax = float(np.min(Ns)), float(np.max(Ns))
    Emin, Emax = float(np.min(Es)), float(np.max(Es))

    def _extend(Nmin, Nmax, Emin, Emax, P):
        if P is None: return Nmin, Nmax, Emin, Emax
        P = _normalize_path_like(P)
        Nmin = min(Nmin, np.min(P[:,0])); Nmax = max(Nmax, np.max(P[:,0]))
        Emin = min(Emin, np.min(P[:,1])); Emax = max(Emax, np.max(P[:,1]))
        return Nmin, Nmax, Emin, Emax

    if paths is not None:
        for P in paths:
            Nmin, Nmax, Emin, Emax = _extend(Nmin, Nmax, Emin, Emax, P)
    if (leg_index_histories is not None) and (path_by_leg_list is not None):
        for leg_map in path_by_leg_list:
            for P in leg_map.values():
                Nmin, Nmax, Emin, Emax = _extend(Nmin, Nmax, Emin, Emax, P)

    dN = max(1.0, 0.05*(Nmax - Nmin)); dE = max(1.0, 0.05*(Emax - Emin))
    ax.set_xlim(Emin - dE, Emax + dE); ax.set_ylim(Nmin - dN, Nmax + dN)

    # ---- tails & bodies ----
    cumdists = []
    for H in Hs:
        n, e = H[:Tlen, 0], H[:Tlen, 1]
        cumdists.append(_cumulative_dist_EN(e, n))

    tail_lines, body_patches = [], []
    base_L = 18.0*uav_scale; base_W = 10.0*uav_scale
    for i, H in enumerate(Hs):
        (tline,) = ax.plot([], [], lw=2, color=colors[i], zorder=2, animated=blit, label=labels[i])
        tail_lines.append(tline)
        verts = _triangle_vertices_EN(H[0,1], H[0,0], H[0,2], L=base_L, W=base_W)
        poly = Polygon(verts, closed=True, ec=colors[i], fc=colors[i], alpha=0.9, zorder=3, animated=blit)
        body_patches.append(poly); ax.add_patch(poly)
    ax.legend(loc="best")

    # ---- dynamic leg-based path line ----
    use_leg = (leg_index_histories is not None) and (path_by_leg_list is not None)
    leg_path_lines = []
    if use_leg:
        for i in range(n_uav):
            line, = ax.plot([], [], path_linestyle, lw=2, alpha=path_alpha, color=colors[i], zorder=1, animated=blit)
            leg_path_lines.append(line)
    else:
        if paths is not None:
            for i in range(n_uav):
                P = paths[i]
                if P is None: continue
                ax.plot(P[:,1], P[:,0], path_linestyle, lw=2, alpha=path_alpha, color=colors[i], zorder=1)

    frames = list(range(0, Tlen, max(1, int(stride))))

    def _update(k):
        if use_leg:
            for i in range(n_uav):
                leg_hist = leg_index_histories[i]
                leg_map  = path_by_leg_list[i]
                if leg_hist is None or len(leg_hist) == 0:
                    leg_path_lines[i].set_data([], [])
                    continue
                idx_k = int(np.floor(k * (len(leg_hist) - 1) / (Tlen - 1))) if Tlen > 1 else 0
                leg = int(leg_hist[idx_k])
                P = leg_map.get(leg, None)
                if P is None or len(P) == 0:
                    leg_path_lines[i].set_data([], [])
                else:
                    P = _normalize_path_like(P)
                    leg_path_lines[i].set_data(P[:,1], P[:,0])  # (E,N)

        for i, H in enumerate(Hs):
            n, e, psi = H[k,0], H[k,1], H[k,2]
            body_patches[i].set_xy(_triangle_vertices_EN(e, n, psi, L=base_L, W=base_W))
            cd = cumdists[i]
            target = cd[k]; start_val = target - float(tail_length_m)
            j = int(np.searchsorted(cd, start_val, side="left"))
            tail_lines[i].set_data(H[j:k+1,1], H[j:k+1,0])  # (E,N)

        if blit:
            artists = [*tail_lines, *body_patches]
            if use_leg: artists.extend(leg_path_lines)
            return artists
        return []

    ani = animation.FuncAnimation(fig, _update, frames=frames, interval=float(interval_ms), blit=blit)
    if show: plt.show()
    return ani, fig
