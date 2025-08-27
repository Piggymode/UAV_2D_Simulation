import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon

# =========================================================
# Internals
# =========================================================

def _normalize_path_like(P):
    if P is None:
        return None

    def _to_NE(a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 2)
        a = a[:, :2]
        return a  # [N, E]

    if isinstance(P, np.ndarray):
        return _to_NE(P)

    if isinstance(P, (list, tuple)):
        parts = []
        for item in P:
            if item is None:
                continue
            parts.append(_to_NE(item))
        if len(parts) == 0:
            return None
        return np.vstack(parts)

    # fallback
    arr = np.asarray(P, dtype=float)
    if arr.ndim >= 2 and arr.shape[1] >= 2:
        return arr[:, :2]
    return None


def _choose_colors(n):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def _triangle_vertices_EN(e, n, psi, L=18.0, W=20.0):
    f_e, f_n = np.sin(psi), np.cos(psi)
    l_e, l_n = -f_n, f_e

    # local (forward, left) frame 기준 삼각형
    verts_local = np.array([
        [ +L,      0.0     ],  # nose
        [ -0.5*L, +0.5*W   ],
        [ -0.5*L, -0.5*W   ],
    ])

    E = e + verts_local[:, 0] * f_e + verts_local[:, 1] * l_e
    N = n + verts_local[:, 0] * f_n + verts_local[:, 1] * l_n
    return np.column_stack([E, N])


def _cumulative_dist_EN(E, N):
    dE = np.diff(E, prepend=E[0])
    dN = np.diff(N, prepend=N[0])
    step = np.hypot(dE, dN)
    return np.cumsum(step)


# =========================================================
# Public API
# =========================================================

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
    path_alpha=0.1,
    blit=False,
    show=True,
):
    assert isinstance(histories, (list, tuple)) and len(histories) > 0, "histories must be a non-empty list of arrays"
    Hs = [np.asarray(H, dtype=float) for H in histories]
    Tlen = min([H.shape[0] for H in Hs])  # sync to shortest
    n_uav = len(Hs)

    if colors is None:
        colors = _choose_colors(n_uav)
    if labels is None:
        labels = [f"UAV{i+1}" for i in range(n_uav)]

    # Normalize paths (per-UAV로 맞춤)
    paths = None
    if path is not None:
        if isinstance(path, (list, tuple)):
            assert len(path) == n_uav, "path list length must match histories"
            paths = [ _normalize_path_like(p) for p in path ]
        else:
            P = _normalize_path_like(path)
            paths = [P for _ in range(n_uav)]

    # Figure & axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("UAV Simulation (E vs N)")

    # Bounds
    allE, allN = [], []
    for H in Hs:
        n, e = H[:Tlen, 0], H[:Tlen, 1]
        allE.append(e)
        allN.append(n)
    if paths is not None:
        for P in paths:
            if P is not None and P.size > 0:
                allN.append(P[:,0])  # N
                allE.append(P[:,1])  # E
    if len(allE) > 0:
        Ecat = np.concatenate([np.asarray(a).ravel() for a in allE])
        Ncat = np.concatenate([np.asarray(a).ravel() for a in allN])
        pad = 100.0
        ax.set_xlim(Ecat.min()-pad, Ecat.max()+pad)
        ax.set_ylim(Ncat.min()-pad, Ncat.max()+pad)

    # Precompute tail cumulative distance
    cumdists = []
    for H in Hs:
        n, e = H[:Tlen, 0], H[:Tlen, 1]
        cumdists.append(_cumulative_dist_EN(e, n))

    # Artists
    tail_lines = []
    body_patches = []

    # ▶ 경로 라인은 '초기화에서 한 번만' 그림 (정적)
    if paths is not None:
        for i in range(n_uav):
            P = paths[i]
            if P is None or P.size == 0:
                continue
            wN, wE = P[:,0], P[:,1]
            # animated=False 로 설정하여 blit 시에도 재렌더 대상에서 제외
            line, = ax.plot(
                wE, wN,
                #linestyle=(0, (4, 6)),
                alpha=path_alpha,
                lw=2,
                color=colors[i],
                zorder=1,
                animated=False
            )
            line.set_linestyle("-")    
            #line.set_dashes([6, 6])

    # 동적 아티스트: 꼬리선 / 기체 폴리곤만 프레임별 업데이트
    for i in range(n_uav):
        color = colors[i]

        # Tail line (recent tail_length_m)
        line, = ax.plot([], [], lw=1.0, label=labels[i], linestyle=':', color=color, zorder=3, animated=blit)
        tail_lines.append(line)

        # Vehicle polygon
        tri = Polygon([[0,0],[0,0],[0,0]], closed=True, ec="none", fc=color, alpha=0.95, zorder=4, animated=blit)
        ax.add_patch(tri)
        body_patches.append(tri)

    # Legend
    if labels is not None:
        ax.legend(loc="best")

    # Interval handling
    if interval_ms < 5.0:
        interval = int(1000.0 * interval_ms)
    else:
        interval = int(interval_ms)

    # Triangle size
    base_L = 18.0 * uav_scale
    base_W = 20.0 * uav_scale

    frames = np.arange(0, Tlen, stride, dtype=int)

    def _update(frame_idx):
        k = int(frame_idx)
        for i, H in enumerate(Hs):
            n, e, psi = H[k, 0], H[k, 1], H[k, 2]
            # body
            verts = _triangle_vertices_EN(e, n, psi, L=base_L, W=base_W)
            body_patches[i].set_xy(verts)

            # tail (최근 tail_length_m 구간만)
            cd = cumdists[i]
            target = cd[k]
            start_val = target - tail_length_m
            j = int(np.searchsorted(cd, start_val, side="left"))
            e_seg = H[j:k+1, 1]
            n_seg = H[j:k+1, 0]
            tail_lines[i].set_data(e_seg, n_seg)

        # 정적 경로 라인은 반환하지 않음
        return (*tail_lines, *body_patches)

    ani = animation.FuncAnimation(
        fig, _update, frames=frames, interval=interval, blit=blit, repeat=False
    )

    # Keep references so GC doesn't kill the animation
    if not hasattr(fig, "_keepalive"):
        fig._keepalive = []
    fig._keepalive.append(ani)

    # Optionally show and block until window is closed
    if show:
        plt.show()

    return ani, fig


def plot_uav_paths(histories, path=None, colors=None, labels=None, figsize=(7,7)):
    Hs = [np.asarray(H, dtype=float) for H in histories]
    n_uav = len(Hs)
    if colors is None:
        colors = _choose_colors(n_uav)
    if labels is None:
        labels = [f"UAV{i+1}" for i in range(n_uav)]

    # Normalize paths
    paths = None
    if path is not None:
        if isinstance(path, (list, tuple)):
            assert len(path) == n_uav, "path list length must match histories"
            paths = [ _normalize_path_like(p) for p in path ]
        else:
            P = _normalize_path_like(path)
            paths = [P for _ in range(n_uav)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("UAV Trajectories (E vs N)")

    for i, H in enumerate(Hs):
        n, e = H[:,0], H[:,1]
        ax.plot(e, n, lw=2.0, label=labels[i], color=colors[i])

    if paths is not None:
        for i, P in enumerate(paths):
            if P is None or P.size == 0:
                continue
            wN, wE = P[:,0], P[:,1]
            line, = ax.plot(wE, wN, linestyle=(0, (4, 6)), lw=1, color=colors[i], alpha=0.1)
            line.set_linestyle(":")
            line.set_dashes([6, 6])

    ax.legend(loc="best")
    return fig, ax


def plot_turn_rates(U_list, times=None, labels=None, figsize=(8,4)):
    Us = [np.asarray(U) for U in U_list]
    n_uav = len(Us)
    if labels is None:
        labels = [f"UAV{i+1}" for i in range(n_uav)]

    fig, ax = plt.subplots(figsize=figsize)
    for i, U in enumerate(Us):
        if U.ndim == 2 and U.shape[1] >= 1:
            y = U[:,0]
        else:
            y = U.ravel()
        if i == 0 and times is not None:
            x = np.asarray(times).ravel()
        else:
            x = np.arange(len(y))
        ax.plot(x, y, lw=1.6, label=labels[i])

    ax.set_xlabel("Time [s]" if times is not None else "Step")
    ax.set_ylabel("Yaw rate command ω [rad/s]")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title("Yaw-rate Commands")
    ax.legend(loc="best")
    return fig, ax
