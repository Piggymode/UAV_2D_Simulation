import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from itertools import cycle

def triangle_points(e, n, psi, L=80.0, W=40.0):
    half_w = W * 0.5
    pts_body = np.array([
        [ +L/2,   0.0 ],
        [ -L/2, +half_w ],
        [ -L/2, -half_w ],
    ])
    c, s = np.cos(np.pi/2-psi), np.sin(np.pi/2-psi)
    R = np.array([[c, -s],[s, c]])     # x=E, y=N
    pts_EN = pts_body @ R.T
    pts_EN[:,0] += e
    pts_EN[:,1] += n
    return pts_EN

def parse_path(p):
    if p is None:
        return None
    if isinstance(p, (tuple, list)) and len(p) == 2:
        return np.asarray(p[0]).ravel(), np.asarray(p[1]).ravel()

    P = np.asarray(p)
    if P.ndim != 2 or (2 not in P.shape):
        raise ValueError("path must be (N,2)[n,e] or (2,N)[[n...],[e...]] or (wN,wE).")

    if P.shape[1] == 2:      # (N,2)
        wN, wE = P[:,0], P[:,1]
    else:                    # (2,N)
        wN, wE = P[0,:], P[1,:]
    return wN.ravel(), wE.ravel()

class MultiUAVAnimator:
    def __init__(self, Hs, paths, stride, tri_len, tri_w, path_style, trail_styles, labels):
        self.NEPs = []
        for H in Hs:
            H = np.asarray(H)
            if H.ndim != 2 or H.shape[1] < 3:
                raise ValueError("Each history must be shape (T,>=3) as [n,e,psi,...].")
            self.NEPs.append((H[:,0], H[:,1], H[:,2]))  # (ns, es, psis)

        # 경로 정규화 (차량 수와 맞춤)
        if paths is None:
            self.paths = [None]*len(self.NEPs)
        elif isinstance(paths, (list, tuple)):
            if len(paths) != len(self.NEPs):
                raise ValueError("Number of paths must match number of histories.")
            self.paths = [parse_path(p) for p in paths]
        else:
            self.paths = [parse_path(paths)]*len(self.NEPs)

        # 스타일/라벨
        if trail_styles is None:
            trail_styles = ['b--','r--','g--','m--','c--','y--']
        self.style_cycle = cycle(trail_styles)

        if labels is None:
            labels = [f"UAV{i+1}" for i in range(len(self.NEPs))]
        if len(labels) != len(self.NEPs):
            raise ValueError("labels length must match number of histories.")
        self.labels = labels

        self.tri_len = tri_len
        self.tri_w   = tri_w
        self.path_style = path_style
        self.stride = max(1, int(stride))

        self.fig, self.ax = plt.subplots()

        n_all = np.concatenate([ns for (ns,_,_) in self.NEPs])
        e_all = np.concatenate([es for (_,es,_) in self.NEPs])
        n_min, n_max = float(np.min(n_all)), float(np.max(n_all))
        e_min, e_max = float(np.min(e_all)), float(np.max(e_all))
        for p in self.paths:
            if p is not None:
                wN, wE = p
                n_min = min(n_min, float(np.min(wN)))
                n_max = max(n_max, float(np.max(wN)))
                e_min = min(e_min, float(np.min(wE)))
                e_max = max(e_max, float(np.max(wE)))

        pad = 500.0
        self.ax.set_xlim(e_min - pad, e_max + pad)
        self.ax.set_ylim(n_min - pad, n_max + pad)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, linestyle=':', alpha=0.3)
        self.path_lines = []
        
        common_id = None
        if any(p is not None for p in self.paths):
            ids = [id(p) for p in self.paths if p is not None]
            if len(set(ids)) == 1:
                common_id = ids[0]

        for i, p in enumerate(self.paths):
            if p is None:
                self.path_lines.append(None)
            else:
                wN, wE = p
                lbl = "Path" if (common_id is not None and id(p) == common_id and i == 0) else None
                (pline,) = self.ax.plot(wE, wN, self.path_style, lw=1.2, label=lbl)
                self.path_lines.append(pline)

        self.trail_lines = []
        self.aircraft_patches = []
        for i, (ns, es, psis) in enumerate(self.NEPs):
            style = next(self.style_cycle)
            (tl,) = self.ax.plot([], [], style, lw=1.8, alpha=0.85, label=f"{self.labels[i]} trail")
            tri_pts = triangle_points(es[0], ns[0], psis[0], L=self.tri_len, W=self.tri_w)
            ac = patches.Polygon(tri_pts,
                     closed=True,
                     ec=tl.get_color(),
                     fc=tl.get_color(),
                     alpha=0.9,
                     zorder=5)
            self.ax.add_patch(ac)
            self.trail_lines.append(tl)
            self.aircraft_patches.append(ac)

        self.ax.legend(loc='best')
        self.max_len = max(len(ns) for (ns,_,_) in self.NEPs)
        self.frames = range(0, self.max_len, self.stride)

    def init(self):
        for tl in self.trail_lines:
            tl.set_data([], [])
        for i, (ns, es, psis) in enumerate(self.NEPs):
            tri0 = triangle_points(es[0], ns[0], psis[0], L=self.tri_len, W=self.tri_w)
            self.aircraft_patches[i].set_xy(tri0)

        artists = []
        artists.extend(self.trail_lines)
        artists.extend(self.aircraft_patches)
        artists.extend([pl for pl in self.path_lines if pl is not None])
        return tuple(artists)

    def update(self, frame):
        for i, (ns, es, psis) in enumerate(self.NEPs):
            f = min(frame, len(ns)-1)
            self.trail_lines[i].set_data(es[:f+1], ns[:f+1])
            tri = triangle_points(es[f], ns[f], psis[f], L=self.tri_len, W=self.tri_w)
            self.aircraft_patches[i].set_xy(tri)

        artists = []
        artists.extend(self.trail_lines)
        artists.extend(self.aircraft_patches)
        artists.extend([pl for pl in self.path_lines if pl is not None])
        return tuple(artists)

def animate_simulation(hists,
                       interval_ms=30,
                       path=None,
                       stride=10,
                       tri_len=80.0,
                       tri_w=40.0,
                       path_style='k:',
                       trail_styles=None,
                       labels=None):
    # hists 정규화
    if isinstance(hists, np.ndarray):
        Hs = [hists]
    elif isinstance(hists, (list, tuple)):
        Hs = [np.asarray(h) for h in hists]
    else:
        raise TypeError("hists must be ndarray or list/tuple of ndarrays.")

    animator = MultiUAVAnimator(Hs, path, stride, tri_len, tri_w, path_style, trail_styles, labels)

    ani = animation.FuncAnimation(
        animator.fig,
        animator.update,
        frames=animator.frames,
        init_func=animator.init,
        interval=interval_ms,
        blit=False,
        repeat=False
    )
    return ani, animator.fig

# ===========================
# NEW: 정적 그래프 함수들
# ===========================

def plot_uav_paths(hists, path=None, labels=None):
    if isinstance(hists, np.ndarray):
        Hs = [hists]
    elif isinstance(hists, (list, tuple)):
        Hs = [np.asarray(h) for h in hists]
    else:
        raise TypeError("hists must be ndarray or list/tuple of ndarrays.")

    if labels is None:
        labels = [f"UAV{i+1}" for i in range(len(Hs))]

    fig, ax = plt.subplots(figsize=(7, 7))

    if path is not None:
        wN, wE = parse_path(path)
        ax.plot(wE, wN, 'k:', lw=1.5, label="Path")

    for i, H in enumerate(Hs):
        n, e = H[:, 0], H[:, 1]
        ax.plot(e, n, lw=2, label=labels[i])

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("UAV Trajectories")
    ax.legend(loc='best')

    return fig, ax


def plot_turn_rates(inputs_list, times=None, dt=None, labels=None):
    if not isinstance(inputs_list, (list, tuple)):
        inputs_list = [inputs_list]

    T0 = inputs_list[0].shape[0]

    if times is None:
        if dt is None:
            raise ValueError("Provide either times or dt.")
        times = np.arange(T0) * float(dt)
    else:
        times = np.asarray(times).ravel()
        if len(times) != T0:
            raise ValueError("times length must match inputs length.")

    if labels is None:
        labels = [f"UAV{i+1}" for i in range(len(inputs_list))]

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, u in enumerate(inputs_list):
        if u.shape[0] != T0 or u.shape[1] < 2:
            raise ValueError(f"inputs_list[{i}] must be shape (T,2)")
        omega = u[:, 0]
        ax.plot(times, omega, lw=1.8, label=labels[i])

    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Yaw rate command ω [rad/s]")
    ax.set_title("Input Yaw Rate (from inputs[:,0])")
    ax.legend(loc='best')
    return fig, ax

