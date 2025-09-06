import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def tf(num, den):
    num = np.atleast_1d(np.asarray(num, float))
    den = np.atleast_1d(np.asarray(den, float))
    if den[0] == 0:
        raise ValueError("Leading denominator coefficient cannot be zero.")
    # 정규화(den[0]=1)
    num = num / den[0]
    den = den / den[0]
    return signal.TransferFunction(num, den)

def _nd(sys):
    # scipy 1.12+ 에서는 sys.num, sys.den가 1D array (SISO)로 온다
    return np.asarray(sys.num, float), np.asarray(sys.den, float)

def series(G, H):
    """G*H"""
    Ng, Dg = _nd(G); Nh, Dh = _nd(H)
    return tf(np.polymul(Ng, Nh), np.polymul(Dg, Dh))

def parallel(G, H):
    """G+H"""
    Ng, Dg = _nd(G); Nh, Dh = _nd(H)
    num = np.polyadd(np.polymul(Ng, Dh), np.polymul(Nh, Dg))
    den = np.polymul(Dg, Dh)
    return tf(num, den)

def feedback(G, H=None, sign=-1):
    """Closed loop TF """
    Ng, Dg = _nd(G)
    if H is None:
        Nh, Dh = np.array([1.0]), np.array([1.0])
    else:
        Nh, Dh = _nd(H)
    num = np.polymul(Ng, Dh)                          # Ng*Dh
    den = np.polyadd(np.polymul(Dg, Dh),              # Dg*Dh
                     sign * np.polymul(Ng, Nh))       # + sign*Ng*Nh
    return tf(num, den)

def step(G, T=None):
    t, y = signal.step(G, T=T)
    return t, y

def impulse(G, T=None):
    t, y = signal.impulse(G, T=T)
    return t, y

def lsim(G, u, t):
    sys_ss = signal.StateSpace(*signal.tf2ss(*_nd(G)))
    tout, y, x = signal.lsim(sys_ss, U=np.asarray(u, float), T=np.asarray(t, float))
    return tout, y, x

def bode(G, w):
    w, mag, phase = signal.bode(G, w=w)
    return w, mag, phase


if __name__ == "__main__":
    wn, zeta = 2.0, 0.7
    G = tf([wn**2], [1, 2*zeta*wn, wn**2])
    
    # Step 응답
    t = np.linspace(0,6, 1000)
    t, y = step(G, T=t)
    plt.figure(); plt.plot(t, y); plt.grid(True); plt.title("Step response"); plt.xlabel("s")
    
    # Bode
    w = np.logspace(-2, 2, 500)
    w, mag, phase = bode(G, w)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax1.semilogx(w, mag); ax1.set_ylabel("Magnitude [dB]"); ax1.set_title("Bode Plot"); ax1.grid(True, which="both")
    ax2.semilogx(w, phase); ax2.set_ylabel("Phase [deg]"); ax2.set_xlabel("Frequency [rad/s]"); ax2.grid(True, which="both")

    # 임의 입력 (사인)
    t = np.linspace(0, 10, 2001)
    u = np.sin(2*np.pi*0.8*t)
    tout, y, _ = lsim(G, u, t)
    plt.figure(); plt.plot(t, u, label="u"); plt.plot(tout, y, label="y"); plt.legend(); plt.grid(True)
    plt.show()