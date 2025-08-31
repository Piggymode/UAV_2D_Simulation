import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation

from Guidance.TrackGuidance.TG import TG_Guidance
from Guidance.NonlinearPathFollowing.NLPF import NLPF_Guidance

from UAV_Simulator import integrate_step, unicycle_dynamics
from Mission_Planner import mission_planner_step, append_path, build_viz_path_from_triplet
from Animation_Plot import animate_simulation

# =========================
# Main
# =========================
if __name__ == "__main__":
    total_time = 300.0
    sim_dt  = 0.01
    ctrl_dt = 0.1
    save_video = False
    
    # =========================
    # Initial State
    # =========================
    num_uav = 4
    x0_list = [
        np.array([-800.0, -400.0, np.deg2rad( 20.0), 40.0]),    # UAV0
        np.array([-500.0,  200.0, np.deg2rad(-10.0), 40.0]),    # UAV1
        np.array([ 300.0, -700.0, np.deg2rad( 45.0), 40.0]),    # UAV2
        np.array([ 300.0, -700.0, np.deg2rad( 45.0), 40.0]),    # UAV3
    ]
    
    # =========================
    # Mission Planning
    # =========================
    # Description
    # HOLD      (n, e, radius, direction(+1/-1), duration[s])
    # LINE      (n, e, mode="FLY_OVER"/"FLY_BY")
    # RACETRACK (n, e, length, width, bearing[deg], direction(+1/-1), laps)
    m0 = [("HOLD", 0.0, 0.0, 400.0, -1, 30.0),
          ("LINE", 1200.0, 800.0, "FLY_OVER"),
          ("HOLD", 800.0, -600.0, 300.0, +1, 30.0),
          ("RACETRACK", 1500.0, -500.0, 250.0, 1800.0, -20.0, -1, 3),
          ("LINE", 400.0, -600.0, "FLY_BY")]
    m1 = [("RACETRACK", 1500.0, -500.0, 250.0, 1800.0, -20.0, +1, 3),
          ("HOLD", 0.0, 0.0, 400.0, -1, 60.0),
          ("LINE", 1200.0, 800.0, "FLY_OVER"),
          ("HOLD", 400.0, -600.0, 300.0, +1, 120.0)]
    m2 = [("HOLD", 800.0, -600.0, 300.0, +1, 120.0),
          ("LINE", 00.0, 00.0, "FLY_BY"),
          ("HOLD", 0.0, 0.0, 400.0, -1, 60.0),
          ("RACETRACK", 1500.0, -500.0, 250.0, 1800.0, -20.0, -1, 3),
          ("LINE", 1200.0, 800.0, "FLY_OVER"),]
    m3 = [("HOLD", 0.0, 0.0, 400.0, -1, 60.0),
          ("RACETRACK", 1500.0, -500.0, 250.0, 1800.0, -20.0, +1, 3),
          ("LINE", 1200.0, 800.0, "FLY_OVER"),
          ("RACETRACK", 1500.0, -500.0, 250.0, 1800.0, -20.0, +1, 3),
          ("LINE", 400.0, -600.0, "FLY_BY"),
          ("HOLD", 400.0, -600.0, 300.0, +1, 120.0)]
    
    missions = [m0, m1, m2, m3]
    mission_states = [None for _ in range(num_uav)] 
    guidance_times = [0.0 for _ in range(num_uav)]
    WP_save = [None for _ in range(num_uav)] 
    LEG_HIST     = [[] for _ in range(num_uav)]   # 프레임별 현재 레그 기록
    PATH_BY_LEG  = [{} for _ in range(num_uav)] 

    # =========================
    # Guidance Method
    # =========================
    def Guidance_Method_0(state, idx=0):
        (triplet, meta), mission_states[idx] = mission_planner_step(t=guidance_times[idx],pos_NE=state[:2],psi=state[2],mission_state=mission_states[idx],mission_spec=missions[idx])
        omega = TG_Guidance(state, triplet, Kp=0.0002, K=100)
        guidance_times[idx] += ctrl_dt
        leg = int(meta["leg_idx"]); LEG_HIST[idx].append(leg)
        if leg not in PATH_BY_LEG[idx]:
            P, _ = build_viz_path_from_triplet(triplet, state=state, num_points=600)
            PATH_BY_LEG[idx][leg] = P
        return float(np.clip(omega, -1.0, 1.0))

    def Guidance_Method_1(state, idx=1):
        (triplet, meta), mission_states[idx] = mission_planner_step(t=guidance_times[idx],pos_NE=state[:2],psi=state[2],mission_state=mission_states[idx],mission_spec=missions[idx])
        omega = NLPF_Guidance(state, triplet, L1=200.0, num_points=600)
        guidance_times[idx] += ctrl_dt
        leg = int(meta["leg_idx"]); LEG_HIST[idx].append(leg)
        if leg not in PATH_BY_LEG[idx]:
            P, _ = build_viz_path_from_triplet(triplet, state=state, num_points=600)
            PATH_BY_LEG[idx][leg] = P
        return float(np.clip(omega, -1.0, 1.0))

    def Guidance_Method_2(state, idx=2):
        (triplet, meta), mission_states[idx] = mission_planner_step(t=guidance_times[idx],pos_NE=state[:2],psi=state[2],mission_state=mission_states[idx],mission_spec=missions[idx])
        omega = TG_Guidance(state, triplet, Kp=0.0002, K=100)
        guidance_times[idx] += ctrl_dt
        leg = int(meta["leg_idx"]); LEG_HIST[idx].append(leg)
        if leg not in PATH_BY_LEG[idx]:
            P, _ = build_viz_path_from_triplet(triplet, state=state, num_points=600)
            PATH_BY_LEG[idx][leg] = P
        return float(np.clip(omega, -1.0, 1.0))

    def Guidance_Method_3(state, idx=3):
        (triplet, meta), mission_states[idx] = mission_planner_step(t=guidance_times[idx],pos_NE=state[:2],psi=state[2],mission_state=mission_states[idx],mission_spec=missions[idx])
        omega = NLPF_Guidance(state, triplet, L1=200.0, num_points=600)
        guidance_times[idx] += ctrl_dt
        leg = int(meta["leg_idx"]); LEG_HIST[idx].append(leg)
        if leg not in PATH_BY_LEG[idx]:
            P, _ = build_viz_path_from_triplet(triplet, state=state, num_points=600)
            PATH_BY_LEG[idx][leg] = P
        return float(np.clip(omega, -1.0, 1.0))
        
    # =========================
    # Simulation (loop 없이 4대 명시)
    # =========================
    T_steps = int(total_time / sim_dt)
    
    X1 = np.zeros((T_steps + 1, 4)); U1 = np.zeros((T_steps + 1, 2))
    X2 = np.zeros((T_steps + 1, 4)); U2 = np.zeros((T_steps + 1, 2))
    X3 = np.zeros((T_steps + 1, 4)); U3 = np.zeros((T_steps + 1, 2))
    X4 = np.zeros((T_steps + 1, 4)); U4 = np.zeros((T_steps + 1, 2))
    x1 = np.asarray(x0_list[0], dtype=float); u1 = np.array([0.0, 0.0], float)
    x2 = np.asarray(x0_list[1], dtype=float); u2 = np.array([0.0, 0.0], float)
    x3 = np.asarray(x0_list[2], dtype=float); u3 = np.array([0.0, 0.0], float)
    x4 = np.asarray(x0_list[3], dtype=float); u4 = np.array([0.0, 0.0], float)
    X1[0] = x1; U1[0] = u1
    X2[0] = x2; U2[0] = u2
    X3[0] = x3; U3[0] = u3
    X4[0] = x4; U4[0] = u4
    
    ctrl_chk = ctrl_dt
    t = 0.0
    
    for k in range(T_steps):
        if t > ctrl_chk - 1e-12:
            omega1 = Guidance_Method_0(x1) # UAV1
            u1[0] = float(np.clip(omega1, -1.0, 1.0));  u1[1] = 0.0  # acc 고정
            omega2 = Guidance_Method_1(x2) # UAV2
            u2[0] = float(np.clip(omega2, -1.0, 1.0));  u2[1] = 0.0
            omega3 = Guidance_Method_2(x3) # UAV3
            u3[0] = float(np.clip(omega3, -1.0, 1.0));  u3[1] = 0.0
            omega4 = Guidance_Method_3(x4) # UAV4
            u4[0] = float(np.clip(omega4, -1.0, 1.0));  u4[1] = 0.0
            ctrl_chk += ctrl_dt
        x1 = integrate_step(unicycle_dynamics, x1, u1, sim_dt, method="rk4", t=t)
        x2 = integrate_step(unicycle_dynamics, x2, u2, sim_dt, method="rk4", t=t)
        x3 = integrate_step(unicycle_dynamics, x3, u3, sim_dt, method="rk4", t=t)
        x4 = integrate_step(unicycle_dynamics, x4, u4, sim_dt, method="rk4", t=t)
        X1[k+1] = x1; U1[k+1] = u1
        X2[k+1] = x2; U2[k+1] = u2
        X3[k+1] = x3; U3[k+1] = u3
        X4[k+1] = x4; U4[k+1] = u4
        t += sim_dt
        
    # =========================
    # Visualize
    # =========================
    ani, fig = animate_simulation(
    histories=[X1, X2, X3, X4],
    leg_index_histories = [np.asarray(LEG_HIST[0], dtype=int),
                           np.asarray(LEG_HIST[1], dtype=int),
                           np.asarray(LEG_HIST[2], dtype=int),
                           np.asarray(LEG_HIST[3], dtype=int)],
    path_by_leg_list    = [PATH_BY_LEG[0],
                           PATH_BY_LEG[1],
                           PATH_BY_LEG[2],
                           PATH_BY_LEG[3]],
    interval_ms=50, stride=60, tail_length_m=800.0,
    uav_scale=3, blit=True, show=True)
    if save_video == True:
        ani.save("uav_simulation.mp4", writer=animation.FFMpegWriter(fps=180))
    
    plt.show(block=False)
    plt.close(fig) 
    
