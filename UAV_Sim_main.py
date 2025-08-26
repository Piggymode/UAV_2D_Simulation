import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation

from Guidance.TG import TG_find_target, TG_Guidance
from Guidance.NLPF import NLPF_find_target, NLPF_Guidance

from UAV_Simulator import simulate_unicycle
from Mission_Planner import mission_planner_step, append_path
from Animation_Plot import animate_simulation, plot_uav_paths, plot_turn_rates

# =========================
# Main
# =========================
if __name__ == "__main__":
    total_time = 100.0
    sim_dt  = 0.002
    ctrl_dt = 0.05
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
    # HOLD     (x, y, radius, direction(+1/-1), duration[s])
    # LINE     (x0, y0, x1, y1, mode="FLY_OVER"/"FLY_BY")
    # RACETRACK(x, y, length, width, bearing[deg], direction(+1/-1), laps)
    
    m0 = [("HOLD", 0.0, 0.0, 400.0, -1, 60.0),
          ("LINE", 0.0, 0.0, 1200.0, 800.0, "FLY_OVER")]
    m1 = [("RACETRACK", 1500.0, -500.0, 250.0, 1800.0, -20.0, +1, 3)]
    m2 = [("HOLD", 800.0, -600.0, 300.0, +1, 120.0),
          ("LINE", -800.0, -600.0, 400.0, -600.0, "FLY_BY")]
    m3 = [("LINE", -800.0, -600.0, 400.0, -600.0, "FLY_BY"),
          ("HOLD", 400.0, -600.0, 300.0, +1, 120.0)]
    
    missions = [m0, m1, m2, m3]
    mission_states = [None for _ in range(num_uav)] 
    guidance_times = [0.0 for _ in range(num_uav)]
    WP_save = [None for _ in range(num_uav)] 

    # =========================
    # Guidance Method
    # =========================
    def Guidance_Method_0(state, idx=0):
        (WP_gen, is_loop, meta), mission_states[idx] = mission_planner_step(t=guidance_times[idx],pos_NE=state[:2],psi=state[2],mission_state=mission_states[idx],mission_spec=missions[idx])
        WP_save[idx] = append_path(WP_save[idx], WP_gen)
        tgt = NLPF_find_target(state[:2], WP_gen, 200.0, is_loop)
        omega = NLPF_Guidance(state, tgt)
        guidance_times[idx] += ctrl_dt
        return float(np.clip(omega, -1.0, 1.0))

    def Guidance_Method_1(state, idx=1):
        (WP_gen, is_loop, meta), mission_states[idx] = mission_planner_step(t=guidance_times[idx],pos_NE=state[:2],psi=state[2],mission_state=mission_states[idx],mission_spec=missions[idx])
        WP_save[idx] = append_path(WP_save[idx], WP_gen)
        tgt = NLPF_find_target(state[:2], WP_gen, 200.0, is_loop)
        omega = NLPF_Guidance(state, tgt)
        guidance_times[idx] += ctrl_dt
        return float(np.clip(omega, -1.0, 1.0))

    def Guidance_Method_2(state, idx=2):
        (WP_gen, is_loop, meta), mission_states[idx] = mission_planner_step(t=guidance_times[idx],pos_NE=state[:2],psi=state[2],mission_state=mission_states[idx],mission_spec=missions[idx])
        WP_save[idx] = append_path(WP_save[idx], WP_gen)
        tgt, heading_path = TG_find_target(state[:2], WP_gen, is_loop)
        omega = TG_Guidance(state, tgt, heading_path, -0.033, 400)
        guidance_times[idx] += ctrl_dt
        return float(np.clip(omega, -1.0, 1.0))

    def Guidance_Method_3(state, idx=3):
        (WP_gen, is_loop, meta), mission_states[idx] = mission_planner_step(t=guidance_times[idx],pos_NE=state[:2],psi=state[2],mission_state=mission_states[idx],mission_spec=missions[idx])
        WP_save[idx] = append_path(WP_save[idx], WP_gen) 
        tgt = NLPF_find_target(state[:2], WP_gen, 200.0, is_loop)
        omega = NLPF_Guidance(state, tgt)
        guidance_times[idx] += ctrl_dt
        return float(np.clip(omega, -1.0, 1.0))
        
    # =========================
    # Simulation
    # =========================
    T, X1, U1 = simulate_unicycle(Guidance_Method_0, x0_list[0] , total_time, sim_dt, ctrl_dt, method="rk4")
    _, X2, U2 = simulate_unicycle(Guidance_Method_1, x0_list[1] , total_time, sim_dt, ctrl_dt, method="rk4")
    _, X3, U3 = simulate_unicycle(Guidance_Method_2, x0_list[2] , total_time, sim_dt, ctrl_dt, method="rk4")
    _, X4, U4 = simulate_unicycle(Guidance_Method_3, x0_list[3] , total_time, sim_dt, ctrl_dt, method="rk4")
    
    # =========================
    # Visualizeㄴ
    # =========================
    ani, fig = animate_simulation(histories=[X1, X2, X3, X4], path=WP_save, interval_ms=0.01, stride=120, tail_length_m=800.0, uav_scale=3, blit=True, show=True)
    if save_video == True:
        ani.save("uav_simulation.mp4", writer=animation.FFMpegWriter(fps=180))
    
    plt.show(block=False)
    plt.close(fig) 


    plot_uav_paths([X1, X2, X3, X4], path=WP_save)
    plot_turn_rates([U1, U2, U3, U4], times=T)
    plt.show()
