############################################################
#
#   - Name : run_core_smoke.py
#
#                                 - KAIST FDCL, 2026.03.11
#
############################################################

from __future__ import annotations

import importlib.util
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathfollowing.core.core_pathfollowing import PathFollowingCore


def vehicle_cfg() -> dict:
    return {
        "model": {
            "mass": 2.0262,
            "Ixx": 0.029125,
            "Iyy": 0.029125,
            "Izz": 0.055225,
            "gravity": 9.81,
        },
        "actuator": {
            "n_motor": 4,
            "max_thrust_per_rotor": 7.0664,
            "rotor_positions": [
                [0.13, 0.21, 0.0],
                [0.13, -0.21, 0.0],
                [-0.13, -0.21, 0.0],
                [-0.13, 0.21, 0.0],
            ],
            "spin_direction": [-1, 1, -1, 1],
        },
        "px4_model": {
            "max_tilt_deg": 30.0,
            "max_accel_xy": 4.0,
            "max_accel_z": 2.5,
            "tau_phi": 0.3,
            "tau_the": 0.3,
            "tau_psi": 0.6,
            "del_psi_cmd_limit_deg": 20.0,
            "tau_wb": 0.05,
            "tau_p": 0.1,
            "tau_q": 0.1,
            "tau_r": 0.2,
            "alpha_p": 0.1,
            "alpha_q": 0.1,
            "alpha_r": 0.1,
            "tau_throttle": 0.01875,
        },
    }


def sim_cfg() -> dict:
    return {
        "path_following": {
            "dt_pf": 0.004,
            "desired_speed": 3.0,
            "lookahead_distance": 4.5,
            "waypoint_switch_distance": 4.5,
            "kp_vel": 1.0,
            "kd_vel": 0.0,
            "kp_speed": 3.0,
            "kd_speed": 0.0,
            "guid_eta": 2.0,
            "final_wp_tolerance_m": 0.5,
            "transition_speed_threshold_mps": 1.0,
            "position_kp": 0.5,
            "position_kv_multiplier": 3.0,
            "min_guid_eta": 0.5,
            "gl_vertical_speed_gain": 2.0,
        },
        "mppi": {
            "dt_mppi": 0.05,
            "num_samples": 64,
            "horizon_steps": 30,
            "cost_min_v_aligned": 0.3,
            "use_gpr_forecast": False,
            "type2_gamma_scale": 0.05,
            "disturbance_var_weight_gain": 40.0,
            "disturbance_var_clip": 5.0e-4,
            "cost_path_error_scale_m": 5.0,
            "gpr_var_bias": 1.0e-2,
            "gpr_var_clip": 1.0,
            "type_1": {
                "state_cost": [0.01, 0.01, 0.01],
                "terminal_cost": [0.05, 0.05, 0.05],
                "noise_std_u0_scale": 0.7,
                "noise_std_u1_scale": 0.7,
                "lambda_weight": 0.000245,
                "u0_init": 0.5,
                "u1_init": 2.0,
            },
            "type_2": {
                "state_cost": [75.0, 50.0, 5.0],
                "terminal_cost": [0.0, 50.0, 5.0],
                "noise_std_u0_scale": 0.04472135955,
                "noise_std_u1_scale": 0.08944271910,
                "lambda_weight": 0.1,
                "u0_init": 0.0,
                "u1_init": 2.0,
            },
        },
        "ndo": {"gain": [5.0, 5.0, 5.0]},
        "numerics": {"eps": 1.0e-9},
        "gpr": {
            "dt_gpr": 0.01,
            "dt_gpr_opt_mul": 20,
            "forecast_steps": 200,
            "measurement_matrix": [1.0, 0.0],
            "meas_noise_var_x": 1.0e-6,
            "meas_noise_var_y": 1.0e-6,
            "hyp_l_init": [1.0, 1.0],
            "hyp_q_init": [1.0, 1.0],
            "hyp_n": 500,
            "hyp_l_min": 1.0e-2,
            "hyp_l_max": 0.5,
            "hyp_q_min": 1.0e-2,
            "hyp_q_max": 0.5,
            "lpf_cutoff_hz": 5.0,
        },
    }


def run_att_core_only() -> None:
    att = PathFollowingCore(
        guid_type=2,
        wp_type=0,
        vehicle_cfg=vehicle_cfg(),
        sim_cfg=sim_cfg(),
        logger_obj=None,
    )
    att.update_heartbeats(True, True, True)
    att.update_waypoints(False, [0.0, 20.0, 40.0], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0])

    outputs = []
    for i in range(60):
        att.update_timesync_timestamp_us(1_000_000 + i * 4_000)
        att.update_local_position(float(i), 0.0, -10.0, 8.0, 0.0, 0.0)
        att.update_attitude_quat(1.0, 0.0, 0.0, 0.0)
        att.update_accel_xyz(0.0, 0.0, 0.0)
        out = att.update()
        if out is not None:
            outputs.append(out)

    if not outputs:
        raise RuntimeError("ATT core smoke test failed: no control output.")

    max_heading = max(int(o["heading_wp_idx"]) for o in outputs)
    print(f"[ATT] outputs={len(outputs)}, max_heading_wp_idx={max_heading}")


def run_att_mppi_bridge() -> None:
    if importlib.util.find_spec("pycuda") is None:
        print("[BRIDGE] skipped: pycuda not installed.")
        return

    from pathfollowing.core.core_mppi import MPPICore

    att = PathFollowingCore(
        guid_type=2,
        wp_type=0,
        vehicle_cfg=vehicle_cfg(),
        sim_cfg=sim_cfg(),
        logger_obj=None,
    )
    mppi = MPPICore(guid_type=2, vehicle_cfg=vehicle_cfg(), sim_cfg=sim_cfg())
    att.update_heartbeats(True, True, True)
    att.update_waypoints(False, [0.0, 20.0, 40.0, 60.0], [0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0])

    seen_reset_zero = False
    for i in range(80):
        att.update_timesync_timestamp_us(1_000_000 + i * 4_000)
        att.update_local_position(float(i), 0.0, -10.0, 8.0, 0.0, 0.0)
        att.update_attitude_quat(1.0, 0.0, 0.0, 0.0)
        att.update_accel_xyz(0.0, 0.0, 0.0)
        out_att = att.update()
        if out_att is None:
            continue

        runtime_flags = att.get_mppi_runtime_flags()
        vehicle_state = att.get_mppi_vehicle_state()
        wp = att.get_mppi_waypoints_ned()
        disturbance_acc = att.get_gpr_disturbance_acc()

        mppi.update_runtime_flags(runtime_flags)
        mppi.update_vehicle_state(vehicle_state)
        if i == 0 and wp is not None:
            mppi.update_waypoints(wp)
        mppi.update_disturbance_acc(disturbance_acc)
        out_mppi = mppi.solve()
        if out_mppi is not None:
            att.update_mppi_output(out_mppi[0], out_mppi[1], out_mppi[2])

        if runtime_flags[3] == 0:
            seen_reset_zero = True

    mppi.close()
    print(f"[BRIDGE] reset_flag_became_zero={seen_reset_zero}")
    if not seen_reset_zero:
        raise RuntimeError("ATT<->MPPI bridge smoke test failed: reset_flag never became 0.")


def main() -> None:
    run_att_core_only()
    run_att_mppi_bridge()
    print("CORE smoke tests finished.")


if __name__ == "__main__":
    main()

