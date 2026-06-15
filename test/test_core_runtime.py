############################################################
#
#   - Name : test_core_runtime.py
#
#                                 - KAIST FDCL, 2026.03.11
#
############################################################

from __future__ import annotations

import math
import numpy as np
import pytest

from pathfollowing.core.core_pathfollowing import PathFollowingCore
from pathfollowing.core.gpr import GPRConfig, GPRCore
from pathfollowing.core.waypoint import build_waypoints


def _vehicle_cfg() -> dict:
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


def _sim_cfg() -> dict:
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


def _update_att_sensor(att_core: PathFollowingCore, step: int, x: float, vx: float) -> None:
    att_core.update_timesync_timestamp_us(1_000_000 + step * 4_000)
    att_core.update_local_position(x, 0.0, -10.0, vx, 0.0, 0.0)
    att_core.update_attitude_quat(1.0, 0.0, 0.0, 0.0)
    att_core.update_accel_xyz(0.0, 0.0, 0.0)


def test_att_core_only_smoke() -> None:
    att = PathFollowingCore(
        guid_type=2,
        wp_type=0,
        vehicle_cfg=_vehicle_cfg(),
        sim_cfg=_sim_cfg(),
        logger_obj=None,
    )
    att.update_heartbeats(True, True, True)
    att.update_waypoints(False, [0.0, 20.0, 40.0], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0])

    outputs = []
    for i in range(60):
        _update_att_sensor(att, i, x=float(i), vx=8.0)
        out = att.update()
        if out is not None:
            outputs.append(out)

    assert outputs
    assert any(int(o["heading_wp_idx"]) >= 2 for o in outputs)

    att.update_waypoints(True, [0.0, 20.0], [0.0, 0.0], [10.0, 10.0])
    assert att.pop_waypoint_ack() is True
    assert att.pop_waypoint_ack() is False


def test_att_mppi_core_bridge_runtime_scenarios() -> None:
    pytest.importorskip("pycuda", reason="pycuda not installed")
    from pathfollowing.core.core_mppi import MPPICore

    att = PathFollowingCore(
        guid_type=2,
        wp_type=0,
        vehicle_cfg=_vehicle_cfg(),
        sim_cfg=_sim_cfg(),
        logger_obj=None,
    )
    mppi = MPPICore(guid_type=2, vehicle_cfg=_vehicle_cfg(), sim_cfg=_sim_cfg())
    att.update_heartbeats(True, True, True)

    att.update_waypoints(
        False,
        [0.0, 20.0, 40.0, 60.0],
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0],
    )

    step = 0

    def bridge_step(x: float, push_wp: bool) -> list[int]:
        nonlocal step
        _update_att_sensor(att, step, x=x, vx=8.0)
        step += 1

        out_att = att.update()
        assert out_att is not None

        runtime_flags = att.get_mppi_runtime_flags()
        vehicle_state = att.get_mppi_vehicle_state()
        wp = att.get_mppi_waypoints_ned()
        disturbance_acc = att.get_gpr_disturbance_acc()

        mppi.update_runtime_flags(runtime_flags)
        mppi.update_vehicle_state(vehicle_state)
        if push_wp and wp is not None:
            mppi.update_waypoints(wp)
        mppi.update_disturbance_acc(disturbance_acc)

        out_mppi = mppi.solve()
        assert out_mppi is not None
        att.update_mppi_output(out_mppi[0], out_mppi[1], out_mppi[2])
        return runtime_flags

    s1 = []
    for i in range(80):
        s1.append(bridge_step(x=float(i), push_wp=(i == 0)))

    s1_reset = [r[3] for r in s1]
    assert any(v == 0 for v in s1_reset)

    att.update_waypoints(
        False,
        [80.0, 100.0, 120.0],
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
    )

    s2 = []
    for i in range(30):
        s2.append(bridge_step(x=80.0 + float(i), push_wp=(i == 0)))

    assert s2[0][3] == 1
    assert any(r[3] == 0 for r in s2)

    att.update_guid_type(1)

    s3 = []
    for i in range(30):
        s3.append(bridge_step(x=110.0 + float(i), push_wp=False))

    assert s3[0][2] == 1
    assert s3[0][3] == 1
    assert any(r[3] == 0 for r in s3[1:])

    mppi.close()


def test_stop_flag_freezes_virtual_target() -> None:
    att = PathFollowingCore(
        guid_type=0,
        wp_type=0,
        vehicle_cfg=_vehicle_cfg(),
        sim_cfg=_sim_cfg(),
        logger_obj=None,
    )
    att.update_heartbeats(True, True, True)
    att.update_waypoints(False, [0.0, 30.0, 60.0], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0])

    _update_att_sensor(att, step=0, x=0.0, vx=6.0)
    out0 = att.update()
    assert out0 is not None
    vt_before = att.path.virtual_target_ned.copy()

    att.update_stop_flag(True)
    _update_att_sensor(att, step=1, x=15.0, vx=6.0)
    out1 = att.update()
    assert out1 is not None
    vt_frozen = att.path.virtual_target_ned.copy()
    assert (vt_frozen == vt_before).all()

    att.update_stop_flag(False)
    _update_att_sensor(att, step=2, x=25.0, vx=6.0)
    out2 = att.update()
    assert out2 is not None
    vt_after = att.path.virtual_target_ned.copy()
    assert not (vt_after == vt_frozen).all()


def test_waypoint_completion_legacy_transition_semantics() -> None:
    att = PathFollowingCore(
        guid_type=0,
        wp_type=0,
        vehicle_cfg=_vehicle_cfg(),
        sim_cfg=_sim_cfg(),
        logger_obj=None,
    )
    waypoints = np.array(
        [
            [0.0, 0.0, -10.0],
            [5.0, 0.0, -10.0],
            [10.0, 0.0, -10.0],
        ],
        dtype=float,
    )

    # Legacy behavior: if close enough at heading transition to final WP, completion becomes true.
    next_heading, done = att._check_waypoint(
        path_waypoints=waypoints,
        heading_wp_idx=1,
        position_ned=np.array([5.2, 0.0, -10.0], dtype=float),
        wp_switch_distance=4.5,
        final_wp_tolerance=0.5,
    )
    assert next_heading == 2
    assert done is True

    # Once heading is already final, completion is also true when final distance is within tolerance.
    next_heading2, done2 = att._check_waypoint(
        path_waypoints=waypoints,
        heading_wp_idx=2,
        position_ned=np.array([9.9, 0.0, -10.0], dtype=float),
        wp_switch_distance=4.5,
        final_wp_tolerance=0.5,
    )
    assert next_heading2 == 2
    assert done2 is True


def test_yaw_wrap_across_pi_boundary_is_continuous() -> None:
    att = PathFollowingCore(
        guid_type=0,
        wp_type=0,
        vehicle_cfg=_vehicle_cfg(),
        sim_cfg=_sim_cfg(),
        logger_obj=None,
    )
    euler_rpy = np.array([0.0, 0.0, -math.pi + 0.01], dtype=float)
    dcm_ned_to_body = np.array(
        [
            [math.cos(euler_rpy[2]), math.sin(euler_rpy[2]), 0.0],
            [-math.sin(euler_rpy[2]), math.cos(euler_rpy[2]), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    waypoints = np.array([[0.0, 0.0, -10.0], [-1.0, 0.0, -10.0]], dtype=float)

    _, _, att_cmd_rpy = att._convert_acc_cmd_to_thrust_and_att_cmd(
        dcm_ned_to_body=dcm_ned_to_body,
        acc_cmd_ned=np.array([0.0, 0.0, -9.81], dtype=float),
        mass=2.0,
        max_total_thrust=100.0,
        path_waypoints=waypoints,
        heading_wp_idx=1,
        position_ned=np.array([0.0, 0.0, -10.0], dtype=float),
        euler_rpy=euler_rpy,
        yaw_cmd_limit_rad=att.cfg.px4_model.del_psi_cmd_limit_rad,
        max_tilt_rad=att.cfg.px4_model.max_tilt_rad,
    )

    yaw_delta = (att_cmd_rpy[2] - euler_rpy[2] + math.pi) % (2.0 * math.pi) - math.pi
    assert abs(yaw_delta) < 0.05


def test_vtp_decision_clips_out_of_range_passed_index() -> None:
    att = PathFollowingCore(
        guid_type=0,
        wp_type=0,
        vehicle_cfg=_vehicle_cfg(),
        sim_cfg=_sim_cfg(),
        logger_obj=None,
    )
    waypoints = np.array(
        [
            [0.0, 0.0, -10.0],
            [10.0, 0.0, -10.0],
            [20.0, 0.0, -10.0],
        ],
        dtype=float,
    )
    vt = att._vtp_decision(
        dist_to_path=1.0,
        lookahead_distance=4.0,
        closest_point_ned=np.array([1.0, 0.0, -10.0], dtype=float),
        passed_wp_idx=999,  # deliberate out-of-range index
        path_waypoints=waypoints,
    )
    assert np.isfinite(vt).all()


def test_runtime_flags_and_plot_complete_are_independent() -> None:
    att = PathFollowingCore(
        guid_type=0,
        wp_type=0,
        vehicle_cfg=_vehicle_cfg(),
        sim_cfg=_sim_cfg(),
        logger_obj=None,
    )
    att.update_heartbeats(True, True, True)
    att.update_waypoints(False, [0.0, 10.0, 20.0], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0])

    # Plotter ack should not overwrite path-following completion state.
    att.update_plot_waypoint_complete(True)
    assert att.path.plot_complete is True
    assert att.path.pf_done is False

    att.update_interrupt_flag(True)
    att.update_stop_flag(True)
    flags = att.get_runtime_flags_int()
    assert len(flags) == 6
    assert flags[0] == 0   # guid_type
    assert flags[1] == 1   # interrupt
    assert flags[2] == 1   # stop
    assert flags[5] == 1   # plot_complete


def test_gpr_core_update_and_forecast() -> None:
    cfg = GPRConfig.from_dict(_sim_cfg())
    gpr = GPRCore(cfg)

    for i in range(300):
        t = i * cfg.dt_gpr
        measurement = np.array(
            [
                math.sin(0.3 * t),
                0.5 * math.cos(0.2 * t),
                0.0,
            ],
            dtype=np.float64,
        )
        gpr.update(t, measurement)
        if i % max(cfg.dt_gpr_opt_mul, 1) == 0:
            gpr.optimize_hyperparams()

    mean, var = gpr.forecast()
    assert mean.shape == (cfg.forecast_steps, 3)
    assert var.shape == (cfg.forecast_steps, 1)
    assert np.isfinite(mean).all()
    assert np.isfinite(var).all()


def test_gpr_config_missing_key_raises() -> None:
    sim_cfg = _sim_cfg()
    del sim_cfg["gpr"]["measurement_matrix"]
    with pytest.raises(RuntimeError, match="missing required gpr keys"):
        GPRConfig.from_dict(sim_cfg)


def test_waypoint_builder_strict_validation() -> None:
    with pytest.raises(RuntimeError, match="unsupported wp_type"):
        build_waypoints(99)

    with pytest.raises(RuntimeError, match="lengths must match"):
        build_waypoints(0, waypoint_x=[0.0, 1.0], waypoint_y=[0.0], waypoint_z=[10.0, 10.0])

    with pytest.raises(RuntimeError, match="count must be >= 2"):
        build_waypoints(0, waypoint_x=[0.0], waypoint_y=[0.0], waypoint_z=[10.0])


def test_mppi_input_payload_validation() -> None:
    pytest.importorskip("pycuda", reason="pycuda not installed")
    from pathfollowing.core.core_mppi import MPPICore

    mppi = MPPICore(guid_type=2, vehicle_cfg=_vehicle_cfg(), sim_cfg=_sim_cfg())
    with pytest.raises(RuntimeError, match="runtime_flags requires 4 values"):
        mppi.update_runtime_flags([1, 2, 3])
    with pytest.raises(RuntimeError, match="vehicle_state requires 10 values"):
        mppi.update_vehicle_state([0.0] * 9)
    with pytest.raises(RuntimeError, match="waypoints_ned requires at least 2 waypoints"):
        mppi.update_waypoints([0.0, 0.0, -10.0])
    with pytest.raises(RuntimeError, match="GPR/in/disturbance_acc requires 4 values"):
        mppi.update_disturbance_acc([0.0, 0.0, 0.0])
    mppi.close()


def test_replan_waypoint_update_relocalizes_without_reset() -> None:
    att = PathFollowingCore(
        guid_type=0,
        wp_type=0,
        vehicle_cfg=_vehicle_cfg(),
        sim_cfg=_sim_cfg(),
        logger_obj=None,
    )
    att.update_heartbeats(True, True, True)

    # First path: full reset snaps tracking to the start.
    att.update_waypoints(
        False,
        [0.0, 20.0, 40.0, 60.0],
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0],
    )
    assert att.path.passed_index == 0
    assert att.path.heading_index == 1

    # Fly forward and hold at x=30 (on segment [20, 40]) so tracking advances.
    for i in range(40):
        _update_att_sensor(att, i, x=min(float(i), 30.0), vx=8.0)
        out = att.update()
        assert out is not None
    assert att.path.passed_index >= 1

    # Stand in for converged NDO/MPPI state to detect an unwanted forced reset.
    att.mppi.reset_flag = 0
    att.ndo.disturbance_acc_ned[:] = [1.0, 2.0, 3.0]

    # Replan: a fresh waypoint stream arrives mid-flight.
    att.update_waypoints(
        False,
        [0.0, 20.0, 40.0, 60.0],
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0],
    )

    # Must NOT snap back to the start; must re-localize to the segment that
    # contains the current position (x=30 -> segment [20, 40]).
    assert att.path.passed_index == 1
    assert att.path.heading_index == 2

    # NDO / MPPI state preserved (no forced reset on a live replan update).
    assert att.mppi.reset_flag == 0
    assert (att.ndo.disturbance_acc_ned == np.array([1.0, 2.0, 3.0])).all()


def test_att_core_without_drag_coeff_is_allowed() -> None:
    vehicle = _vehicle_cfg()
    PathFollowingCore(
        guid_type=0,
        wp_type=0,
        vehicle_cfg=vehicle,
        sim_cfg=_sim_cfg(),
        logger_obj=None,
    )

