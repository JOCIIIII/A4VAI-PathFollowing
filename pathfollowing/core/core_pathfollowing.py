############################################################
#
#   - Name : core_pathfollowing.py
#
#                                 - KAIST FDCL, 2026.03.11
#
############################################################

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np

from .waypoint import build_waypoints
from ..utils.math_utils import (
    quat_from_euler,
    euler_from_quat,
    azim_elev_from_vec3,
    dcm_from_euler321,
)


def _require_sections(cfg: dict[str, Any], required: tuple[str, ...], scope: str) -> None:
    missing = [k for k in required if k not in cfg]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(f"[FATAL] missing required {scope} keys: {missing_list}")


# =====================================================
# config dataclasses
# =====================================================
@dataclass
class VehicleModelConfig:
    mass: float
    ixx: float
    iyy: float
    izz: float
    gravity: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VehicleModelConfig":
        _require_sections(cfg, ("mass", "Ixx", "Iyy", "Izz", "gravity"), "model")

        mass = float(cfg["mass"])
        ixx = float(cfg["Ixx"])
        iyy = float(cfg["Iyy"])
        izz = float(cfg["Izz"])
        gravity = float(cfg["gravity"])
        if mass <= 0.0 or ixx <= 0.0 or iyy <= 0.0 or izz <= 0.0 or gravity <= 0.0:
            raise RuntimeError("[FATAL] model mass/inertia/gravity must be > 0.")

        return cls(
            mass=mass,
            ixx=ixx,
            iyy=iyy,
            izz=izz,
            gravity=gravity,
        )


@dataclass
class ActuatorConfig:
    n_motor: int
    max_thrust_per_rotor: float
    rotor_positions: np.ndarray
    spin_direction: np.ndarray

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ActuatorConfig":
        _require_sections(
            cfg,
            ("n_motor", "max_thrust_per_rotor", "rotor_positions", "spin_direction"),
            "actuator",
        )
        rotor_positions = np.asarray(cfg["rotor_positions"], dtype=float)
        spin_direction = np.asarray(cfg["spin_direction"], dtype=float)
        n_motor = int(cfg["n_motor"])
        max_thrust_per_rotor = float(cfg["max_thrust_per_rotor"])

        if rotor_positions.ndim != 2 or rotor_positions.shape[1] != 3:
            raise RuntimeError("[FATAL] actuator.rotor_positions must be shape (N, 3).")

        if n_motor <= 0 or max_thrust_per_rotor <= 0.0:
            raise RuntimeError("[FATAL] actuator.n_motor and max_thrust_per_rotor must be > 0.")

        if rotor_positions.shape[0] != n_motor:
            raise RuntimeError("[FATAL] rotor_positions row count must match n_motor.")

        if len(spin_direction) != n_motor:
            raise RuntimeError("[FATAL] actuator.spin_direction length must match n_motor.")

        return cls(
            n_motor=n_motor,
            max_thrust_per_rotor=max_thrust_per_rotor,
            rotor_positions=rotor_positions,
            spin_direction=spin_direction,
        )

    @property
    def max_total_thrust(self) -> float:
        return float(self.n_motor) * float(self.max_thrust_per_rotor)


@dataclass
class PX4ModelConfig:
    max_tilt_deg: float
    max_accel_xy: float
    max_accel_z: float

    tau_phi: float
    tau_the: float
    tau_psi: float
    del_psi_cmd_limit_deg: float

    tau_wb: float

    tau_p: float
    tau_q: float
    tau_r: float

    alpha_p: float
    alpha_q: float
    alpha_r: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "PX4ModelConfig":
        _require_sections(
            cfg,
            (
                "max_tilt_deg",
                "max_accel_xy",
                "max_accel_z",
                "tau_phi",
                "tau_the",
                "tau_psi",
                "del_psi_cmd_limit_deg",
                "tau_wb",
                "tau_p",
                "tau_q",
                "tau_r",
                "alpha_p",
                "alpha_q",
                "alpha_r",
            ),
            "px4_model",
        )
        return cls(
            max_tilt_deg=float(cfg["max_tilt_deg"]),
            max_accel_xy=float(cfg["max_accel_xy"]),
            max_accel_z=float(cfg["max_accel_z"]),
            tau_phi=float(cfg["tau_phi"]),
            tau_the=float(cfg["tau_the"]),
            tau_psi=float(cfg["tau_psi"]),
            del_psi_cmd_limit_deg=float(cfg["del_psi_cmd_limit_deg"]),
            tau_wb=float(cfg["tau_wb"]),
            tau_p=float(cfg["tau_p"]),
            tau_q=float(cfg["tau_q"]),
            tau_r=float(cfg["tau_r"]),
            alpha_p=float(cfg["alpha_p"]),
            alpha_q=float(cfg["alpha_q"]),
            alpha_r=float(cfg["alpha_r"]),
        )

    @property
    def max_tilt_rad(self) -> float:
        return math.radians(self.max_tilt_deg)

    @property
    def del_psi_cmd_limit_rad(self) -> float:
        return math.radians(self.del_psi_cmd_limit_deg)


@dataclass
class GuidanceTuningConfig:
    final_wp_tolerance_m: float
    transition_speed_threshold_mps: float
    position_kp: float
    position_kv_multiplier: float
    min_guid_eta: float
    gl_vertical_speed_gain: float
    # Optional speed-regulated pursuit for guid 0. With the defaults the legacy
    # law (|v_cmd| = position_kp * distance to the virtual target) is unchanged.
    pos_speed_mode: bool = False
    lookahead_time_s: float = 0.0
    lookahead_max_m: float = 0.0
    brake_accel_mps2: float = 0.0
    # Speed mode extras: centripetal feed-forward from the path curvature
    # (|a| = gain * v^2 * kappa, taken lead_s ahead to cover the velocity-loop lag,
    # capped at max_accel) and a cross-track velocity term that pulls the vehicle
    # back onto the path even when the speed reference is ~0.
    curvature_ff_gain: float = 0.0
    curvature_ff_lead_s: float = 0.0
    curvature_ff_max_accel: float = 0.0
    xt_gain: float = 0.0
    ndo_max_accel: float = 0.0
    # declare done only below this speed [m/s] (0 = legacy): done hands over to the
    # flight manager's landing, which drifted 0.8-1.8 m when entered at 1-2 m/s
    done_speed_max: float = 0.0
    # terminal convergence [m/s] (0 = off): once the vehicle is at/past the end of the path
    # (no path length left) it is pulled back to the last waypoint at
    # min(terminal_speed_max, position_kp * distance). Without it an overshoot of the end
    # left speed_ref at 0 (no remaining length), so the vehicle hovered just outside the
    # done radius and SUPER kept replanning around the goal (s7l).
    terminal_speed_max: float = 0.0
    # vertical priority (PX4 style, default off): keep the commanded vertical specific force
    # (>= 0.2 g up), fit the horizontal part into the tilt cone and the thrust limit, and size
    # the thrust for the PLANNED attitude (vertical / cos(tilt)). The legacy thrust is the
    # projection of the command onto the CURRENT body axis, which collapses to its 0.2 g floor
    # when the command swings far from the present attitude: a 5 m/s reversal commanded by the
    # noclimb CA policy (nc_c) cut thrust to 0.14 and dropped the vehicle 1.2 m at 1.6 m/s.
    vertical_priority: bool = False
    # MPPI engages on speed (default off): leave the pursuit start-up law once the vehicle moves
    # faster than transition_speed_threshold_mps even before the first waypoint is passed - the
    # rule the MPPI rollout kernel already uses. Off, MPPI waits for passed_wp_idx >= 1, which a
    # path re-sent from the vehicle position every step (learned CA, 20 Hz) never reaches.
    mppi_speed_transition: bool = False
    # yaw command: 'waypoint' (default: toward the heading waypoint), 'velocity' (along the
    # vehicle's horizontal velocity, i.e. the direction of travel; below yaw_min_speed the last
    # yaw is kept) or 'hold' (the yaw when following starts). With the learned CA policy's
    # waypoints jumping every 50 ms, 'waypoint' made the yaw setpoint wobble (~80 deg/s rms);
    # the velocity cannot jump, so 'velocity' points the nose where the vehicle goes, smoothly.
    yaw_mode: str = "waypoint"
    yaw_hold: bool = False             # legacy switch for yaw_mode 'hold'
    yaw_rate_limit_deg: float = 0.0    # max rate of the yaw command [deg/s], 0 = none
    yaw_min_speed: float = 1.0         # 'velocity' mode: below this horizontal speed keep the yaw
    # first-order smoothing of the MPPI output (updated at 20 Hz, applied at 250 Hz) [s], 0 = off
    mppi_output_tau_s: float = 0.0
    # MPPI's lateral law is gain * speed * sin(heading error): with the target behind (a reversal
    # commanded by the learned CA) it vanishes and the vehicle kept flying away, climbing and
    # tilting to 50 deg. Beyond this heading error [deg] the pursuit law takes the step
    # (decelerate and turn); the rollout model applies the same rule. 0 = off.
    mppi_max_heading_error_deg: float = 0.0
    # speed feedback added to MPPI's forward acceleration: u0 + kp * (speed_ref - speed) [1/s].
    # MPPI alone regulated the speed slowly (5 m/s reference, the vehicle drifted to 6.6 m/s with
    # u0 near 0); the rollout model applies the same term. 0 = off.
    mppi_speed_kp: float = 0.0
    # speed cap on the forward acceleration: min(acc, kp * (speed_ref + margin - speed)).
    # With mppi_speed_kp alone the vehicle still settled at 6.6 m/s for a 5 m/s reference: MPPI's
    # u0 random-walked around +1.2 (white per-step exploration barely moves the speed inside the
    # horizon) and cancelled the feedback. Continuous, so no switching; the rollout model applies
    # the same cap. kp 0 = off.
    mppi_speed_guard_kp: float = 0.0
    mppi_speed_guard_margin: float = 0.3
    # corner speed (default off): at a path vertex turning by dpsi the vehicle is meant to fly the
    # arc of radius R = corner_max_error_m / (1 - cos(dpsi/2)) - the arc's mid-point is then
    # corner_max_error_m inside the polyline - at v_c = sqrt(corner_lat_accel_mps2 * R) (floor
    # corner_min_speed_mps). The arc starts d = R tan(dpsi/2) before the vertex (at most half of
    # either segment, R shrinks to fit); before it speed_ref <= sqrt(v_c^2 + 2 a (s - d)) with
    # a = corner_brake_accel_mps2 and s the distance to the vertex, after the vertex the same law
    # on the distance from it (the turn's second half, then speeding up). Kinks below
    # corner_min_angle_deg are ignored. Without it MPPI flew 90 deg corners of a 50 m square at
    # 10 m/s and overshot 8-10 m (35 deg tilt limit: turn radius ~14.5 m). The rollout model
    # applies the same limit (see mppi_rollout_sync).
    corner_speed_enable: bool = False
    corner_lat_accel_mps2: float = 4.0
    corner_max_error_m: float = 1.5
    corner_brake_accel_mps2: float = 3.0
    corner_min_speed_mps: float = 1.0
    corner_min_angle_deg: float = 10.0
    # MPPI rollout = PF guidance (default off): the rollout also uses the PF's speed-dependent
    # lookahead (lookahead_time_s / lookahead_max_m; the rollout otherwise keeps the fixed
    # lookahead_distance - 2 m against the PF's 6 m at 10 m/s) and the end-of-path braking cap
    # (brake_accel_mps2). Read by node_mppi (core_mppi.GuidanceTuningConfig).
    mppi_rollout_sync: bool = False

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        scope: str = "path_following",
    ) -> "GuidanceTuningConfig":
        _require_sections(
            cfg,
            (
                "final_wp_tolerance_m",
                "transition_speed_threshold_mps",
                "position_kp",
                "position_kv_multiplier",
                "min_guid_eta",
                "gl_vertical_speed_gain",
            ),
            scope,
        )

        final_wp_tolerance_m = float(cfg["final_wp_tolerance_m"])
        transition_speed_threshold_mps = float(cfg["transition_speed_threshold_mps"])
        position_kp = float(cfg["position_kp"])
        position_kv_multiplier = float(cfg["position_kv_multiplier"])
        min_guid_eta = float(cfg["min_guid_eta"])
        gl_vertical_speed_gain = float(cfg["gl_vertical_speed_gain"])

        if final_wp_tolerance_m <= 0.0:
            raise RuntimeError(f"[FATAL] {scope}.final_wp_tolerance_m must be > 0.")
        if transition_speed_threshold_mps <= 0.0:
            raise RuntimeError(f"[FATAL] {scope}.transition_speed_threshold_mps must be > 0.")
        if position_kp <= 0.0 or position_kv_multiplier <= 0.0:
            raise RuntimeError(f"[FATAL] {scope} position gains must be > 0.")
        if min_guid_eta <= 0.0:
            raise RuntimeError(f"[FATAL] {scope}.min_guid_eta must be > 0.")
        if gl_vertical_speed_gain <= 0.0:
            raise RuntimeError(f"[FATAL] {scope}.gl_vertical_speed_gain must be > 0.")

        lookahead_time_s = float(cfg.get("lookahead_time_s", 0.0))
        lookahead_max_m = float(cfg.get("lookahead_max_m", 0.0))
        brake_accel_mps2 = float(cfg.get("brake_accel_mps2", 0.0))
        curvature_ff_gain = float(cfg.get("curvature_ff_gain", 0.0))
        curvature_ff_lead_s = float(cfg.get("curvature_ff_lead_s", 0.0))
        curvature_ff_max_accel = float(cfg.get("curvature_ff_max_accel", 0.0))
        xt_gain = float(cfg.get("xt_gain", 0.0))
        ndo_max_accel = float(cfg.get("ndo_max_accel", 0.0))
        done_speed_max = float(cfg.get("done_speed_max", 0.0))
        terminal_speed_max = float(cfg.get("terminal_speed_max", 0.0))
        if min(lookahead_time_s, lookahead_max_m, brake_accel_mps2, curvature_ff_gain,
               curvature_ff_lead_s, curvature_ff_max_accel, xt_gain, ndo_max_accel,
               done_speed_max, terminal_speed_max) < 0.0:
            raise RuntimeError(f"[FATAL] {scope} speed-mode parameters must be >= 0.")

        return cls(
            final_wp_tolerance_m=final_wp_tolerance_m,
            transition_speed_threshold_mps=transition_speed_threshold_mps,
            position_kp=position_kp,
            position_kv_multiplier=position_kv_multiplier,
            min_guid_eta=min_guid_eta,
            gl_vertical_speed_gain=gl_vertical_speed_gain,
            pos_speed_mode=bool(cfg.get("pos_speed_mode", False)),
            lookahead_time_s=lookahead_time_s,
            lookahead_max_m=lookahead_max_m,
            brake_accel_mps2=brake_accel_mps2,
            curvature_ff_gain=curvature_ff_gain,
            curvature_ff_lead_s=curvature_ff_lead_s,
            curvature_ff_max_accel=curvature_ff_max_accel,
            xt_gain=xt_gain,
            ndo_max_accel=ndo_max_accel,
            done_speed_max=done_speed_max,
            terminal_speed_max=terminal_speed_max,
            vertical_priority=bool(cfg.get("vertical_priority", False)),
            mppi_speed_transition=bool(cfg.get("mppi_speed_transition", False)),
            yaw_mode=str(cfg.get("yaw_mode", "hold" if bool(cfg.get("yaw_hold", False)) else "waypoint")),
            yaw_hold=bool(cfg.get("yaw_hold", False)),
            yaw_rate_limit_deg=max(float(cfg.get("yaw_rate_limit_deg", 0.0)), 0.0),
            yaw_min_speed=max(float(cfg.get("yaw_min_speed", 1.0)), 0.0),
            mppi_output_tau_s=max(float(cfg.get("mppi_output_tau_s", 0.0)), 0.0),
            mppi_max_heading_error_deg=max(float(cfg.get("mppi_max_heading_error_deg", 0.0)), 0.0),
            mppi_speed_kp=max(float(cfg.get("mppi_speed_kp", 0.0)), 0.0),
            mppi_speed_guard_kp=max(float(cfg.get("mppi_speed_guard_kp", 0.0)), 0.0),
            mppi_speed_guard_margin=max(float(cfg.get("mppi_speed_guard_margin", 0.3)), 0.0),
            corner_speed_enable=bool(cfg.get("corner_speed_enable", False)),
            corner_lat_accel_mps2=max(float(cfg.get("corner_lat_accel_mps2", 4.0)), 0.1),
            corner_max_error_m=max(float(cfg.get("corner_max_error_m", 1.5)), 0.05),
            corner_brake_accel_mps2=max(float(cfg.get("corner_brake_accel_mps2", 3.0)), 0.1),
            corner_min_speed_mps=max(float(cfg.get("corner_min_speed_mps", 1.0)), 0.0),
            corner_min_angle_deg=max(float(cfg.get("corner_min_angle_deg", 10.0)), 0.1),
            mppi_rollout_sync=bool(cfg.get("mppi_rollout_sync", False)),
        )


@dataclass
class PathFollowingConfig:
    dt_pf: float
    desired_speed: float
    lookahead_distance: float
    waypoint_switch_distance: float

    kp_vel: float
    kd_vel: float

    kp_speed: float
    kd_speed: float
    guid_eta: float
    guidance_tuning: GuidanceTuningConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "PathFollowingConfig":
        _require_sections(
            cfg,
            (
                "dt_pf",
                "desired_speed",
                "lookahead_distance",
                "waypoint_switch_distance",
                "kp_vel",
                "kd_vel",
                "kp_speed",
                "kd_speed",
                "guid_eta",
            ),
            "path_following",
        )
        dt_pf = float(cfg["dt_pf"])
        desired_speed = float(cfg["desired_speed"])
        lookahead_distance = float(cfg["lookahead_distance"])
        waypoint_switch_distance = float(cfg["waypoint_switch_distance"])
        if dt_pf <= 0.0 or desired_speed < 0.0 or lookahead_distance <= 0.0 or waypoint_switch_distance <= 0.0:
            raise RuntimeError("[FATAL] path_following dt/speed/distances are invalid.")

        if "guidance_tuning" in cfg:
            guidance_cfg = cfg["guidance_tuning"]
            if not isinstance(guidance_cfg, dict):
                raise RuntimeError("[FATAL] path_following.guidance_tuning must be a mapping.")
            guidance = GuidanceTuningConfig.from_dict(guidance_cfg, scope="path_following.guidance_tuning")
        else:
            guidance = GuidanceTuningConfig.from_dict(cfg, scope="path_following")

        return cls(
            dt_pf=dt_pf,
            desired_speed=desired_speed,
            lookahead_distance=lookahead_distance,
            waypoint_switch_distance=waypoint_switch_distance,
            kp_vel=float(cfg["kp_vel"]),
            kd_vel=float(cfg["kd_vel"]),
            kp_speed=float(cfg["kp_speed"]),
            kd_speed=float(cfg["kd_speed"]),
            guid_eta=float(cfg["guid_eta"]),
            guidance_tuning=guidance,
        )


@dataclass
class MPPITypeConfig:
    state_cost: np.ndarray
    terminal_cost: np.ndarray
    noise_std_u0_scale: float
    noise_std_u1_scale: float
    lambda_weight: float
    u0_init: float
    u1_init: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "MPPITypeConfig":
        _require_sections(
            cfg,
            (
                "state_cost",
                "terminal_cost",
                "noise_std_u0_scale",
                "noise_std_u1_scale",
                "lambda_weight",
                "u0_init",
                "u1_init",
            ),
            "mppi.type",
        )
        state_cost = np.asarray(cfg["state_cost"], dtype=float).reshape(-1)
        terminal_cost = np.asarray(cfg["terminal_cost"], dtype=float).reshape(-1)
        if state_cost.size < 3 or terminal_cost.size < 3:
            raise RuntimeError("[FATAL] mppi state_cost/terminal_cost must have at least 3 elements.")

        return cls(
            state_cost=state_cost,
            terminal_cost=terminal_cost,
            noise_std_u0_scale=float(cfg["noise_std_u0_scale"]),
            noise_std_u1_scale=float(cfg["noise_std_u1_scale"]),
            lambda_weight=float(cfg["lambda_weight"]),
            u0_init=float(cfg["u0_init"]),
            u1_init=float(cfg["u1_init"]),
        )


@dataclass
class MPPIConfig:
    dt_mppi: float
    num_samples: int
    horizon_steps: int
    cost_min_v_aligned: float
    type_1: MPPITypeConfig
    type_2: MPPITypeConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "MPPIConfig":
        _require_sections(
            cfg,
            ("dt_mppi", "num_samples", "horizon_steps", "cost_min_v_aligned"),
            "mppi",
        )
        dt_mppi = float(cfg["dt_mppi"])
        num_samples = int(cfg["num_samples"])
        horizon_steps = int(cfg["horizon_steps"])
        if dt_mppi <= 0.0 or num_samples <= 0 or horizon_steps <= 0:
            raise RuntimeError("[FATAL] mppi dt_mppi/num_samples/horizon_steps must be > 0.")

        # New keys: type_1/type_2, while still accepting legacy type_3/type_4.
        if "type_1" in cfg and "type_2" in cfg:
            type_1_key = "type_1"
            type_2_key = "type_2"
        elif "type_3" in cfg and "type_4" in cfg:
            type_1_key = "type_3"
            type_2_key = "type_4"
        else:
            raise RuntimeError(
                "[FATAL] mppi requires type_1/type_2 (legacy type_3/type_4 also supported)."
            )

        return cls(
            dt_mppi=dt_mppi,
            num_samples=num_samples,
            horizon_steps=horizon_steps,
            cost_min_v_aligned=float(cfg["cost_min_v_aligned"]),
            type_1=MPPITypeConfig.from_dict(cfg[type_1_key]),
            type_2=MPPITypeConfig.from_dict(cfg[type_2_key]),
        )

    def selected(self, guid_type: int) -> MPPITypeConfig:
        if int(guid_type) == 2:
            return self.type_2
        return self.type_1


@dataclass
class NDOConfig:
    gain: np.ndarray

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "NDOConfig":
        _require_sections(cfg, ("gain",), "ndo")
        gain = np.asarray(cfg["gain"], dtype=float).reshape(-1)
        if gain.size != 3:
            raise RuntimeError("[FATAL] ndo.gain must have 3 elements.")
        return cls(gain=gain)


@dataclass
class NumericsConfig:
    eps: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "NumericsConfig":
        _require_sections(cfg, ("numerics",), "simulation config")
        numerics_cfg = cfg["numerics"]
        _require_sections(numerics_cfg, ("eps",), "numerics")
        eps = float(numerics_cfg["eps"])
        if not np.isfinite(eps) or eps <= 0.0:
            raise RuntimeError("[FATAL] numerics.eps must be finite and > 0.")
        return cls(eps=eps)


@dataclass
class GroundVehicleConfig:
    # Ground (UGV) / surface (USV) vehicle under PX4 rover_differential offboard velocity control:
    # the PX4 setpoint is a speed and a bearing (= yaw setpoint), PX4 closes speed -> throttle and
    # bearing -> yaw (P, yaw_kp) -> yaw rate -> differential drive. The guidance law is unchanged
    # (acceleration command in NED); only the output stage differs from the multicopter's
    # acceleration -> thrust + attitude:
    #   along-track accel  a_t -> speed command   v + a_t * tau_speed      (inverse of the speed lag)
    #   lateral accel      a_n -> yaw rate        a_n / max(v, rate_speed_floor), |r| <= yaw_rate_max
    #                          -> bearing command  yaw + r / yaw_kp          (inverse of PX4's yaw P)
    # velocity_frame       the velocity the MPPI-guided steps (and the MPPI rollout's initial state) use.
    #                      'heading': the forward speed along the yaw, no lateral velocity - a wheeled rover
    #                      cannot move sideways, and what the estimator reports there (0.1-0.3 m/s of
    #                      noise) was fed back as a turn command (pf_ugv_line_g0: +-20 deg of yaw wobble).
    #                      'course': the estimated velocity as it is - a boat slips ~20 deg sideways in
    #                      turns; the rollout models it (sway_tau, mppi_kernel.cu GND_SWAY_TAU).
    #                      The pursuit law (guid 0) always uses 'heading': its velocity-error feedback on the
    #                      boat's sway made it oscillate (offline loop: circle p95 3.0 vs 1.1 m)
    # speed_max            [m/s] PX4 RO_SPEED_LIM
    # tau_speed            [s]   measured lag of PX4's speed loop (step test)
    # yaw_kp               [1/s] PX4 RO_YAW_P
    # yaw_rate_max         [rad/s] <= PX4 RO_YAW_RATE_LIM
    # bearing_lead_max_deg how far the bearing may lead the yaw while driving: below PX4's
    #                      drive -> spot-turn switch (RD_TRANS_DRV_TRN), or the rover stops to turn
    # course_min_speed     [m/s] direction of travel = velocity azimuth above it, the yaw below
    # rate_speed_floor     [m/s] speed floor of a_n / v
    # align_enter_deg / align_exit_deg / align_speed_max: below align_speed_max and more than
    #                      align_enter_deg off the direction to the virtual target, the bearing is
    #                      that direction itself (PX4 turns on the spot or in its smallest circle)
    #                      until within align_exit_deg - a start that is not aligned with the path
    speed_max: float
    tau_speed: float
    yaw_kp: float
    yaw_rate_max: float
    bearing_lead_max_deg: float
    course_min_speed: float
    rate_speed_floor: float
    align_enter_deg: float
    align_exit_deg: float
    align_speed_max: float
    align_speed: float
    velocity_frame: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "GroundVehicleConfig":
        out = cls(
            velocity_frame=str(cfg.get("velocity_frame", "heading")),
            speed_max=float(cfg["speed_max"]),
            tau_speed=float(cfg["tau_speed"]),
            yaw_kp=float(cfg["yaw_kp"]),
            yaw_rate_max=float(cfg["yaw_rate_max"]),
            bearing_lead_max_deg=float(cfg["bearing_lead_max_deg"]),
            course_min_speed=float(cfg.get("course_min_speed", 0.3)),
            rate_speed_floor=float(cfg.get("rate_speed_floor", 0.5)),
            align_enter_deg=float(cfg.get("align_enter_deg", 45.0)),
            align_exit_deg=float(cfg.get("align_exit_deg", 15.0)),
            align_speed_max=float(cfg.get("align_speed_max", 0.5)),
            align_speed=float(cfg.get("align_speed", 0.0)),
        )
        if (out.speed_max <= 0.0 or out.tau_speed <= 0.0 or out.yaw_kp <= 0.0 or out.yaw_rate_max <= 0.0
                or out.bearing_lead_max_deg <= 0.0 or out.rate_speed_floor <= 0.0
                or out.velocity_frame not in ("heading", "course")):
            raise RuntimeError("[FATAL] invalid ground vehicle config.")
        return out


@dataclass
class CoreConfig:
    model: VehicleModelConfig
    actuator: ActuatorConfig
    px4_model: PX4ModelConfig
    path_following: PathFollowingConfig
    mppi: MPPIConfig
    ndo: NDOConfig
    numerics: NumericsConfig


# =====================================================
# state dataclasses
# =====================================================
@dataclass
class VehicleState:
    position_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    euler_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    accel_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    sim_time: float = 0.0


@dataclass
class PathState:
    waypoints_ned: np.ndarray | None = None
    ready: bool = False
    initialized: bool = False
    pf_done: bool = False
    plot_complete: bool = False

    passed_index: int = 0
    heading_index: int = 0
    closest_index: int = 0

    closest_point_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    virtual_target_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    cross_track_error: float = 0.0
    distance_to_heading_wp: float = 0.0
    distance_to_goal: float = float("inf")

    interrupt_active: int = 0
    interrupt_prev: int = 0
    stop_flag: int = 0

    # what the guidance used this step (logged): speed reference after the end-of-path and
    # corner limits (-1 = none) and the virtual-target lookahead [m]
    speed_ref: float = -1.0
    lookahead: float = 0.0


@dataclass
class MPPIState:
    u0: float = 0.0
    u1: float = 0.0
    solve_time: float = 0.0
    reset_flag: int = 0


@dataclass
class NDOState:
    observer_state_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    disturbance_acc_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


@dataclass
class AttitudeCommand:
    q_d: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    thrust_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    euler_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    yaw_sp_move_rate: float = 0.0
    total_thrust_cmd: float = 0.0
    thrust_norm_cmd: float = 0.0


@dataclass
class GroundCommand:
    speed: float = 0.0          # [m/s] speed setpoint
    bearing: float = 0.0        # [rad, NED] bearing = yaw setpoint
    yaw_rate: float = 0.0       # [rad/s] the turn rate the bearing lead asks for
    acc_t: float = 0.0          # [m/s^2] guidance acceleration along the direction of travel
    acc_n: float = 0.0          # [m/s^2] ... and to its right
    align: int = 0              # 1 while turning toward the path before driving


# =====================================================
# core
# =====================================================
class PathFollowingCore:
    def __init__(
        self,
        guid_type: int,
        wp_type: int,
        vehicle_cfg: dict[str, Any],
        sim_cfg: dict[str, Any],
        logger_obj: Any | None = None,
        vehicle_type: int = -1,
    ):
        self.vehicle_type = int(vehicle_type)
        self.guid_type = int(guid_type)
        self.guid_type_init = int(guid_type)
        self.wp_type = int(wp_type)
        if self.guid_type not in (0, 1, 2):
            raise RuntimeError(f"[FATAL] unsupported guid_type: {self.guid_type}")
        if self.wp_type not in set(range(0, 10)):
            raise RuntimeError(f"[FATAL] unsupported wp_type: {self.wp_type}")

        self.logger = logger_obj

        _require_sections(vehicle_cfg, ("model", "actuator", "px4_model"), "vehicle config")
        _require_sections(sim_cfg, ("path_following", "mppi", "ndo", "numerics"), "simulation config")

        self.cfg = CoreConfig(
            model=VehicleModelConfig.from_dict(vehicle_cfg["model"]),
            actuator=ActuatorConfig.from_dict(vehicle_cfg["actuator"]),
            px4_model=PX4ModelConfig.from_dict(vehicle_cfg["px4_model"]),
            path_following=PathFollowingConfig.from_dict(sim_cfg["path_following"]),
            mppi=MPPIConfig.from_dict(sim_cfg["mppi"]),
            ndo=NDOConfig.from_dict(sim_cfg["ndo"]),
            numerics=NumericsConfig.from_dict(sim_cfg),
        )

        # ground / surface vehicle: the vehicle yaml has a "ground" section (ugv.yaml, usv.yaml)
        self.ground = (GroundVehicleConfig.from_dict(vehicle_cfg["ground"])
                       if isinstance(vehicle_cfg.get("ground"), dict) else None)
        self.ground_cmd = GroundCommand()
        self._ground_align = False
        self._ground_end = False
        self._yaw_prev = None
        self._yaw_rate = 0.0

        self.state = VehicleState()
        self.path = PathState()
        self.mppi = MPPIState()
        self.ndo = NDOState()
        self.att_cmd = AttitudeCommand()
        # Legacy node initialized thrust command with 9.81.
        self.att_cmd.total_thrust_cmd = float(self.cfg.model.gravity)

        self.controller_heartbeat = False
        self.path_planning_heartbeat = False
        self.collision_avoidance_heartbeat = False

        self._time_initialized = False
        self._time0 = 0.0
        self._waypoint_ack_pending = False
        self._last_guid_type_used = int(self.guid_type)
        self._yaw_hold_rad = None
        self._yaw_cmd_prev = None
        self._mppi_u_filtered = None

        self._initialize_mppi_state()

        if self.wp_type != 0:
            self.path.waypoints_ned = build_waypoints(self.wp_type)
            self.path.ready = True
            self._reset_path_tracking_state(self.path.waypoints_ned)

    # =====================================================
    # properties
    # =====================================================
    @property
    def dt_gcu(self) -> float:
        return self.cfg.path_following.dt_pf

    @property
    def dt_mppi(self) -> float:
        return self.cfg.mppi.dt_mppi

    # =====================================================
    # public update functions
    # =====================================================
    def update_heartbeats(self, ctrl: bool, plan: bool, ca: bool) -> None:
        self.controller_heartbeat = bool(ctrl)
        self.path_planning_heartbeat = bool(plan)
        self.collision_avoidance_heartbeat = bool(ca)

    def update_timesync_timestamp_us(self, timestamp_us: int) -> None:
        t = float(timestamp_us) * 1.0e-6
        if not self._time_initialized:
            self._time0 = t
            self._time_initialized = True
            self.state.sim_time = 0.0
        else:
            self.state.sim_time = t - self._time0

    def update_local_position(
        self,
        x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        vz: float,
    ) -> None:
        if self.ground is not None:
            # planar: distances and waypoint switching are 3-D, and a ground EKF's z drifts
            z, vz = 0.0, 0.0
        self.state.position_ned[:] = [x, y, z]
        self.state.velocity_ned[:] = [vx, vy, vz]

    def update_attitude_quat(self, q0: float, q1: float, q2: float, q3: float, timestamp_us: int | None = None) -> None:
        roll, pitch, yaw = euler_from_quat(q0, q1, q2, q3)
        if self.ground is not None:
            roll, pitch = 0.0, 0.0
            # yaw rate for the MPPI rollout's initial state (low-passed difference of the attitude)
            if timestamp_us is not None:
                t = float(timestamp_us) * 1.0e-6
                if self._yaw_prev is not None and t > self._yaw_prev[0]:
                    dt = t - self._yaw_prev[0]
                    r = ((yaw - self._yaw_prev[1] + math.pi) % (2.0 * math.pi) - math.pi) / dt
                    a = dt / (0.05 + dt)
                    self._yaw_rate += a * (r - self._yaw_rate)
                self._yaw_prev = (t, yaw)
        self.state.euler_rpy[:] = [roll, pitch, yaw]

    def update_accel_xyz(self, ax: float, ay: float, az: float) -> None:
        self.state.accel_xyz[:] = [ax, ay, az]

    def update_waypoints(
        self,
        path_planning_complete: bool,
        waypoint_x,
        waypoint_y,
        waypoint_z,
    ) -> None:
        waypoints_ned = None
        should_reset_tracking = False

        if self.wp_type == 0:
            if self.ground is not None:
                waypoint_z = [0.0] * len(waypoint_x)
            waypoints_ned = build_waypoints(
                0,
                waypoint_x=waypoint_x,
                waypoint_y=waypoint_y,
                waypoint_z=waypoint_z,
            )
            should_reset_tracking = True
        elif not bool(path_planning_complete):
            # Legacy behavior: non-external waypoint modes also reset tracking on regeneration.
            waypoints_ned = build_waypoints(self.wp_type)
            should_reset_tracking = True

        if waypoints_ned is not None and should_reset_tracking:
            # Distinguish first-time setup from an in-flight replan update.
            # Once following is underway (path.initialized) and a path already
            # exists, a fresh waypoint stream is a replan: keep NDO/MPPI/guidance
            # state and just re-localize tracking indices onto the new path
            # (MPPI-style smooth update). Only the very first path does a full
            # reset that snaps tracking to the start.
            is_live_update = self.path.initialized and (self.path.waypoints_ned is not None)

            self.path.waypoints_ned = waypoints_ned
            self.path.ready = True

            self._ground_end = False
            if is_live_update:
                self._relocalize_on_path(waypoints_ned)
            else:
                self._reset_path_tracking_state(waypoints_ned)
                self.ndo.observer_state_ned[:] = 0.0
                self.ndo.disturbance_acc_ned[:] = 0.0
                self.mppi.reset_flag = 1

        if bool(path_planning_complete):
            self._waypoint_ack_pending = True

    def _mppi_input(self) -> np.ndarray:
        raw = np.array([self.mppi.u0, self.mppi.u1], dtype=float)
        tau = self.cfg.path_following.guidance_tuning.mppi_output_tau_s
        if tau <= 0.0:
            return raw
        if self._mppi_u_filtered is None:
            self._mppi_u_filtered = raw.copy()
        dt = self.cfg.path_following.dt_pf
        self._mppi_u_filtered += dt / (tau + dt) * (raw - self._mppi_u_filtered)
        return self._mppi_u_filtered.copy()

    def update_mppi_output(self, u0: float, u1: float, solve_time: float) -> None:
        values = np.asarray([u0, u1, solve_time], dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError("[FATAL] invalid MPPI output: contains non-finite value.")
        self.mppi.u0 = float(u0)
        self.mppi.u1 = float(u1)
        self.mppi.solve_time = float(solve_time)

    def update_plot_waypoint_complete(self, flag: bool) -> None:
        self.path.plot_complete = bool(flag)

    def update_wp_complete(self, flag: bool) -> None:
        # Backward-compatible alias.
        self.update_plot_waypoint_complete(flag)

    def update_interrupt_flag(self, flag: bool) -> None:
        self.path.interrupt_active = int(bool(flag))

    def update_desired_speed(self, speed: float) -> None:
        if math.isfinite(speed) and speed >= 0.0:
            self.cfg.path_following.desired_speed = float(speed)

    def update_guid_type(self, guid_type: int) -> None:
        new_guid_type = int(guid_type)
        if new_guid_type not in (0, 1, 2):
            raise ValueError(f"Unsupported guid_type: {new_guid_type}")

        if new_guid_type != self.guid_type:
            self.guid_type = new_guid_type
            self._initialize_mppi_state()
            self.mppi.reset_flag = 1

    def update_stop_flag(self, flag: bool) -> None:
        self.path.stop_flag = int(bool(flag))

    # =====================================================
    # main update
    # =====================================================
    def update(self) -> dict[str, Any] | None:
        if not (self.controller_heartbeat and self.path_planning_heartbeat and self.collision_avoidance_heartbeat):
            return None

        if not self.path.ready or self.path.waypoints_ned is None:
            return None

        if not self.path.initialized:
            self.path.initialized = True

        waypoints_ned = self.path.waypoints_ned
        position_ned = self.state.position_ned
        velocity_ned = self._guidance_velocity()
        velocity_pursuit_ned = self._guidance_velocity(pursuit=True) if self.ground is not None else None
        dcm_ned_to_body = dcm_from_euler321(self.state.euler_rpy)

        dist_to_path, closest_point_ned, passed_wp_idx = self._distance_to_path(
            path_waypoints=waypoints_ned,
            heading_wp_idx=self.path.heading_index,
            position_ned=position_ned,
            closest_point_ned=self.path.closest_point_ned.copy(),
            passed_wp_idx=self.path.passed_index,
        )

        self.path.cross_track_error = float(dist_to_path)
        self.path.closest_point_ned[:] = closest_point_ned
        self.path.passed_index = int(passed_wp_idx)

        heading_index, pf_done = self._check_waypoint(
            path_waypoints=waypoints_ned,
            heading_wp_idx=self.path.heading_index,
            position_ned=position_ned,
            wp_switch_distance=self.cfg.path_following.waypoint_switch_distance,
            final_wp_tolerance=self.cfg.path_following.guidance_tuning.final_wp_tolerance_m,
        )
        self.path.heading_index = int(heading_index)
        done_speed_max = self.cfg.path_following.guidance_tuning.done_speed_max
        if pf_done and done_speed_max > 0.0 and float(np.linalg.norm(velocity_ned)) > done_speed_max:
            pf_done = False
        self.path.pf_done = bool(pf_done)

        tuning = self.cfg.path_following.guidance_tuning
        lookahead = self.cfg.path_following.lookahead_distance
        if tuning.lookahead_time_s > 0.0:
            lookahead = max(lookahead, float(np.linalg.norm(velocity_ned)) * tuning.lookahead_time_s)
            if tuning.lookahead_max_m > 0.0:
                lookahead = min(lookahead, tuning.lookahead_max_m)

        speed_ref = -1.0
        if tuning.pos_speed_mode:
            speed_ref = float(self.cfg.path_following.desired_speed)
            if tuning.brake_accel_mps2 > 0.0:
                remaining = self._remaining_path_length(
                    waypoints_ned, self.path.passed_index, self.path.closest_point_ned
                )
                speed_ref = min(speed_ref, math.sqrt(2.0 * tuning.brake_accel_mps2 * remaining))
                if tuning.terminal_speed_max > 0.0 and remaining < 0.05:
                    d_end = float(np.linalg.norm(waypoints_ned[-1] - position_ned))
                    speed_ref = max(speed_ref, min(tuning.terminal_speed_max,
                                                   tuning.position_kp * d_end))
        if tuning.corner_speed_enable:
            if speed_ref < 0.0:
                speed_ref = float(self.cfg.path_following.desired_speed)
            speed_ref = min(speed_ref, self._corner_speed_limit(
                waypoints_ned, self.path.passed_index, self.path.closest_point_ned, speed_ref, tuning))
        self.path.speed_ref = float(speed_ref)
        self.path.lookahead = float(lookahead)

        acc_ff_ned = np.zeros(3, dtype=float)
        if tuning.pos_speed_mode and tuning.curvature_ff_gain > 0.0:
            acc_ff_ned = self._curvature_feedforward(
                waypoints_ned, self.path.passed_index, self.path.closest_point_ned,
                float(np.linalg.norm(velocity_ned)), tuning,
            )

        prev_virtual_target = self.path.virtual_target_ned.copy()
        virtual_target = self._vtp_decision(
            dist_to_path=dist_to_path,
            lookahead_distance=lookahead,
            closest_point_ned=self.path.closest_point_ned,
            passed_wp_idx=self.path.passed_index,
            path_waypoints=waypoints_ned,
        )
        if self.path.stop_flag == 1:
            virtual_target = prev_virtual_target
        self.path.virtual_target_ned[:] = virtual_target

        self.path.distance_to_heading_wp = float(
            np.linalg.norm(waypoints_ned[self.path.heading_index] - position_ned)
        )
        self.path.distance_to_goal = float(np.linalg.norm(waypoints_ned[-1] - position_ned))

        acc_cmd_ned, guid_type_used, interrupt_prev, reset_mppi = self._guidance_modules(
            guid_type=self.guid_type,
            passed_wp_idx=self.path.passed_index,
            heading_wp_idx=self.path.heading_index,
            num_waypoints=waypoints_ned.shape[0],
            virtual_target_ned=virtual_target,
            position_ned=position_ned,
            velocity_ned=velocity_ned,
            guid_eta=self.cfg.path_following.guid_eta,
            mppi_ctrl_input=self._mppi_input(),
            interrupt_active=self.path.interrupt_active,
            interrupt_prev=self.path.interrupt_prev,
            transition_speed_threshold_mps=self.cfg.path_following.guidance_tuning.transition_speed_threshold_mps,
            position_kp=self.cfg.path_following.guidance_tuning.position_kp,
            position_kv_multiplier=self.cfg.path_following.guidance_tuning.position_kv_multiplier,
            min_guid_eta=self.cfg.path_following.guidance_tuning.min_guid_eta,
            gl_vertical_speed_gain=self.cfg.path_following.guidance_tuning.gl_vertical_speed_gain,
            speed_ref=speed_ref,
            off_path=bool(dist_to_path >= lookahead),
            closest_point_ned=self.path.closest_point_ned,
            xt_gain=tuning.xt_gain,
            acc_ff_ned=acc_ff_ned,
            velocity_pursuit_ned=velocity_pursuit_ned,
        )
        self.path.interrupt_prev = int(interrupt_prev)
        self._last_guid_type_used = int(guid_type_used)

        if int(reset_mppi) == 1:
            self.mppi.reset_flag = 1

        if self.ground is not None:
            # ground / surface vehicle: no gravity, no thrust, no disturbance observer.
            # End of the path: the vehicle cannot hover back onto the last waypoint, so the path ends
            # once it is within the tolerance of the end OR has crossed the end line; from then on it
            # brakes on its heading and the path is done when it has (almost) stopped. Without this
            # the rover overshot by 0.8 m, turned round on the spot to drive back and kept hunting.
            end_ned = waypoints_ned[-1]
            seg_ned = end_ned - waypoints_ned[-2]
            if (self.path.heading_index >= waypoints_ned.shape[0] - 1
                    and (float(np.dot(position_ned - end_ned, seg_ned)) >= 0.0
                         or float(np.linalg.norm(end_ned - position_ned)) < tuning.final_wp_tolerance_m)):
                self._ground_end = True
            if self._ground_end:
                self.path.pf_done = bool(done_speed_max <= 0.0
                                         or float(np.linalg.norm(velocity_ned)) <= done_speed_max)
            self.ground_cmd = self._convert_acc_cmd_to_ground_cmd(
                acc_cmd_ned, virtual_target, velocity_pursuit_ned if int(guid_type_used) == 0 else velocity_ned)
            self._log_step()
            return {
                "att_cmd": self.att_cmd,
                "ground_cmd": self.ground_cmd,
                "heading_wp_idx": int(self.path.heading_index),
                "pf_done": bool(self.path.pf_done),
                "guid_type_used": int(guid_type_used),
            }

        acc_cmd_compensated = acc_cmd_ned.copy()
        acc_cmd_compensated = acc_cmd_compensated - self.ndo.disturbance_acc_ned
        acc_cmd_compensated[2] = acc_cmd_compensated[2] - self.cfg.model.gravity

        thrust_cmd_total, thrust_norm_cmd, att_cmd_rpy = self._convert_acc_cmd_to_thrust_and_att_cmd(
            dcm_ned_to_body=dcm_ned_to_body,
            acc_cmd_ned=acc_cmd_compensated,
            mass=self.cfg.model.mass,
            max_total_thrust=self.cfg.actuator.max_total_thrust,
            path_waypoints=waypoints_ned,
            heading_wp_idx=self.path.heading_index,
            position_ned=position_ned,
            euler_rpy=self.state.euler_rpy,
            yaw_cmd_limit_rad=self.cfg.px4_model.del_psi_cmd_limit_rad,
            max_tilt_rad=self.cfg.px4_model.max_tilt_rad,
        )

        # Use current-cycle thrust command for observer update.
        self._update_disturbance_acc(
            thrust_cmd_total=float(thrust_cmd_total),
        )

        q_d = quat_from_euler(att_cmd_rpy[0], att_cmd_rpy[1], att_cmd_rpy[2])

        self.att_cmd.q_d = q_d
        self.att_cmd.thrust_body = np.array([0.0, 0.0, -thrust_norm_cmd], dtype=float)
        self.att_cmd.euler_cmd = att_cmd_rpy
        self.att_cmd.yaw_sp_move_rate = 0.0
        self.att_cmd.total_thrust_cmd = float(thrust_cmd_total)
        self.att_cmd.thrust_norm_cmd = float(thrust_norm_cmd)

        self._log_step()

        return {
            "att_cmd": self.att_cmd,
            "heading_wp_idx": int(self.path.heading_index),
            "pf_done": bool(self.path.pf_done),
            "guid_type_used": int(guid_type_used),
        }

    # =====================================================
    # public getters / builders
    # =====================================================
    def pop_waypoint_ack(self) -> bool:
        if self._waypoint_ack_pending:
            self._waypoint_ack_pending = False
            return True
        return False

    def get_mppi_runtime_flags(self) -> list[int]:
        data = [
            int(self.path.heading_index),
            int(self.path.passed_index),
            int(self.guid_type),
            int(self.mppi.reset_flag),
        ]
        self.mppi.reset_flag = 0
        return data

    def get_mppi_vehicle_state(self) -> list[float]:
        return [
            float(self.state.position_ned[0]),
            float(self.state.position_ned[1]),
            float(self.state.position_ned[2]),
            float(self._guidance_velocity()[0]),
            float(self._guidance_velocity()[1]),
            float(self._guidance_velocity()[2]),
            float(self.state.euler_rpy[0]),
            float(self.state.euler_rpy[1]),
            float(self.state.euler_rpy[2]),
            # ground / surface vehicle: the yaw rate in the thrust slot (mppi_kernel.cu GROUND_VEHICLE)
            float(self._yaw_rate if self.ground is not None else self.att_cmd.total_thrust_cmd),
        ]

    def get_mppi_waypoints_ned(self) -> list[float] | None:
        if self.path.waypoints_ned is None:
            return None
        return self.path.waypoints_ned.reshape(-1).tolist()

    def get_gpr_disturbance_acc(self) -> list[float]:
        return [
            float(self.state.sim_time),
            float(self.ndo.disturbance_acc_ned[0]),
            float(self.ndo.disturbance_acc_ned[1]),
            float(self.ndo.disturbance_acc_ned[2]),
        ]

    def get_waypoints_for_plot(self) -> list[float] | None:
        if self.path.waypoints_ned is None:
            return None

        data = self.path.waypoints_ned.reshape(-1).tolist()
        data.append(float(self.guid_type))
        return data

    def get_runtime_flags_int(self) -> list[int]:
        return [
            int(self.guid_type),
            int(self.path.interrupt_active),
            int(self.path.stop_flag),
            int(self.mppi.reset_flag),
            int(self.path.pf_done),
            int(self.path.plot_complete),
        ]

    # =====================================================
    # initialization helpers
    # =====================================================
    def _initialize_mppi_state(self) -> None:
        selected_mppi = self.cfg.mppi.selected(self.guid_type)
        self.mppi.u0 = float(selected_mppi.u0_init)
        self.mppi.u1 = float(selected_mppi.u1_init)
        self.mppi.solve_time = 0.0
        self.mppi.reset_flag = 0

    def _reset_path_tracking_state(self, waypoints_ned: np.ndarray) -> None:
        self.path.initialized = False
        self.path.pf_done = False
        self.path.plot_complete = False

        self.path.passed_index = 0
        self.path.heading_index = min(1, waypoints_ned.shape[0] - 1)
        self.path.closest_index = 0

        self.path.closest_point_ned[:] = waypoints_ned[0]
        self.path.virtual_target_ned[:] = waypoints_ned[self.path.heading_index]

        self.path.cross_track_error = 0.0
        self.path.distance_to_heading_wp = 0.0
        self.path.distance_to_goal = float("inf")

        self.path.interrupt_active = 0
        self.path.interrupt_prev = 0
        self.path.stop_flag = 0

    def _relocalize_on_path(self, waypoints_ned: np.ndarray) -> None:
        """Re-localize tracking onto a freshly updated (replanned) path.

        Unlike _reset_path_tracking_state(), this is used while following is
        already underway. It preserves NDO/MPPI solver state and the runtime
        flags (interrupt/stop), and recovers passed/heading indices by
        re-projecting the current vehicle position onto the new path. This lets
        a continuously replanned waypoint stream be followed without snapping
        back to the start each time.
        """
        num_waypoints = int(waypoints_ned.shape[0])
        if num_waypoints < 2:
            raise RuntimeError("[FATAL] path following requires at least 2 waypoints.")

        position_ned = self.state.position_ned
        eps = self.cfg.numerics.eps

        best_dist = math.inf
        best_seg_start = 0
        best_point = waypoints_ned[0].copy()

        # Search every segment of the new path (full re-localization), since the
        # incoming path's index frame is unrelated to the previous one.
        for seg_start in range(num_waypoints - 1):
            segment_vec = waypoints_ned[seg_start + 1] - waypoints_ned[seg_start]
            segment_len = np.linalg.norm(segment_vec)
            vec_to_vehicle = position_ned - waypoints_ned[seg_start]

            projection_len = min(
                max(np.dot(segment_vec, vec_to_vehicle) / max(segment_len, eps), 0.0),
                segment_len,
            )
            point_on_segment = (
                waypoints_ned[seg_start] + projection_len * segment_vec / max(segment_len, eps)
            )
            distance_to_segment = np.linalg.norm(point_on_segment - position_ned)

            if distance_to_segment < best_dist:
                best_dist = float(distance_to_segment)
                best_seg_start = int(seg_start)
                best_point = point_on_segment

        self.path.passed_index = int(best_seg_start)
        self.path.heading_index = int(min(best_seg_start + 1, num_waypoints - 1))
        self.path.closest_index = int(best_seg_start)
        self.path.closest_point_ned[:] = best_point
        self.path.virtual_target_ned[:] = waypoints_ned[self.path.heading_index]

        self.path.cross_track_error = float(best_dist)
        self.path.distance_to_heading_wp = float(
            np.linalg.norm(waypoints_ned[self.path.heading_index] - position_ned)
        )
        self.path.distance_to_goal = float(np.linalg.norm(waypoints_ned[-1] - position_ned))

        # A new path means the previous completion no longer holds; resume.
        self.path.pf_done = False
        self.path.plot_complete = False

    # =====================================================
    # legacy path-following geometry helpers
    # =====================================================
    def _distance_to_path(
        self,
        path_waypoints: np.ndarray,
        heading_wp_idx: int,
        position_ned: np.ndarray,
        closest_point_ned: np.ndarray,
        passed_wp_idx: int,
    ) -> tuple[float, np.ndarray, int]:
        num_waypoints = int(path_waypoints.shape[0])
        if num_waypoints < 2:
            raise RuntimeError("[FATAL] path following requires at least 2 waypoints.")

        heading_wp_idx = int(np.clip(heading_wp_idx, 1, num_waypoints - 1))
        passed_wp_idx = int(np.clip(passed_wp_idx, 0, num_waypoints - 2))

        min_dist_to_path = math.inf
        has_updated = False
        passed_wp_idx_prev = int(passed_wp_idx)

        for wp_idx in range(heading_wp_idx, max(passed_wp_idx_prev - 1, 0), -1):
            segment_vec = path_waypoints[wp_idx] - path_waypoints[wp_idx - 1]
            segment_len = np.linalg.norm(segment_vec)
            vec_wp1_to_vehicle = position_ned - path_waypoints[wp_idx - 1]
            eps = self.cfg.numerics.eps

            projection_len = min(
                max(np.dot(segment_vec, vec_wp1_to_vehicle) / max(segment_len, eps), 0.0),
                segment_len,
            )
            point_on_segment = (
                path_waypoints[wp_idx - 1] + projection_len * segment_vec / max(segment_len, eps)
            )
            distance_to_segment = np.linalg.norm(point_on_segment - position_ned)

            if min_dist_to_path > distance_to_segment:
                min_dist_to_path = float(distance_to_segment)
                closest_point_ned = point_on_segment
                passed_wp_idx = max(wp_idx - 1, passed_wp_idx_prev)
                has_updated = True
            elif wp_idx == (passed_wp_idx_prev + 1) and (has_updated is False):
                min_dist_to_path = float(distance_to_segment)
                closest_point_ned = point_on_segment
                passed_wp_idx = passed_wp_idx_prev

        return min_dist_to_path, closest_point_ned, int(passed_wp_idx)

    def _check_waypoint(
        self,
        path_waypoints: np.ndarray,
        heading_wp_idx: int,
        position_ned: np.ndarray,
        wp_switch_distance: float,
        final_wp_tolerance: float,
    ) -> tuple[int, bool]:
        num_waypoints = int(path_waypoints.shape[0])
        if num_waypoints < 2:
            raise RuntimeError("[FATAL] path following requires at least 2 waypoints.")
        heading_wp_idx = int(np.clip(heading_wp_idx, 0, num_waypoints - 1))

        vec_to_heading_wp = path_waypoints[heading_wp_idx] - position_ned
        distance_to_heading_wp = np.linalg.norm(vec_to_heading_wp)

        # Keep legacy completion semantics for landing handoff compatibility:
        # completion check uses distance computed before heading index update.
        path_following_done = False
        if distance_to_heading_wp < wp_switch_distance:
            heading_wp_idx = min(heading_wp_idx + 1, num_waypoints - 1)

            if heading_wp_idx == num_waypoints - 1 and distance_to_heading_wp < final_wp_tolerance:
                path_following_done = True
            heading_wp_idx = min(heading_wp_idx, num_waypoints - 1)

        return int(heading_wp_idx), bool(path_following_done)

    def _vtp_decision(
        self,
        dist_to_path: float,
        lookahead_distance: float,
        closest_point_ned: np.ndarray,
        passed_wp_idx: int,
        path_waypoints: np.ndarray,
    ) -> np.ndarray:
        num_waypoints = int(path_waypoints.shape[0])
        if num_waypoints < 2:
            raise RuntimeError("[FATAL] path following requires at least 2 waypoints.")
        passed_wp_idx = int(np.clip(passed_wp_idx, 0, num_waypoints - 2))

        virtual_target_ned = path_waypoints[min(passed_wp_idx + 1, num_waypoints - 1)].copy()
        # ground / surface vehicle: the target is always lookahead_distance AHEAD of the closest point
        # along the path ("carrot"). The multicopter rule (target lookahead - distance ahead, the
        # closest point itself once off the path) sent a rover that had overshot a corner straight at
        # the path, perpendicular to it: it crossed, turned on the spot, crossed back (pf_ugv_sqsharp_g0,
        # 1.2 m off the path). The kernel's vtp_decision_ does the same (GROUND_VEHICLE).
        ground = self.ground is not None
        if dist_to_path >= lookahead_distance and not ground:
            virtual_target_ned = closest_point_ned.copy()
        else:
            total_len = 0.0 if ground else dist_to_path
            segment_start = closest_point_ned.copy()

            for wp_idx in range(passed_wp_idx + 1, path_waypoints.shape[0]):
                segment_end = path_waypoints[wp_idx]
                segment_vec = segment_end - segment_start
                segment_len = np.linalg.norm(segment_vec)

                if total_len + segment_len > lookahead_distance:
                    lookahead_on_segment = lookahead_distance - total_len
                    virtual_target_ned = (
                        segment_start + lookahead_on_segment * segment_vec / max(segment_len, self.cfg.numerics.eps)
                    )
                    break
                else:
                    segment_start = segment_end
                    total_len = total_len + segment_len
                    if wp_idx == path_waypoints.shape[0] - 1:
                        virtual_target_ned = segment_end.copy()

        return virtual_target_ned

    # =====================================================
    # legacy guidance / observer helpers
    # =====================================================
    def _guidance_modules(
        self,
        guid_type: int,
        passed_wp_idx: int,
        heading_wp_idx: int,
        num_waypoints: int,
        virtual_target_ned: np.ndarray,
        position_ned: np.ndarray,
        velocity_ned: np.ndarray,
        guid_eta: float,
        mppi_ctrl_input: np.ndarray,
        interrupt_active: int,
        interrupt_prev: int,
        transition_speed_threshold_mps: float,
        position_kp: float,
        position_kv_multiplier: float,
        min_guid_eta: float,
        gl_vertical_speed_gain: float,
        speed_ref: float = -1.0,
        off_path: bool = False,
        closest_point_ned: np.ndarray | None = None,
        xt_gain: float = 0.0,
        acc_ff_ned: np.ndarray | None = None,
        velocity_pursuit_ned: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int, int, int]:
        reset_mppi = 0
        transition_mode = 0

        if passed_wp_idx < 1 and not (self.cfg.path_following.guidance_tuning.mppi_speed_transition
                                      and np.linalg.norm(velocity_ned) > transition_speed_threshold_mps):
            transition_mode = 1

        if interrupt_active == 1:
            guid_type = 0
            reset_mppi = 1

        if (interrupt_active == 0) and (interrupt_prev == 1):
            transition_mode = 1

        interrupt_prev = interrupt_active

        if transition_mode == 1:
            guid_type = 0
            reset_mppi = 1
            if np.linalg.norm(velocity_ned) > transition_speed_threshold_mps:
                transition_mode = 0

        if heading_wp_idx >= (num_waypoints - 1):
            guid_type = 0

        max_err = self.cfg.path_following.guidance_tuning.mppi_max_heading_error_deg
        if guid_type in (1, 2) and max_err > 0.0 and math.hypot(velocity_ned[0], velocity_ned[1]) > 1e-3:
            az_v = math.atan2(velocity_ned[1], velocity_ned[0])
            az_t = math.atan2(virtual_target_ned[1] - position_ned[1], virtual_target_ned[0] - position_ned[0])
            if abs((az_t - az_v + math.pi) % (2.0 * math.pi) - math.pi) > math.radians(max_err):
                guid_type = 0

        acc_cmd_ned = np.zeros(3, dtype=float)

        if guid_type == 0:
            if velocity_pursuit_ned is not None:
                velocity_ned = velocity_pursuit_ned
            pos_error_ned = virtual_target_ned - position_ned
            kp_pos = position_kp
            vel_cmd_ned = kp_pos * pos_error_ned

            # Speed-regulated pursuit: keep the direction to the virtual target but
            # fly it at speed_ref (already limited by the braking distance left on
            # the path). Off the path the legacy proportional speed is the cap, so a
            # large cross-track error is closed gently instead of at cruise speed.
            if speed_ref >= 0.0:
                dist = float(np.linalg.norm(pos_error_ned))
                if dist > self.cfg.numerics.eps:
                    v_mag = min(speed_ref, kp_pos * dist) if off_path else speed_ref
                    vel_cmd_ned = pos_error_ned * (v_mag / dist)
                    if xt_gain > 0.0 and closest_point_ned is not None:
                        # cross-track pull toward the path, normal to the pursuit direction
                        t_hat = pos_error_ned / dist
                        e = closest_point_ned - position_ned
                        vel_cmd_ned = vel_cmd_ned + xt_gain * (e - float(np.dot(e, t_hat)) * t_hat)

            vel_error_ned = vel_cmd_ned - velocity_ned
            kp_vel = kp_pos * position_kv_multiplier
            acc_cmd_ned = kp_vel * vel_error_ned
            if speed_ref >= 0.0 and acc_ff_ned is not None:
                acc_cmd_ned = acc_cmd_ned + acc_ff_ned

        elif guid_type in (1, 2):
            guid_eta = max(mppi_ctrl_input[1], min_guid_eta)

            speed_mag = np.linalg.norm(velocity_ned)
            fpa_azim, fpa_elev = azim_elev_from_vec3(velocity_ned)
            dcm_ned_to_wind = dcm_from_euler321(np.array([0.0, fpa_elev, fpa_azim], dtype=float))

            acc_cmd_wind = np.zeros(3, dtype=float)
            acc_cmd_wind[0] = mppi_ctrl_input[0]
            kp_v = self.cfg.path_following.guidance_tuning.mppi_speed_kp
            if kp_v > 0.0 and speed_ref >= 0.0:
                acc_cmd_wind[0] += kp_v * (speed_ref - speed_mag)
            kg = self.cfg.path_following.guidance_tuning.mppi_speed_guard_kp
            if kg > 0.0 and speed_ref >= 0.0:
                margin = self.cfg.path_following.guidance_tuning.mppi_speed_guard_margin
                acc_cmd_wind[0] = min(acc_cmd_wind[0], kg * (speed_ref + margin - speed_mag))

            vec_to_target_ned = virtual_target_ned - position_ned
            vec_to_target_wind = dcm_ned_to_wind @ vec_to_target_ned
            err_azim, err_elev = azim_elev_from_vec3(vec_to_target_wind)

            acc_cmd_wind[1] = guid_eta * speed_mag * math.sin(err_azim)
            acc_cmd_wind[2] = -gl_vertical_speed_gain * speed_mag * math.sin(err_elev)

            dcm_wind_to_ned = dcm_ned_to_wind.T
            acc_cmd_ned = dcm_wind_to_ned @ acc_cmd_wind
            acc_cmd_ned[2] = -guid_eta * speed_mag * math.sin(err_elev)

        return acc_cmd_ned, int(guid_type), int(interrupt_prev), int(reset_mppi)

    @staticmethod
    def _point_along_path(
        path_waypoints: np.ndarray, passed_wp_idx: int, start_ned: np.ndarray, s: float
    ) -> np.ndarray | None:
        # point at arc length s ahead of start_ned (on segment passed_wp_idx), None past the end
        num_waypoints = int(path_waypoints.shape[0])
        i = int(np.clip(passed_wp_idx, 0, num_waypoints - 2)) + 1
        p = np.asarray(start_ned, dtype=float)
        while i < num_waypoints:
            seg = path_waypoints[i] - p
            seg_len = float(np.linalg.norm(seg))
            if seg_len >= s:
                return p + seg * (s / max(seg_len, 1e-9))
            s -= seg_len
            p = path_waypoints[i]
            i += 1
        return None

    def _curvature_feedforward(
        self, path_waypoints: np.ndarray, passed_wp_idx: int, closest_point_ned: np.ndarray,
        speed: float, tuning: "GuidanceTuningConfig",
    ) -> np.ndarray:
        # centripetal acceleration v^2 * kappa * n from a central second difference of
        # three arc-length-spaced path points, centred lead_s ahead of the closest point
        h = max(1.0, 0.15 * speed)
        s0 = max(speed * tuning.curvature_ff_lead_s - h, 0.0)
        pts = [self._point_along_path(path_waypoints, passed_wp_idx, closest_point_ned, s0 + k * h)
               for k in range(3)]
        if any(q is None for q in pts):
            return np.zeros(3, dtype=float)
        acc = tuning.curvature_ff_gain * speed * speed * (pts[0] - 2.0 * pts[1] + pts[2]) / (h * h)
        acc[2] = 0.0   # horizontal only: altitude kinks between replans are not real curvature
        a_mag = float(np.linalg.norm(acc))
        if tuning.curvature_ff_max_accel > 0.0 and a_mag > tuning.curvature_ff_max_accel:
            acc = acc * (tuning.curvature_ff_max_accel / a_mag)
        return acc

    @staticmethod
    def _corner_geometry(
        path_waypoints: np.ndarray, k: int, tuning: "GuidanceTuningConfig"
    ) -> tuple[float, float] | None:
        # (corner speed v_c, turn start before the vertex d) at interior vertex k, None if it is
        # (almost) straight; horizontal geometry only (GuidanceTuningConfig.corner_speed_enable).
        # mppi_kernel.cu corner_geom_ is the same computation.
        n = int(path_waypoints.shape[0])
        if k < 1 or k > n - 2:
            return None
        d1 = path_waypoints[k, :2] - path_waypoints[k - 1, :2]
        d2 = path_waypoints[k + 1, :2] - path_waypoints[k, :2]
        l1 = float(np.hypot(d1[0], d1[1]))
        l2 = float(np.hypot(d2[0], d2[1]))
        if l1 < 1e-3 or l2 < 1e-3:
            return None
        dpsi = math.acos(float(np.clip(np.dot(d1, d2) / (l1 * l2), -1.0, 1.0)))
        if dpsi < math.radians(tuning.corner_min_angle_deg):
            return None
        half = 0.5 * dpsi
        radius = tuning.corner_max_error_m / max(1.0 - math.cos(half), 1e-6)
        tan_half = max(math.tan(half), 1e-6)
        d_turn = radius * tan_half
        d_max = 0.5 * min(l1, l2)
        if d_turn > d_max:
            d_turn = d_max
            radius = d_turn / tan_half
        return max(math.sqrt(tuning.corner_lat_accel_mps2 * radius), tuning.corner_min_speed_mps), d_turn

    def _corner_speed_limit(
        self, path_waypoints: np.ndarray, passed_wp_idx: int, closest_point_ned: np.ndarray,
        speed_cap: float, tuning: "GuidanceTuningConfig",
    ) -> float:
        # speed allowed by the corners around the closest point (see corner_speed_enable):
        # the vertex just passed (second half of its turn) and the ones ahead within braking range
        n = int(path_waypoints.shape[0])
        i = int(np.clip(passed_wp_idx, 0, n - 2))
        a = tuning.corner_brake_accel_mps2
        v_lim = math.inf
        geo = self._corner_geometry(path_waypoints, i, tuning)
        if geo is not None:
            s_back = float(np.hypot(*(closest_point_ned[:2] - path_waypoints[i, :2])))
            v_lim = min(v_lim, math.sqrt(geo[0] ** 2 + 2.0 * a * max(s_back - geo[1], 0.0)))
        s = float(np.hypot(*(path_waypoints[i + 1, :2] - closest_point_ned[:2])))
        s_reach = max(speed_cap, 0.0) ** 2 / (2.0 * a) + 50.0
        for k in range(i + 1, n - 1):
            geo = self._corner_geometry(path_waypoints, k, tuning)
            if geo is not None:
                v_lim = min(v_lim, math.sqrt(geo[0] ** 2 + 2.0 * a * max(s - geo[1], 0.0)))
            s += float(np.hypot(*(path_waypoints[k + 1, :2] - path_waypoints[k, :2])))
            if s > s_reach:
                break
        return v_lim

    @staticmethod
    def _remaining_path_length(
        path_waypoints: np.ndarray, passed_wp_idx: int, closest_point_ned: np.ndarray
    ) -> float:
        num_waypoints = int(path_waypoints.shape[0])
        i = int(np.clip(passed_wp_idx, 0, num_waypoints - 2))
        remaining = float(np.linalg.norm(path_waypoints[i + 1] - closest_point_ned))
        if i + 1 < num_waypoints - 1:
            remaining += float(np.sum(np.linalg.norm(np.diff(path_waypoints[i + 1:], axis=0), axis=1)))
        return remaining

    def _convert_acc_cmd_to_thrust_and_att_cmd(
        self,
        dcm_ned_to_body: np.ndarray,
        acc_cmd_ned: np.ndarray,
        mass: float,
        max_total_thrust: float,
        path_waypoints: np.ndarray,
        heading_wp_idx: int,
        position_ned: np.ndarray,
        euler_rpy: np.ndarray,
        yaw_cmd_limit_rad: float,
        max_tilt_rad: float,
    ) -> tuple[float, float, np.ndarray]:
        if self.cfg.path_following.guidance_tuning.vertical_priority:
            gravity = self.cfg.model.gravity
            f_up = max(-float(acc_cmd_ned[2]), 0.2 * gravity)
            f_h = np.asarray(acc_cmd_ned[:2], dtype=float)
            f_max = max_total_thrust / mass
            h_max = f_up * math.tan(max_tilt_rad)
            h_max = min(h_max, math.sqrt(f_max * f_max - f_up * f_up)) if f_max > f_up else 0.0
            h = float(np.linalg.norm(f_h))
            if h > h_max:
                f_h = f_h * (h_max / h)
            acc_cmd_ned = np.array([f_h[0], f_h[1], -f_up], dtype=float)
            acc_cmd_mag = np.linalg.norm(acc_cmd_ned)
            thrust_cmd_total = min(acc_cmd_mag * mass, max_total_thrust)
        else:
            acc_cmd_mag = np.linalg.norm(acc_cmd_ned)
            acc_cmd_body = dcm_ned_to_body @ acc_cmd_ned
            # Thrust acts along -z_body. The legacy abs() turned a commanded downward body
            # acceleration (wanting to sink faster than gravity) into FULL upward thrust - a
            # runaway climb once the vertical velocity error got large (v10h: 152 m). Keep the
            # sign; floor at 0.2 g so the attitude loop keeps some authority.
            thrust_cmd_total = min(max(-acc_cmd_body[2] * mass, 0.2 * mass * self.cfg.model.gravity),
                                   max_total_thrust)
        thrust_cmd_norm = thrust_cmd_total / max(max_total_thrust, self.cfg.numerics.eps)

        heading_wp_ned = path_waypoints[heading_wp_idx]
        vec_to_heading_wp = heading_wp_ned - position_ned

        if heading_wp_idx < path_waypoints.shape[0] - 1:
            yaw_des, _ = azim_elev_from_vec3(vec_to_heading_wp)
        else:
            prev_wp_idx = max(heading_wp_idx - 1, 0)
            prev_wp_ned = path_waypoints[prev_wp_idx]
            path_segment = heading_wp_ned - prev_wp_ned
            yaw_des, _ = azim_elev_from_vec3(path_segment)

        tuning = self.cfg.path_following.guidance_tuning
        if tuning.yaw_mode == "hold":
            if self._yaw_hold_rad is None:
                self._yaw_hold_rad = float(euler_rpy[2])
            yaw_des = self._yaw_hold_rad
        elif tuning.yaw_mode == "velocity":
            v = self.state.velocity_ned
            if math.hypot(float(v[0]), float(v[1])) > tuning.yaw_min_speed:
                yaw_des = math.atan2(float(v[1]), float(v[0]))
            else:
                yaw_des = float(euler_rpy[2]) if self._yaw_cmd_prev is None else self._yaw_cmd_prev
        if tuning.yaw_rate_limit_deg > 0.0:
            prev = float(euler_rpy[2]) if self._yaw_cmd_prev is None else self._yaw_cmd_prev
            step = math.radians(tuning.yaw_rate_limit_deg) * self.cfg.path_following.dt_pf
            d_yaw = (yaw_des - prev + math.pi) % (2.0 * math.pi) - math.pi
            yaw_des = prev + max(min(d_yaw, step), -step)
        self._yaw_cmd_prev = (yaw_des + math.pi) % (2.0 * math.pi) - math.pi
        yaw_error = (yaw_des - euler_rpy[2] + math.pi) % (2.0 * math.pi) - math.pi
        yaw_error = max(min(yaw_error, yaw_cmd_limit_rad), -yaw_cmd_limit_rad)
        yaw_des = euler_rpy[2] + yaw_error

        yaw_rotation = np.array([0.0, 0.0, yaw_des], dtype=float)
        dcm_yaw_only = dcm_from_euler321(yaw_rotation)
        yaw_aligned_acc_cmd = dcm_yaw_only @ acc_cmd_ned

        if acc_cmd_mag < self.cfg.numerics.eps:
            roll_cmd = 0.0
            pitch_cmd = 0.0
        else:
            roll_cmd = min(max(math.asin(yaw_aligned_acc_cmd[1] / acc_cmd_mag), -max_tilt_rad), max_tilt_rad)
            sin_tilt_limit = math.sin(max_tilt_rad)
            sin_pitch = min(
                max(
                    -yaw_aligned_acc_cmd[0] / max(math.cos(roll_cmd) * acc_cmd_mag, self.cfg.numerics.eps),
                    -sin_tilt_limit,
                ),
                sin_tilt_limit,
            )
            pitch_cmd = math.asin(sin_pitch)

        att_cmd_rpy = np.array([roll_cmd, pitch_cmd, yaw_des], dtype=float)
        return float(thrust_cmd_total), float(thrust_cmd_norm), att_cmd_rpy

    def _guidance_velocity(self, pursuit: bool = False) -> np.ndarray:
        # the velocity the guidance (and the MPPI rollout's initial state) works with; for a ground /
        # surface vehicle the pursuit law always takes the heading-frame one (see GroundVehicleConfig)
        if self.ground is None or (self.ground.velocity_frame != "heading" and not pursuit):
            return self.state.velocity_ned
        yaw = float(self.state.euler_rpy[2])
        v = self.state.velocity_ned
        # floor 1 mm/s: the guidance takes the direction of travel from this vector (MPPI's frame),
        # a zero vector would point it north
        forward = max(float(v[0]) * math.cos(yaw) + float(v[1]) * math.sin(yaw), 1.0e-3)
        return np.array([forward * math.cos(yaw), forward * math.sin(yaw), 0.0], dtype=float)

    def _convert_acc_cmd_to_ground_cmd(
        self,
        acc_cmd_ned: np.ndarray,
        virtual_target_ned: np.ndarray,
        velocity_ned: np.ndarray,
    ) -> GroundCommand:
        g = self.ground
        position_ned = self.state.position_ned
        yaw = float(self.state.euler_rpy[2])
        speed = math.hypot(float(velocity_ned[0]), float(velocity_ned[1]))
        forward_speed = float(velocity_ned[0]) * math.cos(yaw) + float(velocity_ned[1]) * math.sin(yaw)

        # direction of travel: the velocity azimuth once moving, the yaw at rest
        course = math.atan2(float(velocity_ned[1]), float(velocity_ned[0])) if speed > g.course_min_speed else yaw
        t_hat = (math.cos(course), math.sin(course))
        acc_t = float(acc_cmd_ned[0]) * t_hat[0] + float(acc_cmd_ned[1]) * t_hat[1]
        acc_n = -float(acc_cmd_ned[0]) * t_hat[1] + float(acc_cmd_ned[1]) * t_hat[0]

        yaw_rate = acc_n / max(speed, g.rate_speed_floor)
        yaw_rate = max(min(yaw_rate, g.yaw_rate_max), -g.yaw_rate_max)
        lead_max = math.radians(g.bearing_lead_max_deg)
        lead = max(min(yaw_rate / g.yaw_kp, lead_max), -lead_max)
        bearing = yaw + lead

        speed_cmd = max(forward_speed, 0.0) + acc_t * g.tau_speed

        # not aligned with the path at (almost) rest: turn toward the virtual target first
        to_target = virtual_target_ned - position_ned
        if math.hypot(float(to_target[0]), float(to_target[1])) > self.cfg.numerics.eps:
            az_target = math.atan2(float(to_target[1]), float(to_target[0]))
            err = abs((az_target - yaw + math.pi) % (2.0 * math.pi) - math.pi)
            if self._ground_align:
                if err < math.radians(g.align_exit_deg):
                    self._ground_align = False
            elif speed < g.align_speed_max and err > math.radians(g.align_enter_deg):
                self._ground_align = True
            if self._ground_align:
                bearing = az_target
                yaw_rate = 0.0
                speed_cmd = g.align_speed
        speed_cmd = max(min(speed_cmd, g.speed_max), 0.0)
        if self._ground_end or self.path.pf_done:
            self._ground_align = False
            speed_cmd = 0.0
            yaw_rate = 0.0
            bearing = yaw

        return GroundCommand(
            speed=float(speed_cmd),
            bearing=float((bearing + math.pi) % (2.0 * math.pi) - math.pi),
            yaw_rate=float(yaw_rate),
            acc_t=float(acc_t),
            acc_n=float(acc_n),
            align=int(self._ground_align),
        )

    def _update_disturbance_acc(self, thrust_cmd_total: float) -> None:
        gains = self.cfg.ndo.gain
        velocity_ned = self.state.velocity_ned
        dt = self.cfg.path_following.dt_pf
        gravity = self.cfg.model.gravity

        dcm_ned_to_body = dcm_from_euler321(self.state.euler_rpy)

        thrust_acc_body = np.array([0.0, 0.0, -thrust_cmd_total / self.cfg.model.mass], dtype=float)
        thrust_acc_ned = dcm_ned_to_body.T @ thrust_acc_body

        observer_state = self.ndo.observer_state_ned
        observer_state_dot = np.zeros(3, dtype=float)

        observer_state_dot[0] = (
            -gains[0] * observer_state[0]
            - gains[0] * (gains[0] * velocity_ned[0] + thrust_acc_ned[0])
        )
        observer_state_dot[1] = (
            -gains[1] * observer_state[1]
            - gains[1] * (gains[1] * velocity_ned[1] + thrust_acc_ned[1])
        )
        observer_state_dot[2] = (
            -gains[2] * observer_state[2]
            - gains[2] * (gains[2] * velocity_ned[2] + thrust_acc_ned[2] + gravity)
        )

        disturbance_est = np.zeros(3, dtype=float)
        disturbance_est[0] = observer_state[0] + gains[0] * velocity_ned[0]
        disturbance_est[1] = observer_state[1] + gains[1] * velocity_ned[1]
        disturbance_est[2] = observer_state[2] + gains[2] * velocity_ned[2]

        self.ndo.observer_state_ned = observer_state + observer_state_dot * dt
        # bound the estimate: ground contact / thrust saturation read as huge
        # 'disturbances' and the compensation then overshoots after release
        ndo_max = self.cfg.path_following.guidance_tuning.ndo_max_accel
        d_norm = float(np.linalg.norm(disturbance_est))
        if ndo_max > 0.0 and d_norm > ndo_max:
            disturbance_est = disturbance_est * (ndo_max / d_norm)
        self.ndo.disturbance_acc_ned = disturbance_est

    # =====================================================
    # logging helper
    # =====================================================
    def _log_step(self) -> None:
        if self.logger is None:
            return

        speed_total = float(np.linalg.norm(self.state.velocity_ned))
        speed_error = abs(self.cfg.path_following.desired_speed - speed_total)

        try:
            self.logger.update(
                float(self.state.sim_time),
                self.state.position_ned,
                self.state.velocity_ned,
                self.state.euler_rpy,
                [self.mppi.u0, self.mppi.u1],
                float(self.mppi.solve_time),
                float(self.path.cross_track_error),
                float(speed_error),
                float(self.cfg.path_following.desired_speed),
                self.att_cmd.euler_cmd,
                float(self.att_cmd.thrust_norm_cmd),
                thrust_total_cmd=float(self.att_cmd.total_thrust_cmd),
                desired_speed_cfg=float(self.cfg.path_following.desired_speed),
                vehicle_type_cfg=int(self.vehicle_type),
                guid_type_init=int(self.guid_type_init),
                wp_type_cfg=int(self.wp_type),
                guid_type_cfg=int(self.guid_type),
                guid_type_used=int(self._last_guid_type_used),
                lookahead_distance_cfg=float(self.cfg.path_following.lookahead_distance),
                waypoint_switch_distance_cfg=float(self.cfg.path_following.waypoint_switch_distance),
                wp_idx_heading=int(self.path.heading_index),
                wp_idx_passed=int(self.path.passed_index),
                mppi_reset_flag=int(self.mppi.reset_flag),
                interrupt_active=int(self.path.interrupt_active),
                interrupt_prev=int(self.path.interrupt_prev),
                stop_flag=int(self.path.stop_flag),
                pf_done=bool(self.path.pf_done),
                plot_complete=bool(self.path.plot_complete),
                dist_to_heading_wp=float(self.path.distance_to_heading_wp),
                dist_to_goal=float(self.path.distance_to_goal),
                virtual_target=self.path.virtual_target_ned,
                closest_point=self.path.closest_point_ned,
                ndo=self.ndo.disturbance_acc_ned,
                speed_ref=float(self.path.speed_ref),
                lookahead=float(self.path.lookahead),
                ground=(None if self.ground is None else [
                    self.ground_cmd.speed, self.ground_cmd.bearing, self.ground_cmd.yaw_rate,
                    self.ground_cmd.acc_t, self.ground_cmd.acc_n, self.ground_cmd.align]),
            )
        except Exception:
            pass
