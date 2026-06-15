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

        return cls(
            final_wp_tolerance_m=final_wp_tolerance_m,
            transition_speed_threshold_mps=transition_speed_threshold_mps,
            position_kp=position_kp,
            position_kv_multiplier=position_kv_multiplier,
            min_guid_eta=min_guid_eta,
            gl_vertical_speed_gain=gl_vertical_speed_gain,
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
        self.state.position_ned[:] = [x, y, z]
        self.state.velocity_ned[:] = [vx, vy, vz]

    def update_attitude_quat(self, q0: float, q1: float, q2: float, q3: float) -> None:
        roll, pitch, yaw = euler_from_quat(q0, q1, q2, q3)
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

            if is_live_update:
                self._relocalize_on_path(waypoints_ned)
            else:
                self._reset_path_tracking_state(waypoints_ned)
                self.ndo.observer_state_ned[:] = 0.0
                self.ndo.disturbance_acc_ned[:] = 0.0
                self.mppi.reset_flag = 1

        if bool(path_planning_complete):
            self._waypoint_ack_pending = True

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
        velocity_ned = self.state.velocity_ned
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
        self.path.pf_done = bool(pf_done)

        prev_virtual_target = self.path.virtual_target_ned.copy()
        virtual_target = self._vtp_decision(
            dist_to_path=dist_to_path,
            lookahead_distance=self.cfg.path_following.lookahead_distance,
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
            mppi_ctrl_input=np.array([self.mppi.u0, self.mppi.u1], dtype=float),
            interrupt_active=self.path.interrupt_active,
            interrupt_prev=self.path.interrupt_prev,
            transition_speed_threshold_mps=self.cfg.path_following.guidance_tuning.transition_speed_threshold_mps,
            position_kp=self.cfg.path_following.guidance_tuning.position_kp,
            position_kv_multiplier=self.cfg.path_following.guidance_tuning.position_kv_multiplier,
            min_guid_eta=self.cfg.path_following.guidance_tuning.min_guid_eta,
            gl_vertical_speed_gain=self.cfg.path_following.guidance_tuning.gl_vertical_speed_gain,
        )
        self.path.interrupt_prev = int(interrupt_prev)
        self._last_guid_type_used = int(guid_type_used)

        if int(reset_mppi) == 1:
            self.mppi.reset_flag = 1

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
            float(self.state.velocity_ned[0]),
            float(self.state.velocity_ned[1]),
            float(self.state.velocity_ned[2]),
            float(self.state.euler_rpy[0]),
            float(self.state.euler_rpy[1]),
            float(self.state.euler_rpy[2]),
            float(self.att_cmd.total_thrust_cmd),
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
        if dist_to_path >= lookahead_distance:
            virtual_target_ned = closest_point_ned.copy()
        else:
            total_len = dist_to_path
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
    ) -> tuple[np.ndarray, int, int, int]:
        reset_mppi = 0
        transition_mode = 0

        if passed_wp_idx < 1:
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

        acc_cmd_ned = np.zeros(3, dtype=float)

        if guid_type == 0:
            pos_error_ned = virtual_target_ned - position_ned
            kp_pos = position_kp
            vel_cmd_ned = kp_pos * pos_error_ned

            vel_error_ned = vel_cmd_ned - velocity_ned
            kp_vel = kp_pos * position_kv_multiplier
            acc_cmd_ned = kp_vel * vel_error_ned

        elif guid_type in (1, 2):
            guid_eta = max(mppi_ctrl_input[1], min_guid_eta)

            speed_mag = np.linalg.norm(velocity_ned)
            fpa_azim, fpa_elev = azim_elev_from_vec3(velocity_ned)
            dcm_ned_to_wind = dcm_from_euler321(np.array([0.0, fpa_elev, fpa_azim], dtype=float))

            acc_cmd_wind = np.zeros(3, dtype=float)
            acc_cmd_wind[0] = mppi_ctrl_input[0]

            vec_to_target_ned = virtual_target_ned - position_ned
            vec_to_target_wind = dcm_ned_to_wind @ vec_to_target_ned
            err_azim, err_elev = azim_elev_from_vec3(vec_to_target_wind)

            acc_cmd_wind[1] = guid_eta * speed_mag * math.sin(err_azim)
            acc_cmd_wind[2] = -gl_vertical_speed_gain * speed_mag * math.sin(err_elev)

            dcm_wind_to_ned = dcm_ned_to_wind.T
            acc_cmd_ned = dcm_wind_to_ned @ acc_cmd_wind
            acc_cmd_ned[2] = -guid_eta * speed_mag * math.sin(err_elev)

        return acc_cmd_ned, int(guid_type), int(interrupt_prev), int(reset_mppi)

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
        acc_cmd_mag = np.linalg.norm(acc_cmd_ned)
        acc_cmd_body = dcm_ned_to_body @ acc_cmd_ned
        thrust_cmd_total = min(abs(acc_cmd_body[2]) * mass, max_total_thrust)
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
            )
        except Exception:
            pass
