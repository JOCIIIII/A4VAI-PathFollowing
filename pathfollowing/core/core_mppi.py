############################################################
#
#   - Name : core_mppi.py
#
#                                 - KAIST FDCL, 2026.03.11
#
############################################################

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import os
import time
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from .gpr import GPRConfig, GPRCore


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
    max_total_thrust: float
    throttle_hover: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], eps: float) -> "VehicleModelConfig":
        _require_sections(cfg, ("model", "actuator", "px4_model"), "vehicle config")
        model_cfg = cfg["model"]
        actuator_cfg = cfg["actuator"]
        _require_sections(model_cfg, ("mass", "Ixx", "Iyy", "Izz", "gravity"), "model")
        _require_sections(actuator_cfg, ("n_motor", "max_thrust_per_rotor"), "actuator")

        mass = float(model_cfg["mass"])
        ixx = float(model_cfg["Ixx"])
        iyy = float(model_cfg["Iyy"])
        izz = float(model_cfg["Izz"])
        gravity = float(model_cfg["gravity"])
        max_total_thrust = float(actuator_cfg["n_motor"]) * float(actuator_cfg["max_thrust_per_rotor"])
        if mass <= 0.0 or ixx <= 0.0 or iyy <= 0.0 or izz <= 0.0 or gravity <= 0.0:
            raise RuntimeError("[FATAL] model mass/inertia/gravity must be > 0.")
        if max_total_thrust <= 0.0:
            raise RuntimeError("[FATAL] actuator n_motor*max_thrust_per_rotor must be > 0.")

        hover_ratio = mass * gravity / max_total_thrust
        hover_ratio = min(max(hover_ratio, eps), 1.0)
        throttle_hover = math.sqrt(hover_ratio)

        return cls(
            mass=mass,
            ixx=ixx,
            iyy=iyy,
            izz=izz,
            gravity=gravity,
            max_total_thrust=max_total_thrust,
            throttle_hover=throttle_hover,
        )


@dataclass
class PX4ModelConfig:
    max_tilt_deg: float
    tau_phi: float
    tau_the: float
    tau_psi: float
    tau_wb: float
    tau_p: float
    tau_q: float
    tau_r: float
    alpha_p: float
    alpha_q: float
    alpha_r: float
    tau_throttle: float
    del_psi_cmd_limit_deg: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "PX4ModelConfig":
        _require_sections(cfg, ("px4_model",), "vehicle config")
        px4_cfg = cfg["px4_model"]
        _require_sections(
            px4_cfg,
            (
                "max_tilt_deg",
                "tau_phi",
                "tau_the",
                "tau_psi",
                "tau_wb",
                "tau_p",
                "tau_q",
                "tau_r",
                "alpha_p",
                "alpha_q",
                "alpha_r",
                "tau_throttle",
                "del_psi_cmd_limit_deg",
            ),
            "px4_model",
        )
        max_tilt_deg = float(px4_cfg["max_tilt_deg"])
        tau_phi = float(px4_cfg["tau_phi"])
        tau_the = float(px4_cfg["tau_the"])
        tau_psi = float(px4_cfg["tau_psi"])
        tau_wb = float(px4_cfg["tau_wb"])
        tau_p = float(px4_cfg["tau_p"])
        tau_q = float(px4_cfg["tau_q"])
        tau_r = float(px4_cfg["tau_r"])
        alpha_p = float(px4_cfg["alpha_p"])
        alpha_q = float(px4_cfg["alpha_q"])
        alpha_r = float(px4_cfg["alpha_r"])
        tau_throttle = float(px4_cfg["tau_throttle"])
        del_psi_cmd_limit_deg = float(px4_cfg["del_psi_cmd_limit_deg"])

        if (
            max_tilt_deg <= 0.0
            or tau_phi <= 0.0
            or tau_the <= 0.0
            or tau_psi <= 0.0
            or tau_wb <= 0.0
            or tau_p <= 0.0
            or tau_q <= 0.0
            or tau_r <= 0.0
            or alpha_p <= 0.0
            or alpha_q <= 0.0
            or alpha_r <= 0.0
            or tau_throttle <= 0.0
            or del_psi_cmd_limit_deg <= 0.0
        ):
            raise RuntimeError("[FATAL] px4_model time constants/gains/angle limits must be > 0.")

        return cls(
            max_tilt_deg=max_tilt_deg,
            tau_phi=tau_phi,
            tau_the=tau_the,
            tau_psi=tau_psi,
            tau_wb=tau_wb,
            tau_p=tau_p,
            tau_q=tau_q,
            tau_r=tau_r,
            alpha_p=alpha_p,
            alpha_q=alpha_q,
            alpha_r=alpha_r,
            tau_throttle=tau_throttle,
            del_psi_cmd_limit_deg=del_psi_cmd_limit_deg,
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

        if final_wp_tolerance_m <= 0.0 or transition_speed_threshold_mps <= 0.0:
            raise RuntimeError(f"[FATAL] {scope} tolerances/speeds must be > 0.")
        if position_kp <= 0.0 or position_kv_multiplier <= 0.0:
            raise RuntimeError(f"[FATAL] {scope} position gains must be > 0.")
        if min_guid_eta <= 0.0 or gl_vertical_speed_gain <= 0.0:
            raise RuntimeError(f"[FATAL] {scope} eta/vertical gain must be > 0.")

        return cls(
            final_wp_tolerance_m=final_wp_tolerance_m,
            transition_speed_threshold_mps=transition_speed_threshold_mps,
            position_kp=position_kp,
            position_kv_multiplier=position_kv_multiplier,
            min_guid_eta=min_guid_eta,
            gl_vertical_speed_gain=gl_vertical_speed_gain,
        )


@dataclass
class DesiredSpeedRampConfig:
    enabled: bool
    min_mps: float
    slope_mps2: float
    tracking_margin_mps: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], target_speed: float) -> "DesiredSpeedRampConfig":
        ramp_cfg = cfg.get("desired_speed_ramp")
        if ramp_cfg is None:
            return cls(
                enabled=False,
                min_mps=float(target_speed),
                slope_mps2=0.0,
                tracking_margin_mps=0.0,
            )
        if not isinstance(ramp_cfg, dict):
            raise RuntimeError("[FATAL] path_following.desired_speed_ramp must be a mapping.")

        enabled = bool(ramp_cfg.get("enabled", False))
        min_mps = float(ramp_cfg.get("min_mps", target_speed))
        slope_mps2 = float(ramp_cfg.get("slope_mps2", 0.0))
        tracking_margin_mps = float(ramp_cfg.get("tracking_margin_mps", 0.0))
        if min_mps < 0.0:
            raise RuntimeError("[FATAL] desired_speed_ramp.min_mps must be >= 0.")
        if enabled and slope_mps2 <= 0.0:
            raise RuntimeError("[FATAL] desired_speed_ramp.slope_mps2 must be > 0.")
        if not enabled and slope_mps2 < 0.0:
            raise RuntimeError("[FATAL] desired_speed_ramp.slope_mps2 must be >= 0.")
        if tracking_margin_mps < 0.0:
            raise RuntimeError("[FATAL] desired_speed_ramp.tracking_margin_mps must be >= 0.")

        return cls(
            enabled=enabled,
            min_mps=min_mps,
            slope_mps2=slope_mps2,
            tracking_margin_mps=tracking_margin_mps,
        )


@dataclass
class PathFollowingConfig:
    desired_speed: float
    desired_speed_ramp: DesiredSpeedRampConfig
    lookahead_distance: float
    waypoint_switch_distance: float
    guid_eta: float
    guidance_tuning: GuidanceTuningConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "PathFollowingConfig":
        _require_sections(cfg, ("path_following",), "simulation config")
        pf_cfg = cfg["path_following"]
        _require_sections(
            pf_cfg,
            (
                "desired_speed",
                "lookahead_distance",
                "waypoint_switch_distance",
                "guid_eta",
            ),
            "path_following",
        )
        desired_speed = float(pf_cfg["desired_speed"])
        desired_speed_ramp = DesiredSpeedRampConfig.from_dict(pf_cfg, desired_speed)
        lookahead_distance = float(pf_cfg["lookahead_distance"])
        waypoint_switch_distance = float(pf_cfg["waypoint_switch_distance"])
        if desired_speed < 0.0 or lookahead_distance <= 0.0 or waypoint_switch_distance <= 0.0:
            raise RuntimeError("[FATAL] path_following speed/distances are invalid.")

        if "guidance_tuning" in pf_cfg:
            guidance_cfg = pf_cfg["guidance_tuning"]
            if not isinstance(guidance_cfg, dict):
                raise RuntimeError("[FATAL] path_following.guidance_tuning must be a mapping.")
            guidance = GuidanceTuningConfig.from_dict(guidance_cfg, scope="path_following.guidance_tuning")
        else:
            guidance = GuidanceTuningConfig.from_dict(pf_cfg, scope="path_following")

        return cls(
            desired_speed=desired_speed,
            desired_speed_ramp=desired_speed_ramp,
            lookahead_distance=lookahead_distance,
            waypoint_switch_distance=waypoint_switch_distance,
            guid_eta=float(pf_cfg["guid_eta"]),
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
        state_cost = np.asarray(cfg["state_cost"], dtype=np.float32).reshape(-1)
        terminal_cost = np.asarray(cfg["terminal_cost"], dtype=np.float32).reshape(-1)
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
class MPPIRolloutTuningConfig:
    type2_gamma_scale: float
    disturbance_var_weight_gain: float
    disturbance_var_clip: float
    cost_path_error_scale_m: float
    gpr_var_bias: float
    gpr_var_clip: float

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        scope: str = "mppi",
    ) -> "MPPIRolloutTuningConfig":
        _require_sections(
            cfg,
            (
                "disturbance_var_weight_gain",
                "disturbance_var_clip",
                "cost_path_error_scale_m",
                "gpr_var_bias",
                "gpr_var_clip",
            ),
            scope,
        )
        # New key: type2_gamma_scale (legacy type4_gamma_scale also supported).
        if "type2_gamma_scale" in cfg:
            type2_gamma_scale = float(cfg["type2_gamma_scale"])
        elif "type4_gamma_scale" in cfg:
            type2_gamma_scale = float(cfg["type4_gamma_scale"])
        else:
            raise RuntimeError(
                f"[FATAL] {scope}.type2_gamma_scale is required "
                "(legacy key type4_gamma_scale is also supported)."
            )
        disturbance_var_weight_gain = float(cfg["disturbance_var_weight_gain"])
        disturbance_var_clip = float(cfg["disturbance_var_clip"])
        cost_path_error_scale_m = float(cfg["cost_path_error_scale_m"])
        gpr_var_bias = float(cfg["gpr_var_bias"])
        gpr_var_clip = float(cfg["gpr_var_clip"])

        if type2_gamma_scale <= 0.0:
            raise RuntimeError(f"[FATAL] {scope}.type2_gamma_scale must be > 0.")
        if disturbance_var_weight_gain < 0.0 or disturbance_var_clip < 0.0:
            raise RuntimeError(f"[FATAL] {scope} disturbance variance weights must be >= 0.")
        if cost_path_error_scale_m <= 0.0:
            raise RuntimeError(f"[FATAL] {scope}.cost_path_error_scale_m must be > 0.")
        if gpr_var_bias < 0.0 or gpr_var_clip < 0.0:
            raise RuntimeError(f"[FATAL] {scope} gpr variance terms must be >= 0.")

        return cls(
            type2_gamma_scale=type2_gamma_scale,
            disturbance_var_weight_gain=disturbance_var_weight_gain,
            disturbance_var_clip=disturbance_var_clip,
            cost_path_error_scale_m=cost_path_error_scale_m,
            gpr_var_bias=gpr_var_bias,
            gpr_var_clip=gpr_var_clip,
        )


@dataclass
class MPPIConfig:
    dt_mppi: float
    num_samples: int
    horizon_steps: int
    cost_min_v_aligned: float
    rollout_tuning: MPPIRolloutTuningConfig
    type_1: MPPITypeConfig
    type_2: MPPITypeConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "MPPIConfig":
        _require_sections(cfg, ("mppi",), "simulation config")
        mppi_cfg = cfg["mppi"]
        _require_sections(
            mppi_cfg,
            (
                "dt_mppi",
                "num_samples",
                "horizon_steps",
                "cost_min_v_aligned",
            ),
            "mppi",
        )
        dt_mppi = float(mppi_cfg["dt_mppi"])
        num_samples = int(mppi_cfg["num_samples"])
        horizon_steps = int(mppi_cfg["horizon_steps"])
        if dt_mppi <= 0.0 or num_samples <= 0 or horizon_steps <= 0:
            raise RuntimeError("[FATAL] mppi dt_mppi/num_samples/horizon_steps must be > 0.")

        if "rollout_tuning" in mppi_cfg:
            rollout_cfg = mppi_cfg["rollout_tuning"]
            if not isinstance(rollout_cfg, dict):
                raise RuntimeError("[FATAL] mppi.rollout_tuning must be a mapping.")
            rollout = MPPIRolloutTuningConfig.from_dict(rollout_cfg, scope="mppi.rollout_tuning")
        else:
            rollout = MPPIRolloutTuningConfig.from_dict(mppi_cfg, scope="mppi")

        # New keys: type_1/type_2, while still accepting legacy type_3/type_4.
        if "type_1" in mppi_cfg and "type_2" in mppi_cfg:
            type_1_key = "type_1"
            type_2_key = "type_2"
        elif "type_3" in mppi_cfg and "type_4" in mppi_cfg:
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
            cost_min_v_aligned=float(mppi_cfg["cost_min_v_aligned"]),
            rollout_tuning=rollout,
            type_1=MPPITypeConfig.from_dict(mppi_cfg[type_1_key]),
            type_2=MPPITypeConfig.from_dict(mppi_cfg[type_2_key]),
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
        _require_sections(cfg, ("ndo",), "simulation config")
        gain = np.asarray(cfg["ndo"]["gain"], dtype=np.float32).reshape(-1)
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
    px4: PX4ModelConfig
    path_following: PathFollowingConfig
    mppi: MPPIConfig
    ndo: NDOConfig
    gpr: GPRConfig
    numerics: NumericsConfig


# =====================================================
# runtime dataclasses
# =====================================================
@dataclass
class RuntimeFlagsState:
    wp_idx_heading: int = 0
    wp_idx_passed: int = 0
    guid_type: int = 1
    reset_flag: int = 0


@dataclass
class VehicleState:
    position_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    euler_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    thrust_cmd: float = 0.0
    desired_speed: float = 0.0
    desired_speed_target: float = 0.0


@dataclass
class DisturbanceState:
    sim_time: float = 0.0
    measured_acc_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    forecast_acc_ned: np.ndarray = field(default_factory=lambda: np.zeros((1, 3), dtype=np.float32))
    forecast_var_xy: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))


# =====================================================
# core
# =====================================================
class MPPICore:
    def __init__(
        self,
        guid_type: int,
        vehicle_cfg: dict[str, Any],
        sim_cfg: dict[str, Any],
    ):
        self.guid_type = int(guid_type)
        if self.guid_type not in (0, 1, 2):
            raise RuntimeError(f"[FATAL] unsupported guid_type: {self.guid_type}")

        _require_sections(vehicle_cfg, ("model", "actuator", "px4_model"), "vehicle config")
        _require_sections(sim_cfg, ("path_following", "mppi", "ndo", "gpr", "numerics"), "simulation config")

        numerics_cfg = NumericsConfig.from_dict(sim_cfg)

        self.cfg = CoreConfig(
            model=VehicleModelConfig.from_dict(vehicle_cfg, eps=numerics_cfg.eps),
            px4=PX4ModelConfig.from_dict(vehicle_cfg),
            path_following=PathFollowingConfig.from_dict(sim_cfg),
            mppi=MPPIConfig.from_dict(sim_cfg),
            ndo=NDOConfig.from_dict(sim_cfg),
            gpr=GPRConfig.from_dict(sim_cfg),
            numerics=numerics_cfg,
        )

        self.runtime_flags = RuntimeFlagsState(guid_type=self.guid_type)
        self.vehicle_state = VehicleState()
        self.disturbance = DisturbanceState()

        self.waypoints_ned: np.ndarray | None = None
        self.has_runtime_flags = False
        self.has_vehicle_state = False
        self.has_wp_input = False
        self.kernel_ready = False
        self._compiled_n_wp = 0

        selected = self.cfg.mppi.selected(self.guid_type)
        self.u0 = selected.u0_init * np.ones(self.cfg.mppi.horizon_steps, dtype=np.float32)
        self.u1 = selected.u1_init * np.ones(self.cfg.mppi.horizon_steps, dtype=np.float32)

        self.last_output = np.array([selected.u0_init, selected.u1_init], dtype=np.float32)
        self.last_solve_time = 0.0

        self.gpr = GPRCore(self.cfg.gpr)
        self.use_gpr_forecast = self.cfg.gpr.use_gpr_forecast

        self.module = None
        self.func_mc = None

        self.gpu_u0 = None
        self.gpu_u1 = None
        self.gpu_delta_u0 = None
        self.gpu_delta_u1 = None
        self.gpu_const = None
        self.gpu_update = None
        self.gpu_waypoints_ned = None
        self.gpu_disturbance_acc_est = None
        self.gpu_disturbance_acc_var = None
        self.gpu_stk = None

        # ★ desired_speed 런타임 override (외부 /desired_speed 토픽 or 학습 신경망).
        #   None 이면 PF core 가 램프 계산한 vehicle_state.desired_speed 사용 = upstream 동작.
        #   값이 있으면 host_update[13] 에 그 값을 넣어 MPPI cruise 목표속도를 동적 갱신.
        self.desired_speed_override = None

        # Preallocated host buffers (set in _build_cuda_solver).
        self.host_const = None
        self.host_update = None
        self.host_delta_u0 = None
        self.host_delta_u1 = None
        self.host_res_stk = None
        self.host_weight = None
        self.host_du0 = None
        self.host_du1 = None
        self._const_guid_type = None
        self._rng = np.random.default_rng()
        self._supports_float32_normal_out = None

    # =====================================================
    # properties
    # =====================================================
    @property
    def dt_mppi(self) -> float:
        return self.cfg.mppi.dt_mppi

    @property
    def dt_gpr_opt(self) -> float:
        return self.cfg.gpr.dt_gpr * self.cfg.gpr.dt_gpr_opt_mul

    # =====================================================
    # external updates
    # =====================================================
    def update_runtime_flags(self, data) -> None:
        if len(data) < 4:
            raise RuntimeError(f"[FATAL] MPPI/in/runtime_flags requires 4 values, got {len(data)}.")

        self.runtime_flags.wp_idx_heading = int(data[0])
        self.runtime_flags.wp_idx_passed = int(data[1])
        self.runtime_flags.guid_type = int(data[2])
        self.runtime_flags.reset_flag = int(data[3])
        if self.runtime_flags.guid_type not in (0, 1, 2):
            raise RuntimeError(
                f"[FATAL] invalid guid_type in runtime_flags: {self.runtime_flags.guid_type}"
            )
        if self.runtime_flags.reset_flag not in (0, 1):
            raise RuntimeError(
                f"[FATAL] invalid reset_flag in runtime_flags: {self.runtime_flags.reset_flag}"
            )
        self.has_runtime_flags = True

    def update_vehicle_state(self, data) -> None:
        if len(data) < 10:
            raise RuntimeError(f"[FATAL] MPPI/in/vehicle_state requires 10 values, got {len(data)}.")

        self.vehicle_state.position_ned[:] = [float(data[0]), float(data[1]), float(data[2])]
        self.vehicle_state.velocity_ned[:] = [float(data[3]), float(data[4]), float(data[5])]
        self.vehicle_state.euler_rpy[:] = [float(data[6]), float(data[7]), float(data[8])]
        self.vehicle_state.thrust_cmd = float(data[9])
        if len(data) >= 11:
            self.vehicle_state.desired_speed = float(data[10])
        else:
            self.vehicle_state.desired_speed = float(self.cfg.path_following.desired_speed)
        if len(data) >= 12:
            self.vehicle_state.desired_speed_target = float(data[11])
        else:
            self.vehicle_state.desired_speed_target = float(self.cfg.path_following.desired_speed)
        check = np.array(
            [
                self.vehicle_state.position_ned[0],
                self.vehicle_state.position_ned[1],
                self.vehicle_state.position_ned[2],
                self.vehicle_state.velocity_ned[0],
                self.vehicle_state.velocity_ned[1],
                self.vehicle_state.velocity_ned[2],
                self.vehicle_state.euler_rpy[0],
                self.vehicle_state.euler_rpy[1],
                self.vehicle_state.euler_rpy[2],
                self.vehicle_state.thrust_cmd,
                self.vehicle_state.desired_speed,
                self.vehicle_state.desired_speed_target,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(check).all():
            raise RuntimeError("[FATAL] MPPI/in/vehicle_state contains non-finite value.")
        if self.vehicle_state.desired_speed < 0.0:
            raise RuntimeError("[FATAL] MPPI/in/vehicle_state desired_speed must be >= 0.")
        if self.vehicle_state.desired_speed_target < 0.0:
            raise RuntimeError("[FATAL] MPPI/in/vehicle_state desired_speed_target must be >= 0.")
        self.has_vehicle_state = True

    def update_waypoints(self, data) -> None:
        array = np.asarray(data, dtype=np.float32).reshape(-1)
        if array.size < 3 or array.size % 3 != 0:
            raise RuntimeError(
                f"[FATAL] MPPI/in/waypoints_ned must be flat 3N with N>=2, got length {array.size}."
            )

        next_wps = array.reshape(-1, 3)
        next_n_wp = int(next_wps.shape[0])
        if next_n_wp < 2:
            raise RuntimeError("[FATAL] MPPI/in/waypoints_ned requires at least 2 waypoints.")
        if not np.isfinite(next_wps).all():
            raise RuntimeError("[FATAL] MPPI/in/waypoints_ned contains non-finite value.")

        self.waypoints_ned = next_wps
        self.has_wp_input = True

        need_rebuild = (not self.kernel_ready) or (next_n_wp != self._compiled_n_wp)
        if need_rebuild:
            self._release_cuda_buffers()
            self._build_cuda_solver()
            self.kernel_ready = True
        else:
            cuda.memcpy_htod(self.gpu_waypoints_ned, self.waypoints_ned.reshape(-1))

    def update_disturbance_acc(self, data) -> None:
        if len(data) < 4:
            raise RuntimeError(f"[FATAL] GPR/in/disturbance_acc requires 4 values, got {len(data)}.")

        sim_time = float(data[0])
        measurement = np.asarray([data[1], data[2], data[3]], dtype=np.float32)
        if not np.isfinite(np.array([sim_time, measurement[0], measurement[1], measurement[2]])).all():
            raise RuntimeError("[FATAL] GPR/in/disturbance_acc contains non-finite value.")

        self.disturbance.sim_time = sim_time
        self.disturbance.measured_acc_ned[:] = measurement
        self.gpr.update(sim_time, measurement)

    def _update_forecast_state(
        self,
        disturbance_acc_est: np.ndarray,
        disturbance_acc_var: np.ndarray,
    ) -> None:
        self.disturbance.forecast_acc_ned = np.asarray(
            disturbance_acc_est,
            dtype=np.float32,
        ).copy()
        self.disturbance.forecast_var_xy = np.asarray(
            disturbance_acc_var,
            dtype=np.float32,
        ).reshape(-1, 1).copy()

    # =====================================================
    # public api
    # =====================================================
    def optimize_gpr_if_needed(self) -> None:
        if self.runtime_flags.wp_idx_passed >= 1:
            self.gpr.optimize_hyperparams()

    def solve(self) -> list[float] | None:
        if not (self.has_runtime_flags and self.has_vehicle_state and self.has_wp_input and self.kernel_ready):
            return None

        if self.runtime_flags.guid_type == 0:
            horizon = self.cfg.mppi.horizon_steps
            base_est = np.repeat(self.disturbance.measured_acc_ned.reshape(1, 3), horizon, axis=0)
            base_var = np.zeros(horizon, dtype=np.float32)
            self._update_forecast_state(base_est, base_var)
            return [0.0, 0.0, 0.0]

        selected = self.cfg.mppi.selected(self.runtime_flags.guid_type)
        if self.runtime_flags.reset_flag == 1:
            self.u0[:] = selected.u0_init
            self.u1[:] = selected.u1_init
            self.last_output[:] = [selected.u0_init, selected.u1_init]
            self.last_solve_time = 0.0
            self.gpr.reset()
            horizon = self.cfg.mppi.horizon_steps
            base_est = np.repeat(self.disturbance.measured_acc_ned.reshape(1, 3), horizon, axis=0)
            base_var = np.zeros(horizon, dtype=np.float32)
            self._update_forecast_state(base_est, base_var)
            return [float(self.last_output[0]), float(self.last_output[1]), 0.0]

        horizon = self.cfg.mppi.horizon_steps
        disturbance_acc_est = np.repeat(self.disturbance.measured_acc_ned.reshape(1, 3), horizon, axis=0)
        disturbance_acc_var = np.zeros(horizon, dtype=np.float32)

        if self.use_gpr_forecast and self.runtime_flags.wp_idx_passed >= 1:
            mean_forecast, var_forecast = self.gpr.forecast(steps=horizon)
            n = min(horizon, mean_forecast.shape[0])

            if n > 0:
                disturbance_acc_est[:n, 0] = mean_forecast[:n, 0]
                disturbance_acc_est[:n, 1] = mean_forecast[:n, 1]
                disturbance_acc_est[:n, 2] = self.disturbance.measured_acc_ned[2]

                disturbance_acc_var[: min(horizon, var_forecast.shape[0])] = np.minimum(
                    var_forecast[: min(horizon, var_forecast.shape[0]), 0]
                    + self.cfg.mppi.rollout_tuning.gpr_var_bias,
                    self.cfg.mppi.rollout_tuning.gpr_var_clip,
                )

            if n < horizon and n > 0:
                disturbance_acc_est[n:, 0] = disturbance_acc_est[n - 1, 0]
                disturbance_acc_est[n:, 1] = disturbance_acc_est[n - 1, 1]
                disturbance_acc_est[n:, 2] = self.disturbance.measured_acc_ned[2]
                disturbance_acc_var[n:] = disturbance_acc_var[n - 1]

        self._update_forecast_state(disturbance_acc_est, disturbance_acc_var)
        output, solve_time = self._run_cuda_rollout(disturbance_acc_est, disturbance_acc_var)
        self.last_output[:] = output
        self.last_solve_time = solve_time
        return [float(output[0]), float(output[1]), float(solve_time)]

    def close(self) -> None:
        self._release_cuda_buffers()

    # =====================================================
    # cuda helpers
    # =====================================================
    def _control_weights(self, selected: MPPITypeConfig) -> np.ndarray:
        eps = self.cfg.numerics.eps
        beta = max(float(selected.lambda_weight), eps)
        var0 = max(float(selected.noise_std_u0_scale), eps)
        var1 = max(float(selected.noise_std_u1_scale), eps)
        return np.array(
            [
                beta / (var0 * var0),
                beta / (var1 * var1),
                0.0,
            ],
            dtype=np.float32,
        )

    def _control_gamma(self, selected: MPPITypeConfig, guid_type: int) -> float:
        beta = max(float(selected.lambda_weight), self.cfg.numerics.eps)
        if int(guid_type) == 2:
            return self.cfg.mppi.rollout_tuning.type2_gamma_scale * beta
        return beta

    def _make_const_vector(self, guid_type: int) -> np.ndarray:
        selected = self.cfg.mppi.selected(guid_type)
        control_weight = self._control_weights(selected)
        control_gamma = self._control_gamma(selected, guid_type)
        gt = self.cfg.path_following.guidance_tuning
        rt = self.cfg.mppi.rollout_tuning

        return np.array(
            [
                float(self.cfg.mppi.num_samples),
                float(self.cfg.mppi.horizon_steps),
                float(self.cfg.mppi.dt_mppi),
                float(control_gamma),
                float(control_weight[0]),
                float(control_weight[1]),
                float(control_weight[2]),
                float(selected.state_cost[0]),
                float(selected.state_cost[1]),
                float(selected.state_cost[2]),
                float(selected.terminal_cost[0]),
                float(selected.terminal_cost[1]),
                float(selected.terminal_cost[2]),
                float(self.cfg.model.mass),
                float(self.cfg.model.gravity),
                float(self.cfg.path_following.waypoint_switch_distance),
                float(self.cfg.path_following.desired_speed),
                float(self.cfg.path_following.lookahead_distance),
                float(self.cfg.path_following.guid_eta),
                float(self.cfg.px4.del_psi_cmd_limit_rad),
                float(self.cfg.px4.tau_phi),
                float(self.cfg.px4.tau_the),
                float(self.cfg.px4.tau_psi),
                float(self.cfg.mppi.cost_min_v_aligned),
                float(selected.lambda_weight),
                float(self.cfg.model.max_total_thrust),
                float(self.cfg.model.throttle_hover),
                float(self.cfg.px4.max_tilt_rad),
                float(gt.final_wp_tolerance_m),
                float(gt.transition_speed_threshold_mps),
                float(gt.position_kp),
                float(gt.position_kv_multiplier),
                float(gt.min_guid_eta),
                float(gt.gl_vertical_speed_gain),
                float(self.cfg.px4.tau_wb),
                float(self.cfg.px4.tau_p),
                float(self.cfg.px4.tau_q),
                float(self.cfg.px4.tau_r),
                float(self.cfg.px4.alpha_p),
                float(self.cfg.px4.alpha_q),
                float(self.cfg.px4.alpha_r),
                float(self.cfg.px4.tau_throttle),
                float(rt.disturbance_var_weight_gain),
                float(rt.disturbance_var_clip),
                float(rt.cost_path_error_scale_m),
                float(rt.gpr_var_bias),
                float(rt.gpr_var_clip),
                float(1.0 if self.cfg.path_following.desired_speed_ramp.enabled else 0.0),
                float(self.cfg.path_following.desired_speed_ramp.slope_mps2),
                float(self.cfg.path_following.desired_speed_ramp.tracking_margin_mps),
            ],
            dtype=np.float32,
        )

    def _build_cuda_solver(self) -> None:
        if self.waypoints_ned is None:
            raise RuntimeError("[FATAL] waypoints must exist before building CUDA solver.")

        self._compiled_n_wp = int(self.waypoints_ned.shape[0])

        kernel_path = os.path.join(os.path.dirname(__file__), "mppi_kernel.cu")
        with open(kernel_path, "r", encoding="utf-8") as f:
            kernel_body = f.read()

        full_code = (
            f"#define N_WP {self._compiled_n_wp}\n"
            f"#define EPS_USER {float(self.cfg.numerics.eps):.16e}f\n"
            + kernel_body
        )
        self.module = SourceModule(full_code)
        self.func_mc = self.module.get_function("mppi_rollout_kernel")

        k = self.cfg.mppi.num_samples
        n = self.cfg.mppi.horizon_steps

        # Host buffers are preallocated and reused every solve() step.
        self.host_const = self._make_const_vector(self.guid_type)
        self.host_update = np.zeros(16, dtype=np.float32)
        self.host_delta_u0 = np.zeros((n, k), dtype=np.float32)
        self.host_delta_u1 = np.zeros((n, k), dtype=np.float32)
        self.host_res_stk = np.empty(k, dtype=np.float32)
        self.host_weight = np.empty(k, dtype=np.float32)
        self.host_du0 = np.empty(n, dtype=np.float32)
        self.host_du1 = np.empty(n, dtype=np.float32)
        self._const_guid_type = int(self.guid_type)

        arr_waypoints_ned = np.ascontiguousarray(self.waypoints_ned.reshape(-1), dtype=np.float32)
        arr_disturbance_acc_est = np.zeros((n, 3), dtype=np.float32)
        arr_disturbance_acc_var = np.zeros(n, dtype=np.float32)

        self.gpu_u0 = cuda.mem_alloc(self.u0.nbytes)
        self.gpu_u1 = cuda.mem_alloc(self.u1.nbytes)
        self.gpu_delta_u0 = cuda.mem_alloc(self.host_delta_u0.nbytes)
        self.gpu_delta_u1 = cuda.mem_alloc(self.host_delta_u1.nbytes)
        self.gpu_const = cuda.mem_alloc(self.host_const.nbytes)
        self.gpu_update = cuda.mem_alloc(self.host_update.nbytes)
        self.gpu_waypoints_ned = cuda.mem_alloc(arr_waypoints_ned.nbytes)
        self.gpu_disturbance_acc_est = cuda.mem_alloc(arr_disturbance_acc_est.nbytes)
        self.gpu_disturbance_acc_var = cuda.mem_alloc(arr_disturbance_acc_var.nbytes)
        self.gpu_stk = cuda.mem_alloc(self.host_res_stk.nbytes)

        cuda.memcpy_htod(self.gpu_const, self.host_const)
        cuda.memcpy_htod(self.gpu_waypoints_ned, arr_waypoints_ned)

    def _release_cuda_buffers(self) -> None:
        for name in (
            "gpu_u0",
            "gpu_u1",
            "gpu_delta_u0",
            "gpu_delta_u1",
            "gpu_const",
            "gpu_update",
            "gpu_waypoints_ned",
            "gpu_disturbance_acc_est",
            "gpu_disturbance_acc_var",
            "gpu_stk",
        ):
            buf = getattr(self, name, None)
            if buf is not None:
                try:
                    buf.free()
                except Exception:
                    pass
                setattr(self, name, None)

        self.module = None
        self.func_mc = None
        self.kernel_ready = False
        self._compiled_n_wp = 0

        self.host_const = None
        self.host_update = None
        self.host_delta_u0 = None
        self.host_delta_u1 = None
        self.host_res_stk = None
        self.host_weight = None
        self.host_du0 = None
        self.host_du1 = None
        self._const_guid_type = None

    def _run_cuda_rollout(
        self,
        disturbance_acc_est: np.ndarray,
        disturbance_acc_var: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if self.func_mc is None:
            raise RuntimeError("[FATAL] CUDA solver is not initialized.")

        t0 = time.time()

        if (
            self.host_const is None
            or self.host_update is None
            or self.host_delta_u0 is None
            or self.host_delta_u1 is None
            or self.host_res_stk is None
            or self.host_weight is None
            or self.host_du0 is None
            or self.host_du1 is None
        ):
            raise RuntimeError("[FATAL] host buffers are not initialized.")

        k = self.cfg.mppi.num_samples
        n = self.cfg.mppi.horizon_steps
        selected = self.cfg.mppi.selected(self.runtime_flags.guid_type)

        guid_type = int(self.runtime_flags.guid_type)
        if self._const_guid_type != guid_type:
            self.host_const[:] = self._make_const_vector(guid_type)
            cuda.memcpy_htod(self.gpu_const, self.host_const)
            self._const_guid_type = guid_type

        self._fill_standard_normal_float32(self.host_delta_u0)
        self.host_delta_u0 *= float(selected.noise_std_u0_scale)
        self._fill_standard_normal_float32(self.host_delta_u1)
        self.host_delta_u1 *= float(selected.noise_std_u1_scale)

        self.host_update[0] = float(self.runtime_flags.wp_idx_heading)
        self.host_update[1] = float(self.runtime_flags.wp_idx_passed)
        self.host_update[2] = float(guid_type)
        self.host_update[3] = float(self.vehicle_state.position_ned[0])
        self.host_update[4] = float(self.vehicle_state.position_ned[1])
        self.host_update[5] = float(self.vehicle_state.position_ned[2])
        self.host_update[6] = float(self.vehicle_state.velocity_ned[0])
        self.host_update[7] = float(self.vehicle_state.velocity_ned[1])
        self.host_update[8] = float(self.vehicle_state.velocity_ned[2])
        self.host_update[9] = float(self.vehicle_state.euler_rpy[0])
        self.host_update[10] = float(self.vehicle_state.euler_rpy[1])
        self.host_update[11] = float(self.vehicle_state.euler_rpy[2])
        self.host_update[12] = float(self.vehicle_state.thrust_cmd)
        # ★ desired_speed 런타임 override (/desired_speed 토픽). override 가 있으면
        #   시작속도([13])와 목표속도([15]) 모두 override 값으로 넣어 커널 램프를 무력화하고
        #   일정 순항속도로 동작(기존 포크 동작). 없으면 PF core 가 램프 계산한
        #   vehicle_state.desired_speed / desired_speed_target 사용(upstream 동작).
        if self.desired_speed_override is not None:
            _des_spd = float(self.desired_speed_override)
            _des_spd_target = _des_spd
        else:
            _des_spd = float(self.vehicle_state.desired_speed)
            _des_spd_target = float(self.vehicle_state.desired_speed_target)
        self.host_update[13] = _des_spd
        self.host_update[14] = float(self.cfg.path_following.lookahead_distance)
        self.host_update[15] = _des_spd_target

        cuda.memcpy_htod(self.gpu_u0, self.u0)
        cuda.memcpy_htod(self.gpu_u1, self.u1)
        cuda.memcpy_htod(self.gpu_delta_u0, self.host_delta_u0)
        cuda.memcpy_htod(self.gpu_delta_u1, self.host_delta_u1)
        cuda.memcpy_htod(self.gpu_update, self.host_update)
        cuda.memcpy_htod(self.gpu_disturbance_acc_est, disturbance_acc_est)
        cuda.memcpy_htod(self.gpu_disturbance_acc_var, disturbance_acc_var)

        block_size = 32
        grid_size = int(math.ceil(k / block_size))

        self.func_mc(
            self.gpu_u0,
            self.gpu_u1,
            self.gpu_delta_u0,
            self.gpu_delta_u1,
            self.gpu_stk,
            self.gpu_const,
            self.gpu_update,
            self.gpu_waypoints_ned,
            self.gpu_disturbance_acc_est,
            self.gpu_disturbance_acc_var,
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )

        cuda.memcpy_dtoh(self.host_res_stk, self.gpu_stk)

        self.host_res_stk -= np.min(self.host_res_stk)
        eps = self.cfg.numerics.eps
        beta = max(float(selected.lambda_weight), eps)
        np.divide(self.host_res_stk, -beta, out=self.host_weight)
        np.exp(self.host_weight, out=self.host_weight)
        eta = float(np.sum(self.host_weight))
        if eta < eps:
            eta = eps

        np.matmul(self.host_delta_u0, self.host_weight, out=self.host_du0)
        np.matmul(self.host_delta_u1, self.host_weight, out=self.host_du1)
        self.host_du0 /= eta
        self.host_du1 /= eta

        self.u0 += self.host_du0
        self.u1 += self.host_du1

        output = np.array([self.u0[0], self.u1[0]], dtype=np.float32)

        self.u0[:-1] = self.u0[1:]
        self.u1[:-1] = self.u1[1:]
        self.u0[-1] = selected.u0_init
        self.u1[-1] = selected.u1_init

        t1 = time.time()
        return output, t1 - t0

    def _fill_standard_normal_float32(self, out: np.ndarray) -> None:
        if self._supports_float32_normal_out is None:
            probe = np.empty(1, dtype=np.float32)
            try:
                self._rng.standard_normal(size=probe.shape, dtype=np.float32, out=probe)
                self._supports_float32_normal_out = True
            except TypeError:
                self._supports_float32_normal_out = False

        if self._supports_float32_normal_out:
            self._rng.standard_normal(size=out.shape, dtype=np.float32, out=out)
            return

        # Fallback for NumPy builds that only support float64 with `out=`.
        out[:] = self._rng.standard_normal(size=out.shape).astype(np.float32)
