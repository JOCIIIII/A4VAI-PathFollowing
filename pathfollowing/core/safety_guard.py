############################################################
#
#   - Name : safety_guard.py
#
#                                 - KAIST FDCL, 2026.06.30
#
############################################################

from dataclasses import dataclass
import math


@dataclass
class MPPIOutputCheck:
    accepted: bool
    output: tuple[float, float, float] | None
    reason: str


class PathFollowingSafetyGuard:
    def __init__(self, now_func, log_func):
        self._now = now_func
        self._log_output = log_func

        self.enabled = None
        self.strict_startup_check = False
        self.startup_error = ""
        self.preflight_ready = False
        self.preflight_error = "CORE_NOT_READY"
        self._last_preflight_ready = None

        self.startup_monitor_period_s = None
        self.mppi_input_period_ratio = None
        self.path_following_heartbeat_period_s = None
        self.path_following_waypoint_plot_period_s = None
        self.runtime_flags_period_s = None
        self.mppi_waypoint_period_multiplier = None

        self.controller_hb_warn_timeout_s = None
        self.controller_hb_hold_timeout_s = None
        self.path_hb_warn_timeout_s = None
        self.path_hb_hold_timeout_s = None
        self.collision_hb_warn_timeout_s = None
        self.collision_hb_hold_timeout_s = None

        self.path_warn_timeout_s = None
        self.path_degraded_timeout_s = None
        self.path_hold_timeout_s = None
        self.no_valid_path_hold_timeout_s = None

        self.vehicle_state_warn_timeout_s = None
        self.vehicle_state_degraded_timeout_s = None
        self.vehicle_state_hold_timeout_s = None

        self.mppi_nonfinite_degraded_count = None
        self.mppi_nonfinite_hold_count = None
        self.mppi_deadline_degraded_count = None
        self.mppi_deadline_warn_ratio = None
        self.mppi_deadline_reject_ratio = None
        self.mppi_output_stale_ratio = None
        self.mppi_output_stale_hold_ratio = None

        self.safety_log_throttle_s = None
        now = self._now()
        self._node_start_time = now
        self.last_safety_log_time = None

        self.controller_heartbeat_ok = False
        self.path_planning_heartbeat_ok = False
        self.collision_avoidance_heartbeat_ok = False
        self.last_controller_hb_time = None
        self.last_path_planning_hb_time = None
        self.last_collision_hb_time = None
        self.last_controller_hb_ok_time = None
        self.last_path_planning_hb_ok_time = None
        self.last_collision_hb_ok_time = None

        self.last_local_position_time = None
        self.last_attitude_time = None
        self.last_accel_time = None
        self.last_vehicle_state_ok_time = None

        self.path_missing_since = None
        self.last_valid_path_time = None
        self.has_valid_path = False
        self.no_output_since = None

        self.last_mppi_output_time = None
        self.mppi_nonfinite_count = 0
        self.mppi_deadline_miss_count = 0
        self.last_mppi_deadline_loop_id = None
        self.mppi_stale_count = 0

        self.warn_active = False
        self.degraded_active = False
        self.hold_requested = False
        self.reason = ""

    def load_config(self, safety_cfg: dict) -> None:
        if not isinstance(safety_cfg, dict):
            raise RuntimeError("[FATAL] missing required 'safety' section in sim.yaml.")

        bool_keys = ("enabled", "strict_startup_check")
        for key in bool_keys:
            if key not in safety_cfg:
                raise RuntimeError(f"[FATAL] missing safety.{key} in sim.yaml.")
            if not isinstance(safety_cfg[key], bool):
                raise RuntimeError(f"[FATAL] safety.{key} must be bool.")
            attr = "enabled" if key == "enabled" else key
            setattr(self, attr, bool(safety_cfg[key]))

        float_keys = (
            "startup_monitor_period_s",
            "mppi_input_period_ratio",
            "path_following_heartbeat_period_s",
            "path_following_waypoint_plot_period_s",
            "runtime_flags_period_s",
            "mppi_waypoint_period_multiplier",
            "controller_hb_warn_timeout_s",
            "controller_hb_hold_timeout_s",
            "path_hb_warn_timeout_s",
            "path_hb_hold_timeout_s",
            "collision_hb_warn_timeout_s",
            "collision_hb_hold_timeout_s",
            "path_warn_timeout_s",
            "path_degraded_timeout_s",
            "path_hold_timeout_s",
            "no_valid_path_hold_timeout_s",
            "vehicle_state_warn_timeout_s",
            "vehicle_state_degraded_timeout_s",
            "vehicle_state_hold_timeout_s",
            "mppi_deadline_warn_ratio",
            "mppi_deadline_reject_ratio",
            "mppi_output_stale_ratio",
            "mppi_output_stale_hold_ratio",
            "safety_log_throttle_s",
        )
        for key in float_keys:
            if key not in safety_cfg:
                raise RuntimeError(f"[FATAL] missing safety.{key} in sim.yaml.")
            value = float(safety_cfg[key])
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(f"[FATAL] safety.{key} must be finite and > 0.")
            setattr(self, key, value)

        int_keys = (
            "mppi_nonfinite_degraded_count",
            "mppi_nonfinite_hold_count",
            "mppi_deadline_degraded_count",
        )
        for key in int_keys:
            if key not in safety_cfg:
                raise RuntimeError(f"[FATAL] missing safety.{key} in sim.yaml.")
            value = int(safety_cfg[key])
            if value < 1:
                raise RuntimeError(f"[FATAL] safety.{key} must be >= 1.")
            setattr(self, key, value)

    def mark_core_ready(self, now: float, has_path: bool) -> None:
        if has_path:
            self.mark_path_valid(now)

    def mark_startup_error(self, error: str) -> None:
        self.startup_error = error
        self.set_hold("STARTUP_CONFIG_INVALID", self.startup_error)

    def finite_all(self, values) -> bool:
        return all(math.isfinite(float(value)) for value in values)

    def attitude_command_is_finite(self, attitude_command) -> bool:
        values = list(attitude_command.q_d)
        values.extend(attitude_command.thrust_body.tolist())
        values.extend(attitude_command.euler_cmd.tolist())
        values.append(float(attitude_command.yaw_sp_move_rate))
        values.append(float(attitude_command.total_thrust_cmd))
        values.append(float(attitude_command.thrust_norm_cmd))
        return self.finite_all(values)

    def set_warn(self, reason: str, detail: str = "") -> None:
        self.warn_active = True
        if not self.hold_requested:
            self.reason = reason
        suffix = f": {detail}" if detail else ""
        self._log("warning", f"[SAFETY] WARN {reason}{suffix}")

    def set_degraded(self, reason: str, detail: str = "") -> None:
        self.warn_active = True
        self.reason = reason
        suffix = f": {detail}" if detail else ""
        self._log("warning", f"[SAFETY] DEGRADED SUPPRESSED {reason}{suffix}")

    def set_hold(self, reason: str, detail: str = "") -> None:
        self.warn_active = True
        self.reason = reason
        suffix = f": {detail}" if detail else ""
        self._log(
            "warning",
            f"[SAFETY] HOLD SUPPRESSED {reason}{suffix}",
        )

    def request_hold(self, reason: str) -> None:
        self.set_hold(reason)

    def mark_heartbeat(self, name: str, ok: bool, now: float) -> None:
        if name == "controller":
            self.controller_heartbeat_ok = bool(ok)
            self.last_controller_hb_time = now
            if ok:
                self.last_controller_hb_ok_time = now
        elif name == "path_planning":
            self.path_planning_heartbeat_ok = bool(ok)
            self.last_path_planning_hb_time = now
            if ok:
                self.last_path_planning_hb_ok_time = now
        elif name == "collision_avoidance":
            self.collision_avoidance_heartbeat_ok = bool(ok)
            self.last_collision_hb_time = now
            if ok:
                self.last_collision_hb_ok_time = now

    def mark_local_position(self, now: float) -> None:
        self.last_local_position_time = now
        self.last_vehicle_state_ok_time = now

    def mark_attitude(self, now: float) -> None:
        self.last_attitude_time = now
        self.last_vehicle_state_ok_time = now

    def mark_accel(self, now: float) -> None:
        self.last_accel_time = now

    def mark_path_valid(self, now: float) -> None:
        self.has_valid_path = True
        self.last_valid_path_time = now
        self.path_missing_since = None

    def mark_waypoint_rejected(self, now: float) -> None:
        if not self.has_valid_path:
            self.path_missing_since = self.path_missing_since or now

    def evaluate(
        self,
        core_valid: bool,
        core_has_path: bool,
        mppi_guidance_active: bool,
        dt_mppi: float | None,
        now: float,
    ) -> None:
        self.warn_active = False
        self.degraded_active = False

        if self.enabled is False:
            return

        if not core_valid:
            detail = self.startup_error if self.startup_error else "core is not valid"
            self.set_hold("STARTUP_CONFIG_INVALID", detail)
            return

        self._evaluate_heartbeat(
            "controller",
            self.controller_heartbeat_ok,
            self.last_controller_hb_ok_time,
            self.last_controller_hb_time,
            self.controller_hb_warn_timeout_s,
            self.controller_hb_hold_timeout_s,
        )
        self._evaluate_heartbeat(
            "path_planning",
            self.path_planning_heartbeat_ok,
            self.last_path_planning_hb_ok_time,
            self.last_path_planning_hb_time,
            self.path_hb_warn_timeout_s,
            self.path_hb_hold_timeout_s,
        )
        self._evaluate_heartbeat(
            "collision_avoidance",
            self.collision_avoidance_heartbeat_ok,
            self.last_collision_hb_ok_time,
            self.last_collision_hb_time,
            self.collision_hb_warn_timeout_s,
            self.collision_hb_hold_timeout_s,
        )
        self._evaluate_vehicle_state(now)
        self._evaluate_path_state(core_has_path, now)
        self._evaluate_mppi_state(mppi_guidance_active, dt_mppi, now)
        self.update_preflight_status(core_valid, core_has_path)

    def update_preflight_status(self, core_valid: bool, core_has_path: bool) -> None:
        if not core_valid:
            ready = False
            reason = "CORE_INVALID"
        elif not core_has_path:
            ready = False
            reason = "PATH_NOT_READY"
        elif self.last_local_position_time is None or self.last_attitude_time is None:
            ready = False
            reason = "VEHICLE_STATE_NOT_READY"
        elif not (
            self.controller_heartbeat_ok
            and self.path_planning_heartbeat_ok
            and self.collision_avoidance_heartbeat_ok
        ):
            ready = False
            reason = "HEARTBEAT_NOT_READY"
        elif self.hold_requested:
            ready = False
            reason = self.reason or "HOLD_REQUESTED"
        else:
            ready = True
            reason = ""

        self.preflight_ready = ready
        self.preflight_error = reason
        if self._last_preflight_ready is None or self._last_preflight_ready != ready:
            self._last_preflight_ready = ready
            if ready:
                self._log("info", "[PREFLIGHT] READY", force=True)
            else:
                self._log("warning", f"[PREFLIGHT] NOT READY: {reason}", force=True)

    def check_mppi_output(
        self,
        data,
        core_valid: bool,
        dt_mppi: float | None,
    ) -> MPPIOutputCheck:
        if not core_valid:
            self.set_warn("MPPI_OUTPUT_REJECTED", "core is not valid")
            return MPPIOutputCheck(False, None, "core is not valid")

        if len(data) < 3:
            self.mppi_nonfinite_count += 1
            self.set_warn("MPPI_OUTPUT_REJECTED", f"requires 3 values, got {len(data)}")
            return MPPIOutputCheck(False, None, "payload too short")

        try:
            u0 = float(data[0])
            u1 = float(data[1])
            solve_time = float(data[2])
        except Exception:
            self.mppi_nonfinite_count += 1
            self.set_warn("MPPI_OUTPUT_REJECTED", "payload conversion failed")
            return MPPIOutputCheck(False, None, "payload conversion failed")

        solve_loop_id = None
        if len(data) >= 4:
            try:
                solve_loop_id_value = float(data[3])
            except Exception:
                self.set_warn("MPPI_OUTPUT_REJECTED", "invalid solve loop id")
                return MPPIOutputCheck(False, None, "invalid solve loop id")

            if not math.isfinite(solve_loop_id_value) or solve_loop_id_value < 0.0:
                self.set_warn("MPPI_OUTPUT_REJECTED", "invalid solve loop id")
                return MPPIOutputCheck(False, None, "invalid solve loop id")
            solve_loop_id = int(solve_loop_id_value)

        if not self.finite_all([u0, u1, solve_time]):
            self.mppi_nonfinite_count += 1
            self.set_warn("MPPI_OUTPUT_NONFINITE", f"count={self.mppi_nonfinite_count}")
            if self.mppi_nonfinite_count >= self.mppi_nonfinite_hold_count:
                self.set_hold("MPPI_OUTPUT_NONFINITE")
            elif self.mppi_nonfinite_count >= self.mppi_nonfinite_degraded_count:
                self.set_degraded("MPPI_OUTPUT_NONFINITE")
            return MPPIOutputCheck(False, None, "nonfinite")

        if dt_mppi is None or not math.isfinite(float(dt_mppi)) or float(dt_mppi) <= 0.0:
            self.set_warn("MPPI_OUTPUT_REJECTED", "invalid dt_mppi")
            return MPPIOutputCheck(False, None, "invalid dt_mppi")

        dt_mppi = float(dt_mppi)
        is_new_solve_loop = True
        if solve_loop_id is not None:
            is_new_solve_loop = solve_loop_id != self.last_mppi_deadline_loop_id
            if is_new_solve_loop:
                self.last_mppi_deadline_loop_id = solve_loop_id

        deadline_missed = solve_time > (self.mppi_deadline_warn_ratio * dt_mppi)
        severe_deadline_miss = solve_time > (self.mppi_deadline_reject_ratio * dt_mppi)
        if is_new_solve_loop:
            if deadline_missed:
                self.mppi_deadline_miss_count += 1
                detail = (
                    f"solve_time={solve_time:.4f}s dt={dt_mppi:.4f}s "
                    f"count={self.mppi_deadline_miss_count}"
                )
                if solve_loop_id is not None:
                    detail += f" loop_id={solve_loop_id}"
                self.set_warn("MPPI_DEADLINE_MISS", detail)
                if self.mppi_deadline_miss_count >= self.mppi_deadline_degraded_count:
                    self.set_degraded("MPPI_DEADLINE_MISS")
            else:
                self.mppi_deadline_miss_count = 0

        if severe_deadline_miss:
            self.set_warn("MPPI_OUTPUT_REJECTED", "severe deadline miss")
            return MPPIOutputCheck(False, None, "severe deadline miss")

        return MPPIOutputCheck(True, (u0, u1, solve_time), "")

    def mark_mppi_output_accepted(self) -> None:
        self.last_mppi_output_time = self._now()
        self.mppi_nonfinite_count = 0
        self.mppi_stale_count = 0

    def handle_missing_control_output(self, now: float) -> None:
        if self.no_output_since is None:
            self.no_output_since = now
        no_output_time = now - self.no_output_since
        detail = f"duration={no_output_time:.3f}s"
        if no_output_time > self.path_hold_timeout_s:
            self.set_hold("CONTROL_OUTPUT_MISSING", detail)
        elif no_output_time > self.path_degraded_timeout_s:
            self.set_degraded("CONTROL_OUTPUT_MISSING", detail)
        elif no_output_time > self.path_warn_timeout_s:
            self.set_warn("CONTROL_OUTPUT_MISSING", detail)

    def clear_missing_control_output(self) -> None:
        self.no_output_since = None

    def _log(self, level: str, message: str, force: bool = False) -> None:
        now = self._now()
        if (
            not force
            and self.last_safety_log_time is not None
            and self.safety_log_throttle_s is not None
            and (now - self.last_safety_log_time) < self.safety_log_throttle_s
        ):
            return

        self.last_safety_log_time = now
        self._log_output(level, message)

    def _heartbeat_age(self, last_ok_time: float | None, last_msg_time: float | None) -> float:
        reference_time = last_ok_time
        if reference_time is None:
            reference_time = last_msg_time
        if reference_time is None:
            reference_time = self._node_start_time
        return self._now() - reference_time

    def _evaluate_heartbeat(
        self,
        name: str,
        ok: bool,
        last_ok_time: float | None,
        last_msg_time: float | None,
        warn_timeout: float,
        hold_timeout: float,
    ) -> None:
        age = self._heartbeat_age(last_ok_time, last_msg_time)
        if ok and last_msg_time is not None and age <= warn_timeout:
            return

        if age > hold_timeout:
            self.set_warn(f"HEARTBEAT_TIMEOUT:{name}", f"age={age:.3f}s")
        elif not ok:
            self.set_warn(f"HEARTBEAT_FALSE:{name}")
        elif age > warn_timeout:
            self.set_warn(f"HEARTBEAT_STALE:{name}", f"age={age:.3f}s")

    def _evaluate_vehicle_state(self, now: float) -> None:
        if self.last_local_position_time is None or self.last_attitude_time is None:
            age = now - self._node_start_time
            if age > self.vehicle_state_hold_timeout_s:
                self.set_hold("VEHICLE_STATE_NOT_READY", f"age={age:.3f}s")
            elif age > self.vehicle_state_degraded_timeout_s:
                self.set_degraded("VEHICLE_STATE_NOT_READY", f"age={age:.3f}s")
            elif age > self.vehicle_state_warn_timeout_s:
                self.set_warn("VEHICLE_STATE_NOT_READY", f"age={age:.3f}s")
            return

        state_time = min(self.last_local_position_time, self.last_attitude_time)
        age = now - state_time
        if age > self.vehicle_state_hold_timeout_s:
            self.set_hold("VEHICLE_STATE_STALE", f"age={age:.3f}s")
        elif age > self.vehicle_state_degraded_timeout_s:
            self.set_degraded("VEHICLE_STATE_STALE", f"age={age:.3f}s")
        elif age > self.vehicle_state_warn_timeout_s:
            self.set_warn("VEHICLE_STATE_STALE", f"age={age:.3f}s")

    def _evaluate_path_state(self, core_has_path: bool, now: float) -> None:
        if core_has_path:
            self.mark_path_valid(now)
            return

        if self.path_missing_since is None:
            self.path_missing_since = now

        missing_time = now - self.path_missing_since
        if not self.has_valid_path and missing_time > self.no_valid_path_hold_timeout_s:
            self.set_hold("NO_VALID_PATH", f"missing={missing_time:.3f}s")
            return

        if missing_time > self.path_hold_timeout_s:
            self.set_hold("PATH_MISSING", f"missing={missing_time:.3f}s")
        elif missing_time > self.path_degraded_timeout_s:
            self.set_degraded("PATH_MISSING", f"missing={missing_time:.3f}s")
        elif missing_time > self.path_warn_timeout_s:
            self.set_warn("PATH_MISSING", f"missing={missing_time:.3f}s")

    def _evaluate_mppi_state(
        self,
        mppi_guidance_active: bool,
        dt_mppi: float | None,
        now: float,
    ) -> None:
        if not mppi_guidance_active:
            return

        if dt_mppi is None or not math.isfinite(float(dt_mppi)) or float(dt_mppi) <= 0.0:
            self.set_hold("MPPI_DT_INVALID")
            return

        stale_timeout = self.mppi_output_stale_ratio * float(dt_mppi)
        reference_time = self.last_mppi_output_time
        if reference_time is None:
            reference_time = self._node_start_time

        age = now - reference_time
        if age <= stale_timeout:
            self.mppi_stale_count = 0
            return

        self.mppi_stale_count += 1
        self.set_degraded("MPPI_OUTPUT_STALE", f"age={age:.3f}s")
        if age > (self.mppi_output_stale_hold_ratio * stale_timeout):
            self.set_hold("MPPI_OUTPUT_STALE", f"age={age:.3f}s")
