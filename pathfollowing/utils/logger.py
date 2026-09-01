############################################################
#
#   - Name : logger.py
#
#                                 - KAIST FDCL, 2026.03.11
#
############################################################

import os
import csv
from datetime import datetime
from zoneinfo import ZoneInfo


class Logger:
    def __init__(self, log_dir: str | None = None):
        log_dir = "/home/user/workspace/ros2/logs"

        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now(
            tz=ZoneInfo("Asia/Seoul")
        ).strftime("%Y-%m-%d_%H-%M-%S")

        self.filename = os.path.join(log_dir, f"{timestamp}.csv")

        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z",
                "eul_roll", "eul_pitch", "eul_yaw",
                "mppi_u0", "mppi_u1",
                "mppi_solve_time",
                "cross_track_error",
                "speed_error",
                "desired_speed",
                "desired_speed_target",
                "desired_speed_cfg",
                "cmd_roll", "cmd_pitch", "cmd_yaw",
                "thrust_norm_cmd",
                "thrust_total_cmd",
                "vehicle_type_cfg",
                "guid_type_init",
                "wp_type_cfg",
                "guid_type_cfg",
                "guid_type_used",
                "vehicle_arming_state",
                "vehicle_nav_state",
                "vehicle_is_armed",
                "lookahead_distance_cfg",
                "waypoint_switch_distance_cfg",
                "wp_idx_heading",
                "wp_idx_passed",
                "mppi_reset_flag",
                "interrupt_active",
                "interrupt_prev",
                "stop_flag",
                "pf_done",
                "plot_complete",
                "dist_to_heading_wp",
                "dist_to_goal",
                "vt_x", "vt_y", "vt_z",
                "closest_x", "closest_y", "closest_z",
                "ndo_x", "ndo_y", "ndo_z",
            ])

    def update(
        self,
        t: float,
        pos,
        vel,
        eul,
        mppi_cmd,
        solve_time: float,
        cross_track_error: float,
        speed_error: float,
        desired_speed: float,
        desired_speed_target: float,
        att_cmd,
        thrust_norm_cmd: float,
        *,
        thrust_total_cmd: float = 0.0,
        desired_speed_cfg: float = float("nan"),
        vehicle_type_cfg: int = -1,
        guid_type_init: int = -1,
        wp_type_cfg: int = -1,
        guid_type_cfg: int = 0,
        guid_type_used: int = 0,
        vehicle_arming_state: int = -1,
        vehicle_nav_state: int = -1,
        vehicle_is_armed: bool = False,
        lookahead_distance_cfg: float = float("nan"),
        waypoint_switch_distance_cfg: float = float("nan"),
        wp_idx_heading: int = 0,
        wp_idx_passed: int = 0,
        mppi_reset_flag: int = 0,
        interrupt_active: int = 0,
        interrupt_prev: int = 0,
        stop_flag: int = 0,
        pf_done: bool = False,
        plot_complete: bool = False,
        dist_to_heading_wp: float = 0.0,
        dist_to_goal: float = 0.0,
        virtual_target=None,
        closest_point=None,
        ndo=None,
    ) -> None:
        def _vec3(vec):
            if vec is None:
                return 0.0, 0.0, 0.0
            try:
                return float(vec[0]), float(vec[1]), float(vec[2])
            except Exception:
                return 0.0, 0.0, 0.0

        vt_x, vt_y, vt_z = _vec3(virtual_target)
        cp_x, cp_y, cp_z = _vec3(closest_point)
        ndo_x, ndo_y, ndo_z = _vec3(ndo)

        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                t,
                pos[0], pos[1], pos[2],
                vel[0], vel[1], vel[2],
                eul[0], eul[1], eul[2],
                mppi_cmd[0], mppi_cmd[1],
                solve_time,
                cross_track_error,
                speed_error,
                desired_speed,
                desired_speed_target,
                desired_speed_cfg,
                att_cmd[0], att_cmd[1], att_cmd[2],
                thrust_norm_cmd,
                thrust_total_cmd,
                int(vehicle_type_cfg),
                int(guid_type_init),
                int(wp_type_cfg),
                int(guid_type_cfg),
                int(guid_type_used),
                int(vehicle_arming_state),
                int(vehicle_nav_state),
                int(bool(vehicle_is_armed)),
                lookahead_distance_cfg,
                waypoint_switch_distance_cfg,
                int(wp_idx_heading),
                int(wp_idx_passed),
                int(mppi_reset_flag),
                int(interrupt_active),
                int(interrupt_prev),
                int(stop_flag),
                int(bool(pf_done)),
                int(bool(plot_complete)),
                dist_to_heading_wp,
                dist_to_goal,
                vt_x, vt_y, vt_z,
                cp_x, cp_y, cp_z,
                ndo_x, ndo_y, ndo_z,
            ])


logger = Logger()
