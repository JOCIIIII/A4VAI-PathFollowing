############################################################
#
#   - Name : node_mppi.py
#
#                                 - KAIST FDCL, 2026.03.11
#
############################################################

import os
import yaml
import rclpy

from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray

from .core.core_mppi import MPPICore


# per-launch overrides of the corner speed limit (as node_pathfollowing.CORNER_FLOAT_PARAMS)
CORNER_FLOAT_PARAMS = (
    ('corner_lat_accel_mps2', 4.0),
    ('corner_max_error_m', 1.5),
    ('corner_brake_accel_mps2', 3.0),
    ('corner_min_speed_mps', 1.0),
    ('corner_min_angle_deg', 10.0),
)


class NodeMPPI(Node):
    def __init__(self):
        super().__init__('node_mppi')

        # config file of this run (default sim.yaml): e.g. sim_ca_mppi.yaml for the learned CA + MPPI
        sim_cfg = self._load_yaml(str(self.declare_parameter('sim_yaml', 'sim.yaml').value))

        self.declare_parameter('vehicle_type')
        self.declare_parameter('guid_type')

        vehicle_type = self._require_int(
            'vehicle_type',
            allowed={1, 2},
            default=sim_cfg.get('vehicle_type'),
        )
        guid_type = self._require_int(
            'guid_type',
            allowed={0, 1, 2},
            default=sim_cfg.get('guid_type'),
        )
        vehicle_yaml = 'quad.yaml' if vehicle_type == 1 else 'octo.yaml'
        vehicle_cfg = self._load_yaml(vehicle_yaml)
        # the sim yaml may override vehicle-model entries per vehicle file (sim_ca_mppi.yaml: the
        # MPPI rollout's attitude time constants): {quad|octo: {section: {key: value}}}
        overrides = (sim_cfg.get('vehicle_override') or {}).get(vehicle_yaml.split('.')[0]) or {}
        for section, values in overrides.items():
            if isinstance(values, dict) and isinstance(vehicle_cfg.get(section), dict):
                vehicle_cfg[section].update(values)
        # real vehicle: the normalised hover thrust measured on it (PX4 hover thrust estimate in
        # its log) sets the thrust scale - only mass*g/(n_motor*max_thrust_per_rotor) enters the
        # thrust command, the disturbance observer and the MPPI rollout - and its tilt limit
        # replaces max_tilt_deg. 0 = keep the yaml. Give node_pathfollowing and node_mppi the same.
        hover_thrust = float(self.declare_parameter('hover_thrust', 0.0).value)
        if hover_thrust > 0.0:
            model, actuator = vehicle_cfg['model'], vehicle_cfg['actuator']
            actuator['max_thrust_per_rotor'] = (
                float(model['mass']) * float(model['gravity']) / (int(actuator['n_motor']) * hover_thrust))
        max_tilt_deg = float(self.declare_parameter('max_tilt_deg', 0.0).value)
        if max_tilt_deg > 0.0:
            vehicle_cfg['px4_model']['max_tilt_deg'] = max_tilt_deg

        # corner speed limit / rollout = PF guidance: same per-launch parameters as
        # node_pathfollowing (see core_pathfollowing.GuidanceTuningConfig)
        pf_cfg = sim_cfg.get('path_following', {})
        tuning_cfg = pf_cfg.get('guidance_tuning', pf_cfg) if isinstance(pf_cfg, dict) else {}
        if isinstance(tuning_cfg, dict):
            for name in ('corner_speed_enable', 'mppi_rollout_sync'):
                tuning_cfg[name] = bool(self.declare_parameter(name, bool(tuning_cfg.get(name, False))).value)
            for name, default in CORNER_FLOAT_PARAMS:
                tuning_cfg[name] = float(self.declare_parameter(name, float(tuning_cfg.get(name, default))).value)

        self.core = MPPICore(
            guid_type=guid_type,
            vehicle_cfg=vehicle_cfg,
            sim_cfg=sim_cfg,
        )

        # =====================================================
        # subscribers
        # =====================================================
        self.mppi_runtime_flags_subscription = self.create_subscription(
            Int32MultiArray,
            'MPPI/in/runtime_flags',
            self._mppi_runtime_flags_callback,
            qos_profile_sensor_data,
        )

        self.mppi_vehicle_state_subscription = self.create_subscription(
            Float32MultiArray,
            'MPPI/in/vehicle_state',
            self._mppi_vehicle_state_callback,
            qos_profile_sensor_data,
        )

        self.mppi_waypoints_ned_subscription = self.create_subscription(
            Float32MultiArray,
            'MPPI/in/waypoints_ned',
            self._mppi_waypoints_ned_callback,
            qos_profile_sensor_data,
        )

        self.gpr_disturbance_acc_subscription = self.create_subscription(
            Float32MultiArray,
            'GPR/in/disturbance_acc',
            self._gpr_disturbance_acc_callback,
            qos_profile_sensor_data,
        )

        # ★ 동적 desired_speed override (/desired_speed). 학습 속도 신경망(speed_policy_node)
        #   또는 수동 pub 이 목표 순항속도를 실시간 갱신. 안 오면 config 기본값 사용.
        self.desired_speed_subscription = self.create_subscription(
            Float32,
            '/desired_speed',
            self._desired_speed_callback,
            qos_profile_sensor_data,
        )

        # =====================================================
        # publishers
        # =====================================================
        self.mppi_output_publisher = self.create_publisher(
            Float32MultiArray,
            'MPPI/out/guidance_cmd',
            10,
        )

        # =====================================================
        # timers
        # =====================================================
        self.solve_timer = self.create_timer(
            self.core.dt_mppi,
            self._solve_timer_callback,
        )

        self.gpr_opt_timer = self.create_timer(
            self.core.dt_gpr_opt,
            self._gpr_opt_timer_callback,
        )

    # =====================================================
    # parameter helpers
    # =====================================================
    def _require_int(
        self,
        name: str,
        allowed: set[int] | None = None,
        default: int | None = None,
    ) -> int:
        parameter = self.get_parameter(name)

        if default is None:
            if parameter.type_ == Parameter.Type.NOT_SET:
                raise RuntimeError(
                    f"[FATAL] Required parameter '{name}' is not set. "
                    f"Pass a YAML file via: --ros-args --params-file <file.yaml>"
                )
            value = int(parameter.value)
        else:
            if parameter.type_ == Parameter.Type.NOT_SET:
                value = int(default)
            else:
                value = int(parameter.value)

        if allowed is not None and value not in allowed:
            raise RuntimeError(
                f"[FATAL] Invalid '{name}'={value}. Allowed: {sorted(list(allowed))}"
            )

        return value

    def _load_yaml(self, filename: str) -> dict:
        share_directory = get_package_share_directory('pathfollowing')
        yaml_path = os.path.join(share_directory, 'config', filename)

        if not os.path.exists(yaml_path):
            raise RuntimeError(f"[FATAL] yaml not found: {yaml_path}")

        with open(yaml_path, 'r') as file:
            document = yaml.safe_load(file) or {}
        if not isinstance(document, dict):
            raise RuntimeError(f"[FATAL] invalid yaml root in {yaml_path}: expected mapping.")

        section_candidates = (self.get_name(), 'node_pathfollowing')
        node_cfg = None
        selected_section = None
        for section_name in section_candidates:
            if section_name in document:
                node_cfg = document[section_name]
                selected_section = section_name
                break
        if node_cfg is None:
            raise RuntimeError(
                f"[FATAL] missing node section in {yaml_path}. "
                f"Tried: {section_candidates}"
            )
        if not isinstance(node_cfg, dict) or 'ros__parameters' not in node_cfg:
            raise RuntimeError(
                f"[FATAL] missing '{selected_section}.ros__parameters' section in {yaml_path}"
            )
        params = node_cfg['ros__parameters']
        if not isinstance(params, dict):
            raise RuntimeError(f"[FATAL] '{selected_section}.ros__parameters' must be a mapping.")
        return params

    # =====================================================
    # subscriber callbacks
    # =====================================================
    def _mppi_runtime_flags_callback(self, msg: Int32MultiArray) -> None:
        self.core.update_runtime_flags(msg.data)

    def _mppi_vehicle_state_callback(self, msg: Float32MultiArray) -> None:
        self.core.update_vehicle_state(msg.data)

    def _mppi_waypoints_ned_callback(self, msg: Float32MultiArray) -> None:
        self.core.update_waypoints(msg.data)

    def _gpr_disturbance_acc_callback(self, msg: Float32MultiArray) -> None:
        self.core.update_disturbance_acc(msg.data)

    def _desired_speed_callback(self, msg: Float32) -> None:
        # 음수/비정상은 무시(안전). 유효값이면 MPPI cruise 목표속도 override.
        v = float(msg.data)
        if v >= 0.0:
            self.core.desired_speed_override = v

    # =====================================================
    # timer callbacks
    # =====================================================
    def _solve_timer_callback(self) -> None:
        output = self.core.solve()
        if output is None:
            return

        msg = Float32MultiArray()
        msg.data = output
        self.mppi_output_publisher.publish(msg)

    def _gpr_opt_timer_callback(self) -> None:
        self.core.optimize_gpr_if_needed()

    # =====================================================
    # main
    # =====================================================
    def destroy_node(self):
        try:
            self.core.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NodeMPPI()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
