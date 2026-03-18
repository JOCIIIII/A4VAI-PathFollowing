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

from std_msgs.msg import Float32MultiArray, Int32MultiArray

from .core.core_mppi import MPPICore


class NodeMPPI(Node):
    def __init__(self):
        super().__init__('node_mppi')

        sim_cfg = self._load_yaml('sim.yaml')

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
