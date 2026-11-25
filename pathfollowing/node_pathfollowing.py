############################################################
#
#   - Name : node_pathfollowing.py
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

from std_msgs.msg import Float32, Float32MultiArray, Float64MultiArray, Int32MultiArray, Bool, Int32

from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleAcceleration
from px4_msgs.msg import TimesyncStatus
from px4_msgs.msg import VehicleAttitudeSetpoint

from custom_msgs.msg import LocalWaypointSetpoint
from custom_msgs.msg import ConveyLocalWaypointComplete

from .core.core_pathfollowing import PathFollowingCore
from .utils.logger import logger


# vehicle_type -> vehicle yaml. 3 / 4: ground (UGV) and surface (USV) vehicles under PX4
# rover_differential - their yaml has a "ground" section, which switches the core's output stage
# from thrust + attitude to speed + bearing (core_pathfollowing.GroundVehicleConfig)
VEHICLE_YAML = {1: 'quad.yaml', 2: 'octo.yaml', 3: 'ugv.yaml', 4: 'usv.yaml'}

# per-launch overrides of the corner speed limit (defaults as GuidanceTuningConfig)
CORNER_FLOAT_PARAMS = (
    ('corner_lat_accel_mps2', 4.0),
    ('corner_max_error_m', 1.5),
    ('corner_brake_accel_mps2', 3.0),
    ('corner_min_speed_mps', 1.0),
    ('corner_min_angle_deg', 10.0),
)


class NodePathFollowing(Node):
    def __init__(self):
        super().__init__('node_pathfollowing')

        # config file of this run (default sim.yaml): e.g. sim_ca_mppi.yaml for the learned CA + MPPI
        sim_cfg = self._load_yaml(str(self.declare_parameter('sim_yaml', 'sim.yaml').value))

        # =====================================================
        # parameters
        # =====================================================
        self.declare_parameter('vehicle_type')
        self.declare_parameter('guid_type')
        self.declare_parameter('wp_type')

        # RealGazebo 는 차량을 vehicle{id+1} 네임스페이스로 띄운다 (예: /vehicle1/fmu/...).
        # px4_ns 파라미터로 PX4 출력 토픽 접두사를 지정한다 (기본 '' = 네임스페이스 없음).
        self.declare_parameter('px4_ns', '')
        px4_ns = self.get_parameter('px4_ns').get_parameter_value().string_value
        px4_ns = px4_ns.rstrip('/')

        vehicle_type = self._require_int(
            'vehicle_type',
            allowed=set(VEHICLE_YAML),
            default=sim_cfg.get('vehicle_type'),
        )
        guid_type = self._require_int(
            'guid_type',
            allowed={0, 1, 2},
            default=sim_cfg.get('guid_type'),
        )
        wp_type = self._require_int(
            'wp_type',
            allowed=set(range(0, 10)),
            default=sim_cfg.get('wp_type'),
        )

        vehicle_yaml = VEHICLE_YAML[vehicle_type]
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

        # per-launch override of path_following.vertical_priority (see GuidanceTuningConfig)
        pf_cfg = sim_cfg.get('path_following', {})
        tuning_cfg = pf_cfg.get('guidance_tuning', pf_cfg) if isinstance(pf_cfg, dict) else {}
        if isinstance(tuning_cfg, dict):
            for name in ('vertical_priority', 'mppi_speed_transition', 'yaw_hold'):
                tuning_cfg[name] = bool(self.declare_parameter(name, bool(tuning_cfg.get(name, False))).value)
            tuning_cfg['yaw_mode'] = str(self.declare_parameter(
                'yaw_mode', str(tuning_cfg.get('yaw_mode', 'hold' if tuning_cfg.get('yaw_hold') else 'waypoint'))).value)
            # corner speed limit / rollout = PF guidance (GuidanceTuningConfig); node_mppi takes
            # the same parameters - give both nodes the same values (pf_test.sh PF_TEST_CORNER)
            for name in ('corner_speed_enable', 'mppi_rollout_sync'):
                tuning_cfg[name] = bool(self.declare_parameter(name, bool(tuning_cfg.get(name, False))).value)
            for name, default in CORNER_FLOAT_PARAMS:
                tuning_cfg[name] = float(self.declare_parameter(name, float(tuning_cfg.get(name, default))).value)

        # =====================================================
        # core
        # =====================================================
        self.core = PathFollowingCore(
            guid_type=guid_type,
            wp_type=wp_type,
            vehicle_cfg=vehicle_cfg,
            sim_cfg=sim_cfg,
            logger_obj=logger,
            vehicle_type=vehicle_type,
        )

        # =====================================================
        # node states
        # =====================================================
        self.controller_heartbeat_ok = False
        self.path_planning_heartbeat_ok = False
        self.collision_avoidance_heartbeat_ok = False
        self.plot_waypoint_complete = False

        # =====================================================
        # subscribers
        # =====================================================
        self.controller_heartbeat_subscription = self.create_subscription(
            Bool,
            '/controller_heartbeat',
            self._controller_heartbeat_callback,
            10,
        )

        self.path_planning_heartbeat_subscription = self.create_subscription(
            Bool,
            '/path_planning_heartbeat',
            self._path_planning_heartbeat_callback,
            10,
        )

        self.collision_avoidance_heartbeat_subscription = self.create_subscription(
            Bool,
            '/collision_avoidance_heartbeat',
            self._collision_avoidance_heartbeat_callback,
            10,
        )

        self.local_waypoint_subscription = self.create_subscription(
            LocalWaypointSetpoint,
            '/local_waypoint_setpoint_to_PF',
            self._local_waypoint_callback,
            1,
        )

        self.vehicle_local_position_subscription = self.create_subscription(
            VehicleLocalPosition,
            f'{px4_ns}/fmu/out/vehicle_local_position',
            self._vehicle_local_position_callback,
            qos_profile_sensor_data,
        )

        self.vehicle_attitude_subscription = self.create_subscription(
            VehicleAttitude,
            f'{px4_ns}/fmu/out/vehicle_attitude',
            self._vehicle_attitude_callback,
            qos_profile_sensor_data,
        )

        self.vehicle_acceleration_subscription = self.create_subscription(
            VehicleAcceleration,
            f'{px4_ns}/fmu/out/vehicle_acceleration',
            self._vehicle_acceleration_callback,
            qos_profile_sensor_data,
        )

        self.timesync_status_subscription = self.create_subscription(
            TimesyncStatus,
            f'{px4_ns}/fmu/out/timesync_status',
            self._timesync_status_callback,
            qos_profile_sensor_data,
        )

        self.mppi_output_subscription = self.create_subscription(
            Float32MultiArray,
            'MPPI/out/guidance_cmd',
            self._mppi_output_callback,
            qos_profile_sensor_data,
        )

        self.plot_waypoint_complete_subscription = self.create_subscription(
            Bool,
            '/plot_wp_complete',
            self._plot_waypoint_complete_callback,
            10,
        )

        # Runtime flag/control channels for safer integration testing.
        self.interrupt_flag_subscription = self.create_subscription(
            Bool,
            '/path_following/set_interrupt_flag',
            self._interrupt_flag_callback,
            10,
        )

        self.stop_flag_subscription = self.create_subscription(
            Bool,
            '/path_following/set_stop_flag',
            self._stop_flag_callback,
            10,
        )

        self.guid_type_subscription = self.create_subscription(
            Int32,
            '/path_following/set_guid_type',
            self._guid_type_callback,
            10,
        )

        # Runtime cruise-speed override (same topic node_mppi listens to); used by
        # the speed-regulated guid 0 law (path_following.pos_speed_mode).
        self.desired_speed_subscription = self.create_subscription(
            Float32,
            '/desired_speed',
            self._desired_speed_callback,
            10,
        )

        # =====================================================
        # publishers
        # =====================================================
        self.attitude_setpoint_publisher = self.create_publisher(
            VehicleAttitudeSetpoint,
            '/pf_att_2_control',
            10,
        )

        # ground / surface vehicle: [speed m/s, bearing rad (NED), yaw rate rad/s, acc_t, acc_n, align,
        # guid_type_used] - the driver turns speed + bearing into PX4's offboard velocity setpoint
        self.ground_cmd_publisher = self.create_publisher(
            Float32MultiArray,
            '/pf_ground_cmd',
            10,
        )

        self.waypoint_ack_publisher = self.create_publisher(
            ConveyLocalWaypointComplete,
            '/convey_local_waypoint_complete',
            10,
        )

        self.heading_waypoint_index_publisher = self.create_publisher(
            Int32,
            '/heading_waypoint_index',
            1,
        )

        self.path_following_heartbeat_publisher = self.create_publisher(
            Bool,
            '/path_following_heartbeat',
            10,
        )

        self.path_following_complete_publisher = self.create_publisher(
            Bool,
            '/path_following_complete',
            1,
        )

        self.path_following_waypoint_plot_publisher = self.create_publisher(
            Float64MultiArray,
            '/path_following_waypoint_to_plotter',
            1,
        )

        self.mppi_runtime_flags_publisher = self.create_publisher(
            Int32MultiArray,
            'MPPI/in/runtime_flags',
            10,
        )

        self.mppi_vehicle_state_publisher = self.create_publisher(
            Float32MultiArray,
            'MPPI/in/vehicle_state',
            10,
        )

        self.mppi_waypoints_ned_publisher = self.create_publisher(
            Float32MultiArray,
            'MPPI/in/waypoints_ned',
            10,
        )

        self.gpr_disturbance_acc_publisher = self.create_publisher(
            Float32MultiArray,
            'GPR/in/disturbance_acc',
            10,
        )

        self.runtime_flags_publisher = self.create_publisher(
            Int32MultiArray,
            '/path_following/runtime_flags',
            10,
        )

        # =====================================================
        # timers
        # =====================================================
        self.heartbeat_timer = self.create_timer(
            1.0,
            self._publish_heartbeat,
        )

        self.control_timer = self.create_timer(
            self.core.dt_gcu,
            self._control_timer_callback,
        )

        period_mppi_input = 0.5 * self.core.dt_mppi

        self.mppi_runtime_flags_timer = self.create_timer(
            period_mppi_input,
            self._publish_mppi_runtime_flags,
        )

        self.mppi_vehicle_state_timer = self.create_timer(
            period_mppi_input,
            self._publish_mppi_vehicle_state,
        )

        self.mppi_waypoints_ned_timer = self.create_timer(
            period_mppi_input * 10.0,
            self._publish_mppi_waypoints_ned,
        )

        self.gpr_disturbance_acc_timer = self.create_timer(
            period_mppi_input,
            self._publish_gpr_disturbance_acc,
        )

        self.path_following_waypoint_plot_timer = self.create_timer(
            5.0,
            self._publish_path_following_waypoints_for_plot,
        )

        self.runtime_flags_timer = self.create_timer(
            1.0,
            self._publish_runtime_flags,
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
            raise RuntimeError(f"[FATAL] vehicle yaml not found: {yaml_path}")

        with open(yaml_path, 'r') as file:
            document = yaml.safe_load(file) or {}
        if not isinstance(document, dict):
            raise RuntimeError(f"[FATAL] invalid yaml root in {yaml_path}: expected mapping.")

        node_name = self.get_name()
        if node_name not in document:
            raise RuntimeError(f"[FATAL] missing node section '{node_name}' in {yaml_path}")

        node_cfg = document[node_name]
        if not isinstance(node_cfg, dict) or 'ros__parameters' not in node_cfg:
            raise RuntimeError(
                f"[FATAL] missing '{node_name}.ros__parameters' section in {yaml_path}"
            )
        params = node_cfg['ros__parameters']
        if not isinstance(params, dict):
            raise RuntimeError(f"[FATAL] '{node_name}.ros__parameters' must be a mapping.")
        return params

    # =====================================================
    # internal helpers
    # =====================================================
    def _update_core_heartbeats(self) -> None:
        self.core.update_heartbeats(
            self.controller_heartbeat_ok,
            self.path_planning_heartbeat_ok,
            self.collision_avoidance_heartbeat_ok,
        )

    # =====================================================
    # subscriber callbacks
    # =====================================================
    def _controller_heartbeat_callback(self, msg: Bool) -> None:
        self.controller_heartbeat_ok = bool(msg.data)
        self._update_core_heartbeats()

    def _path_planning_heartbeat_callback(self, msg: Bool) -> None:
        self.path_planning_heartbeat_ok = bool(msg.data)
        self._update_core_heartbeats()

    def _collision_avoidance_heartbeat_callback(self, msg: Bool) -> None:
        self.collision_avoidance_heartbeat_ok = bool(msg.data)
        self._update_core_heartbeats()

    def _local_waypoint_callback(self, msg: LocalWaypointSetpoint) -> None:
        self.core.update_waypoints(
            path_planning_complete=bool(msg.path_planning_complete),
            waypoint_x=msg.waypoint_x,
            waypoint_y=msg.waypoint_y,
            waypoint_z=msg.waypoint_z,
        )

    def _vehicle_local_position_callback(self, msg: VehicleLocalPosition) -> None:
        self.core.update_local_position(
            msg.x,
            msg.y,
            msg.z,
            msg.vx,
            msg.vy,
            msg.vz,
        )

    def _vehicle_attitude_callback(self, msg: VehicleAttitude) -> None:
        self.core.update_attitude_quat(
            msg.q[0],
            msg.q[1],
            msg.q[2],
            msg.q[3],
            timestamp_us=int(msg.timestamp),
        )

    def _vehicle_acceleration_callback(self, msg: VehicleAcceleration) -> None:
        self.core.update_accel_xyz(
            msg.xyz[0],
            msg.xyz[1],
            msg.xyz[2],
        )

    def _timesync_status_callback(self, msg: TimesyncStatus) -> None:
        self.core.update_timesync_timestamp_us(int(msg.timestamp))

    def _mppi_output_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 3:
            raise RuntimeError(f"[FATAL] MPPI/out/guidance_cmd requires 3 values, got {len(msg.data)}.")
        self.core.update_mppi_output(
            msg.data[0],
            msg.data[1],
            msg.data[2],
        )

    def _plot_waypoint_complete_callback(self, msg: Bool) -> None:
        self.plot_waypoint_complete = bool(msg.data)
        self.core.update_plot_waypoint_complete(self.plot_waypoint_complete)

    def _interrupt_flag_callback(self, msg: Bool) -> None:
        self.core.update_interrupt_flag(bool(msg.data))

    def _stop_flag_callback(self, msg: Bool) -> None:
        self.core.update_stop_flag(bool(msg.data))

    def _guid_type_callback(self, msg: Int32) -> None:
        self.core.update_guid_type(int(msg.data))

    def _desired_speed_callback(self, msg: Float32) -> None:
        self.core.update_desired_speed(float(msg.data))

    # =====================================================
    # timer callbacks
    # =====================================================
    def _control_timer_callback(self) -> None:
        output = self.core.update()
        if output is None:
            return

        ground_command = output.get("ground_cmd")
        if ground_command is not None:
            ground_msg = Float32MultiArray()
            ground_msg.data = [
                float(ground_command.speed), float(ground_command.bearing), float(ground_command.yaw_rate),
                float(ground_command.acc_t), float(ground_command.acc_n), float(ground_command.align),
                float(output["guid_type_used"]),
            ]
            self.ground_cmd_publisher.publish(ground_msg)
            self._publish_progress(output)
            return

        attitude_command = output["att_cmd"]

        attitude_setpoint_msg = VehicleAttitudeSetpoint()
        attitude_setpoint_msg.q_d[0] = attitude_command.q_d[0]
        attitude_setpoint_msg.q_d[1] = attitude_command.q_d[1]
        attitude_setpoint_msg.q_d[2] = attitude_command.q_d[2]
        attitude_setpoint_msg.q_d[3] = attitude_command.q_d[3]
        attitude_setpoint_msg.thrust_body[0] = 0.0
        attitude_setpoint_msg.thrust_body[1] = 0.0
        attitude_setpoint_msg.thrust_body[2] = float(attitude_command.thrust_body[2])
        self.attitude_setpoint_publisher.publish(attitude_setpoint_msg)
        self._publish_progress(output)

    def _publish_progress(self, output) -> None:
        heading_waypoint_index_msg = Int32()
        heading_waypoint_index_msg.data = int(output["heading_wp_idx"])
        self.heading_waypoint_index_publisher.publish(heading_waypoint_index_msg)

        if bool(output["pf_done"]):
            path_following_complete_msg = Bool()
            path_following_complete_msg.data = True
            self.path_following_complete_publisher.publish(path_following_complete_msg)

        if self.core.pop_waypoint_ack():
            waypoint_ack_msg = ConveyLocalWaypointComplete()
            waypoint_ack_msg.convey_local_waypoint_is_complete = True
            self.waypoint_ack_publisher.publish(waypoint_ack_msg)

    # =====================================================
    # publishers
    # =====================================================
    def _publish_heartbeat(self) -> None:
        heartbeat_msg = Bool()
        heartbeat_msg.data = True
        self.path_following_heartbeat_publisher.publish(heartbeat_msg)

    def _publish_mppi_runtime_flags(self) -> None:
        input_msg = Int32MultiArray()
        input_msg.data = self.core.get_mppi_runtime_flags()
        self.mppi_runtime_flags_publisher.publish(input_msg)

    def _publish_mppi_vehicle_state(self) -> None:
        input_msg = Float32MultiArray()
        input_msg.data = self.core.get_mppi_vehicle_state()
        self.mppi_vehicle_state_publisher.publish(input_msg)

    def _publish_mppi_waypoints_ned(self) -> None:
        waypoint_data = self.core.get_mppi_waypoints_ned()
        if waypoint_data is None:
            return

        input_msg = Float32MultiArray()
        input_msg.data = waypoint_data
        self.mppi_waypoints_ned_publisher.publish(input_msg)

    def _publish_gpr_disturbance_acc(self) -> None:
        input_msg = Float32MultiArray()
        input_msg.data = self.core.get_gpr_disturbance_acc()
        self.gpr_disturbance_acc_publisher.publish(input_msg)

    def _publish_path_following_waypoints_for_plot(self) -> None:
        if self.plot_waypoint_complete:
            return

        waypoint_plot_data = self.core.get_waypoints_for_plot()
        if waypoint_plot_data is None:
            return

        waypoint_plot_msg = Float64MultiArray()
        waypoint_plot_msg.data = waypoint_plot_data
        self.path_following_waypoint_plot_publisher.publish(waypoint_plot_msg)

    def _publish_runtime_flags(self) -> None:
        msg = Int32MultiArray()
        msg.data = self.core.get_runtime_flags_int()
        self.runtime_flags_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = NodePathFollowing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
