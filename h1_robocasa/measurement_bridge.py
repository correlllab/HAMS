"""Publish RoboCasa task measurements (goal / success / reward) onto ROS.

The RoboCasa env shares our MjData (we step it ourselves), so after each
``mj_step`` the loop must call ``env.update_state()`` (refreshes fixture caches:
temperature, timer, door, rack) BEFORE reading ``_check_success``. This bridge
wraps that read and publishes:

  /robocasa/task_name   std_msgs/String   (the task env name, e.g. "TurnOnToasterOven", latched)
  /robocasa/task_goal   std_msgs/String   (the language instruction, latched)
  /robocasa/success     std_msgs/Bool     (sustained success, after debounce)
  /robocasa/reward      std_msgs/Float32  (sparse instantaneous = float(success))

Success is debounced over N consecutive successful steps (RoboCasa teleop uses
~15) so a single transient frame does not count as task completion.
"""
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger


class MeasurementBridge(Node):
    def __init__(self, env, debounce_steps=15, elastic_band=None, task_name=""):
        super().__init__("robocasa_measurement")
        self.env = env
        self.debounce_steps = int(debounce_steps)
        self._streak = 0
        self.lang = ""
        self.task_name = task_name

        # Optional elastic-band toggle service (same contract as the legacy
        # RosSensorBridge: std_srvs/Trigger -> ElasticBand.toggle()). Lets the
        # band be released without the windowed viewer / SPACE key.
        self.elastic_band = elastic_band
        if elastic_band is not None:
            self.create_service(Trigger, "/elastic_band/toggle", self._on_elastic_band_toggle)

        # transient_local so late subscribers still get the latched goal
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile

        latched = QoSProfile(depth=1)
        latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._task_pub = self.create_publisher(String, "/robocasa/task_name", latched)
        self._goal_pub = self.create_publisher(String, "/robocasa/task_goal", latched)
        self._succ_pub = self.create_publisher(Bool, "/robocasa/success", 10)
        self._rew_pub = self.create_publisher(Float32, "/robocasa/reward", 10)

    def _on_elastic_band_toggle(self, request, response):
        enabled = self.elastic_band.toggle()
        response.success = True
        response.message = "enabled" if enabled else "disabled"
        return response

    def publish_goal(self):
        self._task_pub.publish(String(data=self.task_name))   # task env name (latched)
        try:
            self.lang = self.env.get_ep_meta().get("lang", "") or ""
        except Exception as e:  # pragma: no cover - container only
            self.get_logger().warn(f"get_ep_meta failed: {e}")
            self.lang = ""
        self._goal_pub.publish(String(data=self.lang))
        self.get_logger().info(f"task {self.task_name!r} goal: {self.lang!r}")
        return self.lang

    def reset_episode(self):
        self._streak = 0

    def tick(self):
        """Read success/reward off the (already update_state()'d) env. Returns the
        debounced 'task complete' boolean."""
        try:
            success = bool(self.env._check_success())
        except Exception as e:  # pragma: no cover - container only
            self.get_logger().warn(f"_check_success failed: {e}")
            success = False
        self._streak = self._streak + 1 if success else 0
        held = self._streak >= self.debounce_steps
        self._succ_pub.publish(Bool(data=held))
        self._rew_pub.publish(Float32(data=float(success)))
        return held
