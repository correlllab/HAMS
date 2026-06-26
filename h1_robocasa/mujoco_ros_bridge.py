"""ROS 2 publishers for MuJoCo RGBD cameras + downward half-sphere lidar.

One RGBD camera set is published per configured camera under
/realsense/<namespace>/* — by default the head camera, plus the two eye-in-hand
gripper cameras (namespaces head, left_hand, right_hand). Topics published
(shown for `head`; left_hand/right_hand mirror the same five topics; frame IDs
match URDF link names):
  /clock                                                        rosgraph_msgs/Clock (sim time)
  /realsense/head/color/image_raw                               sensor_msgs/Image (rgb8)
  /realsense/head/color/image_raw/compressed                    sensor_msgs/CompressedImage (jpeg)
  /realsense/head/aligned_depth_to_color/image_raw              sensor_msgs/Image (16UC1 mm)
  /realsense/head/aligned_depth_to_color/image_raw/compressedDepth
                                                                sensor_msgs/CompressedImage (16UC1 png)
  /realsense/head/color/camera_info                             sensor_msgs/CameraInfo
  /livox/lidar                                                  livox_ros_driver2/CustomMsg
  /livox/pointcloud                                             sensor_msgs/PointCloud2 (same scan, driver-native layout)
  /livox/imu                                                    sensor_msgs/Imu (co-located w/ lidar)

tick() is called once per sim step from the main loop. High-rate publishers
(clock, IMU) run at every call; slow publishers (camera, lidar) are
rate-throttled internally. MuJoCo's EGL renderer context is thread-affine,
so rendering must happen on the main thread.
"""
import io
import os
import struct
from array import array as _array
from collections import namedtuple

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from PIL import Image as PILImage
from livox_ros_driver2.msg import CustomMsg, CustomPoint
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, PointCloud2, PointField
from std_srvs.srv import Trigger


def _sim_time_to_msg(sim_time: float) -> TimeMsg:
    """Convert MuJoCo sim time (float seconds) to a ROS Time message."""
    sec = int(sim_time)
    nanosec = int((sim_time - sec) * 1e9)
    msg = TimeMsg()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


# PointCloud2 layout mirrors livox_ros_driver2's native xfer_format=0 output
# (LivoxPointXyzrtlt in lddc.cpp:262-296): 26-byte unaligned point. Keep this
# in sync with the real driver so downstream nodes don't care whether the
# scan comes from sim or hardware.
PC2_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("intensity", np.float32),
        ("tag", np.uint8),
        ("line", np.uint8),
        ("timestamp", np.float64),
    ],
    align=False,
)
assert PC2_DTYPE.itemsize == 26
PC2_POINT_STEP = PC2_DTYPE.itemsize
PC2_FIELDS = [
    PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="tag",       offset=16, datatype=PointField.UINT8,   count=1),
    PointField(name="line",      offset=17, datatype=PointField.UINT8,   count=1),
    PointField(name="timestamp", offset=18, datatype=PointField.FLOAT64, count=1),
]


# One render target + its five RealSense-style publishers. All cameras share the
# bridge's renderer/resolution/rate; each carries its own MuJoCo camera name,
# header frame_id, intrinsics, and /realsense/<namespace>/* publishers.
_CameraPub = namedtuple(
    "_CameraPub",
    "name frame cam_id info_msg pub_rgb pub_depth pub_rgb_raw pub_depth_raw pub_info",
)


def _build_camera_info(width: int, height: int, fovy_deg: float, frame_id: str) -> CameraInfo:
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    info = CameraInfo()
    info.width = width
    info.height = height
    info.distortion_model = "plumb_bob"
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    info.header.frame_id = frame_id
    return info


class RosSensorBridge(Node):
    def __init__(
        self,
        model,
        data,
        cameras=None,
        lidar_body: str = "livox_link",
        lidar_frame: str = "lidar_link",
        lidar_exclude_body: str = "torso_link",
        cam_width: int = 1280,
        cam_height: int = 720,
        cam_rate_hz: float = 15.0,
        lidar_az_rays: int = 180,
        lidar_el_rays: int = 28,
        lidar_rate_hz: float = 5.0,
        lidar_max_range: float = 30.0,
        lidar_min_range: float = 0.1,
        imu_quat_sensor: str = "livox_imu_quat",
        imu_gyro_sensor: str = "livox_imu_gyro",
        imu_acc_sensor: str = "livox_imu_acc",
        imu_frame: str = "lidar_link",
        imu_rate_hz: float = 100.0,
        elastic_band=None,
        sim_lock=None,
    ):
        super().__init__("mujoco_sensors")
        self.model = model
        self.data = data
        self.elastic_band = elastic_band
        self.sim_lock = sim_lock

        self.lidar_frame = lidar_frame

        # cameras: list of (mujoco_cam_name, ros_namespace, frame_id). Each entry
        # publishes the full RealSense topic set under /realsense/<namespace>/*.
        if cameras is None:
            cameras = [("head_cam", "head", "camera_color_optical_frame")]
        self._camera_specs = cameras

        self.lidar_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lidar_body)
        if self.lidar_body_id < 0:
            raise RuntimeError(f"body '{lidar_body}' not found in MJCF")
        # Lidar self-occlusion mask: ignore returns whose geom is attached to
        # `torso_link` (the body the lidar is mounted on — its outer shell
        # would otherwise show as a phantom near-zero return). Limbs hanging
        # off the torso (arms, head, legs) are deliberately NOT in this mask
        # so they occlude the scan like real obstacles.
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lidar_exclude_body)
        if torso_id < 0:
            raise RuntimeError(f"body '{lidar_exclude_body}' not found in MJCF")
        self.lidar_exclude_body_id = int(torso_id)
        self.geom_excluded = (model.geom_bodyid == self.lidar_exclude_body_id)
        # mj_multiRay / mj_ray require a (6, 1) column vector here.
        self.lidar_geomgroup = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8).reshape(6, 1)

        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_period = 1.0 / cam_rate_hz
        # Far-plane distance in world units. MuJoCo's depth buffer reports
        # zfar for rays that miss; mask those pixels to "no return" so
        # consumers don't treat the far plane as a real surface.
        self.zfar = float(model.vis.map.zfar * model.stat.extent)

        self.lidar_period = 1.0 / lidar_rate_hz
        self.lidar_max_range = lidar_max_range
        self.lidar_min_range = lidar_min_range
        # Livox MID-360 FOV in WORLD coordinates is el ∈ [-7°, +52°] (mostly
        # above horizon). The sensor is mounted upside-down in the MJCF, so
        # local +Z = world -Z; flipping signs gives a LOCAL elevation range
        # of [-52°, +7°] that produces the correct world coverage.
        # Defaults of 360 az × 56 el rays at 10 Hz ≈ 200,000 pts/s, matching
        # the real sensor's per-second density. The uniform grid is a
        # simplification of the MID-360's non-repetitive scan pattern.
        az = np.linspace(0.0, 2.0 * np.pi, lidar_az_rays, endpoint=False, dtype=np.float64)
        el = np.linspace(-np.deg2rad(52.0), np.deg2rad(7.0), lidar_el_rays, endpoint=False, dtype=np.float64)
        az_grid, el_grid = np.meshgrid(az, el, indexing="xy")
        cos_el = np.cos(el_grid)
        self.lidar_local_dirs = np.stack(
            [cos_el * np.cos(az_grid), cos_el * np.sin(az_grid), np.sin(el_grid)],
            axis=-1,
        ).reshape(-1, 3)
        self.lidar_rays = self.lidar_local_dirs.shape[0]
        self.lidar_az_rays = lidar_az_rays
        # Mid-360 has 6 laser lines. Map each synthetic ray's elevation index
        # mod 6 to a line number so FAST-LIO's N_SCANS=6 filter accepts all
        # points evenly distributed across scans.
        self.lidar_lines = ((np.arange(self.lidar_rays) // lidar_az_rays) % 6).astype(np.uint8)
        # Per-point acquisition time within a scan: model a constant-angular-
        # velocity azimuth sweep (rotary approximation — all elevations at a
        # given azimuth fire together), so offset_time depends only on the
        # azimuth index. The grid above is elevation-major / azimuth-fast, so
        # ray i has azimuth index (i % lidar_az_rays). FAST-LIO's AVIA path
        # reads CustomPoint.offset_time [ns] and divides by 1e6 → per-point
        # time [ms] for motion deskew. Span = one scan period (1 / rate).
        az_idx = np.arange(self.lidar_rays) % lidar_az_rays
        self.lidar_offset_time_ns = (
            (az_idx / lidar_az_rays) * self.lidar_period * 1e9).astype(np.uint32)
        # Preallocated mj_multiRay output buffers (overwritten each tick).
        # mujoco 3.3.1's pybind binding requires column-vector shapes
        # (m, 1) for the geomid / dist / vec / pnt / geomgroup args.
        self._lidar_dists = np.full((self.lidar_rays, 1), -1.0, dtype=np.float64)
        self._lidar_geomids = np.full((self.lidar_rays, 1), -1, dtype=np.int32)

        # /clock stays on default reliable QoS so subscribers don't drop ticks.
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)

        # Build one _CameraPub per requested camera. Intrinsics (fovy) are read
        # from each camera's own MJCF definition; resolution/rate are shared.
        self.cameras = []
        for mj_name, ns, frame in self._camera_specs:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, mj_name)
            if cam_id < 0:
                raise RuntimeError(f"camera '{mj_name}' not found in MJCF")
            fovy = float(model.cam_fovy[cam_id])
            self.cameras.append(_CameraPub(
                name=mj_name,
                frame=frame,
                cam_id=cam_id,
                info_msg=_build_camera_info(cam_width, cam_height, fovy, frame),
                pub_rgb=self.create_publisher(
                    CompressedImage, f"/realsense/{ns}/color/image_raw/compressed", qos_profile_sensor_data),
                pub_depth=self.create_publisher(
                    CompressedImage, f"/realsense/{ns}/aligned_depth_to_color/image_raw/compressedDepth", qos_profile_sensor_data),
                pub_rgb_raw=self.create_publisher(
                    Image, f"/realsense/{ns}/color/image_raw", qos_profile_sensor_data),
                pub_depth_raw=self.create_publisher(
                    Image, f"/realsense/{ns}/aligned_depth_to_color/image_raw", qos_profile_sensor_data),
                pub_info=self.create_publisher(
                    CameraInfo, f"/realsense/{ns}/color/camera_info", qos_profile_sensor_data),
            ))
        # FAST-LIO subscribes to /livox/lidar (CustomMsg) and /livox/imu (Imu)
        # with RELIABLE QoS (laserMapping.cpp:923,929). Match it so DDS doesn't
        # drop scans on a QoS mismatch.
        self.pub_lidar = self.create_publisher(
            CustomMsg, "/livox/lidar",
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=20))
        self.pub_pc2 = self.create_publisher(
            PointCloud2, "/livox/pointcloud",
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=20))
        self.pub_imu = self.create_publisher(
            Imu, "/livox/imu",
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=10))

        self.imu_frame = imu_frame
        self.imu_period = 1.0 / imu_rate_hz
        self.imu_quat_adr, self.imu_quat_dim = self._sensor_adr(imu_quat_sensor)
        self.imu_gyro_adr, self.imu_gyro_dim = self._sensor_adr(imu_gyro_sensor)
        self.imu_acc_adr, self.imu_acc_dim = self._sensor_adr(imu_acc_sensor)

        if elastic_band is not None:
            self.create_service(
                Trigger, "/elastic_band/toggle", self._on_elastic_band_toggle
            )

        # Lazy renderer — created on first tick() call, which runs on the
        # same (main) thread as the viewer and subsequent renders.
        self._renderer = None

        # Rate throttling uses sim time (data.time) not wall-clock, so topic
        # rates are consistent regardless of real-time factor.
        self._last_cam_sim_t = 0.0
        self._last_lidar_sim_t = 0.0
        self._last_imu_sim_t = 0.0

    def _sensor_adr(self, name: str):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f"sensor '{name}' not found in MJCF")
        return int(self.model.sensor_adr[sid]), int(self.model.sensor_dim[sid])

    def _on_elastic_band_toggle(self, request, response):
        if self.sim_lock is not None:
            with self.sim_lock:
                enabled = self.elastic_band.toggle()
        else:
            enabled = self.elastic_band.toggle()
        response.success = True
        response.message = "enabled" if enabled else "disabled"
        return response

    def shutdown(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def tick(self) -> None:
        """Call once per sim step from the main loop.

        /clock is published every call. TF and IMU are published at their
        target rates keyed off sim time. Camera and lidar are rate-throttled
        the same way but run less frequently (they involve rendering / ray
        casting on the main thread).
        """
        sim_t = float(self.data.time)
        stamp = _sim_time_to_msg(sim_t)

        # Always publish /clock so subscriber nodes can track sim time.
        clock_msg = Clock()
        clock_msg.clock = stamp
        self.pub_clock.publish(clock_msg)

        if sim_t - self._last_imu_sim_t >= self.imu_period:
            self._last_imu_sim_t = sim_t
            try:
                self._publish_imu(stamp)
            except Exception as e:
                self.get_logger().warn(f"imu publish failed: {e}")

        if sim_t - self._last_cam_sim_t >= self.cam_period:
            self._last_cam_sim_t = sim_t
            try:
                self._publish_camera_frame(stamp)
            except Exception as e:
                self.get_logger().warn(f"camera publish failed: {e}")

        if sim_t - self._last_lidar_sim_t >= self.lidar_period:
            self._last_lidar_sim_t = sim_t
            try:
                self._publish_lidar_scan(stamp)
            except Exception as e:
                self.get_logger().warn(f"lidar publish failed: {e}")

    def _publish_imu(self, stamp: TimeMsg) -> None:
        sd = self.data.sensordata
        # framequat sensor returns (w, x, y, z) world-frame; gyro/accel are site-local.
        q = sd[self.imu_quat_adr : self.imu_quat_adr + self.imu_quat_dim]
        w = sd[self.imu_gyro_adr : self.imu_gyro_adr + self.imu_gyro_dim]
        a = sd[self.imu_acc_adr : self.imu_acc_adr + self.imu_acc_dim]

        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self.imu_frame
        msg.orientation.w = float(q[0])
        msg.orientation.x = float(q[1])
        msg.orientation.y = float(q[2])
        msg.orientation.z = float(q[3])
        msg.angular_velocity.x = float(w[0])
        msg.angular_velocity.y = float(w[1])
        msg.angular_velocity.z = float(w[2])
        msg.linear_acceleration.x = float(a[0])
        msg.linear_acceleration.y = float(a[1])
        msg.linear_acceleration.z = float(a[2])
        self.pub_imu.publish(msg)

    def _publish_camera_frame(self, stamp: TimeMsg) -> None:
        if self._renderer is None:
            if self.model.vis.global_.offwidth < self.cam_width:
                self.model.vis.global_.offwidth = self.cam_width
            if self.model.vis.global_.offheight < self.cam_height:
                self.model.vis.global_.offheight = self.cam_height
            self._renderer = mujoco.Renderer(self.model, height=self.cam_height, width=self.cam_width)
            # Group 0: red collision primitives (joint spheres, mesh
            # duplicates) — hide. Group 1: H1 body visual meshes — show.
            # Group 2: magpie hand visuals + h12 wrist mount — show.
            # Group 3: magpie collision meshes + finger pad boxes — hide.
            self._scene_opt = mujoco.MjvOption()
            self._scene_opt.geomgroup[:] = 0
            self._scene_opt.geomgroup[1] = 1
            self._scene_opt.geomgroup[2] = 1
            # Hide all site markers from camera output. Sites are non-physical
            # debug markers (camera/lidar mounts, robosuite's required grip_site
            # at the grasp center, etc.); a hand camera looking down the gripper
            # axis stares straight at grip_site, whose semi-transparent geom is
            # faint in RGB but writes a solid blob into the depth image.
            self._scene_opt.sitegroup[:] = 0
        r = self._renderer

        # Render every camera through the shared renderer (one render context,
        # selected per camera by name). All share resolution/rate; each publishes
        # to its own /realsense/<namespace>/* topics with its own frame_id.
        for cam in self.cameras:
            r.disable_depth_rendering()
            r.update_scene(self.data, camera=cam.name, scene_option=self._scene_opt)
            rgb_u8 = np.ascontiguousarray(r.render(), dtype=np.uint8)

            rgb_jpeg = io.BytesIO()
            PILImage.fromarray(rgb_u8, mode="RGB").save(rgb_jpeg, format="JPEG", quality=80)
            rgb_msg = CompressedImage()
            rgb_msg.header.stamp = stamp
            rgb_msg.header.frame_id = cam.frame
            rgb_msg.format = "jpeg"
            # Wrap byte payloads in array.array('B', …) — rclpy's auto-generated
            # message setters validate the data field element-by-element when fed
            # raw bytes (one isinstance check per byte → ~17 ms/frame at 320×240
            # RGB+depth). array.array('B') tells the binding "uint8 array, no
            # validation needed" and the setter copies in O(N) C.
            rgb_msg.data = _array("B", rgb_jpeg.getvalue())
            cam.pub_rgb.publish(rgb_msg)

            rgb_raw = Image()
            rgb_raw.header.stamp = stamp
            rgb_raw.header.frame_id = cam.frame
            rgb_raw.height = self.cam_height
            rgb_raw.width = self.cam_width
            rgb_raw.encoding = "rgb8"
            rgb_raw.is_bigendian = 0
            rgb_raw.step = self.cam_width * 3
            rgb_raw.data = _array("B", rgb_u8.tobytes())
            cam.pub_rgb_raw.publish(rgb_raw)

            r.enable_depth_rendering()
            r.update_scene(self.data, camera=cam.name, scene_option=self._scene_opt)
            depth = r.render()

            # Mask "no hit" pixels (far-plane sentinel + non-finite/<=0) to 0
            # before quantizing to mm. RealSense uses 0 as the invalid value
            # in 16UC1 aligned-depth-to-color, which compressed_depth honors.
            invalid = ~np.isfinite(depth) | (depth <= 0) | (depth >= 0.999 * self.zfar)
            depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            depth_mm[invalid] = 0

            # 16UC1 mm → PNG → compressedDepth (12-byte ConfigHeader + PNG).
            # RealSense publishes aligned_depth_to_color in 16UC1 mm, so this
            # matches their format exactly; image_transport's compressed_depth
            # plugin decodes it directly.
            depth_png = io.BytesIO()
            PILImage.fromarray(depth_mm, mode="I;16").save(depth_png, format="PNG")
            # ConfigHeader: compressionFormat (uint32, 0=PNG), depthQuantA (f32), depthQuantB (f32).
            # 16UC1 uses raw PNG with no quantization, so A=B=0.
            depth_header = struct.pack("<Iff", 0, 0.0, 0.0)
            depth_msg = CompressedImage()
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = cam.frame
            depth_msg.format = "16UC1; compressedDepth"
            depth_msg.data = _array("B", depth_header + depth_png.getvalue())
            cam.pub_depth.publish(depth_msg)

            depth_raw = Image()
            depth_raw.header.stamp = stamp
            depth_raw.header.frame_id = cam.frame
            depth_raw.height = self.cam_height
            depth_raw.width = self.cam_width
            depth_raw.encoding = "16UC1"
            depth_raw.is_bigendian = 0
            depth_raw.step = self.cam_width * 2
            depth_raw.data = _array("B", np.ascontiguousarray(depth_mm).tobytes())
            cam.pub_depth_raw.publish(depth_raw)

            info = cam.info_msg
            info.header.stamp = stamp
            cam.pub_info.publish(info)

    def _publish_lidar_scan(self, stamp: TimeMsg) -> None:
        origin = np.ascontiguousarray(self.data.xpos[self.lidar_body_id], dtype=np.float64).reshape(3, 1)
        rot = np.array(self.data.xmat[self.lidar_body_id], dtype=np.float64).reshape(3, 3)

        # Rotate local ray directions to world frame.
        # rot maps local→world (column vecs); for row-vec array: world = local @ rot.T
        world_dirs = np.ascontiguousarray(self.lidar_local_dirs @ rot.T, dtype=np.float64)

        # Single batched cast. bodyexclude=torso skips the torso shell at the
        # source: the lidar sits inside the torso visual mesh and otherwise
        # every ray would self-hit at ~2 cm, forcing a per-ray retry loop
        # (~280 ms/scan). Excluding here gives ~80 ms/scan.
        dists = self._lidar_dists
        geomids = self._lidar_geomids
        dists.fill(-1.0)
        geomids.fill(-1)
        mujoco.mj_multiRay(
            self.model, self.data,
            origin, world_dirs.reshape(-1, 1),
            self.lidar_geomgroup,
            1,                              # include static geoms
            self.lidar_exclude_body_id,     # skip torso geoms
            geomids, dists,
            self.lidar_rays,
            self.lidar_max_range,
        )
        hit_dists = dists.ravel()

        # Points in lidar-local frame: local_dir * distance.
        valid = (hit_dists >= self.lidar_min_range) & (hit_dists <= self.lidar_max_range)
        pts = (self.lidar_local_dirs[valid] * hit_dists[valid, np.newaxis]).astype(np.float32)
        lines = self.lidar_lines[valid]
        offs = self.lidar_offset_time_ns[valid]

        msg = CustomMsg()
        msg.header.stamp = stamp
        msg.header.frame_id = self.lidar_frame
        msg.timebase = int(self.data.time * 1e9)
        msg.point_num = int(pts.shape[0])
        msg.lidar_id = 0
        msg.rsvd = [0, 0, 0]
        msg.points = [
            CustomPoint(
                offset_time=int(offs[i]),
                x=float(pts[i, 0]), y=float(pts[i, 1]), z=float(pts[i, 2]),
                reflectivity=0, tag=0, line=int(lines[i]),
            )
            for i in range(msg.point_num)
        ]
        self.pub_lidar.publish(msg)

        # Parallel sensor_msgs/PointCloud2 on /livox/pointcloud — same scan,
        # same stamp/frame, driver-native field layout. timestamp carries the
        # per-point offset_time [ns, as float64] for parity with the real driver
        # (nothing in FAST-LIO subscribes to this topic); intensity/tag stay
        # zero (sim has no reflectivity, matching the CustomMsg's reflectivity=0).
        arr = np.zeros(pts.shape[0], dtype=PC2_DTYPE)
        arr["x"] = pts[:, 0]
        arr["y"] = pts[:, 1]
        arr["z"] = pts[:, 2]
        arr["line"] = lines
        arr["timestamp"] = offs.astype(np.float64)
        pc2 = PointCloud2()
        pc2.header.stamp = stamp
        pc2.header.frame_id = self.lidar_frame
        pc2.height = 1
        pc2.width = int(pts.shape[0])
        pc2.fields = PC2_FIELDS
        pc2.is_bigendian = False
        pc2.point_step = PC2_POINT_STEP
        pc2.row_step = PC2_POINT_STEP * pc2.width
        pc2.is_dense = True
        pc2.data = arr.tobytes()
        self.pub_pc2.publish(pc2)


def init_ros() -> bool:
    raw = os.getenv("ROS_DOMAIN_ID")
    if raw is None or raw == "":
        raise RuntimeError(
            "ROS_DOMAIN_ID is not set. Export a positive integer "
            "(domain 0 is reserved for the real robot), e.g. "
            "`export ROS_DOMAIN_ID=1`."
        )
    try:
        domain = int(raw)
    except ValueError as e:
        raise RuntimeError(
            f"ROS_DOMAIN_ID must be an integer, got {raw!r}."
        ) from e
    if domain <= 0:
        raise RuntimeError(
            f"ROS_DOMAIN_ID must be a positive integer (>0), got {domain}. "
            "Domain 0 is reserved for the real robot."
        )
    if not rclpy.ok():
        rclpy.init()
    return True


def shutdown_ros() -> None:
    if rclpy.ok():
        rclpy.shutdown()
