"""
hi, this are some of the classes for the realsense simulation
"""
import collections 

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import omni.usd
import omni.graph.core as og
import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera
import omni.syntheticdata._syntheticdata as sd
from isaacsim.ros2.bridge import read_camera_info
import isaacsim.core.utils.numpy.rotations as rot_utils

def log_func(fn: callable):
    def log(*args, **kwargs):
        name = fn.__name__

        print(f"{__file__}: currently calling: {name}...")
        res = fn(*args, **kwargs)
        print(f"{__file__}: {name} result: {res}")
        print(f"{__file__}: returning from {name} call...")
        return res

    return log



class CamType(Enum):
    DEPTH = auto()
    RGB = auto()

#def get_spec_defaults(key):
#    defaults = {"frequency": 30, "dt": 0.033, "res_width": 1280, "res_height": 720}
#    return defaults[key]

@dataclass
class Spec:
    name: str
    path: str
    role: CamType
    frequency: int = 20
    dt: float = 0.033
    res_width: int =1280
    res_height: int = 720

    @log_func
    def __post_init__(self):
        # just type checking and making sure lol
        for key, value in self.__dict__.items():
            expected_type = self.__annotations__[key]
            try:
                assert isinstance(value, expected_type)
            except AssertionError:
                return f"key: {key} expected type: {expected_type}, but recieved value: {value} of type: {type(value)}"
        return

    @log_func
    def __eq__(self, object):
        return True if self.name == object.name or self.path == object.path else False


class CameraObjectFactory:

    def __init__(self, camera_spec_list: list[Spec]):

        self.spec_stash = camera_spec_list
        self.camera_stash = {}
        self.stage = omni.usd.get_context().get_stage()


        for spec in self.spec_stash:
            self.initialize_sim_camera(spec)

        return

    @log_func
    def initialize_sim_camera(self, spec: Spec):
        position, orientation, translation = self.get_prim_translations(self.stage.GetPrimAtPath(spec.path))
        print(f"{position=}")
        print(f"{orientation=}")
        print(f"{translation=}")
        self.camera_stash[spec.name] = Camera(
            spec.path,
            name=spec.name,
            frequency=spec.frequency,
            #dt=spec.dt,
            resolution=(spec.res_width, spec.res_height),
            render_product_path=self.create_cam_render_product(spec.path, spec.res_width, spec.res_height).path,
            position = position,
            orientation = orientation)
            #translation = translation)
        return

    @log_func
    def export(self):
        return (sorted(self.spec_stash, key=lambda obj: obj.name), collections.OrderedDict(sorted(self.camera_stash.items())))

    @log_func
    def create_cam_render_product(self, camera_path, res_width, res_height):
        return rep.create.render_product(camera_path, (res_width, res_height))

    @log_func
    def assert_no_duplicates(self, camera_spec_list: list[Spec]):
        temp = []
        for i in camera_spec_list:
            assert i not in temp
            temp.append(i)

    @log_func
    def get_prim_translations(self, prim):
        global_matrix = omni.usd.get_world_transform_matrix(prim)
        global_translate_pos = global_matrix.ExtractTranslation()
        tmp = global_matrix.ExtractRotationQuat()
        w = tmp.GetReal()
        x, y, z = tmp.GetImaginary()
        try:
            global_translate_orient = np.array([w, x, y, z])
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            assert 0 > 1
        try:
            local_translate_pos = omni.usd.get_local_transform_SRT(prim)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            assert 0 > 1
        return (global_translate_pos, global_translate_orient, local_translate_pos)

class BaseCameraPublisher:
    def __init__(self, camera_object: Camera, spec: Spec):

        self.camera = camera_object
        self.spec = spec

        try:
            self.rp_path = self.camera._render_product_path
        except AttibuteError:
            self.rp_path = self.camera.render_product_path
        self.step_size = int(60 / self.spec.frequency)
        self.frame_id = self.camera.name
        self.namespace = self.camera.name
        self.queue_size = 10

    @log_func
    def publish_data(self, role: CamType):
        rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
            sd.SensorType.DistanceToImagePlane.name
        )

        assert role is not None

        if role == CamType.DEPTH:
            fetch = "ROS2PublishPointCloud"
            topic = "/aligned_depth_to_color/image_raw/compressedDepth"
        elif role == CamType.RGB:
            fetch = "ROS2PublishImage"
            topic = "/color/image_raw/compressed"
        else:
            assert isinstance(role, CamType)

        writer = rep.writers.get(rv + fetch)
        writer.initialize(
            frameId=self.frame_id,
            nodeNamespace=self.namespace,
            queueSize=self.queue_size,
            topicName=topic,
        )
        writer.attach([self.rp_path])
        gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv + "IsaacSimulationGate", self.rp_path
        )
        og.Controller.attribute(gate_path + ".inputs:step").set(self.step_size)

    @log_func
    def publish_camera_info(self, role):
        assert isinstance(role, CamType)

        if role == CamType.DEPTH:
            fetch = "ROS2PublishPointCloud"
            topic = "/aligned_depth_to_color/camera_info"
        elif role == CamType.RGB:
            fetch = "ROS2PublishImage"
            topic = "/color/camera_info"
        else:
            assert isinstance(role, CamType)

        writer = rep.writers.get("ROS2PublishCameraInfo")
        camera_info, _ = read_camera_info(self.rp_path)
        writer.initialize(
            frameId=self.frame_id,
            nodeNamespace=self.namespace,
            queueSize=self.queue_size,
            topicName=topic,
            width=camera_info.width,
            height=camera_info.height,
            projectionType=camera_info.distortion_model,
            k=camera_info.k.reshape([1, 9]),
            r=camera_info.r.reshape([1, 9]),
            p=camera_info.p.reshape([1, 12]),
            physicalDistortionModel=camera_info.distortion_model,
            physicalDistortionCoefficients=camera_info.d,
        )
        writer.attach([self.rp_path])
        gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            "PostProcessDispatch" + "IsaacSimulationGate", self.rp_path
        )
        og.Controller.attribute(gate_path + ".inputs:step").set(self.step_size)

    

class ROS2CameraFactory:
    def __init__(self, cameras: list[Camera], specs: list[Spec]):
        self.cameras = cameras
        self.specs = specs
        self.ros2_cameras = {}
        print(f"{cameras=}")

        print(zip(cameras, specs))
        for spec in specs:
            cam = cameras[spec.name]
            self.init_ros2_camera(cam, spec)

    @log_func
    def init_ros2_camera(self, camera, spec):
        self.ros2_cameras[spec.name] = (pub := BaseCameraPublisher(camera, spec))
        pub.publish_data(spec.role)
        pub.publish_camera_info(spec.role)



