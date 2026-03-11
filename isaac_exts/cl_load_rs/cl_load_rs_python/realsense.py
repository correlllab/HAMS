#realsense camera manager class
#Written by Mateo Feit, Jan 29 2026
import omni.graph.core as og
import numpy as np
import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera
import omni.syntheticdata._syntheticdata as sd
import logging
from dataclasses import dataclass
from typing import Tuple 
import omni.usd
logger = logging.getLogger(__name__)

@dataclass
class CameraSpecs():
    name: str = "Camera"
    cam_path: str = None
    #depth_path: str = base_path + "rsd455/RSD455/Camera_Pseudo_Depth"
    #rgb_path: str = base_path + "rsd455/RSD455/Camera_OmniVision_*_Color"
    frequency: int = 30
    dt: float = 1.0 / frequency
    res_width: int = 1280
    res_height: int = 720
    _stage = omni.usd.get_context().get_stage()

    assert frequency > 0, f"frequency: {frequency}; must be positive val"
    assert 0 < res_width <= 1920, f"res_width: {res_width}; res_width must be a positive integer less than or equal to 1920"
    assert 0 < res_height <= 1080, f"res_height: {res_height}; res_height must be a positive integer less than or equal to 1080"
    if dt != (1.0 / frequency):
        logger.warning(f"dt: {dt}; dt is not 1.0 / {frequency=}")
    #assert _stage.GetPrimAtPath(cam_path), f"prim: {path} doesn't exist on stage"
    
    def __repr__(self):
        attrs = ', '.join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({attrs})"

class RealsenseCM:
    def __init__(self, specs: Tuple[CameraSpecs, CameraSpecs, CameraSpecs, CameraSpecs]):
        for spec in specs:
            self.init_camera(spec)
    def __repr__(self):
        return f"{specs}"

    def init_camera(self, specs):
        camera = Camera(specs.cam_path, name=specs.name)
        camera.frequency = specs.frequency
        camera.dt = specs.dt
        camera.resolution = (specs.res_width, specs.res_height)
        cam_render_product = rep.create.render_product(specs.cam_path, (specs.res_width, specs.res_height))
        camera.render_product_path = cam_render_product.path
        prim = specs._stage.GetPrimAtPath(specs.cam_path)
        camera.position, camera.orientation, camera.translation = RealsenseCM.get_pos_orient(prim)
        if "Depth" in specs.cam_path.split("/")[-1]:
            RealsenseCM.publish_pointcloud_from_depth(camera)
        elif "Color" in specs.cam_path.split("/")[-1]:
            RealsenseCM.publish_rgb_stream(camera)


    @staticmethod
    def get_pos_orient(prim):
        global_matrix = omni.usd.get_world_transform_matrix(prim)
        global_translate_pos = global_matrix.ExtractTranslation()
        global_translate_orient = global_matrix.ExtractRotation()
        local_translate_pos = omni.usd.get_local_transform_SRT(prim)                
        return (global_translate_pos, global_translate_orient, local_translate_pos)
    
    @staticmethod
    def publish_rgb_stream(camera: Camera, freq = 10):
        render_product = camera.render_product_path
        step_size = int(60/freq)
        topic_name = camera.name
        queue_size = 10
        node_namespace = "/h12_camera"
        frame_id = camera.prim_path.split("/")[-1] 
        rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
            sd.SensorType.DistanceToImagePlane.name
        )
        rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
        writer = rep.writers.get(rv + "ROS2PublishImage")
        writer.initialize(
            frameId=str(frame_id),
            nodeNamespace=str(node_namespace),
            queueSize=int(queue_size),
            topicName=str(topic_name),
        )

        writer.attach([render_product])
        gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv + "IsaacSimulationGate", render_product
        )
        og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

        return
    @staticmethod
    def publish_pointcloud_from_depth(camera: Camera, freq = 10):
        # The following code will link the camera's render product and publish the data to the specified topic name.
        render_product = camera.render_product_path
        step_size = int(60/freq)
        topic_name = camera.name+"_pointcloud" # Set topic name to the camera's name
        queue_size = 10
        node_namespace = "/h12_camera"
        frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.
        # Note, this pointcloud publisher will convert the Depth image to a pointcloud using the Camera intrinsics.
        # This pointcloud generation method does not support semantic labeled objects.
        rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
            sd.SensorType.DistanceToImagePlane.name
        )
        writer = rep.writers.get(rv + "ROS2PublishPointCloud")
        writer.initialize(
            frameId=str(frame_id),
            nodeNamespace=str(node_namespace),
            queueSize=int(queue_size),
            topicName=str(topic_name),
        )
        writer.attach([render_product])
        gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv + "IsaacSimulationGate", render_product
        )
        og.Controller.attribute(gate_path + ".inputs:step").set(step_size)
        
        return

