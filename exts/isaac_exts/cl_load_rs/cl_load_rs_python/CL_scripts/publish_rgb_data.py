import omni.graph.core as og
import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera
import omni.syntheticdata._syntheticdata as sd
def publish_rgb(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera.render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_rgb"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return
    
camera_path = "/World/envs/env_0/Robot/h1_2_26dof_with_inspire_rev_1_0_with_CL_realsense/R_hand_base_link/CL_R_realsense/rsd455/RSD455/Camera_Pseudo_Depth"
stage=omni.usd.get_context().get_stage()
camera = Camera(prim_path=camera_path,)
camera.frequency = 20
camera.dt = 20
camera.resolution = (1280, 720)
prim = stage.GetPrimAtPath(camera_path)
matrix = omni.usd.get_world_transform_matrix(prim)
global_translate_pos = matrix.ExtractTranslation()
global_translate_orient = matrix.ExtractRotation()
local_translate_pos = omni.usd.get_local_transform_SRT(prim)
camera.position = global_translate_pos
camera.orientation = global_translate_orient
camera.translation = local_translate_pos
cam_render_product = rep.create.render_product(camera_path, (1280, 720))
camera.render_product_path = cam_render_product.path
publish_rgb(camera,30)

