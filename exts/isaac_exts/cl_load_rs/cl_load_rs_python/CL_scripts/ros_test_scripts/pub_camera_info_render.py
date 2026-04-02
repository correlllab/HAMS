import omni.replicator.core as rep
import omni.graph.core as og
from isaacsim.sensors.camera import Camera
def publish_camera_info(camera: Camera, freq):
    from isaacsim.ros2.bridge import read_camera_info
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera.render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_camera_info"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    writer = rep.writers.get("ROS2PublishCameraInfo")
    print(1)
    camera_info, _ = read_camera_info(render_product_path=render_product)
    print(2)
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
        width=camera_info.width,
        height=camera_info.height,
        projectionType=camera_info.distortion_model,
        k=camera_info.k.reshape([1, 9]),
        r=camera_info.r.reshape([1, 9]),
        p=camera_info.p.reshape([1, 12]),
        physicalDistortionModel=camera_info.distortion_model,
        physicalDistortionCoefficients=camera_info.d,
    )
    print(3)
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatch" + "IsaacSimulationGate", render_product
    )

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)
    return

print("HUH")
import omni.usd
stage=omni.usd.get_context().get_stage()
camera = Camera(prim_path="/World/envs/env_0/Robot/h1_2_26dof_with_inspire_rev_1_0_with_CL_realsense/R_hand_base_link/CL_R_realsense/rsd455/RSD455/Camera_OmniVision_OV9782_Color",)
camera.frequency = 20
camera.dt = 20
camera.resolution = (1280, 800)
prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/h1_2_26dof_with_inspire_rev_1_0_with_CL_realsense/R_hand_base_link/CL_R_realsense/rsd455/RSD455/Camera_OmniVision_OV9782_Color")
matrix = omni.usd.get_world_transform_matrix(prim)
global_translate_pos = matrix.ExtractTranslation()
global_translate_orient = matrix.ExtractRotation()
local_translate_pos = omni.usd.get_local_transform_SRT(prim)
camera.position = global_translate_pos
camera.orientation = global_translate_orient
camera.translation = local_translate_pos
cam_render_product = rep.create.render_product("/World/envs/env_0/Robot/h1_2_26dof_with_inspire_rev_1_0_with_CL_realsense/R_hand_base_link/CL_R_realsense/rsd455/RSD455/Camera_OmniVision_OV9782_Color", (1280, 800))
camera.render_product_path = cam_render_product.path
#print(f'{cam_render_product=}')
#import inspect
#print(inspect.getmembers(camera))
print("AHA")
#rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
#rgb_annotator.attach([cam_render_product])
#test = cam_render_product.path
#print(test)
#camera.annotator_device = rgb_annotator

print("AAAA")
print(f"{rep.annotators.get_registered_annotators()}=")
print("Active Annotators:", rep.AnnotatorRegistry.get_registered_annotators())
# The colorize parameter is typically passed to the annotator attachment
#frame_dict = camera.get_current_frame()
print(f'{frame_dict}')
print("TEST1")
print(f"{camera.get_rgb()=}")
#camera.add_rgb_to_frame({"colorize":True})
writer
try:
    camera.add_semantic_segmentation_to_frame({"colorize": True})
    camera.add_instance_segmentation_to_frame({"colorize": True})
except Exception as e:
    print(e)
    import traceback
    traceback.print_exc()
print("TET2")
#camera.add_semantic_segmentation_to_frame({"colorize": True})
camera.add_instance_id_segmentation_to_frame({"colorize": True})
print("AAAAAAA")
print(f"Prim Path: {camera.prim_path}")
print(f"Name: {camera.name}")
print(f"Frequency: {camera.frequency}")
print(f"Time Step (dt): {camera.dt}")
print(f"Resolution: {camera.resolution}")
print(f"Position: {camera.position}")
print(f"Orientation: {camera.orientation}")
print(f"Translation: {camera.translation}")
print(f"Render Product Path: {camera.render_product_path}")
print(f"Annotator Device: {camera.annotator_device}")
#print(inspect.getmembers(camera))

frame_dict = camera.get_current_frame()
print(f'{frame_dict}')

rep_writer = rep.BasicWriter(
        output_dir="/home/code/test_images/",
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )
    


#publish_camera_info(camera, 2)
