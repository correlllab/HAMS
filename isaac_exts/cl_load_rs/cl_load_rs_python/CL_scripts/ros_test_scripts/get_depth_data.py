import omni.replicator.core as rep
import omni.graph.core as og
from isaacsim.sensors.camera import Camera
async def my_simulation():
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
    import inspect
    print(inspect.getmembers(camera))

    cam_render_product = rep.create.render_product(camera_path, (1280, 720))
    camera.render_product_path = cam_render_product.path

    #writer = rep.WriterRegistry.get("BasicWriter")
    #writer.initialize(output_dir=OUTPUT_DIR, rgb=True)
    #writer.attach([cam_render_product])
    pc_annot = rep.AnnotatorRegistry.get_annotator("pointcloud", init_params={"includeUnlabelled": True})
    pc_annot.attach(cam_render_product)
    #points = pc_annot.get_data()['data'].flatten()
    
    print(((pc_annot.get_data())['data']).flatten().dtype)
    point_step = 12
    row_step = 12
    data = pc_annot.get_data()['data'].flatten()
    bigendian = True
    print(data.dtype.byteorder)
    print(data)
    print(cam_render_product.path)
    print(cam_render_product.camera)
    
    writer = rep.writers.get("ROS2PublishPointCloud")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        qosProfile=qos_profile,
        data=data,
        context=1,
    )
    print(cam_render_product.path)
    
  
#
    #await rep.orchestrator.step_async()
    #with open("/home/code/test.txt", 'w') as f:
    #    f.write("HI")

import asyncio
    # Run it
asyncio.ensure_future(my_simulation())


