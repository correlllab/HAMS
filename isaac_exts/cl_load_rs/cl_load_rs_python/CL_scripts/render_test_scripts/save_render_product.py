import omni.replicator.core as rep
import omni.graph.core as og
from isaacsim.sensors.camera import Camera
async def my_simulation():
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
    #cam_render_product = rep.create.render_product("/World/envs/env_0/Robot/h1_2_26dof_with_inspire_rev_1_0_with_CL_realsense/R_hand_base_link/CL_R_realsense/rsd455/RSD455/Camera_OmniVision_OV9782_Color", (1280, 800))
    #camera.render_product_path = cam_render_product.path
    import os
    OUTPUT_DIR = "/home/code/test_imgs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rep.orchestrator.step_async(rt_subframes=4)
    cam_render_product = rep.create.render_product("/World/envs/env_0/Robot/h1_2_26dof_with_inspire_rev_1_0_with_CL_realsense/R_hand_base_link/CL_R_realsense/rsd455/RSD455/Camera_OmniVision_OV9782_Color", (1280, 800))
    camera.render_product_path = cam_render_product.path
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=OUTPUT_DIR, rgb=True)
    writer.attach([cam_render_product])
    await rep.orchestrator.run_until_complete_async()
    for _ in range(10):
        await rep.orchestrator.step_async()
        absolute_path = os.path.abspath(OUTPUT_DIR)
        print(absolute_path)
        print(f"Image saved to {OUTPUT_DIR}/replicator_0/RenderProduct_0/rgb_00{_}.png")
    writer.detach()
    

    
import asyncio
# Run it
asyncio.ensure_future(my_simulation())


