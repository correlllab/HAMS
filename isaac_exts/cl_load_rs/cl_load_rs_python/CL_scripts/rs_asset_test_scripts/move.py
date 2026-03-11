from pxr import UsdGeom, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

# Replace with your actual child prim path
child_path = "/World/h1_2_26dof_with_inspire_rev_1_0/left_hand_camera_base_link/realsense_asset"  # or whatever it is now

child_prim = stage.GetPrimAtPath(child_path)
xformable = UsdGeom.Xformable(child_prim)

# Clear and set transforms
xformable.ClearXformOpOrder()

# Rotate 90 degrees counterclockwise - depends on which axis:
# Around Z axis (top-down view): 
#xformable.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, 0))

# OR around Y axis (side view):
# xformable.AddRotateXYZOp().Set(Gf.Vec3d(0, 90, 0))

# OR around X axis (front view):
xformable.AddRotateXYZOp().Set(Gf.Vec3d(90, 0, 0))

# Position very close to parent origin
xformable.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
