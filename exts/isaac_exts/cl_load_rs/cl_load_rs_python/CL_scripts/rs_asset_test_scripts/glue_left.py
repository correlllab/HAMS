from pxr import UsdGeom, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

# Replace these with your actual prim paths
parent_path ="/World/h1_2_26dof_with_inspire_rev_1_0/L_hand_base_link"  # the one you want to attach TO
child_path="/World/realsense"   # the one you want to attach

# 1. First, move child under parent in hierarchy
child_prim = stage.GetPrimAtPath(child_path)
new_child_path = f"{parent_path}/{child_prim.GetName()}"

omni.kit.commands.execute('MovePrim',
    path_from=child_path,
    path_to=new_child_path)

# 2. Now set the child's position relative to parent
child_prim = stage.GetPrimAtPath(new_child_path)
xformable = UsdGeom.Xformable(child_prim)

# Clear existing transforms and set new ones
xformable.ClearXformOpOrder()
#xformable.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.1))  # 10cm above parent origin
#xformable.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, 0))    # no rotation

