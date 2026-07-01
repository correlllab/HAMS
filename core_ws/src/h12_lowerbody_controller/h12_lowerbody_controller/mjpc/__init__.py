"""MJPC (MuJoCo MPC) deploy node for the Unitree H1-2 — Python skeleton.

This subpackage is the Python analog of the fork's C++ deploy node
(``mjpc/deploy/h12_control_node.cc``). Where the C++ node embeds ``mjpc::Agent``
in-process and runs the planner in a background thread, this version drives the
planner through the installed ``mujoco_mpc`` gRPC ``Agent`` (the "Python gRPC
bridge" the C++ header references). It talks to the robot / digital twin over raw
Cyclone DDS via ``unitree_sdk2py`` (whole-body, 27 joints), exactly like the C++
node's I/O.

It is a *skeleton*: the load-bearing control path is implemented (state
reconstruction, gravity feed-forward, bring-up ramp, torque-aware clamp, planner
wiring, whole-body LowCmd publish), while the advanced extras are TODO stubs.

Modules
-------
constants          27-joint gains / limits / IMU offset (≙ the C++ KP[]/KV[]/... tables)
state_estimation   pelvis_from_site: free-joint qpos/qvel from IMU-site pose + IMU (≙ fill_state)
agent_client       MjpcAgentClient: gRPC Agent wrapper + background planner thread
control            GravityFeedforward, BringUpRamp, target_clamp, build_lowcmd
"""
