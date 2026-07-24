"""Battery-workcell env for the H1-2 sim runner (`h12_mujoco.py --task BatteryWorkcell`).

`CL_Assets/battery/scene_h1_2_battery.xml` is a SELF-CONTAINED MuJoCo scene —
it `<include>`s the raw h1_2_magpie robot and adds the ARPA battery cell with a
row of free-joint screws — so unlike the kitchen tasks there is no
robosuite/RoboCasa assembly step. We load the XML directly and duck-type the
small robosuite-env surface the runner and bridges actually consume:

    env.reset()                          build model/data (idempotent restart)
    env.sim.model._model / .data._data   raw MjModel / MjData for the loop
    env.sim.model.get_joint_qpos_addr    place_robot_freejoint()
    env.sim.data.qpos / env.sim.forward  place_robot_* helpers
    env.robots[0].robot_model.naming_prefix   "" — raw XML names, no robosuite
                                              prefixes (NameResolver supports
                                              this by design)
    env.init_robot_base_pos / _ori       the XML's authored spawn pose
    env.obj_body_id                      GT poses for /robocasa/object_poses
    env.update_state() / _check_success() / reward() / get_ep_meta()
                                         RoboCasa task-signal stubs (the
                                         battery task has no scripted success;
                                         grading lives in the benchmark)

Because the raw model has no `robot0_`/`gripper0_` prefixes, the collision-
aware spawn back-off in place_robot_collision_free degrades to a plain
free-joint placement (its robot-geom filters match nothing) — fine here: the
scene XML already places the robot at the cell, and the HAMS_SPAWN_FORWARD /
HAMS_SPAWN_LATERAL / HAMS_STANCE knobs still apply on top.
"""
import os

import mujoco
import numpy as np

DEFAULT_SCENE_XML = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'CL_Assets', 'battery', 'scene_h1_2_battery.xml'))
FREE_JOINT_NAME = 'floating_base_joint'


class _ModelShim:
    """mujoco-py-flavoured model accessor used by the place_robot_* helpers.
    Proxies every other attribute straight to the raw MjModel, so the bridges'
    direct reads (jnt_*, body_*, etc.) work transparently."""

    def __init__(self, m):
        object.__setattr__(self, '_model', m)

    def get_joint_qpos_addr(self, name):
        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f'joint {name!r} not in model')
        adr = int(self._model.jnt_qposadr[jid])
        if self._model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return (adr, adr + 7)
        return adr

    def __getattr__(self, name):        # only called when not found normally
        return getattr(object.__getattribute__(self, '_model'), name)


class _DataShim:
    """Raw MjData wrapper. Proxies all reads (body_xpos/body_xquat/qvel/xpos/
    site_xpos/...) to the real MjData; the runner still uses ._data explicitly."""

    def __init__(self, d):
        object.__setattr__(self, '_data', d)

    # mujoco-py / robosuite compat aliases the bridges expect on env.sim.data.
    # Raw MjData names them xpos/xquat (indexed by body id); the wrapper the
    # kitchen envs use exposes body_xpos/body_xquat.
    @property
    def body_xpos(self):
        return self._data.xpos

    @property
    def body_xquat(self):
        return self._data.xquat

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_data'), name)


class _SimShim:
    def __init__(self, m, d):
        self.model = _ModelShim(m)
        self.data = _DataShim(d)

    def forward(self):
        mujoco.mj_forward(self.model._model, self.data._data)


class _RobotModelShim:
    naming_prefix = ''


class _RobotShim:
    robot_model = _RobotModelShim()


class BatteryWorkcellEnv:
    """Duck-typed 'env' around the self-contained battery scene XML."""

    def __init__(self, xml_path=None):
        self._xml = xml_path or DEFAULT_SCENE_XML
        if not os.path.isfile(self._xml):
            raise FileNotFoundError(
                f'battery scene XML not found at {self._xml} — is CL_Assets '
                f'checked out (git lfs) at the battery-assets pin?')
        self.robots = [_RobotShim()]
        self.sim = None
        self.obj_body_id = {}
        self.init_robot_base_pos = None
        self.init_robot_base_ori = None   # None -> identity via _euler_to_wxyz

    def reset(self):
        m = mujoco.MjModel.from_xml_path(self._xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        self.sim = _SimShim(m, d)

        # The XML's authored spawn = the free joint's compiled default pose.
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, FREE_JOINT_NAME)
        if jid < 0:
            raise RuntimeError(
                f'{FREE_JOINT_NAME!r} missing from {self._xml} — the scene no '
                f'longer includes the h1_2_magpie robot?')
        adr = int(m.jnt_qposadr[jid])
        self.init_robot_base_pos = d.qpos[adr:adr + 3].copy()

        # GT-pose objects for MeasurementBridge (/robocasa/object_poses):
        # every free-joint body except the robot base (the screw row), plus the
        # battery pack. The bridge adds '__pelvis__' itself.
        self.obj_body_id = {}
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE or j == jid:
                continue
            bid = int(m.jnt_bodyid[j])
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or f'body_{bid}'
            self.obj_body_id[bname] = bid
        bat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'battery_link')
        if bat >= 0:
            self.obj_body_id['battery_link'] = int(bat)
        print(f'[battery_env] loaded {os.path.basename(self._xml)}: '
              f'nq={m.nq} nu={m.nu}; GT objects: {sorted(self.obj_body_id)}')
        return {}

    # ------------------------- RoboCasa task-signal stubs (no scripted task)
    def update_state(self):
        pass

    def _check_success(self):
        return False

    def reward(self):
        return 0.0

    def get_ep_meta(self):
        return {'lang': 'battery workcell: pick parts from the ARPA cell screw row'}
