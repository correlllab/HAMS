"""SkillGrasp: gemini (box) -> sam (mask) -> graspgen (6-DOF) -> frame_task."""

import numpy as np

from custom_ros_messages.action import SkillGrasp

from ..base import _Run, GRASP_FRAMES
from ..perception_utils import extract_json, pose_to_matrix


# Ask Gemini for a box around the GRASP POINT (a graspable part), not the whole
# object — so SAM segments just the region GraspGenX should plan on. Same
# [y1,x1,y2,x2]-normalized-to-1000 convention the vision pipeline uses.
GEMINI_GRASP_PROMPT = (
    'Find the single best place to grasp the {obj} with a two-finger parallel '
    'gripper: a graspable part such as a handle, knob, stem, neck, rim, or narrow '
    'protrusion — or the center of mass for a small simple object. Return a JSON '
    'array with exactly one TIGHT bounding box around that grasp region only (not '
    'the whole {obj}): [{{"box_2d": [y1, x1, y2, x2], "label": "{obj} grasp point", '
    '"score": 0.9}}], integer coordinates normalized to 0-1000 with y first. '
    'Always return a 4-number "box_2d" bounding box — never a single point. '
    'If the {obj} is not visible, return [].'
)

# GraspGenX emits each grasp as the pose of its gripper-BASE frame in its own
# convention (+Z = approach into the object, +X = finger-closing axis, origin at
# the gripper base). The frame_task server carries a matching URDF frame
# (GRASP_FRAMES[arm] = left/right_graspgen_site) placed at exactly that
# gripper-base pose, so a grasp is executed by driving that frame to the RAW
# GraspGenX pose — no axis permutation or base->fingertip (TCP-depth) fix-up here.
# STANDOFF_M backs the pre-grasp off along the approach axis (graspgen +Z).
STANDOFF_M = 0.08


def _translate(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


class GraspSkill:
    def _exec_grasp(self, gh):
        """gemini (locate) -> sam (mask) -> graspgen (6-DOF grasp) -> approach +
        close. No lift (by design). Gemini gives a box to focus SAM; SAM's mask is
        back-projected to an object cloud; GraspGenX picks the grasp; frame_task
        drives the grip_site there."""
        goal = gh.request
        run = _Run(self, gh, SkillGrasp, 'grasp')
        arm = self._validated_arm(run, goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')
        obj = goal.target_object

        # --- detect: gemini box (optional) -> sam mask -> object cloud ---------
        if not run.phase('detect', 0.0):
            return run.result
        # The box (when present) targets the grasp region, so let it drive SAM;
        # fall back to the whole-object text only when Gemini returns no box.
        box = self._gemini_box(obj)            # grasp-region box (pixel xyxy), or None
        mask = self.segment(text='' if box else obj, positive_boxes=box)
        if mask is None:
            return run.abort(f'no mask for {obj!r}')
        cloud = self.mask_to_cloud(mask, target_frame='pelvis')
        if cloud is None:
            return run.abort(f'{obj!r} mask produced no usable cloud')

        # --- plan: graspgen on the object cloud --------------------------------
        if not run.phase('approach', 0.4):
            return run.result
        resp = self.plan_grasp(cloud, frame='pelvis')
        if resp is None:
            return run.abort(f'no grasp planned for {obj!r}')
        target, pre, quat = self._grasp_to_targets(resp.grasps[0].pose)
        width_mm = float(resp.gripper_width) * 1000.0

        # --- approach: pre-open + move to the pre-grasp standoff ---------------
        if not self.set_gripper(arm, width_mm):
            return run.abort('gripper pre-open failed')
        if not self.move_frame_to(arm, pre[0], pre[1], pre[2],
                                  duration_sec=4, quat=quat, outer_gh=gh,
                                  frame=GRASP_FRAMES[arm]):
            return run.abort('approach motion failed')

        # --- grasp: move to contact + close ------------------------------------
        if not run.phase('grasp', 0.75):
            return run.result
        if not self.move_frame_to(arm, target[0], target[1], target[2],
                                  duration_sec=2, quat=quat, outer_gh=gh,
                                  frame=GRASP_FRAMES[arm]):
            return run.abort('contact motion failed')
        if not self.close_gripper(arm):
            return run.abort('gripper close failed')

        return run.succeed(
            f'grasped {obj!r} (graspgen score {resp.scores[0]:.2f})')

    def _gemini_box(self, obj):
        """Query Gemini for one bounding box of `obj`; return pixel [x1,y1,x2,y2]
        or None (SAM then falls back to the text prompt alone)."""
        txt = self.query_gemini(GEMINI_GRASP_PROMPT.format(obj=obj))
        data = extract_json(txt)
        entry = None
        if isinstance(data, list) and data:
            entry = data[0]
        elif isinstance(data, dict):
            entry = data
        if not isinstance(entry, dict) or 'box_2d' not in entry:
            return None
        info = self.latest_caminfo()
        if info is None or not info.width or not info.height:
            return None
        try:
            y1, x1, y2, x2 = (float(v) for v in entry['box_2d'])
        except (ValueError, TypeError):
            return None
        w, h = info.width, info.height
        px1, px2 = sorted((x1 / 1000.0 * w, x2 / 1000.0 * w))
        py1, py2 = sorted((y1 / 1000.0 * h, y2 / 1000.0 * h))
        return [px1, py1, px2, py2]

    def _grasp_to_targets(self, grasp_pose):
        """Raw GraspGenX grasp Pose (pelvis) -> (target_xyz, pre_xyz, quat) for the
        *_graspgen_site frame. GraspGenX plans in this exact frame convention
        (+Z approach, +X close, origin at the gripper base), so the orientation is
        passed through unchanged and the grasp position is the pose origin; the
        pre-grasp is backed off STANDOFF_M along the approach axis (graspgen +Z)."""
        t = pose_to_matrix(grasp_pose)
        t_pre = t @ _translate(0.0, 0.0, -STANDOFF_M)
        quat = (grasp_pose.orientation.x, grasp_pose.orientation.y,
                grasp_pose.orientation.z, grasp_pose.orientation.w)
        return t[:3, 3], t_pre[:3, 3], quat
