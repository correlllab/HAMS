"""SkillGrasp: gemini (box) -> sam (mask) -> graspgen (6-DOF) -> frame_task."""

import copy

from geometry_msgs.msg import TransformStamped

from custom_ros_messages.action import SkillGrasp

from ..base import _Run, GRASP_FRAMES, GEMINI_TIMEOUT_SEC
from ..perception_utils import extract_json, pose_to_matrix


# Ask Gemini for a box that fully ENCOMPASSES the whole object — this box is a
# prompt for a segmentation model (SAM), which needs a box around the entire
# target so its mask captures the full object cloud GraspGenX plans on. Same
# [y1,x1,y2,x2]-normalized-to-1000 convention the vision pipeline uses.
GEMINI_GRASP_PROMPT = (
    'Locate the {obj} so a segmentation model can be prompted with a bounding box. '
    'Return a JSON array with exactly one box that FULLY ENCOMPASSES the entire '
    '{obj} — every visible part of it (including any handle, spout, lid, or other '
    'protrusion) must fall inside the box, with the edges snug to the object\'s '
    'outermost extent. Do not box just one part and do not crop the object: '
    '[{{"box_2d": [y1, x1, y2, x2], "label": "{obj}", "score": 0.9}}], integer '
    'coordinates normalized to 0-1000 with y first. Always return a 4-number '
    '"box_2d" bounding box — never a single point. '
    'If the {obj} is not visible, return [].'
)

# GraspGenX emits each grasp as the pose of its gripper-BASE frame in its own
# convention (+Z = approach into the object, +X = finger-closing axis, origin at
# the gripper base). The frame_task server carries a matching URDF frame
# (GRASP_FRAMES[arm] = left/right_graspgenx_frame) placed at exactly that
# gripper-base pose, so a grasp is executed by driving that frame to the RAW
# GraspGenX pose — no axis permutation or base->fingertip (TCP-depth) fix-up here.


# How many of the ranked GraspGenX grasps to try (best-first) before giving up:
# the top grasp may be IK-unreachable, so fall through to the next one.
MAX_GRASP_ATTEMPTS = 5

# Single TF frame the planned pre-grasp approach is broadcast to, updated as the
# loop walks the ranked candidates, so RViz shows the target currently being tried.
TARGET_FRAME = 'graspgenx_target_frame'



class GraspSkill:
    def _exec_grasp(self, gh):
        """gemini (locate) -> sam (mask) -> graspgen (6-DOF grasp) -> approach +
        close. No lift (by design). Gemini gives a box to focus SAM; SAM's mask is
        back-projected to an object cloud; GraspGenX picks the grasp; frame_task
        drives the grip_site there."""
        goal = gh.request
        run = _Run(self, gh, SkillGrasp, 'grasp')
        arm = self._validated_arm(goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')
        obj = goal.target_object

        # --- detect: gemini box (optional) -> sam mask -> object cloud ---------
        if not run.phase('detect', 0.0):
            return run.result
        # Always prompt SAM with the object name; when Gemini returns a box that
        # encompasses the whole object, pass it too as a positive exemplar so text
        # + box together pin down the right instance. The box is optional — the
        # text alone (concept segmentation) is the fallback when Gemini has no box.
        box = self._gemini_box(obj, run, gh)   # whole-object box (pixel xyxy), or None
        # Gemini (gemini-robotics-er) latency is highly variable (seconds to
        # minutes). _gemini_box caps the call at the skill's remaining budget and
        # returns None on cancel/timeout — but None is also a legitimate "no box".
        # Disambiguate here: if the budget is spent or a cancel landed, abort now
        # instead of wasting the SAM call before the next phase check catches it.
        if gh.is_cancel_requested or run.remaining() <= 0.0:
            return run.abort('detection canceled or timed out')
        mask = self.segment(text=obj, positive_boxes=box, outer_gh=gh)
        if mask is None:
            return run.abort(f'no mask for {obj!r}')
        obj_cloud = self.mask_to_cloud(mask, target_frame='pelvis')
        if obj_cloud is None:
            return run.abort(f'{obj!r} mask produced no usable cloud')
        # Whole-frame cloud as obstacle context so graspgen can collision-filter
        # grasps against the surroundings (optional — None just skips filtering).
        scene = self.scene_to_cloud(target_frame='pelvis')

        # --- plan: graspgen on the object cloud --------------------------------
        if not run.phase('approach', 0.4):
            return run.result
        resp = self.plan_grasp(obj_cloud, gripper_name = "magpie", frame='pelvis', scene_cloud=scene)
        if resp is None:
            return run.abort(f'no grasp planned for {obj!r}')
        width_mm = float(resp.gripper_width) * 1000.0

        # --- approach: pre-open the gripper to the planned width ---------------
        if not self.set_gripper(arm, width_mm):
            return run.abort('gripper pre-open failed')

        # GraspGenX returns grasps ranked best-first. The top grasp can be
        # IK-unreachable (out of arm range / singular). Now that frame_task reports
        # real convergence, walk the ranked list and commit to the first pre-grasp
        # approach that actually lands, instead of blindly closing on the top one.
        n = min(len(resp.grasps), MAX_GRASP_ATTEMPTS)
        idx = -1
        grasp_pose = None
        for i in range(n):
            grasp_pose = resp.grasps[i].pose
            approach_pose = get_approach_pose(grasp_pose, approach_dist=-0.05)
            # Register the candidate we're about to drive to as TARGET_FRAME (one
            # frame, updated each iteration). publish_tf keeps re-broadcasting it
            # so RViz shows the live target instead of it expiring between sends.
            self.publish_tf(_approach_target_tf(
                resp.grasps[i].header.frame_id, approach_pose,
                self.get_clock().now().to_msg()))
            self.get_logger().info(
                f'grasp {i} for {obj!r}: score {resp.scores[i]:.2f}, width {width_mm:.1f}mm, \n approach {approach_pose}')
            if self.move_frame_to(GRASP_FRAMES[arm], approach_pose, duration_sec=10, outer_gh=gh):
                idx = i
                break
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return run.abort('canceled or timed out during approach')
            self.get_logger().warn(
                f'grasp {i} pre-grasp unreachable; trying next-ranked grasp')
        if idx < 0:
            return run.abort(
                f'no reachable grasp for {obj!r} (tried {n} of {len(resp.grasps)})')

        # --- grasp: move to contact + close ------------------------------------
        if not run.phase('grasp', 0.75):
            return run.result
        if not self.move_frame_to(GRASP_FRAMES[arm], grasp_pose, duration_sec=10, outer_gh=gh):
            return run.abort('contact motion failed')
        if not self.close_gripper(arm):
            return run.abort('gripper close failed')

        return run.succeed(
            f'grasped {obj!r} (graspgen score {resp.scores[idx]:.2f})')

    def _gemini_box(self, obj, run, gh):
        """Query Gemini for one bounding box of `obj`; return pixel [x1,y1,x2,y2]
        or None (SAM then uses the text prompt alone). The call is capped
        at the skill's remaining time budget (never longer than GEMINI_TIMEOUT_SEC)
        and `gh` is threaded through so a goal cancel aborts it promptly."""
        timeout = min(GEMINI_TIMEOUT_SEC, run.remaining())
        txt = self.query_gemini(GEMINI_GRASP_PROMPT.format(obj=obj),
                                timeout_sec=timeout, outer_gh=gh)
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


def _approach_target_tf(parent_frame, pose, stamp):
    """Build the TARGET_FRAME TransformStamped for `pose` (a Pose in
    `parent_frame`, stamped `stamp`) so the base node's broadcaster can publish
    the currently-targeted pre-grasp approach for RViz."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = TARGET_FRAME
    t.transform.translation.x = pose.position.x
    t.transform.translation.y = pose.position.y
    t.transform.translation.z = pose.position.z
    t.transform.rotation = pose.orientation
    return t


def get_approach_pose(pose, approach_dist):
    """Translate `pose` along its OWN local +Z axis by `approach_dist` metres,
    returning a NEW Pose with the orientation unchanged (input left untouched).

    GraspGenX poses use +Z as the approach axis (into the object), so a POSITIVE
    `approach_dist` slides the pose forward along that approach (deeper toward the
    object) and a NEGATIVE value backs it off — e.g. pass a negative standoff to
    get a pre-grasp pose behind the grasp. The pose's local +Z, expressed in the
    parent frame, is the third column of its rotation matrix."""
    z_axis = pose_to_matrix(pose)[:3, 2]          # pose's local +Z in the parent frame
    out = copy.deepcopy(pose)
    out.position.x += approach_dist * float(z_axis[0])
    out.position.y += approach_dist * float(z_axis[1])
    out.position.z += approach_dist * float(z_axis[2])
    return out
