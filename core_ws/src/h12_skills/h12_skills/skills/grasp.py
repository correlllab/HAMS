"""SkillGrasp: gemini (box) -> sam (mask) -> graspgen (6-DOF) -> frame_task."""

import copy

from geometry_msgs.msg import TransformStamped

from custom_ros_messages.action import SkillGrasp

from ..base import _Run, GRASP_FRAMES, GEMINI_TIMEOUT_SEC, WORLD_FRAME
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


APPROACH_DIST = 0.075  # metres to back off along the grasp's +Z approach axis for pre-grasp
# Single TF frame the planned pre-grasp approach is broadcast to, updated as the
# loop walks the ranked candidates, so RViz shows the target currently being tried.
TARGET_FRAME = 'graspgenx_target_frame'

# How hard to try to physically REACH each candidate pose before falling through
# to the next ranked grasp. These push past servo_frame_to_world's defaults
# (10s primary move / SERVO_ITER refinement passes): give a near-but-not-yet
# reached pose MORE TIME on the main IK move and MORE ITERATIONS of world-frame
# drift correction to settle within tolerance. The iter-0 unreachable fast-fail
# in servo_frame_to_world still bails genuinely out-of-reach candidates quickly,
# so the extra budget is only spent on poses that are actually close to reachable.
SERVO_DURATION_SEC = 15   # primary (iter-0) approach/contact IK move budget [s]
SERVO_MAX_ITER = 6        # world-frame servo refinement passes per pose
# Convergence tolerances for the grasp servo, relaxed from base.py's defaults
# (5 mm / ~1.15 deg). Real-robot IK + pelvis drift rarely settle a 6-DOF grasp
# pose that tight within SERVO_MAX_ITER passes, so accept a looser world-frame
# fit as "reached" instead of burning the whole iteration budget and proceeding
# best-effort anyway. The iter-0 unreachable fast-fail (lin>5cm/ang>0.2rad) still
# rejects genuinely out-of-reach candidates, so this only loosens the final fit.
SERVO_LIN_TOL = 0.015     # 15 mm world-position convergence tol (base: 5 mm)
SERVO_ANG_TOL = 0.05      # ~2.9 deg world-orientation convergence tol (base: ~1.15 deg)



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
        resp = self.plan_grasp(obj_cloud, gripper_name="magpie", frame='pelvis',
                               scene_cloud=scene, arm=arm)
        if resp is None:
            return run.abort(f'no grasp planned for {obj!r}')
        width_mm = float(resp.gripper_width) * 1000.0

        # --- approach: pre-open the gripper to the planned width ---------------
        if not self.set_gripper(arm, width_mm):
            return run.abort('gripper pre-open failed')

        # GraspGenX returns grasps ranked best-first; the top grasp can be
        # IK-unreachable, so we walk the ranked list. Snapshot every candidate's
        # pre-grasp + grasp pose from the CURRENT (static) pelvis frame into the
        # world (WORLD_FRAME) frame ONCE, up front. World-anchored, these targets
        # stay correct as the pelvis drifts during the arm motions:
        # servo_frame_to_world re-resolves them into the live pelvis frame each
        # iteration. If the world TF is unavailable (navigation/odom not running)
        # we drive the raw pelvis poses with no drift compensation.
        n = min(len(resp.grasps), MAX_GRASP_ATTEMPTS)
        grasps_p = [resp.grasps[i].pose for i in range(n)]
        approaches_p = [get_approach_pose(g, approach_dist=-APPROACH_DIST) for g in grasps_p]
        grasps_w = [self._transform_pose(g, 'pelvis', WORLD_FRAME) for g in grasps_p]
        approaches_w = [self._transform_pose(a, 'pelvis', WORLD_FRAME)
                        for a in approaches_p]
        have_world = all(p is not None for p in grasps_w + approaches_w)
        if not have_world:
            self.get_logger().warn(
                f'grasp: {WORLD_FRAME} TF unavailable; pelvis-drift servoing OFF, '
                'driving raw pelvis poses (start navigation/FAST-LIO to enable)')

        idx = -1
        for i in range(n):
            # Register the candidate we're about to drive to as TARGET_FRAME (one
            # frame, updated each iteration). publish_tf keeps re-broadcasting it so
            # RViz shows the live target instead of it expiring between sends. When
            # servoing it is the stable world goal (parented to WORLD_FRAME).
            parent, dbg = ((WORLD_FRAME, approaches_w[i]) if have_world
                           else (resp.grasps[i].header.frame_id, approaches_p[i]))
            self.publish_tf(_approach_target_tf(
                parent, dbg, self.get_clock().now().to_msg()))
            self.get_logger().info(
                f'grasp {i} for {obj!r}: score {resp.scores[i]:.2f}, '
                f'width {width_mm:.1f}mm')
            if self.servo_frame_to_world(
                    GRASP_FRAMES[arm], approaches_w[i] if have_world else None,
                    approaches_p[i], outer_gh=gh,
                    duration_sec=SERVO_DURATION_SEC, max_iter=SERVO_MAX_ITER,
                    lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL):
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
        parent, dbg = ((WORLD_FRAME, grasps_w[idx]) if have_world
                       else (resp.grasps[idx].header.frame_id, grasps_p[idx]))
        self.publish_tf(_approach_target_tf(
            parent, dbg, self.get_clock().now().to_msg()))
        # Drive to contact, then close even if the servo never reaches tolerance:
        # we've already committed to a reachable grasp, so a best-effort contact
        # pose is still worth closing on. servo_frame_to_world logs the residual
        # world error; we deliberately don't abort on non-convergence here.
        self.servo_frame_to_world(
            GRASP_FRAMES[arm], grasps_w[idx] if have_world else None,
            grasps_p[idx], outer_gh=gh,
            duration_sec=SERVO_DURATION_SEC, max_iter=SERVO_MAX_ITER,
            lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL)
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
