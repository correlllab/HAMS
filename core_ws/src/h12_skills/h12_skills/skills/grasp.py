"""SkillGrasp: two grasp behaviors dispatched by target object.

Generic grasp (default): gemini (box) -> sam (mask) -> graspgen (6-DOF) ->
frame_task. Battery grasp (BATTERY_OBJECTS): YOLO detection -> top-down grasp,
skipping sam/graspgen entirely."""

import copy
import math
import time
from dataclasses import dataclass

import numpy as np

from geometry_msgs.msg import Pose, TransformStamped

from custom_ros_messages.action import SkillGrasp

from ..base import _Run, GRASP_FRAMES, WORLD_FRAME
from ..perception_utils import pose_to_matrix, matrix_to_pose


# Battery-workcell parts the fine-tuned YOLO-World checkpoint detects (mirrors
# yolo_server.DEFAULT_QUERIES; 'OrageCover' matches the checkpoint's label spelling).
# When target_object is one of these, _exec_grasp routes to the top-down battery
# grasp (YOLO detection, no gemini/sam/graspgen) instead of the generic pipeline.
BATTERY_OBJECTS = ('Bolt', 'BusBar', 'InteriorScrew', 'Nut', 'OrageCover',
                   'Screw', 'ScrewHole')
_BATTERY_OBJECTS_LC = frozenset(o.lower() for o in BATTERY_OBJECTS)

# Battery grasp standoff [m]: how far ABOVE the detected part to place the
# GraspGenX frame. For now the battery grasp does ONLY this approach (no descend,
# no close) — it parks the frame straight-down this far above the object and stops.
BATTERY_APPROACH_DIST = 0.4
# After parking above the object, how long [s] to wait for a FRESH hand-camera YOLO
# detection at the new pose (the hand camera wasn't looking at the part before the
# move, so its cached bundle is stale). Bounded by the skill's remaining budget.
BATTERY_ARM_DETECT_TIMEOUT = 5.0


# GraspGenX emits each grasp as the pose of its gripper-BASE frame in its own
# convention (+Z = approach into the object, +X = finger-closing axis, origin at
# the gripper base). The frame_task server carries a matching URDF frame
# (GRASP_FRAMES[arm] = left/right_graspgenx_frame) placed at exactly that
# gripper-base pose, so a grasp is executed by driving that frame to the RAW
# GraspGenX pose — no axis permutation or base->fingertip (TCP-depth) fix-up here.


# Gripper aperture used while grasping, as a fraction of the gripper's full
# range (see base.open_gripper/close_gripper). OPEN_PERCENT is how far to open
# before approaching (1 = fully open); CLOSED_PERCENT is how far to close on the
# object (1 = fully closed).
OPEN_PERCENT = 1.0
CLOSED_PERCENT = 1.0

# How many of the ranked GraspGenX grasps to try (best-first) before giving up:
# the top grasp may be IK-unreachable, so fall through to the next one.
MAX_GRASP_ATTEMPTS = 5
# Diversity thresholds for choosing those MAX_GRASP_ATTEMPTS candidates. GraspGenX
# often returns a cluster of near-identical top-scored grasps; taking the first 5
# blindly gives the IK fallback loop five poses that share the same reachability
# fate. Instead we skip a candidate that DUPLICATES an already-selected one — within
# BOTH thresholds (close in position AND orientation). A grasp far in position, or
# rotated well away (e.g. a different roll at the same spot), is kept as a genuine
# alternative. Loosen to spread attempts wider; tighten to allow closer repeats.
GRASP_DIVERSITY_LIN_M = 0.02     # positions within 2 cm ...
GRASP_DIVERSITY_ANG_DEG = 20.0   # ... AND orientations within 20 deg = duplicate


APPROACH_DIST = 0.1  # metres to back off along the grasp's +Z approach axis for pre-grasp
# Metres to shift every GraspGenX grasp along its OWN +Z approach axis before
# executing it. POSITIVE drives the grasp DEEPER into the object (further along
# the approach); NEGATIVE backs it off. The pre-grasp standoff (APPROACH_DIST) is
# measured from this shifted grasp, so the whole approach->grasp pair moves together.
GRASP_OFFSET = 0.0
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


@dataclass
class GraspOutcome:
    """What an in-process caller (pick_place) needs from an executed generic
    grasp, handed back through the node's `_last_grasp_outcome` attribute — the
    SkillGrasp result message itself only carries success/message, and the
    action definition stays unchanged (see _exec_grasp for the side-channel
    contract)."""
    pose: Pose              # executed grasp pose (GraspGenX gripper-base), in `frame`
    frame: str              # WORLD_FRAME when the world TF was available, else 'pelvis'
    centroid: np.ndarray    # object-cloud centroid [m], same frame
    rest_height: float      # centroid height above the object's own bottom [m]
    gripper_width: float    # GraspGenX planned opening [m]
    score: float            # confidence of the executed grasp


class GraspSkill:
    def _exec_grasp(self, gh):
        """Dispatch to the battery top-down grasp for known battery-workcell parts
        (BATTERY_OBJECTS), else the generic gemini->sam->graspgen grasp. Both share
        the SkillGrasp action contract, so the router is transparent to callers.

        In-process side channel (for pick_place, which sends /skill/grasp goals to
        this same node): `self._grasp_overrides` — a dict set by the caller right
        before sending a goal — is consumed HERE, once, so a later standalone
        grasp is unaffected; `self._last_grasp_outcome` is cleared here and only
        repopulated by a SUCCESSFUL generic grasp, so a failed or battery grasp
        can never leak a stale outcome. Assumes one grasp goal in flight at a
        time (true in practice: one robot, skills invoked serially).

        Supported overrides:
          min_downward_pitch_deg — keep only grasps whose approach axis points at
            least this far below horizontal (see the filter in
            _exec_generic_grasp)."""
        overrides = getattr(self, '_grasp_overrides', None) or {}
        self._grasp_overrides = None
        self._last_grasp_outcome = None
        if gh.request.target_object.strip().lower() in _BATTERY_OBJECTS_LC:
            return self._exec_battery_grasp(gh)
        return self._exec_generic_grasp(gh, overrides)

    def _exec_generic_grasp(self, gh, overrides=None):
        """gemini (locate) -> sam (mask) -> graspgen (6-DOF grasp) -> approach +
        close. No lift (by design). Gemini gives a box to focus SAM; SAM's mask is
        back-projected to an object cloud; GraspGenX picks the grasp; frame_task
        drives the grip_site there. `overrides` come from the in-process side
        channel documented on _exec_grasp."""
        goal = gh.request
        run = _Run(self, gh, SkillGrasp, 'grasp')
        arm = self._validated_arm(goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')
        obj = goal.target_object
        overrides = overrides or {}

        # --- detect: gemini box (optional) -> sam mask -> object cloud ---------
        if not run.phase('detect', 0.0):
            return run.result
        obj_cloud, scene, err = self.detect_object_cloud(obj, run, gh)
        if err:
            return run.abort(err)
        # Held-object geometry for the pick_place side channel: the cloud centroid
        # and how high that centroid sits above the object's own bottom (its rest
        # height on a support surface) — measured now, while the object still sits
        # undisturbed on the table.
        centroid_p = obj_cloud.mean(axis=0)
        rest_height = float(centroid_p[2] - np.percentile(obj_cloud[:, 2], 5))

        # --- plan: graspgen on the object cloud --------------------------------
        if not run.phase('approach', 0.4):
            return run.result
        resp = self.plan_grasp(obj_cloud, gripper_name="magpie", frame='pelvis',
                               scene_cloud=scene, arm=arm)
        if resp is None:
            return run.abort(f'no grasp planned for {obj!r}')
        width_mm = float(resp.gripper_width) * 1000.0

        # --- candidate selection: optional downward-pitch gate -----------------
        # GraspGenX ranks best-first; keep that order, but when the caller asked
        # for it (pick_place stacking), drop grasps that approach too flat. The
        # gate is a MINIMUM downward pitch, not proximity to vertical: the H1 arm
        # cannot reach straight-down approaches (~45 deg below horizontal is its
        # practical steepest), so demanding verticality would reject everything
        # reachable. Approach axis = column 2 (+Z) of the grasp rotation in the
        # z-up pelvis frame; its pitch below horizontal is asin(-a_z).
        cand = list(range(len(resp.grasps)))
        min_pitch = float(overrides.get('min_downward_pitch_deg', 0.0) or 0.0)
        if min_pitch > 0.0:
            thr = math.sin(math.radians(min_pitch))
            kept = [i for i in cand
                    if -pose_to_matrix(resp.grasps[i].pose)[2, 2] >= thr]
            self.get_logger().info(
                f'grasp: downward-pitch gate >= {min_pitch:.0f} deg kept '
                f'{len(kept)}/{len(cand)} candidates')
            if not kept:
                return run.abort(
                    f'no grasp for {obj!r} pitched >= {min_pitch:.0f} deg down')
            cand = kept
        # Pick the MAX_GRASP_ATTEMPTS grasps to actually try, enforcing diversity:
        # walk the ranked list best-first and keep a candidate only when it isn't a
        # near-duplicate of one already kept (see GRASP_DIVERSITY_* / _select_diverse).
        # This spreads the IK fallback attempts across genuinely different grasps
        # instead of five near-identical top-scored ones.
        all_poses = [g.pose for g in resp.grasps]
        cand = _select_diverse(cand, all_poses, MAX_GRASP_ATTEMPTS,
                               GRASP_DIVERSITY_LIN_M,
                               math.radians(GRASP_DIVERSITY_ANG_DEG))
        self.get_logger().info(
            f'grasp: selected {len(cand)} diverse candidate(s) for {obj!r} '
            f'(>= {GRASP_DIVERSITY_LIN_M * 100:.0f}cm or '
            f'{GRASP_DIVERSITY_ANG_DEG:.0f}deg apart) from {len(resp.grasps)} ranked')
        cand_grasps = [resp.grasps[i] for i in cand]
        cand_scores = [float(resp.scores[i]) for i in cand]

        # --- approach: pre-open the gripper before reaching for the object -----
        if not self.open_gripper(arm, OPEN_PERCENT):
            return run.abort('gripper pre-open failed')

        # The top candidate can be IK-unreachable, so we walk the (gated) ranked
        # list. Snapshot every candidate's pre-grasp + grasp pose from the CURRENT
        # (static) pelvis frame into the world (WORLD_FRAME) frame ONCE, up front.
        # World-anchored, these targets stay correct as the pelvis drifts during
        # the arm motions: servo_frame_to_world re-resolves them into the live
        # pelvis frame each iteration. If the world TF is unavailable
        # (navigation/odom not running) we drive the raw pelvis poses with no
        # drift compensation.
        n = len(cand_grasps)
        # Shift each raw grasp along its approach axis by GRASP_OFFSET before use.
        grasps_p = [get_approach_pose(cand_grasps[i].pose, approach_dist=GRASP_OFFSET)
                    for i in range(n)]
        approaches_p = [get_approach_pose(g, approach_dist=-APPROACH_DIST) for g in grasps_p]
        grasps_w = [self._transform_pose(g, 'pelvis', WORLD_FRAME) for g in grasps_p]
        approaches_w = [self._transform_pose(a, 'pelvis', WORLD_FRAME)
                        for a in approaches_p]
        have_world = all(p is not None for p in grasps_w + approaches_w)
        if not have_world:
            self.get_logger().warn(
                f'grasp: {WORLD_FRAME} TF unavailable; pelvis-drift servoing OFF, '
                'driving raw pelvis poses (start navigation/FAST-LIO to enable)')
        # Snapshot the object centroid into the world frame from the SAME static
        # pelvis moment as the grasp poses, so the GraspOutcome's pose and
        # centroid stay mutually consistent for pick_place's offset math.
        centroid_w = (self._transform_pose(_pose_at(centroid_p), 'pelvis',
                                           WORLD_FRAME) if have_world else None)

        # Walk the candidates in GraspGenX's own best-first confidence ranking: the
        # reachability loop below tries the highest-scoring grasp first and falls
        # through to lower-ranked ones when a pose is IK-unreachable.
        order = sorted(range(n), key=lambda i: cand_scores[i], reverse=True)

        idx = -1
        for i in order:
            # Register the candidate we're about to drive to as TARGET_FRAME (one
            # frame, updated each iteration). publish_tf keeps re-broadcasting it so
            # RViz shows the live target instead of it expiring between sends. When
            # servoing it is the stable world goal (parented to WORLD_FRAME).
            parent, dbg = ((WORLD_FRAME, approaches_w[i]) if have_world
                           else (cand_grasps[i].header.frame_id, approaches_p[i]))
            self.publish_tf(_approach_target_tf(
                parent, dbg, self.get_clock().now().to_msg()))
            self.get_logger().info(
                f'grasp {i} for {obj!r}: score {cand_scores[i]:.2f}, '
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
                       else (cand_grasps[idx].header.frame_id, grasps_p[idx]))
        self.publish_tf(_approach_target_tf(
            parent, dbg, self.get_clock().now().to_msg()))
        # Drive to contact, then close even if the servo never reaches tolerance:
        # we've already committed to a reachable grasp, so a best-effort contact
        # pose is still worth closing on. servo_frame_to_world logs the residual
        # world error; we deliberately don't abort on non-convergence here.
        # do_plan=False: this is the short, committed pre-grasp -> contact move
        # straight along the approach axis, so drive it directly (no collision-aware
        # planning, which would fight the intended approach into the object).
        self.servo_frame_to_world(
            GRASP_FRAMES[arm], grasps_w[idx] if have_world else None,
            grasps_p[idx], outer_gh=gh,
            duration_sec=SERVO_DURATION_SEC, max_iter=SERVO_MAX_ITER,
            lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL, do_plan=False)

        if not self.close_gripper(arm, CLOSED_PERCENT):
            return run.abort('gripper close failed')

        # Hand the executed grasp + held-object geometry to any in-process caller
        # (pick_place) via the side channel documented on _exec_grasp. Pose and
        # centroid share one frame: the world snapshot when it was available,
        # else the detect-time pelvis frame.
        if have_world and centroid_w is not None:
            out_pose, out_frame, out_c = grasps_w[idx], WORLD_FRAME, centroid_w
        else:
            out_pose, out_frame, out_c = grasps_p[idx], 'pelvis', _pose_at(centroid_p)
        self._last_grasp_outcome = GraspOutcome(
            pose=out_pose, frame=out_frame,
            centroid=np.array([out_c.position.x, out_c.position.y,
                               out_c.position.z]),
            rest_height=rest_height, gripper_width=float(resp.gripper_width),
            score=cand_scores[idx])

        return run.succeed(
            f'grasped {obj!r} (graspgen score {cand_scores[idx]:.2f})')

    # ============================== battery grasp ==========================
    def _exec_battery_grasp(self, gh):
        """Battery-workcell parts: APPROACH ONLY (no descend/close yet). Uses the
        head-camera YOLO DetectionBundle (skips gemini/sam/graspgen): pick the best
        detection of the target class, back-project its bbox center to a 3-D point,
        and park the GraspGenX frame straight-down BATTERY_APPROACH_DIST above it."""
        goal = gh.request
        run = _Run(self, gh, SkillGrasp, 'grasp')
        arm = self._validated_arm(goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')
        obj = goal.target_object

        # --- detect: among the HEAD-camera YOLO detections of `obj`, pick the one
        # CLOSEST TO THE ORIGIN (nearest to the robot in the pelvis frame) --------
        if not run.phase('detect', 0.0):
            return run.result
        # When several of the same part are in view, grasp the physically nearest
        # instance rather than the most confident one. _closest_detection back-
        # projects every matching detection to a pelvis-frame point and returns the
        # nearest one WITH that point (so no second back-projection is needed); a
        # detection whose bbox has no usable depth can't be ranged and is skipped.
        det, point = self._closest_detection(obj, self.latest_detections(),
                                             target_frame='pelvis')
        if det is None:
            return run.abort(f'no head YOLO detection for {obj!r} with usable depth')

        # --- approach: park the frame BATTERY_APPROACH_DIST straight above -----
        if not run.phase('approach', 0.4):
            return run.result
        # Top-down pose at the part (approach axis +Z points straight down), then
        # backed off along that axis by BATTERY_APPROACH_DIST -> the frame sits that
        # far ABOVE the object. That is the whole battery behavior for now: no
        # descend to contact, no gripper close.
        approach_p = get_approach_pose(_top_down_grasp_pose(point),
                                       approach_dist=-BATTERY_APPROACH_DIST)
        self.get_logger().info(
            f'battery grasp {obj!r}: parking {BATTERY_APPROACH_DIST * 100:.0f}cm '
            f'above pelvis ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}), '
            f'dist {float(np.linalg.norm(point)):.2f}m, score {det.prob:.2f}')
        self.publish_tf(_approach_target_tf(
            'pelvis', approach_p, self.get_clock().now().to_msg()))
        if not self.move_frame_to(GRASP_FRAMES[arm], approach_p, outer_gh=gh,
                                  duration_sec=SERVO_DURATION_SEC):
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return run.abort('canceled or timed out during approach')
            return run.abort(f'battery approach for {obj!r} unreachable')

        # --- refine: now that we're parked above, read the ARM hand camera ----
        if not run.phase('refine', 0.75):
            return run.result
        # The hand camera now looks down at the part; wait for a fresh detection
        # from it (the pre-move cache is stale). Not acted on yet — logged for the
        # forthcoming close-in step — so a miss warns rather than aborts.
        arm_det = self._wait_for_arm_detection(
            obj, arm, min(BATTERY_ARM_DETECT_TIMEOUT, run.remaining()), gh)
        if arm_det is None:
            self.get_logger().warn(
                f'battery grasp {obj!r}: no {arm}-hand YOLO detection from above')
        else:
            self.get_logger().info(
                f'battery grasp {obj!r}: {arm}-hand detection score '
                f'{arm_det.prob:.2f} bbox '
                f'({arm_det.bbox_min.x:.0f},{arm_det.bbox_min.y:.0f})-'
                f'({arm_det.bbox_max.x:.0f},{arm_det.bbox_max.y:.0f})')
        return run.succeed(
            f'battery approach for {obj!r}: frame parked '
            f'{BATTERY_APPROACH_DIST * 100:.0f}cm above (head score {det.prob:.2f}, '
            f'{arm}-hand={"yes" if arm_det else "no"})')

    def _yolo_detection(self, obj, bundle):
        """Highest-confidence detection of class `obj` in `bundle` (a
        DetectionBundle), or None. Class match is case-insensitive. Used for the
        arm-hand path, which has no depth back-projection to range detections;
        the head-camera grasp target is instead chosen by _closest_detection."""
        if bundle is None or not bundle.detections:
            return None
        key = obj.strip().lower()
        matches = [d for d in bundle.detections if d.cls.strip().lower() == key]
        if not matches:
            return None
        return max(matches, key=lambda d: d.prob)

    def _closest_detection(self, obj, bundle, target_frame='pelvis'):
        """Among all detections of class `obj` in `bundle` (a DetectionBundle),
        back-project each to a 3-D point in `target_frame` and return the
        (detection, point) pair whose point is CLOSEST TO THE ORIGIN of that frame
        — i.e. the instance nearest the robot when `target_frame` is 'pelvis'. This
        picks the physically nearest part when several of the same class are in
        view, instead of the most confident one. A detection whose bbox yields no
        usable depth can't be ranged, so it's skipped; returns (None, None) when no
        matching detection has usable depth. Class match is case-insensitive."""
        if bundle is None or not bundle.detections:
            return None, None
        key = obj.strip().lower()
        best_det, best_point, best_dist = None, None, None
        for d in bundle.detections:
            if d.cls.strip().lower() != key:
                continue
            pt = self._detection_point(d, target_frame=target_frame)
            if pt is None:                        # no depth under this bbox -> unrangeable
                continue
            dist = float(np.linalg.norm(pt))      # Euclidean distance to the frame origin
            if best_dist is None or dist < best_dist:
                best_det, best_point, best_dist = d, pt, dist
        return best_det, best_point

    def _wait_for_arm_detection(self, obj, arm, timeout_sec, gh):
        """Wait up to `timeout_sec` for a FRESH `arm` hand-camera DetectionBundle
        (one published after this call starts, so it reflects the just-reached pose)
        that contains `obj`, and return its best detection, or None on timeout /
        cancel. Polls the cache — the subscription keeps updating it on other
        executor threads."""
        stale = self.latest_arm_detections(arm)   # bundle present before we waited
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while time.monotonic() < deadline:
            if gh.is_cancel_requested:
                return None
            bundle = self.latest_arm_detections(arm)
            if bundle is not None and bundle is not stale:
                det = self._yolo_detection(obj, bundle)
                if det is not None:
                    return det
            time.sleep(0.1)
        return None

    def _detection_point(self, det, target_frame='pelvis'):
        """Back-project a Detection's bounding box to a single (x, y, z) point in
        `target_frame`, or None. Deprojects the central quarter of the box (avoids
        background depth at the edges) and takes the median for robustness. The
        bbox pixels come from the head color frame, whose aligned depth shares the
        same grid, so they index the depth cache directly."""
        info = self.latest_caminfo()
        if info is None or not info.width or not info.height:
            return None
        w, h = int(info.width), int(info.height)
        cx = (det.bbox_min.x + det.bbox_max.x) / 2.0
        cy = (det.bbox_min.y + det.bbox_max.y) / 2.0
        half_w = max(1.0, abs(det.bbox_max.x - det.bbox_min.x) * 0.25)
        half_h = max(1.0, abs(det.bbox_max.y - det.bbox_min.y) * 0.25)
        u0, u1 = max(0, int(round(cx - half_w))), min(w, int(round(cx + half_w)))
        v0, v1 = max(0, int(round(cy - half_h))), min(h, int(round(cy + half_h)))
        if u1 <= u0 or v1 <= v0:
            return None
        mask = np.zeros((h, w), dtype=bool)
        mask[v0:v1, u0:u1] = True
        pts = self._depth_to_cloud(mask, target_frame)
        if pts is None or len(pts) == 0:
            return None
        return np.median(pts, axis=0)


def _top_down_grasp_pose(point):
    """Build a straight-DOWN grasp Pose at `point` (x, y, z in a z-up frame such as
    pelvis), oriented to obey the SAME axis-alignment rules graspgen_server applies
    to the grasps it returns. The battery path bypasses graspgen, so those rules are
    never actually run on this pose — we bake the same convention in by hand so a
    battery grasp is indistinguishable from a graspgen-filtered top-down grasp.

    GraspGenX/frame convention: +Z = approach axis (into the object), +X =
    finger-closing axis, origin at the gripper base. The two server rules
    (graspgen_server._plan_cb) and how this pose satisfies them, in the parent frame
    (pelvis: +X robot-forward, +Y robot-left, +Z up):
      * approach-direction filter — a straight-down +Z ([0,0,-1]) has zero forward
        (R[0,2]=0) and zero lateral (R[1,2]=0) component, so it clears the forward
        gate (>=0) and BOTH arms' lateral gates; the vertical axis is never gated.
      * finger-closing-axis canonicalization — the server flips grasps so the local
        +X finger axis points toward robot-left (+Y), i.e. R[1,0] >= 0. We put +X
        straight ALONG +Y (R[1,0]=1) so the battery grasp lands on that canonical
        side rather than on the +X fore/aft boundary (R[1,0]=0) the old pose used.

    Columns are the frame's local axes expressed in the parent: col0 (+X finger
    close) = parent +Y, col1 (+Y) = parent +X, col2 (+Z approach) = parent -Z."""
    R = np.array([[0.0,  1.0,  0.0],    # col0 +X (finger close) -> parent +Y (left)
                  [1.0,  0.0,  0.0],    # col1 +Y               -> parent +X (fwd)
                  [0.0,  0.0, -1.0]])   # col2 +Z (approach)    -> straight down
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [float(point[0]), float(point[1]), float(point[2])]
    return matrix_to_pose(T)


def _pose_at(point):
    """Identity-orientation Pose at `point` (x, y, z array-like) — for running a
    bare 3-D point through the Pose-based TF helpers."""
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (
        float(point[0]), float(point[1]), float(point[2]))
    pose.orientation.w = 1.0
    return pose


def _approach_target_tf(parent_frame, pose, stamp, child=TARGET_FRAME):
    """Build a debug TransformStamped for `pose` (a Pose in `parent_frame`,
    stamped `stamp`) so the base node's broadcaster can publish it for RViz —
    by default the currently-targeted pre-grasp approach (TARGET_FRAME); pass
    `child` for other targets (e.g. pick_place's commanded place pose)."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child
    t.transform.translation.x = pose.position.x
    t.transform.translation.y = pose.position.y
    t.transform.translation.z = pose.position.z
    t.transform.rotation = pose.orientation
    return t


def _grasp_poses_close(pa, pb, lin_thr, ang_thr):
    """True when Poses `pa` and `pb` are near-duplicate grasps: within `lin_thr`
    metres in position AND `ang_thr` radians in orientation. Either difference
    exceeding its threshold makes them distinct (a shifted grasp OR a re-oriented
    one — e.g. a different roll at the same spot — is a genuine alternative)."""
    Ta, Tb = pose_to_matrix(pa), pose_to_matrix(pb)
    if float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])) >= lin_thr:
        return False
    # Geodesic angle of the relative rotation Ra^T Rb: arccos((trace - 1) / 2).
    cos = (float(np.trace(Ta[:3, :3].T @ Tb[:3, :3])) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0))) < ang_thr


def _select_diverse(cand, poses, max_n, lin_thr, ang_thr):
    """Greedily pick up to `max_n` indices from `cand` (indices into `poses`,
    already ranked best-first) that are mutually distinct: a candidate is skipped
    when it is a near-duplicate (_grasp_poses_close) of one already picked. The
    best-scored grasp is always taken first, so at least one index is returned
    (fewer than max_n when the ranked list holds no more distinct grasps)."""
    picked = []
    for i in cand:
        if any(_grasp_poses_close(poses[i], poses[j], lin_thr, ang_thr)
               for j in picked):
            continue
        picked.append(i)
        if len(picked) >= max_n:
            break
    return picked


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
