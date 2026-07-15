"""SkillGrasp: one grasp pipeline, two bounding-box sources.

Both paths run the SAME pipeline — box -> sam (mask) -> graspgen (6-DOF) -> Y-up
re-roll + arm-side/levelable filter -> tier order -> diversity -> IK fallback ->
approach + close. The ONLY difference is where the SAM box comes from: Gemini for
a generic object, or the head-camera YOLO detector for a battery-workcell part
(BATTERY_OBJECTS)."""

import copy
import math
import os
from dataclasses import dataclass

import numpy as np


def _envf(name, default):
    """A float tunable overridable at runtime via env var `name` (default kept
    when unset/empty/unparseable). Lets a benchmark sweep vary the grasp
    reachability band per bringup without a rebuild — the real robot leaves the
    vars unset and gets the tuned defaults below. See the reachable-band sweep."""
    raw = os.environ.get(name, '').strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose, Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy

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


# GraspGenX emits each grasp as the pose of its gripper-BASE frame in its own
# convention (+Z = approach into the object, +X = finger-closing axis, origin at
# the gripper base). The frame_task server carries a matching URDF frame
# (GRASP_FRAMES[arm] = left/right_graspgenx_frame) placed at exactly that
# gripper-base pose, so a grasp is executed by driving that frame to the RAW
# GraspGenX pose — no axis permutation or base->fingertip (TCP-depth) fix-up here.


# Fraction of the gripper's full range to pre-open to before approaching (see
# base.open_gripper); 1 = fully open. The grasp CLOSE is a force-based /close at
# base.GRIP_FORCE_N (see base.close_gripper), so there is no closed-fraction knob.
OPEN_PERCENT = 1.0

# How many of the ranked GraspGenX grasps to try (best-first) before giving up:
# the top grasp may be IK-unreachable, so fall through to the next one. NOTE: the
# candidates are sorted tier-major, so if the top tier alone holds >= this many
# diverse grasps, every attempt is spent inside that one tier — a mis-set pitch
# band is then fatal, not merely deprioritised. Overridable for the sweep.
MAX_GRASP_ATTEMPTS = int(_envf('HAMS_MAX_GRASP_ATTEMPTS', 5))
# Diversity thresholds for choosing those MAX_GRASP_ATTEMPTS candidates. GraspGenX
# often returns a cluster of near-identical top-scored grasps; taking the first 5
# blindly gives the IK fallback loop five poses that share the same reachability
# fate. Instead we skip a candidate that DUPLICATES an already-selected one — within
# BOTH thresholds (close in position AND orientation). A grasp far in position, or
# rotated well away (e.g. a different roll at the same spot), is kept as a genuine
# alternative. Loosen to spread attempts wider; tighten to allow closer repeats.
GRASP_DIVERSITY_LIN_M = 0.02     # positions within 2 cm ...
GRASP_DIVERSITY_ANG_DEG = 20.0   # ... AND orientations within 20 deg = duplicate

# Grasp-orientation priority tiers (_grasp_priority_tier): candidates are tried
# tier-by-tier instead of by raw GraspGenX confidence (confidence still breaks
# ties within a tier). Tiers 1-3 want the approach pitched 20-45 deg below
# horizontal ("slightly down" — steeper than ~45 deg is beyond the H1 arm's
# practical reach, flatter tends to rake the support surface); tiers 4-6 are
# the same azimuth classes with the pitch requirement dropped.
# Overridable via HAMS_TIER_PITCH_MIN_DEG / _MAX_DEG for the reachable-band sweep.
TIER_PITCH_MIN_DEG = _envf('HAMS_TIER_PITCH_MIN_DEG', 20.0)
TIER_PITCH_MAX_DEG = _envf('HAMS_TIER_PITCH_MAX_DEG', 45.0)
# "Forward" tiers (1/4) alignment tolerances: how far the approach azimuth may
# stray from pelvis +X, and the finger-closing +X axis from pelvis +Y, while
# still counting as "closely aligned". The same azimuth tolerance is granted on
# the far side of +X in the center-fan tiers (3/6).
TIER1_AZ_TOL_DEG = 15.0
TIER1_X_ALIGN_TOL_DEG = 15.0
# "Diagonal" tiers (2/5): approach azimuth band centered part-way between
# forward and fully-toward-center — 45 +/- 15 deg toward the robot's midline.
TIER2_AZ_CENTER_DEG = 45.0
TIER2_AZ_HALFWIDTH_DEG = 15.0

# Y-up CANONICALIZATION tolerance (applied in _exec_generic_grasp). graspgen
# samples every roll about the approach axis, so rather than DROP grasps whose
# gripper +Y isn't up (which threw away almost everything for objects whose
# surviving grasps are rolled ~90 deg, fingers vertical), the skill RE-ROLLS each
# grasp about its own approach axis to a level wrist — +Y as close to pelvis +Z as
# the approach pitch allows — leaving graspgen's position + approach untouched
# (see _roll_to_yup). GRASP_YUP_TOL_DEG is then the max approach tilt off
# horizontal that can still be leveled to a Y-up within tolerance (the achievable
# +Y-from-up equals |asin(az)|, so keep |az| <= sin(this)); steeper approaches
# can't be made level and are dropped. Arm side is enforced alongside it by
# dropping tier-7 (out-of-fan azimuth) grasps so a left grasp never runs on the
# right arm. Overridable via HAMS_GRASP_YUP_TOL_DEG: this gate DROPS approaches
# steeper than it before IK ever sees them, and the benchmark's one verified lift
# came from a ~70-deg-below-horizontal approach that 55 deg rejects — so the sweep
# needs to raise it to find where the arm actually reaches.
GRASP_YUP_TOL_DEG = _envf('HAMS_GRASP_YUP_TOL_DEG', 55.0)
# RViz viz of the KEPT (post-filter) grasps, published on 'graspgen_markers'
# (latched — the topic and marker style the graspgen server used to draw its
# full ranked pool; the sim.rviz display already listens there). Each kept grasp
# draws an arrow from the GraspGenX pose ORIGIN (the gripper base, where the IK
# pins *_graspgenx_frame) along +Z (approach) to the CONTACT point where the
# fingers close — so the arrow TIP sits on the object, i.e. "where the gripper
# point is" — plus a small sphere at that contact. graspgen_server still renders
# the FULL generated pool in viser (http://localhost:8080); this shows only the
# subset the skill will actually try. GRASPGEN_MARKER_LENGTH_M is the
# base->contact arrow length; magpie = 0.1146 m (its config fingertip =
# [~0, 0.0022, 0.1146]).
KEPT_MARKER_MAX = 20             # cap markers so a big kept set doesn't flood RViz
GRASPGEN_MARKER_LENGTH_M = 0.1146


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
SERVO_LIN_TOL = 0.025     # 25 mm world-position convergence tol (base: 5 mm)
SERVO_ANG_TOL = 0.10      # ~5.7 deg world-orientation convergence tol (base: ~1.15 deg)


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
        """Run the grasp pipeline, choosing perception by target. Small battery-
        workcell parts (BATTERY_OBJECTS) use the head-camera YOLO detector for the
        box and the RAW bounding-box point cloud (SAM is unreliable at that scale);
        everything else uses a Gemini box + SAM mask. Both feed the SAME
        _exec_generic_grasp pipeline from graspgen onward, so only the box source
        and the box->cloud step differ; the router is transparent to callers.

        In-process side channel (for pick_place, which sends /skill/grasp goals to
        this same node): `self._grasp_overrides` — a dict set by the caller right
        before sending a goal — is consumed HERE, once, so a later standalone
        grasp is unaffected; `self._last_grasp_outcome` is cleared here and only
        repopulated by a SUCCESSFUL grasp, so a failed grasp can never leak a stale
        outcome. Assumes one grasp goal in flight at a time (true in practice: one
        robot, skills invoked serially).

        Supported overrides:
          min_downward_pitch_deg — keep only grasps whose approach axis points at
            least this far below horizontal (see the filter in
            _exec_generic_grasp)."""
        overrides = getattr(self, '_grasp_overrides', None) or {}
        self._grasp_overrides = None
        self._last_grasp_outcome = None
        is_small = gh.request.target_object.strip().lower() in _BATTERY_OBJECTS_LC
        return self._exec_generic_grasp(
            gh, overrides,
            box_provider=(self._yolo_box if is_small else None),  # else Gemini
            use_sam=not is_small)                                 # small parts: raw box cloud

    def _exec_generic_grasp(self, gh, overrides=None, box_provider=None, use_sam=True):
        """box -> object cloud -> graspgen (6-DOF grasp) -> approach + close. No lift
        (by design). `box_provider` locates the object and supplies a box (Gemini by
        default; self._yolo_box for battery parts). `use_sam` picks how the box
        becomes a cloud: SAM mask (default) or the raw bounding-box cloud for small
        parts (see detect_object_cloud). GraspGenX then picks the grasp and
        frame_task drives the GraspGenX gripper-base frame there. `overrides` come
        from the in-process side channel documented on _exec_grasp."""
        goal = gh.request
        run = _Run(self, gh, SkillGrasp, 'grasp')
        arm = self._validated_arm(goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')
        obj = goal.target_object
        overrides = overrides or {}

        # --- detect: box (gemini or yolo) -> object cloud (sam mask or raw box) ---
        if not run.phase('detect', 0.0):
            return run.result
        obj_cloud, scene, err = self.detect_object_cloud(
            obj, run, gh, box_provider=box_provider, use_sam=use_sam)
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
        # --- Y-up canonicalization + arm-side filter ---------------------------
        # graspgen samples every ROLL about the approach axis (moe_num_yaws), and
        # for many objects the grasps that survive scoring are rolled ~90 deg
        # (fingers closing vertically) — so a hard "gripper +Y up" DROP threw them
        # all away. Roll is a free parameter, though: rotating a grasp about its
        # own approach axis (+Z) leaves graspgen's chosen POSITION and APPROACH
        # untouched and only changes the finger-closing direction. So instead of
        # dropping mis-rolled grasps we RE-ROLL each to a level wrist below. Two
        # gates remain, both on the approach axis (col 2, which the re-roll never
        # moves):
        #   * arm side — drop tier 7 (out-of-fan azimuth: cross-body / behind), so
        #     a left grasp never runs on the right arm and vice-versa;
        #   * levelable — the best +Y-from-up the re-roll can reach equals the
        #     approach's tilt off horizontal (|asin(az)|), so keep only approaches
        #     within GRASP_YUP_TOL_DEG of horizontal; steeper ones can't be leveled.
        tier0 = {i: _grasp_priority_tier(resp.grasps[i].pose, arm) for i in cand}
        yup_sin = math.sin(math.radians(GRASP_YUP_TOL_DEG))
        def _levelable(i):
            return abs(float(pose_to_matrix(resp.grasps[i].pose)[2, 2])) <= yup_sin
        kept = [i for i in cand if tier0[i] != 7 and _levelable(i)]
        n_side = sum(1 for i in cand if tier0[i] != 7)
        n_level = sum(1 for i in cand if _levelable(i))
        self.get_logger().info(
            f'grasp: {arm}-arm + levelable(<={GRASP_YUP_TOL_DEG:.0f}deg pitch) kept '
            f'{len(kept)}/{len(cand)} (on-side {n_side}, levelable {n_level})')
        if not kept:
            return run.abort(
                f'no levelable {arm}-arm grasp for {obj!r}: all {len(cand)} '
                f'filtered (on-side {n_side}, levelable {n_level}) — approaches too '
                f'steep to level, or all cross-body')
        # Re-roll each kept grasp about its approach axis to a level (Y-up) wrist,
        # IN PLACE, so execution and the RViz markers use the leveled pose.
        for i in kept:
            resp.grasps[i].pose = _roll_to_yup(resp.grasps[i].pose)
        cand = kept
        # Tiers recomputed AFTER the re-roll so the forward-tier finger-axis check
        # sees the leveled finger direction.
        tier_of = {i: _grasp_priority_tier(resp.grasps[i].pose, arm) for i in cand}
        # (Grasp markers are published AFTER the reachability loop commits to a
        # grasp — see below — so RViz shows the pose the arm actually drives to,
        # not the top-scored candidate that may never be executed.)
        # Re-order the survivors tier-major (see _grasp_priority_tier): forward/
        # diagonal/center pitched-down (1-3), then the same azimuth classes flat
        # (4-6). The sort is stable, so GraspGenX confidence keeps breaking ties
        # within each tier. Done BEFORE the diversity pick so the
        # MAX_GRASP_ATTEMPTS slots go to the best tier available.
        cand.sort(key=lambda i: tier_of[i])
        hist = {t: 0 for t in range(1, 8)}
        for i in cand:
            hist[tier_of[i]] += 1
        self.get_logger().info(
            f'grasp: priority tiers over {len(cand)} kept candidate(s) [{arm}]: '
            f'fwd/diag/center down={hist[1]}/{hist[2]}/{hist[3]}, '
            f'flat={hist[4]}/{hist[5]}/{hist[6]}, other={hist[7]}')
        # Pick the MAX_GRASP_ATTEMPTS grasps to actually try, enforcing diversity:
        # walk the tier-ordered list and keep a candidate only when it isn't a
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
        cand_tiers = [tier_of[i] for i in cand]

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

        # Walk the candidates strictly tier-major (1..7), highest GraspGenX
        # confidence first within a tier: the reachability loop below tries
        # them in that order and falls through when a pose is IK-unreachable.
        order = sorted(range(n), key=lambda i: (cand_tiers[i], -cand_scores[i]))

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
                f'grasp {i} for {obj!r}: tier {cand_tiers[i]}, '
                f'score {cand_scores[i]:.2f}, width {width_mm:.1f}mm')
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
        # Now that a grasp is committed, publish the markers so RViz matches where
        # the arm is actually going: the EXECUTED pose (grasps_p[idx] — exactly what
        # we drive GRASP_FRAMES[arm] to, GRASP_OFFSET included) is bright green, the
        # other tried candidates dim. Pelvis frame (same as grasps_p); the full
        # generated pool still shows in graspgen_server's viser (localhost:8080).
        self._publish_grasp_markers(
            cand_grasps[idx].header.frame_id, grasps_p, cand_scores, highlight=idx)
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

        if not self.close_gripper(arm):
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

    def _publish_grasp_markers(self, frame, poses, scores, highlight=None):
        """Publish grasp markers on 'graspgen_markers' (latched, in `frame` =
        pelvis), best-score-first up to KEPT_MARKER_MAX — the same topic and marker
        style the graspgen server used to draw its full ranked pool. `highlight`, if
        given, is the index into `poses` of the grasp actually being driven to; that
        one is painted bright green so RViz matches the executed grasp (falling back
        to the highest-score grasp when `highlight` is None). For each grasp:
          - an ARROW from the GraspGenX pose ORIGIN (gripper base, where the IK
            pins *_graspgenx_frame) along +Z to the CONTACT point, so the arrow TIP
            lands on the object = where the fingers close ("the gripper point");
          - a small SPHERE at that contact point.
        The base (arrow tail) is where the driven frame — and ~8 cm behind it, the
        wrist — ends up; the tip is where the gripper actually grasps. This makes
        the base-vs-contact distinction visible so the wrist sitting near the base
        isn't mistaken for the gripper being short. The `highlight`ed (executed)
        grasp is bright green; the rest are dim. Best-effort — viz must never break
        the grasp."""
        try:
            if not hasattr(self, '_grasp_marker_pub'):
                self._grasp_marker_pub = self.create_publisher(
                    MarkerArray, 'graspgen_markers',
                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            order = sorted(range(len(poses)), key=lambda i: scores[i], reverse=True)
            # Zero stamp = "use the latest available transform". The markers live in
            # the moving `pelvis` frame, so a now() stamp races ahead of the latest
            # pelvis->map TF and RViz drops them ("extrapolation into the future").
            stamp = Time()
            arr = MarkerArray()
            # Clear any markers from a previous grasp so stale ones don't linger.
            clear = Marker()
            clear.header.frame_id = frame
            clear.action = Marker.DELETEALL
            arr.markers.append(clear)
            for n, i in enumerate(order[:KEPT_MARKER_MAX]):
                T = pose_to_matrix(poses[i])
                base = T[:3, 3]
                contact = base + GRASPGEN_MARKER_LENGTH_M * T[:3, 2]  # +Z approach -> TCP
                best = (i == highlight) if highlight is not None else (n == 0)
                r, g, b = (0.0, 1.0, 0.0) if best else (0.5, 0.5, 0.0)
                a = 1.0 if best else 0.5

                arrow = Marker()
                arrow.header.frame_id = frame
                arrow.header.stamp = stamp
                arrow.ns = 'graspgen_approach'
                arrow.id = n
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.points = [
                    Point(x=float(base[0]), y=float(base[1]), z=float(base[2])),
                    Point(x=float(contact[0]), y=float(contact[1]),
                          z=float(contact[2]))]
                arrow.scale.x = 0.008                  # shaft diameter
                arrow.scale.y = 0.018                  # head diameter
                arrow.scale.z = 0.03                   # head length
                arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = r, g, b, a
                arr.markers.append(arrow)

                dot = Marker()
                dot.header.frame_id = frame
                dot.header.stamp = stamp
                dot.ns = 'graspgen_contact'
                dot.id = n
                dot.type = Marker.SPHERE
                dot.action = Marker.ADD
                dot.pose.position = Point(
                    x=float(contact[0]), y=float(contact[1]), z=float(contact[2]))
                dot.pose.orientation.w = 1.0
                dot.scale.x = dot.scale.y = dot.scale.z = 0.02
                dot.color.r, dot.color.g, dot.color.b, dot.color.a = r, g, b, a
                arr.markers.append(dot)
            self._grasp_marker_pub.publish(arr)
        except Exception as e:
            self.get_logger().warn(f'grasp marker publish failed: {e}')

    # ==================== battery parts: YOLO bounding box =================
    def _yolo_box(self, obj, run, gh):
        """Bounding box (pixel xyxy) of the nearest head-camera YOLO detection of
        `obj`, or None — a drop-in `box_provider` for detect_object_cloud that
        supplies the SAM box from YOLO instead of Gemini. Swapping Gemini for this
        is the ONLY difference between a battery-part grasp and a generic grasp.
        Picks the instance CLOSEST to the robot when several are in view (as the
        battery grasp always did). `run`/`gh` complete the box-provider signature
        but go unused here — YOLO is a cached read with no service wait. Returns
        None when no matching detection has usable depth, so SAM falls back to the
        text prompt alone."""
        det, _ = self._closest_detection(obj, self.latest_detections(),
                                         target_frame='pelvis')
        if det is None:
            self.get_logger().warn(
                f'grasp: no head YOLO detection for {obj!r} with usable depth; '
                f'SAM will fall back to the text prompt alone')
            return None
        return [float(det.bbox_min.x), float(det.bbox_min.y),
                float(det.bbox_max.x), float(det.bbox_max.y)]

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


def _grasp_priority_tier(pose, arm):
    """Selection tier of a GraspGenX grasp `pose` (in the z-up pelvis frame:
    +X forward, +Y left) for `arm` ('left' or 'right') — lower tries first.

    Azimuth classes of the approach axis (+Z, col 2), mirrored per hand so
    "toward center" means toward +Y for the right hand and toward -Y for the
    left; they cascade in this order:
      forward  — azimuth within TIER1_AZ_TOL_DEG of pelvis +X AND the
                 finger-closing axis (+X, col 0) within TIER1_X_ALIGN_TOL_DEG
                 of pelvis +Y (either hand);
      diagonal — azimuth TIER2_AZ_CENTER +/- HALFWIDTH deg toward center;
      center   — the full "toward the center" fan: azimuth from
                 -TIER1_AZ_TOL_DEG up to 90 deg toward center. No finger-axis
                 constraint, so a rolled forward approach also lands here.
    The class then combines with pitch into the tier:
      1/2/3 — forward/diagonal/center pitched TIER_PITCH_MIN..MAX deg below
              horizontal ("slightly down"), measured exactly like the
              min_downward_pitch override gate: asin(-az) of the approach;
      4/5/6 — the same classes at ANY other pitch (steep-down or upward
              approaches are admitted by azimuth; they typically fail IK
              fast, so ordering ahead of tier 7 is harmless);
      7     — everything else, kept as a last resort, not discarded."""
    eps = 1e-6      # absorb Pose->quaternion->matrix round-trip at boundaries
    R = pose_to_matrix(pose)[:3, :3]
    ax, ay, az = float(R[0, 2]), float(R[1, 2]), float(R[2, 2])
    azim = math.degrees(math.atan2(ay, ax))     # 0 = +X, positive toward +Y
    toward_center = azim if arm == 'right' else -azim
    if (abs(azim) <= TIER1_AZ_TOL_DEG + eps
            and float(R[1, 0])
            >= math.cos(math.radians(TIER1_X_ALIGN_TOL_DEG)) - eps):
        az_class = 0                                            # forward
    elif (abs(toward_center - TIER2_AZ_CENTER_DEG)
            <= TIER2_AZ_HALFWIDTH_DEG + eps):
        az_class = 1                                            # diagonal
    elif -TIER1_AZ_TOL_DEG - eps <= toward_center <= 90.0 + eps:
        az_class = 2                                            # center fan
    else:
        return 7
    pitch_down = math.degrees(math.asin(np.clip(-az, -1.0, 1.0)))
    pitched = (TIER_PITCH_MIN_DEG - eps <= pitch_down
               <= TIER_PITCH_MAX_DEG + eps)
    return (1 if pitched else 4) + az_class


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


def _roll_to_yup(pose):
    """Return `pose` rolled about its OWN approach axis (+Z, col 2) so the gripper
    +Y axis (col 1) points as close to pelvis +Z (up) as the approach allows — a
    level wrist. ONLY the finger-closing direction changes; the approach axis and
    origin are untouched, so graspgen's chosen position + approach are preserved.
    The achievable +Y-from-up angle equals the approach's tilt off horizontal
    (0 for a level approach). NOTE: this is a genuine regrasp — it swings the pinch
    from whatever roll graspgen emitted (often ~vertical) to horizontal, valid when
    the site affords a horizontal pinch (which a level wrist implies)."""
    T = pose_to_matrix(pose)
    R = T[:3, :3]
    up = np.array([0.0, 0.0, 1.0])
    x_up, y_up = float(R[:, 0] @ up), float(R[:, 1] @ up)   # up-components of +X, +Y
    # Rolling the frame about +Z by theta gives Y'.up = cos(th)*y_up - sin(th)*x_up,
    # maximised at theta = atan2(-x_up, y_up).
    theta = math.atan2(-x_up, y_up)
    c, s = math.cos(theta), math.sin(theta)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]])
    T[:3, :3] = R @ Rz
    return matrix_to_pose(T)


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
