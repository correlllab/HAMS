"""SkillGrasp: one pipeline, two bounding-box sources, two grasp modes.

Flow (see _exec_generic_grasp, the orchestrator):

    prep -> box -> object cloud -> candidates -> pre-grasp approach
         -> contact -> force close

With goal.visual_servo the last two steps become a hand-camera loop instead:
align on the object at VISUAL_SERVO_DEPTH_M and close there (_visual_servo_refine).
That loop failing FAILS the skill — there is no open-loop fallback.

The box comes from Gemini for a generic object, or from the head-camera YOLO
detector for a battery-workcell part (BATTERY_OBJECTS) — that routing is the
ONLY difference between the two paths (_exec_grasp). Candidates come from
GraspGenX, filtered and tier-ordered (_graspgen_candidates), or — when the
goal sets top_down — a single synthetic steep-from-above grasp
(_top_down_candidates). Optional goal fields are documented on _exec_grasp;
the module constants below are the tuning knobs, grouped by pipeline stage.
"""

import copy
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from builtin_interfaces.msg import Time
from geometry_msgs.msg import (Pose, PoseStamped, Point, TransformStamped,
                               Vector3Stamped)
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import QoSProfile, DurabilityPolicy

from custom_ros_messages.action import SkillGrasp

from ..base import _Run, GRASP_FRAMES, SLOW_MODE_TIME_SCALE, WORLD_FRAME
from ..perception_utils import pose_to_matrix, matrix_to_pose


# ========================== perception routing ===============================
# Battery-workcell parts the fine-tuned YOLO-World checkpoint detects (mirrors
# yolo_server.DEFAULT_QUERIES; 'OrageCover' matches the checkpoint's label
# spelling). When target_object is one of these, _exec_grasp swaps the Gemini
# box for a YOLO one and skips SAM (raw bounding-box cloud) — small parts are
# below SAM's reliable scale.
BATTERY_OBJECTS = ('Bolt', 'BusBar', 'InteriorScrew', 'Nut', 'OrageCover',
                   'Screw', 'ScrewHole')
_BATTERY_OBJECTS_LC = frozenset(o.lower() for o in BATTERY_OBJECTS)


# ======================= GraspGenX frame convention ==========================
# GraspGenX emits each grasp as the pose of its gripper-BASE frame in its own
# convention (+Z = approach into the object, +X = finger-closing axis, origin
# at the gripper base). The frame_task server carries a matching URDF frame
# (GRASP_FRAMES[arm] = left/right_graspgenx_frame) placed at exactly that
# gripper-base pose, so a grasp is executed by driving that frame to the RAW
# GraspGenX pose — no axis permutation or base->fingertip (TCP-depth) fix-up.
#
# Distance from that gripper base to the point where the fingers close, along
# +Z — the magpie config fingertip is [~0, 0.0022, 0.1146]. Used both to draw
# the base->contact marker arrows and to stand the base off the object in the
# synthetic top-down grasp.
GRIPPER_BASE_TO_CONTACT_M = 0.1146


# ======================= candidate selection knobs ===========================
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

# Grasp-orientation priority tiers (_grasp_priority_tier): candidates are tried
# tier-by-tier instead of by raw GraspGenX confidence (confidence still breaks
# ties within a tier). Tiers 1-3 want the approach pitched 20-45 deg below
# horizontal ("slightly down" — steeper than ~45 deg is beyond the H1 arm's
# practical reach, flatter tends to rake the support surface); tiers 4-6 are
# the same azimuth classes with the pitch requirement dropped.
TIER_PITCH_MIN_DEG = 20.0
TIER_PITCH_MAX_DEG = 45.0
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

# Y-up CANONICALIZATION tolerance (applied in _graspgen_candidates). graspgen
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
# right arm.
GRASP_YUP_TOL_DEG = 55.0

# Pitch below horizontal of the synthetic top-down grasp (goal.top_down; see
# _top_down_pose). NOT 90 deg: a literally vertical approach puts the wrist in a
# posture the H1 arm can't reach (its practical steepest is ~45 deg), so the
# override instead reuses the TIER-1 orientation — forward azimuth, fingers
# along pelvis +Y, wrist as level as the pitch allows — and only steepens the
# pitch past the tier's TIER_PITCH_MAX_DEG. Lower this toward TIER_PITCH_MAX_DEG
# for a more reachable (flatter) approach; 90 restores the straight-down grasp.
TOP_DOWN_PITCH_DEG = 80.0


# ===================== failure retreat (_retreat_after_failure) ==============
# Straight-up (pelvis +Z) lift run before a skill reports a failure, so a failed
# grasp does not leave the hand sitting in the parts where the next attempt — or
# a person — has to work around it. Larger than the visual servo's own 5 cm
# reacquire hop: that one is trying to keep servoing, this one is leaving.
# Excluded on purpose from the prep-config failure, which fires before the
# skill has moved the arm itself.
FAILURE_RETRACT_M = 0.15
FAILURE_RETRACT_SEC = 3.0


# ========================= motion / execution knobs ==========================
# Fraction of the gripper's full range to pre-open to before approaching (see
# base.open_gripper); 1 = fully open. The grasp CLOSE is a force-based /close at
# base.GRIP_FORCE_N (see base.close_gripper), so there is no closed-fraction knob.
OPEN_PERCENT = 1.0

# Metres to back off along the grasp's +Z approach axis for the pre-grasp pose.
APPROACH_DIST = 0.1
# Metres to shift every grasp along its OWN +Z approach axis before executing
# it. POSITIVE drives the grasp DEEPER into the object (further along the
# approach); NEGATIVE backs it off. The pre-grasp standoff (APPROACH_DIST) is
# measured from this shifted grasp, so the whole approach->grasp pair moves
# together.
GRASP_OFFSET = 0.0
# Pause between the last commanded motion and the gripper close [s]. frame_task
# reports 'done' the moment its IK converges, but the arm is still settling — its
# streamed error visibly wanders several mm mid-move and takes a beat to come to
# rest. Closing into that residual motion drags the part. Uses the node clock, so
# it respects sim time like the other gripper waits.
PRE_CLOSE_SETTLE_SEC = 1.0
# Pause after the close returns, before the skill reports and anything moves the
# arm again [s]. This is ON TOP of the GRIPPER_SETTLE_SEC (1.0 s) close_gripper
# already waits internally, and on top of the grip-hold loop it starts: it gives
# the grip a moment to establish at full force before pick_place's lift takes the
# part's weight. Walked up 0.5 -> 1.0 -> 2.0: the lift was still starting before
# the grip was set.
POST_CLOSE_SETTLE_SEC = 2.0
# Metres to STOP SHORT of the (GRASP_OFFSET-shifted) grasp pose on the contact
# move, measured back along the grasp's own +Z approach axis. Tuning knob for
# how deep the contact move actually drives: 0.0 = the full GraspGenX pose.
CONTACT_OFFSET = 0.25 * APPROACH_DIST
# When True the skill HOLDS at the pre-grasp approach pose, blocking on a
# literal input() until Enter is pressed on the skills node's terminal, then
# executes the contact move + close. Testing switch: needs an interactive
# stdin on the node's process, and the blocked thread ignores cancel/timeout
# while waiting (the goal clock keeps running, so a wait longer than the goal
# timeout aborts right after Enter).
CONFIRM_BEFORE_CONTACT = False

# ======================= visual servo (goal.visual_servo) ====================
# Closed-loop hand-camera alignment run at the pre-grasp, before the gripper
# closes (see _visual_servo_refine). The hand camera looks straight DOWN the
# approach axis — its optical +Z is the graspgenx +Z of the frame being driven
# (the URDF puts *_hand_link at (0, -0.0015, +0.044) in *_graspgenx_frame, i.e.
# on the approach axis, 4.4 cm ahead of the gripper base) — so "centered in the
# image at depth D" means "on the approach ray, D metres ahead of the camera".
#
# TARGET depth from the CAMERA to the object [m], and the Z of the setpoint the
# loop drives to (see the SETPOINT note further down — X and Y are both 0, i.e.
# straight down the boresight). NOTE the 4.4 cm camera lead: the finger contact
# point sits GRIPPER_BASE_TO_CONTACT_M - 0.044 = 0.0706 m in front of the camera,
# so the geometrically consistent "fingers on the object" value is 0.0706. At
# 0.10 the object ends up ~2.9 cm BEYOND the fingertips — conservative by design:
# the loop can never drive the hand into the object.
#
# Do NOT expect to tune all the way down to 0.0706: the hand cameras are D405s
# (min-Z ~0.07 m, optimal 0.07-0.5 m), so that target sits AT the sensor floor
# where depth drops out and the loop would stop converging. ~0.08 m is the
# practical limit. Closing the remaining couple of cm wants a short fixed
# descent after convergence, not a lower target.
VISUAL_SERVO_DEPTH_M = 0.10
# Convergence: the object must be within this far of the image center (metres in
# the camera's XY plane, NOT pixels) AND this close to the target depth.
# Set from a real run (2026-07-21), NOT from theory — an earlier 1 mm guess was
# unreachable and the loop limit-cycled until the budget expired.
#
# What that run showed, from iteration ~28 to 39+: measurement pinned at lateral
# 3.5 +/- 0.2 mm and range -1.0 mm, the commanded pose byte-identical every
# iteration, and frame_task reporting 'done: lin=1.2mm' after wandering to 6.7 mm
# mid-move and returning. Net motion per iteration: zero. The operator confirmed
# that pose was good enough to close on.
#
# The binding limit is the ARM, not the camera or the control law. frame_task
# settles to ~1.2 mm residual with ~6.7 mm excursions during the move, so once
# the error is small the correction (VISUAL_SERVO_GAIN x error, ~2.1 mm at 3.5 mm
# of error) is below what the arm can resolve: it commands a step, nothing
# measurable happens, and the loop spins. Anything under ~4 mm lateral is
# therefore unreachable no matter how much budget it is given.
#
# Tolerance is PER AXIS, not a radius, because the three camera axes mean
# different things to a parallel gripper:
#
#   optical X (1 mm) — perpendicular to the jaws. The part has to sit between
#     them, so this is what a grasp depends on. Tight — and now AT the arm's own
#     resolution: frame_task settles to ~1.2 mm residual, and a correction of
#     GAIN x 1 mm is smaller than that, so the loop cannot deliberately place
#     within this band; it converges when it happens to land there. Halved from
#     2 mm after the setpoint was shifted +2 mm to cancel the observed X bias.
#     If runs start stalling with X named in the message, this is the cause and
#     2 mm is the honest value.
#   optical Z (2 mm) — range along the approach, i.e. how far down the hand
#     comes. Tight: it decides whether the jaws close around the part or above
#     it.
#   optical Y (5 mm) — ALONG the jaw span (optical Y = -graspgenx X = the
#     finger-closing direction). The jaws are pre-opened to
#     GRIPPER_MAX_WIDTH_MM = 106 mm, so being off along the span is far more
#     forgiving than being off across it — but only relatively: 5 mm still holds
#     it near the middle of the grip rather than out toward one jaw.
#
# A single hypot() tolerance conflated X and Y and failed real runs on Y alone
# while X was ~2 mm — demanding precision in the direction the hardware needs it
# least. Keep these ordered X <= Y: inverting them asks for accuracy along the
# span and tolerates it across, which is backwards for a parallel gripper.
VISUAL_SERVO_X_TOL_M = 0.001
VISUAL_SERVO_Y_TOL_M = 0.005
VISUAL_SERVO_RANGE_TOL_M = 0.002
# Wall-clock budget for the whole loop [s] — this, not an iteration count, is
# what bounds the number of corrections: one iteration costs about
# VISUAL_SERVO_MOVE_SEC plus a frame wait, so 90 s is dozens of
# corrections. Walked up from 20 s (~11) as the tolerances tightened — runs
# converge slowly but steadily, so the budget, not the control law, was the
# limit. NOTE the goal's own timeout still wins: the loop checks run.remaining()
# every iteration, so on a 180 s goal the servo only gets whatever detection and
# the approach left behind. On expiry the skill WARNS and falls back to the
# open-loop contact descent rather than aborting, so a longer budget costs time
# on a failing grasp but never changes the outcome.
VISUAL_SERVO_TIMEOUT_SEC = 90.0
# How long to keep waiting for a usable hand-camera detection before treating
# the object as LOST [s] — it may not be in frame yet when the arm arrives, and
# yolo_server only publishes at ~5 Hz.
VISUAL_SERVO_DETECT_WAIT_SEC = 5.0
# Losing sight is usually a geometry problem, not a detector problem: the hand
# has come in close enough that the part has left the frame, is occluded by the
# fingers, or has fallen inside the D405's ~7 cm minimum range. Backing STRAIGHT
# UP (pelvis +Z, not along the approach) widens the view and restores depth
# without giving up the lateral alignment already achieved, so the loop retracts
# and keeps servoing instead of failing outright.
VISUAL_SERVO_LOST_RETRACT_M = 0.05
# Depth at or below which the object is TOO CLOSE to keep servoing on [m], and
# the loop backs off exactly as it does for a lost object. The D405 stops
# returning trustworthy depth under ~0.07 m, so a reading from inside that band
# is not something to steer by — and by then the fingers are practically on the
# part. The loop's own correction would ease back only GAIN x error per tick and
# would keep believing those readings on the way; this gets clear in one move.
#
# Deliberately ABOVE VISUAL_SERVO_MIN_DEPTH_M (0.04), the back-projection's
# floor: the deprojection has to still return points below this threshold, or
# being too close would just look like losing sight and cost the full
# VISUAL_SERVO_DETECT_WAIT_SEC before anything happened.
#
# LOWERED from 0.070 after it fired repeatedly on a perfectly visible screw: the
# FINGER CONTACT POINT sits 70.6 mm in front of the camera (0.1146 - 0.044), so
# a 70 mm trip was 0.6 mm away from the gripper's own hardware, and any depth
# sample catching finger material read as "too close". That failure could not
# heal, either — the fingers are rigid to the camera, so backing off does not
# change their depth; it just retracted twice and aborted. Keep this clearly
# BELOW 0.0706.
VISUAL_SERVO_MIN_RANGE_M = 0.055
# Consecutive too-close readings before acting. One frame must never be able to
# command a 5 cm retreat; the condition is real only if it persists.
VISUAL_SERVO_TOO_CLOSE_STRIKES = 2
# Plausibility band on the depth of a new measurement [m]. Each iteration
# predicts where the object's depth should land (previous depth minus the depth
# component of the correction just commanded) and accepts anything between "the
# arm did not move at all" and "the arm moved fully", plus this slack. Rejects
# the pathological reading — a finger, a shadow, the surface behind the part —
# that would otherwise drive a large bogus correction, WITHOUT rejecting honest
# motion: the largest legitimate change is VISUAL_SERVO_MAX_STEP_M (50 mm), and
# that is inside the band by construction because the prediction accounts for it.
VISUAL_SERVO_RANGE_BAND_M = 0.02
# Cap on retracts per servo run, SHARED by the lost-sight and too-close paths.
# Needed: each retract is gated only by VISUAL_SERVO_DETECT_WAIT_SEC (or by an
# immediate too-close reading), so an uncapped loop would climb ~5 cm at a time
# and walk the arm out of the workspace before the budget ran out. After this
# many, the object is genuinely not servo-able and the skill fails.
VISUAL_SERVO_MAX_RETRACTS = 2
# Stall detection: give up once the error stops improving, instead of spending
# the rest of VISUAL_SERVO_TIMEOUT_SEC re-commanding a pose the arm has already
# reached. Observed twice on the robot — the loop pinned at a fixed residual and
# an unchanging commanded pose for 30+ iterations (~75 s) before the budget
# expired, because below ~2 mm the correction (GAIN x error) is smaller than the
# arm's own ~1.2 mm IK residual and nothing measurable happens.
#
# An iteration counts as progress only if it beats the best error so far by more
# than EPS; STALL_ITERS consecutive non-improving iterations end the run. Failing
# here rather than at the timeout turns "waited 90 s, no idea why" into a message
# carrying the residual, which is what tells you the tolerance is unreachable.
# Both counters reset after a back-off, since that genuinely changes the setup.
VISUAL_SERVO_STALL_ITERS = 5
VISUAL_SERVO_STALL_EPS_M = 0.001
# Proportional gain: fraction of the measured error corrected per iteration.
# Below 1.0 so a bad depth sample can't throw the hand across the workspace and
# so the loop damps rather than oscillates around the target.
VISUAL_SERVO_GAIN = 0.6
# Largest single correction [m]; a mis-detection (wrong instance, background
# depth) can produce a huge error, and this bounds what one iteration can do.
VISUAL_SERVO_MAX_STEP_M = 0.05
# Per-iteration frame_task move duration [s]. Sent with slow_mode (the
# corrections happen with the gripper already at the pre-grasp, centimetres from
# the object); NOT scaled by SLOW_MODE_TIME_SCALE at the call site — this value
# is already sized for a quarter-speed move to converge AND reach the server's
# steady-state hold. That hold matters: each new frame_task goal re-seeds its
# command from the MEASURED joints, so a goal cut off before the hold's integral
# stage cancels the arm's gravity droop leaves the droop baked in — at 1 s per
# move the servo visibly ratcheted the hand downward ~5 mm per iteration while
# lateral corrections went unexecuted.
VISUAL_SERVO_MOVE_SEC = 4.0
# Pause after each servo move before the next measurement [s]. The freshness
# rule only requires a frame NEWER than the last one consumed (see
# _servo_measure), so without this the loop can act on a frame captured
# mid-move, from a vantage point the arm has already left. Node clock, so it
# respects sim time.
VISUAL_SERVO_POST_MOVE_SLEEP_SEC = 0.25
# SETPOINT: the object is driven to (X, Y, VISUAL_SERVO_DEPTH_M) in the camera
# OPTICAL frame, with the lateral part fixed here. Nothing is derived from TF and
# nothing is fitted per-run.
#
# Sign convention for tuning this on the robot: POSITIVE X sits the object
# further RIGHT in the image and parks the hand further BACK in pelvis +X
# (optical +X = graspgenx +Y ~= pelvis +X at the 80-deg approach, and raising the
# setpoint pulls the hand back). Negative does the reverse. Values tried so far:
# anywhere in -0.020 .. +0.012 — all set from observed grasps, not geometry.
#
# The loop previously aimed at the FINGER CONTACT POINT instead, resolved from
# live TF (~+1.9 mm off the boresight) plus an empirical trim tuned on the robot.
# That is the geometrically "correct" target — converged then meant "the object
# is where the fingers will close" rather than "the object is centred". It was
# abandoned because it did not behave: the trim had to grow to +4 mm (more than
# twice TF's own answer, itself a sign the hand-entered camera mount transform in
# cl_realsense's h12_hand_cameras.launch.py is materially wrong), then over-shot
# at that value, and a Y term fitted the same way inverted sign between
# consecutive runs. Aiming at the boresight gives up the finger-axis correction
# in exchange for a setpoint that is exact, stable, and has no fitted constants.
#
# What that costs: any real camera-to-finger offset now shows up as a constant
# lateral bias in the grasp. Recover it by MEASURING the camera mount rather than
# by re-introducing a trim here.
VISUAL_SERVO_TARGET_XY_M = (0.012, 0.0)
# Floor on valid depth for the servo's back-projection [m], overriding base's
# DEPTH_MIN_M (0.1) — that head-camera default sits ABOVE the target depth and
# would reject exactly the measurements the loop needs.
VISUAL_SERVO_MIN_DEPTH_M = 0.04


# How hard to try to physically REACH each candidate pose before falling through
# to the next ranked grasp. These push past servo_frame_to_world's defaults
# (10s primary move / SERVO_ITER refinement passes): give a near-but-not-yet
# reached pose MORE TIME on the main IK move and MORE ITERATIONS of world-frame
# drift correction to settle within tolerance. The iter-0 unreachable fast-fail
# in servo_frame_to_world still bails genuinely out-of-reach candidates quickly,
# so the extra budget is only spent on poses that are actually close to reachable.
SERVO_DURATION_SEC = 15   # primary (iter-0) approach/contact IK move budget [s]
# [s] timeout for the pre-grasp approach move (per candidate, per servo
# iteration). Deliberately generous so a long transit into the workspace is
# never cut off mid-move. It is a TIMEOUT, not a commanded trajectory time:
# reachable moves finish and return as soon as they converge, and dead
# candidates are rejected at plan time in well under a second, so the full
# budget is only ever consumed by motion that is genuinely in progress.
APPROACH_DURATION_SEC = 180.0
SERVO_MAX_ITER = 6        # world-frame servo refinement passes per pose
# Convergence tolerances for the grasp servo, relaxed from base.py's defaults
# (5 mm / ~1.15 deg). Real-robot IK + pelvis drift rarely settle a 6-DOF grasp
# pose that tight within SERVO_MAX_ITER passes, so accept a looser world-frame
# fit as "reached" instead of burning the whole iteration budget and proceeding
# best-effort anyway. The iter-0 unreachable fast-fail (lin>5cm/ang>0.2rad) still
# rejects genuinely out-of-reach candidates, so this only loosens the final fit.
SERVO_LIN_TOL = 0.025     # 25 mm world-position convergence tol (base: 5 mm)
SERVO_ANG_TOL = 0.10      # ~5.7 deg world-orientation convergence tol (base: ~1.15 deg)


# ============================== visualization ================================
# Single TF frame the currently-driven target is broadcast to (updated as the
# loop walks the ranked candidates), so RViz shows the target being tried.
TARGET_FRAME = 'graspgenx_target_frame'
# Cap on 'graspgen_markers' markers so a big kept set doesn't flood RViz.
KEPT_MARKER_MAX = 20


# ================================ data types =================================
@dataclass
class GraspOutcome:
    """What an in-process caller (pick_place) needs from an executed generic
    grasp, handed back through the node's `_last_grasp_outcome` attribute — the
    SkillGrasp result message itself only carries success/message (see
    _exec_grasp for the side-channel contract)."""
    arm: str                # arm that executed it ('left'/'right') — the resolved
                            # one, so a caller that asked for auto learns the pick
    pose: Pose              # executed grasp pose (GraspGenX gripper-base), in `frame`
    frame: str              # WORLD_FRAME when the world TF was available, else 'pelvis'
    centroid: np.ndarray    # object-cloud centroid [m], same frame
    rest_height: float      # centroid height above the object's own bottom [m]
    gripper_width: float    # GraspGenX planned opening [m]
    score: float            # confidence of the executed grasp


@dataclass
class _Candidates:
    """Ranked, ready-to-execute grasp candidates — the output of
    _graspgen_candidates or _top_down_candidates."""
    grasps: list            # PoseStamped per candidate (planning frame = pelvis)
    scores: list            # confidence per candidate, same order
    tiers: list             # _grasp_priority_tier per candidate (1 best .. 7)
    total: int              # size of the ranked pool these were picked from
    gripper_width: float    # planned opening [m]; 0.0 when unknown (top_down)


@dataclass
class _Targets:
    """Per-candidate executable poses, frozen once by _snapshot_targets."""
    frame: str              # the candidates' planning frame ('pelvis')
    grasps_p: list          # GRASP_OFFSET-shifted grasp poses, planning frame
    approaches_p: list      # pre-grasp standoffs (APPROACH_DIST back), same frame
    grasps_w: list          # world-frame snapshots (entries None if TF was down)
    approaches_w: list      # world-frame snapshots of the pre-grasps
    have_world: bool        # all snapshots resolved -> drift-compensated servoing
    centroid_w: object      # object centroid as a world Pose (None without TF)


class GraspSkill:
    # ------------------------------------------------------ action entry point
    def _exec_grasp(self, gh):
        """Run the grasp pipeline, choosing perception by target. Small battery-
        workcell parts (BATTERY_OBJECTS) use the head-camera YOLO detector for the
        box and the RAW bounding-box point cloud (SAM is unreliable at that scale);
        everything else uses a Gemini box + SAM mask. Both feed the SAME
        _exec_generic_grasp pipeline from planning onward, so only the box source
        and the box->cloud step differ; the router is transparent to callers.

        Optional goal fields (see SkillGrasp.action; zero/false = off):
          top_down — skip graspgen and grasp from above with the tier-1
            orientation pitched TOP_DOWN_PITCH_DEG down (_top_down_candidates).
          min_downward_pitch_deg — keep only grasps whose approach axis points
            at least this far below horizontal (see _graspgen_candidates).
          visual_servo — close the loop on the hand camera at the pre-grasp:
            align until the object is centered at VISUAL_SERVO_DEPTH_M and close
            THERE, skipping the contact move (_visual_servo_refine). ABORTS the
            skill if it can't converge — no open-loop fallback.

        In-process RESULT side channel (for pick_place, which sends /skill/grasp
        goals to this same node): `self._last_grasp_outcome` is cleared here and
        only repopulated by a SUCCESSFUL grasp, so a failed grasp can never leak
        a stale outcome. Assumes one grasp goal in flight at a time (true in
        practice: one robot, skills invoked serially)."""
        self._last_grasp_outcome = None
        is_small = gh.request.target_object.strip().lower() in _BATTERY_OBJECTS_LC
        return self._exec_generic_grasp(
            gh,
            box_provider=(self._yolo_box if is_small else None),  # else Gemini
            use_sam=not is_small)                                 # small parts: raw box cloud

    def _exec_generic_grasp(self, gh, box_provider=None, use_sam=True):
        """Orchestrate one grasp, delegating each stage to a helper below:
        detect -> candidates -> pre-grasp approach -> contact + force close.
        No lift afterwards (by design). `box_provider` locates the object and
        supplies a box (Gemini by default; self._yolo_box for battery parts);
        `use_sam` picks how the box becomes a cloud: SAM mask (default) or the
        raw bounding-box cloud for small parts (see detect_object_cloud)."""
        goal = gh.request
        run = _Run(self, gh, SkillGrasp, 'grasp')
        arm = self._validated_arm(goal)
        if arm is None and goal.arm.strip().lower() not in ('', 'none'):
            return run.abort(f'invalid arm {goal.arm!r}')
        obj = goal.target_object

        # --- detect: box (gemini or yolo) -> object cloud (sam or raw box) ----
        if not run.phase('detect_object', 0.0):
            return run.result
        # Arms to the 'prep' config FIRST, before the detection image is
        # grabbed, so they are clear of the head camera's view (an arm across
        # the object corrupts the box/mask/cloud). PLANNED (plan=True): the
        # arms start wherever the last skill left them — often near the
        # workspace or each other — so this long opening move goes through the
        # planner rather than a direct IK descent that can sweep the arms
        # through obstacles.
        if not self.goto_named_config('prep', plan=True, outer_gh=gh):
            # The ONLY failure that does not retract: this one fires before the
            # skill has moved the arm anywhere of its own, and a prep pose that
            # could not be reached is exactly the case where commanding another
            # motion on top is the wrong move.
            return run.abort("move to 'prep' before detection failed")

        def fail(message):
            """Abort, but lift the hand clear of the workspace first. Every
            failure from here on leaves the arm somewhere it was put — reaching
            into the parts, at a pre-grasp, or on the object — and reporting the
            failure without retreating strands it there for the next attempt."""
            self._retreat_after_failure(arm, gh)
            return run.abort(message)

        obj_cloud, scene, err = self.detect_object_cloud(
            obj, run, gh, box_provider=box_provider, use_sam=use_sam)
        if err:
            return fail(err)
        # Held-object geometry for the pick_place side channel: the cloud
        # centroid and how high it sits above the object's own bottom (its rest
        # height on a support surface) — measured now, while the object still
        # sits undisturbed on the table.
        centroid_p = obj_cloud.mean(axis=0)
        rest_height = float(centroid_p[2] - np.percentile(obj_cloud[:, 2], 5))
        if arm is None:   # auto-select: the arm currently closest to the object
            arm = self._closest_arm(centroid_p)

        # --- plan: build the ranked candidate list -----------------------------
        if not run.phase('approach_grasp', 0.4):
            return run.result
        if goal.top_down:
            cands = self._top_down_candidates(obj, centroid_p)
        else:
            cands, err = self._graspgen_candidates(goal, obj, arm, obj_cloud,
                                                   scene)
            if err:
                return fail(err)

        # --- approach: pre-open, then servo to the first reachable pre-grasp --
        if not self.open_gripper(arm, OPEN_PERCENT):
            return fail('gripper pre-open failed')
        targets = self._snapshot_targets(cands, centroid_p)
        idx, err = self._servo_to_first_reachable(run, gh, obj, arm, cands,
                                                  targets)
        if err:
            return fail(err)
        # Markers only now that a grasp is committed, so RViz matches where the
        # arm is actually going: the committed pose (GRASP_OFFSET included; the
        # contact move stops CONTACT_OFFSET short of it) bright green, the
        # other tried candidates dim. The full generated pool still shows in
        # graspgen_server's viser (localhost:8080).
        self._publish_grasp_markers(targets.frame, targets.grasps_p,
                                    cands.scores, highlight=idx)
        # Optional closed-loop alignment, now that the arm is holding at the
        # pre-grasp and the hand camera is looking down the approach. On success
        # this REPLACES the contact move: the loop parks the gripper at
        # VISUAL_SERVO_DEPTH_M from the object, so the grasp closes right there.
        # A failure ABORTS rather than falling back: the open-loop descent would
        # drive to the ORIGINAL planned target, discarding every correction the
        # loop made, and quietly returning that as success would hide exactly the
        # misalignment the caller asked to eliminate.
        servo_pose_p = None
        if goal.visual_servo:
            servo_pose_p, err = self._visual_servo_refine(run, gh, arm, obj,
                                                          targets, idx)
            if err:
                return fail(err)
        if CONFIRM_BEFORE_CONTACT:
            # Hold at the pre-grasp standoff until a human confirms. Logged AND
            # prompted: the logger line reaches rosout even when stdout is
            # swallowed; the input() prompt shows on an interactive terminal.
            self.get_logger().info(
                f'grasp: holding at pre-grasp for {obj!r} — press Enter on the '
                f'skills node terminal to execute the grasp')
            input(f'at pre-grasp for {obj!r} — press Enter to execute the grasp ')

        # --- grasp: move to contact + force close ----------------------------
        if not run.phase('close_gripper', 0.75):
            return run.result
        if servo_pose_p is not None:
            # Already parked by the visual servo — close where we stand.
            contact_p = servo_pose_p
            contact_w = (self._transform_pose(contact_p, 'pelvis', WORLD_FRAME)
                         if targets.have_world else None)
        else:
            # Only reached when visual servoing wasn't requested — a servo that
            # ran and failed has already aborted the skill above.
            contact_p, contact_w = self._drive_to_contact(gh, arm, targets, idx)
        # Let the arm come to rest before the jaws move — see PRE_CLOSE_SETTLE_SEC.
        self.get_clock().sleep_for(RclpyDuration(seconds=PRE_CLOSE_SETTLE_SEC))
        if not self.close_gripper(arm):
            return fail('gripper close failed')
        # Let the grip establish before anything moves the arm again.
        self.get_clock().sleep_for(RclpyDuration(seconds=POST_CLOSE_SETTLE_SEC))

        # Hand the executed grasp + held-object geometry to any in-process
        # caller (pick_place) via the side channel documented on _exec_grasp.
        # Pose and centroid share one frame: the world snapshot when it was
        # available, else the detect-time pelvis frame. The pose is the CONTACT
        # target (CONTACT_OFFSET short of the planned grasp) — where the hand
        # actually is — so pick_place's grip-to-centroid offset math matches
        # reality.
        if targets.have_world and targets.centroid_w is not None:
            out_pose, out_frame, out_c = contact_w, WORLD_FRAME, targets.centroid_w
        else:
            out_pose, out_frame, out_c = contact_p, 'pelvis', _pose_at(centroid_p)
        self._last_grasp_outcome = GraspOutcome(
            arm=arm, pose=out_pose, frame=out_frame,
            centroid=np.array([out_c.position.x, out_c.position.y,
                               out_c.position.z]),
            rest_height=rest_height, gripper_width=cands.gripper_width,
            score=cands.scores[idx])

        return run.succeed(
            f'grasped {obj!r} (graspgen score {cands.scores[idx]:.2f})')

    # ---------------------------------------------------- candidate generation
    def _top_down_candidates(self, obj, centroid_p):
        """One synthetic candidate above the object (goal.top_down): the tier-1
        orientation pitched TOP_DOWN_PITCH_DEG below horizontal, gripper BASE
        one base->contact length back along that approach so the finger contact
        lands on the centroid (see _top_down_pose). Skips graspgen AND every
        orientation gate in _graspgen_candidates (this pitch fails the levelable
        filter by construction). NOTE: still steeper than the arm's ~45-deg
        practical approach — expect best-effort convergence from the servo."""
        g = PoseStamped()
        g.header.frame_id = 'pelvis'
        g.pose = _top_down_pose(centroid_p)
        self.get_logger().info(
            f'grasp: top-down override for {obj!r} '
            f'({TOP_DOWN_PITCH_DEG:.0f}deg below horizontal), gripper base at '
            f'({g.pose.position.x:.3f}, {g.pose.position.y:.3f}, '
            f'{g.pose.position.z:.3f}) pelvis')
        return _Candidates(grasps=[g], scores=[1.0], tiers=[1], total=1,
                           gripper_width=0.0)

    def _graspgen_candidates(self, goal, obj, arm, obj_cloud, scene):
        """GraspGenX plan + the whole candidate-selection chain. Returns
        (_Candidates, None), or (None, abort-reason) when planning or a gate
        leaves nothing. Stages, in order:
          1. plan_grasp on the object cloud (server ranks best-first);
          2. optional downward-pitch gate (goal.min_downward_pitch_deg);
          3. arm-side + levelable filter, then re-roll every survivor to a
             level (Y-up) wrist;
          4. tier-major ordering (_grasp_priority_tier), GraspGenX confidence
             breaking ties within a tier;
          5. diversity pick of the MAX_GRASP_ATTEMPTS poses actually tried."""
        resp = self.plan_grasp(obj_cloud, gripper_name="magpie", frame='pelvis',
                               scene_cloud=scene, arm=arm)
        if resp is None:
            return None, f'no grasp planned for {obj!r}'

        # -- optional downward-pitch gate ------------------------------------
        # Keep the server's best-first order, but when the goal asks for it
        # (e.g. stacking), drop grasps that approach too flat. The gate is a
        # MINIMUM downward pitch, not proximity to vertical: the H1 arm cannot
        # reach straight-down approaches (~45 deg below horizontal is its
        # practical steepest), so demanding verticality would reject everything
        # reachable. Approach axis = column 2 (+Z) of the grasp rotation in the
        # z-up pelvis frame; its pitch below horizontal is asin(-a_z).
        cand = list(range(len(resp.grasps)))
        min_pitch = float(goal.min_downward_pitch_deg or 0.0)
        if min_pitch > 0.0:
            thr = math.sin(math.radians(min_pitch))
            kept = [i for i in cand
                    if -pose_to_matrix(resp.grasps[i].pose)[2, 2] >= thr]
            self.get_logger().info(
                f'grasp: downward-pitch gate >= {min_pitch:.0f} deg kept '
                f'{len(kept)}/{len(cand)} candidates')
            if not kept:
                return None, (
                    f'no grasp for {obj!r} pitched >= {min_pitch:.0f} deg down')
            cand = kept

        # -- arm-side + levelable filter, then Y-up re-roll ------------------
        # graspgen samples every ROLL about the approach axis (moe_num_yaws),
        # and for many objects the grasps that survive scoring are rolled
        # ~90 deg (fingers closing vertically) — so a hard "gripper +Y up" DROP
        # threw them all away. Roll is a free parameter, though: rotating a
        # grasp about its own approach axis (+Z) leaves graspgen's chosen
        # POSITION and APPROACH untouched and only changes the finger-closing
        # direction. So instead of dropping mis-rolled grasps we RE-ROLL each
        # to a level wrist. Two gates remain, both on the approach axis (col 2,
        # which the re-roll never moves):
        #   * arm side — drop tier 7 (out-of-fan azimuth: cross-body / behind),
        #     so a left grasp never runs on the right arm and vice-versa;
        #   * levelable — the best +Y-from-up the re-roll can reach equals the
        #     approach's tilt off horizontal (|asin(az)|), so keep only
        #     approaches within GRASP_YUP_TOL_DEG of horizontal; steeper ones
        #     can't be leveled.
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
            return None, (
                f'no levelable {arm}-arm grasp for {obj!r}: all {len(cand)} '
                f'filtered (on-side {n_side}, levelable {n_level}) — approaches too '
                f'steep to level, or all cross-body')
        # Re-roll each kept grasp about its approach axis to a level (Y-up)
        # wrist, IN PLACE, so execution and the RViz markers use the leveled pose.
        for i in kept:
            resp.grasps[i].pose = _roll_to_yup(resp.grasps[i].pose)
        cand = kept

        # -- tier-major ordering ---------------------------------------------
        # Tiers recomputed AFTER the re-roll so the forward-tier finger-axis
        # check sees the leveled finger direction. Order the survivors
        # tier-major (see _grasp_priority_tier): forward/diagonal/center
        # pitched-down (1-3), then the same azimuth classes flat (4-6). The
        # sort is stable, so GraspGenX confidence keeps breaking ties within
        # each tier. Done BEFORE the diversity pick so the MAX_GRASP_ATTEMPTS
        # slots go to the best tier available.
        tier_of = {i: _grasp_priority_tier(resp.grasps[i].pose, arm) for i in cand}
        cand.sort(key=lambda i: tier_of[i])
        hist = {t: 0 for t in range(1, 8)}
        for i in cand:
            hist[tier_of[i]] += 1
        self.get_logger().info(
            f'grasp: priority tiers over {len(cand)} kept candidate(s) [{arm}]: '
            f'fwd/diag/center down={hist[1]}/{hist[2]}/{hist[3]}, '
            f'flat={hist[4]}/{hist[5]}/{hist[6]}, other={hist[7]}')

        # -- diversity pick ---------------------------------------------------
        # Pick the MAX_GRASP_ATTEMPTS grasps to actually try: walk the
        # tier-ordered list and keep a candidate only when it isn't a
        # near-duplicate of one already kept (GRASP_DIVERSITY_* /
        # _select_diverse). This spreads the IK fallback attempts across
        # genuinely different grasps instead of five near-identical top-scored
        # ones.
        all_poses = [g.pose for g in resp.grasps]
        cand = _select_diverse(cand, all_poses, MAX_GRASP_ATTEMPTS,
                               GRASP_DIVERSITY_LIN_M,
                               math.radians(GRASP_DIVERSITY_ANG_DEG))
        self.get_logger().info(
            f'grasp: selected {len(cand)} diverse candidate(s) for {obj!r} '
            f'(>= {GRASP_DIVERSITY_LIN_M * 100:.0f}cm or '
            f'{GRASP_DIVERSITY_ANG_DEG:.0f}deg apart) from {len(resp.grasps)} ranked')
        return _Candidates(
            grasps=[resp.grasps[i] for i in cand],
            scores=[float(resp.scores[i]) for i in cand],
            tiers=[tier_of[i] for i in cand],
            total=len(resp.grasps),
            gripper_width=float(resp.gripper_width)), None

    # ---------------------------------------------------------------- execution
    def _snapshot_targets(self, cands, centroid_p):
        """Freeze every candidate's grasp + pre-grasp pose (and the object
        centroid) from the CURRENT (static) pelvis frame into the world frame,
        ONCE, up front. World-anchored, the targets stay correct as the pelvis
        drifts during the arm motions: servo_frame_to_world re-resolves them
        into the live pelvis frame each iteration. If the world TF is
        unavailable (navigation/odom not running), `have_world` is False and
        the raw pelvis poses are driven with no drift compensation."""
        # Shift each raw grasp along its approach axis by GRASP_OFFSET, then
        # back off APPROACH_DIST from the shifted grasp for the pre-grasp.
        grasps_p = [get_approach_pose(g.pose, approach_dist=GRASP_OFFSET)
                    for g in cands.grasps]
        approaches_p = [get_approach_pose(g, approach_dist=-APPROACH_DIST)
                        for g in grasps_p]
        grasps_w = [self._transform_pose(g, 'pelvis', WORLD_FRAME)
                    for g in grasps_p]
        approaches_w = [self._transform_pose(a, 'pelvis', WORLD_FRAME)
                        for a in approaches_p]
        have_world = all(p is not None for p in grasps_w + approaches_w)
        if not have_world:
            self.get_logger().warn(
                f'grasp: {WORLD_FRAME} TF unavailable; pelvis-drift servoing OFF, '
                'driving raw pelvis poses (start navigation/FAST-LIO to enable)')
        # Snapshot the object centroid into the world frame from the SAME
        # static pelvis moment as the grasp poses, so the GraspOutcome's pose
        # and centroid stay mutually consistent for pick_place's offset math.
        centroid_w = (self._transform_pose(_pose_at(centroid_p), 'pelvis',
                                           WORLD_FRAME) if have_world else None)
        return _Targets(frame=cands.grasps[0].header.frame_id,
                        grasps_p=grasps_p, approaches_p=approaches_p,
                        grasps_w=grasps_w, approaches_w=approaches_w,
                        have_world=have_world, centroid_w=centroid_w)

    def _servo_to_first_reachable(self, run, gh, obj, arm, cands, targets):
        """Walk the candidates strictly tier-major (1..7), highest GraspGenX
        confidence first within a tier, servoing GRASP_FRAMES[arm] to each
        pre-grasp until one is reachable (the top candidate can be
        IK-unreachable, hence the fallback). The approach moves run on the
        generous APPROACH_DURATION_SEC timeout so a long transit into the
        workspace is never cut off mid-move. Dead candidates still fail fast:
        these are PLANNED moves, and a rejected plan (z-floor, unreachable
        goal) returns well before the timeout.
        Returns (index, None) of the committed candidate, or
        (None, abort-reason) on cancel/timeout or when every candidate is
        unreachable."""
        order = sorted(range(len(cands.grasps)),
                       key=lambda i: (cands.tiers[i], -cands.scores[i]))
        width_mm = cands.gripper_width * 1000.0
        for i in order:
            self._publish_target_tf(targets, targets.approaches_w[i],
                                    targets.approaches_p[i])
            self.get_logger().info(
                f'grasp {i} for {obj!r}: tier {cands.tiers[i]}, '
                f'score {cands.scores[i]:.2f}, width {width_mm:.1f}mm')
            if self.servo_frame_to_world(
                    GRASP_FRAMES[arm],
                    targets.approaches_w[i] if targets.have_world else None,
                    targets.approaches_p[i], outer_gh=gh,
                    duration_sec=APPROACH_DURATION_SEC,
                    max_iter=SERVO_MAX_ITER,
                    lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL):
                return i, None
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return None, 'canceled or timed out during approach'
            self.get_logger().warn(
                f'grasp {i} pre-grasp unreachable; trying next-ranked grasp')
        return None, (f'no reachable grasp for {obj!r} '
                      f'(tried {len(order)} of {cands.total})')

    def _visual_servo_refine(self, run, gh, arm, obj, targets, idx):
        """Closed-loop hand-camera alignment at the pre-grasp (goal.visual_servo).

        Drives GRASP_FRAMES[arm] in PURE TRANSLATION until `obj` sits at
        (VISUAL_SERVO_TARGET_XY_M, VISUAL_SERVO_DEPTH_M) in the hand camera's
        optical frame — a fixed lateral offset from the boresight, at the target
        range. Because the camera looks straight down the approach axis, the
        depth part is the same as "D metres ahead along the approach". The PLANNED
        orientation is re-commanded every iteration and never adjusted, so the
        approach the candidate was chosen for survives the loop and the wrist
        can't drift somewhere unreachable.

        One iteration: take a hand-camera DetectionBundle newer than the last
        move, back-project the target's box to a 3-D point in the camera optical
        frame (bundle_detection_point — the bundle carries its own aligned depth
        + intrinsics), and step the gripper by GAIN x (measured - target),
        rotated into the pelvis frame and capped at MAX_STEP. Requiring a FRESH
        frame matters: re-using the frame that motivated the last move would
        apply the same correction twice and overshoot.

        Two situations are RECOVERABLE rather than fatal, and both are answered
        by backing the hand straight up (pelvis +Z) by
        VISUAL_SERVO_LOST_RETRACT_M and carrying on, up to
        VISUAL_SERVO_MAX_RETRACTS times between them: losing the object for
        VISUAL_SERVO_DETECT_WAIT_SEC, and finding it closer than
        VISUAL_SERVO_MIN_RANGE_M (where the depth can no longer be trusted).

        Returns (pose, None) with the pelvis-frame gripper Pose it converged at —
        the caller closes the gripper THERE, with no contact descent — or
        (None, reason) when it could not converge. A failure FAILS THE SKILL:
        having asked for visual servoing, silently completing an unaligned
        open-loop grasp is worse than a clear abort, especially since the
        fallback would drive to the original planned target and discard every
        correction the loop had already made."""
        frame = GRASP_FRAMES[arm]
        planned_q = targets.grasps_p[idx].orientation
        # Fixed lateral offset from the boresight at the target range — no TF
        # lookup, no fitted constants (see the SETPOINT note above).
        target = np.array([VISUAL_SERVO_TARGET_XY_M[0],
                           VISUAL_SERVO_TARGET_XY_M[1],
                           VISUAL_SERVO_DEPTH_M])
        deadline = time.monotonic() + VISUAL_SERVO_TIMEOUT_SEC
        last_seen = time.monotonic()
        retracts = 0
        best_err, stalled = None, 0        # stall detection, see VISUAL_SERVO_STALL_*
        too_close_hits = 0                 # consecutive sub-MIN_RANGE readings
        last_z = pred_z = None             # depth plausibility band, see below
        # Only frames captured after this instant count, so the first measurement
        # describes the arm where it actually is, not mid-approach.
        after = _stamp_tuple(self.get_clock().now().to_msg())
        self.get_logger().info(
            f'grasp: visual servo for {obj!r} on the {arm} hand camera — setpoint '
            f'({target[0] * 1000:+.0f}, {target[1] * 1000:+.0f}, '
            f'{target[2] * 1000:.0f})mm optical '
            f'(tol {VISUAL_SERVO_X_TOL_M * 1000:.0f}mm across the jaws / '
            f'{VISUAL_SERVO_Y_TOL_M * 1000:.0f}mm along them / '
            f'{VISUAL_SERVO_RANGE_TOL_M * 1000:.0f}mm range)')
        it = 0
        while True:
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return None, 'canceled or timed out during visual servo'
            if time.monotonic() >= deadline:
                return None, (
                    f'visual servo did not converge on {obj!r} within '
                    f'{VISUAL_SERVO_TIMEOUT_SEC:.0f}s ({it} iteration(s), tol '
                    f'{VISUAL_SERVO_X_TOL_M * 1000:.0f}/'
                    f'{VISUAL_SERVO_Y_TOL_M * 1000:.0f}mm X/Y)')
            p_cam, cam_frame, stamp, bundle = self._servo_measure(arm, obj, after)
            # Plausibility gate: a reading whose DEPTH lands outside the band
            # around what the last correction predicted is not this object — a
            # finger, a shadow, the surface behind it — and acting on it would
            # command a large bogus correction. Drop the frame and wait for the
            # next one; if they are all implausible the lost-sight path below
            # eventually fires, which is the right outcome.
            if p_cam is not None and pred_z is not None:
                lo = min(last_z, pred_z) - VISUAL_SERVO_RANGE_BAND_M
                hi = max(last_z, pred_z) + VISUAL_SERVO_RANGE_BAND_M
                if not lo <= float(p_cam[2]) <= hi:
                    self.get_logger().warn(
                        f'grasp: visual servo ignoring implausible depth '
                        f'{p_cam[2] * 1000:.0f}mm (expected '
                        f'{lo * 1000:.0f}-{hi * 1000:.0f}mm)',
                        throttle_duration_sec=2.0)
                    after = stamp          # consumed; do not re-test this frame
                    self.get_clock().sleep_for(RclpyDuration(seconds=0.1))
                    continue
            # Two ways the hand ends up somewhere it cannot servo from, both
            # answered the same way — back straight up and look again:
            #   LOST      nothing usable for DETECT_WAIT (out of frame, occluded
            #             by the fingers, or depth gone entirely);
            #   TOO CLOSE inside MIN_RANGE on STRIKES consecutive frames, where
            #             the depth can no longer be trusted. One frame is never
            #             enough — a single bad sample must not command a retreat.
            if p_cam is not None and float(p_cam[2]) < VISUAL_SERVO_MIN_RANGE_M:
                too_close_hits += 1
            else:
                too_close_hits = 0
            lost = (p_cam is None
                    and time.monotonic() - last_seen > VISUAL_SERVO_DETECT_WAIT_SEC)
            too_close = too_close_hits >= VISUAL_SERVO_TOO_CLOSE_STRIKES
            if lost or too_close:
                why = (f'lost sight of {obj!r} in the {arm} hand camera for '
                       f'{VISUAL_SERVO_DETECT_WAIT_SEC:.0f}s' if lost else
                       f'{obj!r} at {p_cam[2] * 1000:.0f}mm is inside the '
                       f'{VISUAL_SERVO_MIN_RANGE_M * 1000:.0f}mm minimum range')
                if retracts >= VISUAL_SERVO_MAX_RETRACTS:
                    return None, (
                        f'{why}; still not servo-able after {retracts} retract(s) of '
                        f'{VISUAL_SERVO_LOST_RETRACT_M * 100:.0f}cm')
                retracts += 1
                self.get_logger().warn(
                    f'grasp: {why} — backing off '
                    f'{VISUAL_SERVO_LOST_RETRACT_M * 100:.0f}cm up (pelvis +Z) '
                    f'({retracts}/{VISUAL_SERVO_MAX_RETRACTS})')
                cur = self._frame_pose_in_pelvis(frame)
                if cur is None:
                    return None, (f'{why} and TF {frame!r} -> pelvis failed, so the '
                                  'back-off pose is unknown')
                up = Pose()
                up.position.x = cur.position.x
                up.position.y = cur.position.y
                up.position.z = cur.position.z + VISUAL_SERVO_LOST_RETRACT_M
                up.orientation = planned_q
                self.move_frame_to(frame, up, outer_gh=gh,
                                   duration_sec=VISUAL_SERVO_MOVE_SEC, do_plan=False)
                # A back-off is the one place `after` becomes a CLOCK instant
                # rather than a consumed-frame stamp: frames captured before it
                # describe the old vantage point outright. The detect wait
                # restarts too, so another back-off needs its own full wait.
                # Stall counters and the depth prediction reset as well — the
                # hand is somewhere new, so neither the old errors nor the old
                # depth say anything about what should happen next (and a 5 cm
                # jump would fail the plausibility band by construction).
                after = _stamp_tuple(self.get_clock().now().to_msg())
                last_seen = time.monotonic()
                best_err, stalled = None, 0
                last_z = pred_z = None
                too_close_hits = 0
                continue
            if p_cam is None:
                self.get_clock().sleep_for(RclpyDuration(seconds=0.1))
                continue
            last_seen = time.monotonic()
            err = p_cam - target
            ex, ey = float(abs(err[0])), float(abs(err[1]))
            lat = float(math.hypot(err[0], err[1]))
            rng = float(abs(err[2]))
            # Move the camera BY the error: the object sits at p_cam and we want
            # it at `target`, and translating the camera by d shifts the object's
            # camera coordinates by -d. Gain-scaled and step-capped so one bad
            # depth sample can't fling the hand across the workspace. Computed
            # HERE, before the convergence check, only so the debug image can
            # draw the correction this iteration would apply.
            step = VISUAL_SERVO_GAIN * err
            norm = float(np.linalg.norm(step))
            if norm > VISUAL_SERVO_MAX_STEP_M:
                step *= VISUAL_SERVO_MAX_STEP_M / norm
                self.get_logger().warn(
                    f'grasp: visual servo step capped at '
                    f'{VISUAL_SERVO_MAX_STEP_M * 100:.0f}cm (wanted {norm * 100:.1f}cm)')
            # Stamped with the frame's CAPTURE time, not now(), so plots line
            # up with the camera stream it was measured from.
            self.servo_error_pub.publish(_vec3_stamped(
                Time(sec=stamp[0], nanosec=stamp[1]), cam_frame, err))
            self._publish_servo_image(bundle, p_cam, target, step)
            self.get_logger().info(
                f'grasp: visual servo iter {it}: object at ({p_cam[0] * 1000:+.0f}, '
                f'{p_cam[1] * 1000:+.0f}, {p_cam[2] * 1000:.0f})mm in {cam_frame} — '
                f'err X {err[0] * 1000:+.1f}mm (tol {VISUAL_SERVO_X_TOL_M * 1000:.0f}) '
                f'Y {err[1] * 1000:+.1f}mm (tol {VISUAL_SERVO_Y_TOL_M * 1000:.0f}) '
                f'range {err[2] * 1000:+.1f}mm (tol {VISUAL_SERVO_RANGE_TOL_M * 1000:.0f})')
            if (ex <= VISUAL_SERVO_X_TOL_M and ey <= VISUAL_SERVO_Y_TOL_M
                    and rng <= VISUAL_SERVO_RANGE_TOL_M):
                pose = self._frame_pose_in_pelvis(frame)
                if pose is None:
                    return None, (f'visual servo converged but TF {frame!r} -> pelvis '
                                  'failed, so the close pose is unknown')
                pose.orientation = planned_q
                self.get_logger().info(
                    f'grasp: visual servo converged for {obj!r} in {it} iteration(s)')
                return pose, None
            # Not converged — is it still getting closer? A run that stops
            # improving will not start again on its own (see VISUAL_SERVO_STALL_*),
            # so end it here with the residual rather than at the timeout.
            err_norm = float(np.linalg.norm(err))
            if best_err is None or err_norm < best_err - VISUAL_SERVO_STALL_EPS_M:
                best_err, stalled = err_norm, 0
            else:
                stalled += 1
                if stalled >= VISUAL_SERVO_STALL_ITERS:
                    # Name the axis that is actually out of tolerance — X (across
                    # the jaws) is a real problem, Y (along them) usually is not.
                    out = ', '.join(
                        f'{ax} {v * 1000:+.1f}mm (tol {t * 1000:.0f})'
                        for ax, v, t in (('X', err[0], VISUAL_SERVO_X_TOL_M),
                                         ('Y', err[1], VISUAL_SERVO_Y_TOL_M),
                                         ('range', err[2], VISUAL_SERVO_RANGE_TOL_M))
                        if abs(v) > t)
                    return None, (
                        f'visual servo stalled on {obj!r}: no improvement over '
                        f'{stalled} iterations, still out of tolerance on {out} — '
                        'the tolerance is below what this arm can resolve, or the '
                        'setpoint is off')
            delta = self._rotate_into_pelvis(step, cam_frame)
            cur = self._frame_pose_in_pelvis(frame)
            if delta is None or cur is None:
                return None, (f'visual servo TF lookup for {cam_frame!r}/{frame!r} '
                              'failed mid-loop')
            goal_pose = Pose()
            goal_pose.position.x = cur.position.x + float(delta[0])
            goal_pose.position.y = cur.position.y + float(delta[1])
            goal_pose.position.z = cur.position.z + float(delta[2])
            goal_pose.orientation = planned_q
            # Servo targets are pelvis-frame by construction, so publish
            # TARGET_FRAME directly rather than through _publish_target_tf
            # (which would want a world snapshot that doesn't exist here).
            self.publish_tf(_approach_target_tf(
                targets.frame, goal_pose, self.get_clock().now().to_msg()))
            # Undershoot is fine and expected — the next measurement simply sees
            # the error that is left, so a failed/partial move is not fatal here.
            self.move_frame_to(
                frame, goal_pose, outer_gh=gh,
                duration_sec=VISUAL_SERVO_MOVE_SEC,
                do_plan=False, slow_mode=True)
            # Executed move = TF pose after the move minus before (`cur` above,
            # measured just before commanding). Its residual vs the commanded
            # correction `delta` equals post - goal_pose: the tracking
            # shortfall. Debug-only, so a failed TF lookup just skips the pair.
            post = self._frame_pose_in_pelvis(frame)
            if post is not None:
                executed = np.array([post.position.x - cur.position.x,
                                     post.position.y - cur.position.y,
                                     post.position.z - cur.position.z])
                residual = executed - delta
                self.get_logger().info(
                    f'grasp: visual servo iter {it}: '
                    f'commanded ({delta[0] * 1000:+.1f}, {delta[1] * 1000:+.1f}, '
                    f'{delta[2] * 1000:+.1f})mm, '
                    f'executed ({executed[0] * 1000:+.1f}, {executed[1] * 1000:+.1f}, '
                    f'{executed[2] * 1000:+.1f})mm, '
                    f'residual ({residual[0] * 1000:+.1f}, {residual[1] * 1000:+.1f}, '
                    f'{residual[2] * 1000:+.1f})mm (pelvis frame)')
                now = self.get_clock().now().to_msg()
                self.servo_move_pub.publish(
                    _vec3_stamped(now, 'pelvis', executed))
                self.servo_move_residual_pub.publish(
                    _vec3_stamped(now, 'pelvis', residual))
            # Where the depth should land next: moving the camera by +step along
            # its own axes shifts the object's camera coordinates by -step, so a
            # fully-executed move lands at z - step[2] and an ignored one stays
            # at z. The band between those two (plus slack) is what the next
            # measurement is checked against.
            last_z, pred_z = float(p_cam[2]), float(p_cam[2]) - float(step[2])
            # Require a frame NEWER than the one just consumed — not one captured
            # after the move (see _servo_measure).
            after = stamp
            # Give the camera/detector a beat to publish a frame from the new
            # vantage point before measuring again.
            self.get_clock().sleep_for(
                RclpyDuration(seconds=VISUAL_SERVO_POST_MOVE_SLEEP_SEC))
            it += 1

    def _publish_servo_image(self, bundle, p_cam, target, step=None):
        """Draw one visual-servo frame and publish it for RViz: the measured
        object as a RED dot, the setpoint as a GREEN box whose half-extents are
        the X/Y tolerances, both PROJECTED into pixel space through the bundle's
        own intrinsics.

        The tolerance is drawn as a RECTANGLE because that is exactly what the
        convergence test is: per axis, |dx| <= X_TOL AND |dy| <= Y_TOL. An
        ellipse would be inscribed in it and would under-report convergence — a
        dot in a corner (say 1.5 mm X, 4.0 mm Y against 2/5 mm tolerances) passes
        the real test while sitting outside the ellipse. Range is the criterion
        this view cannot show at all, so it goes in the caption.

        The box is drawn at the SETPOINT's depth, so its pixel size is fixed
        regardless of where the object currently is.

        Best-effort throughout: visualisation must never break a grasp."""
        try:
            info = bundle.camera_info
            if not bundle.rgb_image.data or not info.width:
                return
            img = cv2.imdecode(
                np.frombuffer(bytes(bundle.rgb_image.data), dtype=np.uint8),
                cv2.IMREAD_COLOR)
            if img is None:
                return
            fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]

            def project(pt):
                z = float(pt[2])
                if z <= 1e-6:
                    return None
                return (int(round(fx * float(pt[0]) / z + cx)),
                        int(round(fy * float(pt[1]) / z + cy)))

            tgt_px, obj_px = project(target), project(p_cam)
            if tgt_px is not None:
                # Tolerances are metres at the setpoint depth; convert to pixels
                # with the same pinhole scale (u = fx*X/Z) that projected it.
                ax = max(2, int(round(fx * VISUAL_SERVO_X_TOL_M / target[2])))
                ay = max(2, int(round(fy * VISUAL_SERVO_Y_TOL_M / target[2])))
                cv2.rectangle(img, (tgt_px[0] - ax, tgt_px[1] - ay),
                              (tgt_px[0] + ax, tgt_px[1] + ay), (0, 255, 0), 2)
                cv2.drawMarker(img, tgt_px, (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
            if obj_px is not None:
                cv2.circle(img, obj_px, 6, (0, 0, 255), -1)
            if tgt_px is not None and obj_px is not None:
                cv2.line(img, obj_px, tgt_px, (0, 200, 255), 1)
            # Commanded correction, drawn where it is legible: as the object's
            # PREDICTED next position. Translating the camera by +step shifts the
            # object's camera coordinates by -step, so the dot should land at
            # p_cam - step. Projecting that instead of drawing the raw vector
            # gets the perspective right for free — in particular a pure-range
            # correction barely moves the dot, which is the truth.
            cmd_color = _step_color(step[2] if step is not None else 0.0)
            if obj_px is not None and step is not None:
                nxt_px = project(np.asarray(p_cam) - np.asarray(step))
                if nxt_px is not None and nxt_px != obj_px:
                    cv2.arrowedLine(img, obj_px, nxt_px, cmd_color, 2,
                                    tipLength=0.3)
            if tgt_px is not None and obj_px is not None:
                err = np.asarray(p_cam) - np.asarray(target)
                cv2.putText(
                    img, f'err X{err[0] * 1000:+.1f} Y{err[1] * 1000:+.1f} '
                         f'Z{err[2] * 1000:+.1f} mm',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if step is not None:
                    cv2.putText(
                        img, f'cmd X{step[0] * 1000:+.1f} Y{step[1] * 1000:+.1f} '
                             f'Z{step[2] * 1000:+.1f} mm',
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cmd_color, 2)

            img = np.ascontiguousarray(img)
            msg = Image()
            msg.header = bundle.rgb_image.header
            msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
            msg.encoding = 'bgr8'
            msg.is_bigendian = 0
            msg.step = int(img.shape[1] * 3)
            msg.data = img.tobytes()
            self.servo_image_pub.publish(msg)
        except Exception as e:                       # noqa: BLE001 - viz only
            self.get_logger().warn(f'servo image publish failed: {e}',
                                   throttle_duration_sec=5.0)

    def _servo_measure(self, arm, obj, after):
        """One visual-servo measurement: the position of `obj` in the `arm` hand
        camera's optical frame, as (point, frame_id, stamp, bundle), or a tuple
        of Nones. The bundle comes back so the caller can draw the servo
        overlay on the EXACT frame the measurement came from.

        Only accepts a bundle stamped strictly after `after` (a (sec, nanosec)
        tuple). The caller sets `after` to the stamp of the frame it last USED,
        not to a wall-clock instant — that still makes it impossible to apply the
        same frame's correction twice (which would overshoot), while no longer
        demanding a frame captured after the move finished. The stricter rule
        starved the loop whenever yolo_server's inference latency exceeded the
        gap between moves: every arriving bundle carried a capture stamp from
        before the move ended, all were rejected, and after DETECT_WAIT the loop
        declared a plainly visible object lost. The caller does still bump
        `after` to a clock instant after a BACK-OFF, where pre-move frames really
        are describing the wrong vantage point.

        Among detections of the class, picks the one CLOSEST TO THE BORESIGHT
        rather than the most confident: the loop must stay locked on the instance
        it is aligning to, not hop to a neighbouring part that happens to score
        higher on one frame."""
        bundle = self.latest_arm_detections(arm)
        if bundle is None or not bundle.detections:
            return None, None, None, None
        stamp = _stamp_tuple(bundle.rgb_image.header.stamp)
        if stamp <= after:
            return None, None, None, None  # already consumed, or pre-dates a back-off
        key = obj.strip().lower()
        best_p, best_lat = None, None
        for d in bundle.detections:
            if d.cls.strip().lower() != key:
                continue
            p = self.bundle_detection_point(bundle, d,
                                            min_depth=VISUAL_SERVO_MIN_DEPTH_M)
            if p is None:                       # no usable depth under this box
                continue
            lat = math.hypot(float(p[0]), float(p[1]))
            if best_lat is None or lat < best_lat:
                best_p, best_lat = p, lat
        if best_p is None:
            return None, None, None, None
        return best_p, self.bundle_camera_frame(bundle), stamp, bundle

    def _retreat_after_failure(self, arm, gh):
        """Lift the hand FAILURE_RETRACT_M straight up (pelvis +Z) before a skill
        reports a failure, so the arm does not stay parked in the workspace.

        Shared by the grasp skill and pick_place. Keeps the CURRENT orientation
        (read from TF) rather than any planned one — a failure can happen at any
        stage, including ones where no grasp orientation was ever settled, and
        re-orienting a stuck arm is how a bad situation gets worse.

        Entirely best-effort: no arm chosen yet, TF missing, or the move itself
        failing all just warn. The skill is already failing; the retreat must
        never replace the real reason with a second one."""
        if arm is None:
            return
        frame = GRASP_FRAMES[arm]
        cur = self._frame_pose_in_pelvis(frame)
        if cur is None:
            self.get_logger().warn(
                f'retreat: TF {frame!r} -> pelvis unavailable; leaving the arm in place')
            return
        up = copy.deepcopy(cur)
        up.position.z += FAILURE_RETRACT_M
        self.get_logger().info(
            f'retreat: lifting {frame} {FAILURE_RETRACT_M * 100:.0f}cm (pelvis +Z) '
            'before reporting the failure')
        if not self.move_frame_to(frame, up, outer_gh=gh,
                                  duration_sec=FAILURE_RETRACT_SEC * SLOW_MODE_TIME_SCALE,
                                  do_plan=False, slow_mode=True):
            self.get_logger().warn('retreat: lift incomplete')

    def _frame_pose_in_pelvis(self, frame):
        """Current pose of URDF `frame` in the pelvis frame, or None."""
        return self._transform_pose(_pose_at((0.0, 0.0, 0.0)), frame, 'pelvis')

    def _rotate_into_pelvis(self, vec, src_frame):
        """Rotate the 3-vector `vec` from `src_frame` into the pelvis frame (a
        pure direction change — no translation), or None if TF is unavailable.
        Done by transforming the origin and the vector's tip and differencing, so
        it reuses the same TF path as every other lookup in the skill."""
        o = self._transform_pose(_pose_at((0.0, 0.0, 0.0)), src_frame, 'pelvis')
        t = self._transform_pose(_pose_at(vec), src_frame, 'pelvis')
        if o is None or t is None:
            return None
        return np.array([t.position.x - o.position.x,
                         t.position.y - o.position.y,
                         t.position.z - o.position.z])

    def _drive_to_contact(self, gh, arm, targets, idx):
        """Servo from the committed pre-grasp to the contact target — the
        committed grasp pose backed off CONTACT_OFFSET along its own approach
        axis. The backoff is rigid geometry along the pose's local +Z, so
        applying it to the world SNAPSHOT keeps the detect-time anchoring (no
        re-transform through the now-drifted pelvis). Best-effort by design:
        we've already committed to a reachable grasp, so a contact pose that
        never reaches tolerance is still worth closing on —
        servo_frame_to_world logs the residual world error and we deliberately
        don't abort on non-convergence. do_plan=False because this is the
        short, committed pre-grasp -> contact move straight along the approach
        axis (collision-aware planning would fight the intended approach into
        the object). Returns (contact_p, contact_w) for the outcome record."""
        contact_p = get_approach_pose(targets.grasps_p[idx],
                                      approach_dist=-CONTACT_OFFSET)
        contact_w = (get_approach_pose(targets.grasps_w[idx],
                                       approach_dist=-CONTACT_OFFSET)
                     if targets.have_world else None)
        self._publish_target_tf(targets, contact_w, contact_p)
        self.servo_frame_to_world(
            GRASP_FRAMES[arm], contact_w if targets.have_world else None,
            contact_p, outer_gh=gh,
            duration_sec=SERVO_DURATION_SEC, max_iter=SERVO_MAX_ITER,
            lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL, do_plan=False)
        return contact_p, contact_w

    def _publish_target_tf(self, targets, pose_w, pose_p):
        """Broadcast TARGET_FRAME at the given target pose — the stable world
        snapshot when drift-compensated servoing is on, else the raw pelvis
        pose. publish_tf keeps re-broadcasting it so the frame stays alive in
        RViz instead of expiring between sends."""
        parent, pose = ((WORLD_FRAME, pose_w) if targets.have_world
                        else (targets.frame, pose_p))
        self.publish_tf(_approach_target_tf(
            parent, pose, self.get_clock().now().to_msg()))

    # ------------------------------------------------------------ arm selection
    def _closest_arm(self, centroid_p):
        """Arm whose gripper is currently nearest the object centroid
        (`centroid_p`, xyz in the pelvis frame): express the centroid in each
        arm's GraspGenX gripper-base frame via live TF and compare distances.
        When either hand TF is unavailable, fall back to the side of the pelvis
        midline the object sits on (+Y = left)."""
        dists = {}
        for arm, frame in GRASP_FRAMES.items():
            p = self._transform_pose(_pose_at(centroid_p), 'pelvis', frame)
            if p is not None:
                dists[arm] = math.sqrt(p.position.x ** 2 + p.position.y ** 2
                                       + p.position.z ** 2)
        if len(dists) == len(GRASP_FRAMES):
            arm = min(dists, key=dists.get)
            self.get_logger().info(
                'grasp: auto-selected %s arm (hand-to-object %s)' % (arm,
                ', '.join(f'{a} {d:.3f}m' for a, d in sorted(dists.items()))))
        else:
            arm = 'left' if centroid_p[1] > 0.0 else 'right'
            self.get_logger().warn(
                f'grasp: hand TF unavailable for auto arm choice; picked {arm} '
                f'from centroid side (y={centroid_p[1]:+.3f}m)')
        return arm

    # ------------------------------------------------------------ visualization
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
        isn't mistaken for the gripper being short. Best-effort — viz must never
        break the grasp."""
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
                contact = base + GRIPPER_BASE_TO_CONTACT_M * T[:3, 2]  # +Z -> TCP
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

    # ------------------------------------- battery parts: YOLO bounding box --
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


# ============================ grasp-pose geometry ============================
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


def _top_down_pose(centroid):
    """GraspGenX gripper-base Pose (pelvis frame) for the steep top-down grasp of
    the object at `centroid` (x, y, z array-like): the TIER-1 orientation
    (forward azimuth, fingers closing along pelvis +Y, wrist as level as the
    pitch allows) pitched TOP_DOWN_PITCH_DEG below horizontal instead of the
    tier's 20-45 deg band. The base sits GRIPPER_BASE_TO_CONTACT_M BACK ALONG
    the approach axis from the centroid, so the finger CONTACT point (base +
    that length along +Z) lands exactly on the centroid.

    At TOP_DOWN_PITCH_DEG = 90 this collapses to the original straight-down
    grasp (+Z = pelvis -Z, +Y = pelvis +X, base directly above the centroid)."""
    pitch = math.radians(TOP_DOWN_PITCH_DEG)
    # +Z (approach): pitched `pitch` below horizontal at zero azimuth, so it
    # stays in the pelvis XZ plane (straight ahead, angled down).
    z_axis = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    # +X (fingers): pelvis +Y, the tier-1 finger preference. Perpendicular to
    # the approach for any pitch, since the approach has no +Y component.
    x_axis = np.array([0.0, 1.0, 0.0])
    # +Y completes the right-handed frame; it comes out at cos(pitch) from
    # pelvis up — exactly the level-wrist that _roll_to_yup would pick.
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4)
    T[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    T[:3, 3] = np.asarray(centroid, dtype=float) - GRIPPER_BASE_TO_CONTACT_M * z_axis
    return matrix_to_pose(T)


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


def _stamp_tuple(stamp):
    """(sec, nanosec) of a builtin_interfaces/Time, for ordering message stamps
    without pulling in rclpy's Time arithmetic (and without caring whether the
    clock is sim or wall — both sides of every comparison share one clock)."""
    return (stamp.sec, stamp.nanosec)


def _step_color(dz):
    """BGR for the servo image's commanded-step arrow, coding its DEPTH
    component: RED as the step drives IN along the boresight (+Z), GREEN as it
    backs OFF (-Z). At the top-down approach the boresight points down, so red
    reads as "going down onto the part" and green as "coming back up". Scaled
    over +/- VISUAL_SERVO_MAX_STEP_M, so saturated colour = the biggest step the
    loop is allowed to command; a step with no depth component comes out olive."""
    t = (float(np.clip(dz / VISUAL_SERVO_MAX_STEP_M, -1.0, 1.0)) + 1.0) / 2.0
    return (0, int(round(255 * (1.0 - t))), int(round(255 * t)))


def _vec3_stamped(stamp, frame_id, v):
    """A geometry_msgs/Vector3Stamped from a 3-sequence, for the visual-servo
    debug topics (/skill/grasp/servo_*)."""
    msg = Vector3Stamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.vector.x = float(v[0])
    msg.vector.y = float(v[1])
    msg.vector.z = float(v[2])
    return msg


def _pose_at(point):
    """Identity-orientation Pose at `point` (x, y, z array-like) — for running a
    bare 3-D point through the Pose-based TF helpers."""
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (
        float(point[0]), float(point[1]), float(point[2]))
    pose.orientation.w = 1.0
    return pose


# ========================== candidate ranking helpers ========================
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
              min_downward_pitch_deg goal gate: asin(-az) of the approach;
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


# ============================== debug TF helper ==============================
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
