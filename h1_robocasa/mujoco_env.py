"""Elastic-band balance tether used by the sim runner (h12_mujoco.sim_loop).

Two modes, selected at construction:
  - hang band (default, the legacy behavior): a spring-damper that pulls the
    torso toward a fixed anchor above its spawn — holds the robot but also lets
    it HANG (it can pull down/sideways), and cannot constrain tilt.
  - gantry harness (tension_only + posture gains + release_fade): the twin's
    validated handoff rig (h1_mujoco mujoco_env.py) — a TAUT catch-only strap
    that can pull toward the anchor but never push, plus a "steadying hands"
    posture assist (world-frame upright spring + angular damping) applied by
    the sim loop, released by EASING the support off over a quadratic fade
    instead of an instant cut (a cut hands the policy whatever load the strap
    still carried as a step disturbance).

Toggle with SPACE in the viewer or the /elastic_band/toggle service; with a
fade configured the first toggle requests the eased release, a second cuts
instantly.
"""
import mujoco
import numpy as np


class ElasticBand:
    def __init__(self, point=np.array([0, 0, 3]), length=0, stiffness=200, damping=100,
                 tension_only=False, posture_kp=0.0, posture_kd=0.0, release_fade=0.0):
        self.point = point.astype(np.float64)
        self.length = length
        self.stiffness = stiffness
        self.damping = damping
        self.tension_only = tension_only   # a strap can PULL toward the anchor, never push
        self.posture_kp = posture_kp       # upright tilt spring (N*m/rad) — "steadying hands"
        self.posture_kd = posture_kd       # angular damping (N*m*s/rad)
        self.release_fade = release_fade   # >0: release EASES support off over this many sim-s
        self.release_requested = False
        self._fade_t0 = None
        self.enabled = True

    def evaluate_force(self, x, v):
        # displacement / direction from the attached body to the anchor
        displacement = np.linalg.norm(x - self.point)
        if displacement < 1e-9:
            return np.zeros(3)
        direction = (x - self.point) / displacement
        # tangential velocity along the band, for damping
        v_tan = np.dot(v, direction)
        # spring-damper (mag < 0 = pull toward anchor = rope tension)
        mag = -self.stiffness * (displacement - self.length) - self.damping * v_tan
        if self.tension_only and mag > 0.0:
            mag = 0.0   # a slack strap exerts nothing (catch-only)
        return mag * direction

    def fade_scale(self, sim_time):
        """Support scale in [0, 1] while a fade release is in progress; flips
        enabled=False when the fade completes. Quadratic ease-out: most of the
        fade is spent at LOW support — while the strap carries weight the
        planner cannot pre-center the body (its lean commands fight an
        invisible rope), so the re-centering has to happen in the under-loaded
        tail (twin bench 2026-06-12: a linear fade left ~1.7 s of that tail and
        toppled exactly at fade end)."""
        if not self.release_requested or self.release_fade <= 0.0:
            return 1.0
        if self._fade_t0 is None:
            self._fade_t0 = sim_time
        s = 1.0 - (sim_time - self._fade_t0) / self.release_fade
        if s <= 0.0:
            self.enabled = False
            return 0.0
        return s * s

    def request_release(self):
        """Begin the eased release (or cut instantly when no fade is set)."""
        if self.release_fade > 0.0:
            self.release_requested = True
        else:
            self.enabled = False

    def toggle(self) -> bool:
        # harness mode (fade configured): first toggle EASES the support off; a
        # second toggle during the fade cuts it instantly.
        if self.release_fade > 0.0 and self.enabled and not self.release_requested:
            self.release_requested = True
            print(f"Harness release: easing support off over {self.release_fade:.0f}s "
                  "(toggle again = instant cut)")
            return True
        self.enabled = not self.enabled
        if self.enabled:   # re-enable resets a previous fade
            self.release_requested = False
            self._fade_t0 = None
        print(f"Elastic band {'enabled' if self.enabled else 'disabled'}")
        return self.enabled

    def key_callback(self, key):
        glfw = mujoco.glfw.glfw

        if key == glfw.KEY_SPACE:
            self.toggle()
            return
        if not self.enabled:
            return

        # Gantry hoist: raise/lower the anchor like a real winch (U=up, L=lower,
        # 5 cm/press). Lower the limp robot until the feet load and the knees
        # fold -> a physically-reached crouch start; hoist back up as it rises.
        # (P is taken by pause in the twin; kept off here too.) Verbatim from the
        # twin's h1_mujoco/mujoco_env.py so RoboCasa drives identically.
        if key == glfw.KEY_U or key == glfw.KEY_L:
            self.point[2] += 0.05 if key == glfw.KEY_U else -0.05
            print(f"Hoist {'raised' if key == glfw.KEY_U else 'lowered'} to "
                  f"z={self.point[2]:.2f} (U=up L=lower, 5cm/press)")
            return
        # Pay out / reel in rope by changing the strap REST LENGTH (. = pay out
        # +10 cm, , = reel in -10 cm). A tension-only strap goes SLACK/GREEN once
        # the rope is longer than the torso-to-anchor distance = the honest
        # release gesture (the robot then stands on its own feet, rope a pure
        # catch). This is the manual analog of --band-auto-release.
        if key == glfw.KEY_PERIOD:
            self.length += 0.1
            print(f"Rope paid out to rest length {self.length:.2f} "
                  "(. = +10cm; strap slackens toward GREEN)")
            return
        if key == glfw.KEY_COMMA:
            self.length = max(0.0, self.length - 0.1)
            print(f"Rope reeled in to rest length {self.length:.2f} (, = -10cm)")
            return
        # Steadying-hands tilt assist on/off (T): a point strap can't constrain
        # tilt, so the world-frame upright spring + angular damping (300/60, the
        # twin's proven-stable gains) are the operator's hands. Toggle off to see
        # the legs hold tilt unaided.
        if key == glfw.KEY_T:
            if self.posture_kp > 0.0 or self.posture_kd > 0.0:
                self.posture_kp = self.posture_kd = 0.0
                print("Tilt assist OFF")
            else:
                self.posture_kp, self.posture_kd = 300.0, 60.0
                if self.release_fade <= 0.0:
                    self.release_fade = 3.0
                print("Tilt assist ON (300/60 steadying hands) -- "
                      "SPACE now eases support off over 3s")
            return
        # Anchor XY nudge (arrow keys) -- VERBATIM from the twin so RoboCasa's
        # manual harness drives identically: reposition / recenter the strap
        # anchor over the robot while setting up the hoisted crouch start.
        if key == glfw.KEY_LEFT:
            self.point[0] -= 0.1
            print(f"Anchor X -0.1 -> {self.point[0]:.2f} (<-/-> move X)")
            return
        if key == glfw.KEY_RIGHT:
            self.point[0] += 0.1
            print(f"Anchor X +0.1 -> {self.point[0]:.2f} (<-/-> move X)")
            return
        if key == glfw.KEY_UP:
            self.point[1] += 0.1
            print(f"Anchor Y +0.1 -> {self.point[1]:.2f} (up/down move Y)")
            return
        if key == glfw.KEY_DOWN:
            self.point[1] -= 0.1
            print(f"Anchor Y -0.1 -> {self.point[1]:.2f} (up/down move Y)")
            return

    def print_instructions(self):
        print('Elastic Band / Harness Controls:')
        print('  SPACE  - toggle / ease-off release (fade when tilt assist on)')
        print('  U / L  - hoist anchor Up / Lower (5cm/press) -- crouch start')
        print('  . / ,  - pay out / reel in rope (10cm/press) -- . to GREEN/slack = free-stand')
        print('  T      - tilt assist (steadying hands) on/off')
        print('  arrows - move band anchor (<-/-> = X, up/down = Y)')
