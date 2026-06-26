"""Elastic-band balance tether used by the sim runner (h12_mujoco.sim_loop).

A vertical spring-damper that pulls a body (the torso) toward a fixed anchor
point above its spawn, holding the free-floating biped upright until the ROS
walking policy takes over. Toggle with SPACE in the viewer or the
/elastic_band/toggle service.
"""
import mujoco
import numpy as np


class ElasticBand:
    def __init__(self, point=np.array([0, 0, 3]), length=0, stiffness=200, damping=100):
        self.point = point.astype(np.float64)
        self.length = length
        self.stiffness = stiffness
        self.damping = damping
        self.enabled = True

    def evaluate_force(self, x, v):
        # displacement / direction from the attached body to the anchor
        displacement = np.linalg.norm(x - self.point)
        direction = (x - self.point) / displacement
        # tangential velocity along the band, for damping
        v_tan = np.dot(v, direction)
        # spring-damper
        return (-self.stiffness * (displacement - self.length) - self.damping * v_tan) * direction

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        print(f"Elastic band {'enabled' if self.enabled else 'disabled'}")
        return self.enabled

    def key_callback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_SPACE:
            self.toggle()

    def print_instructions(self):
        print('Elastic Band Controls:')
        print('  SPACE - Toggle elastic band on/off')
