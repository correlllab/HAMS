"""Wrapper around h12_ros2_controller's joint_state_publisher that initialises
the Unitree DDS channel on domain 1 (matching h1_mujoco) before the upstream
code calls ChannelFactoryInitialize() with the default domain 0.

The upstream joint_state_publisher hardcodes ChannelFactoryInitialize() with no
args and the Unitree SDK offers no env-var override, so we must call it first
(the factory is a singleton — the first call wins).
"""
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# Singleton — must be called before any other import that touches the channel.
ChannelFactoryInitialize(id=1)

from h12_ros2_controller.ros2.joint_state_publisher import main  # noqa: E402

if __name__ == '__main__':
    main()
