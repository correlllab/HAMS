import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'h12_lowerbody_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Policy weights + configs, installed under share/<pkg>/policies/<name>/.
        (os.path.join('share', package_name, 'policies', 'walk'),
            glob('policies/walk/*')),
        (os.path.join('share', package_name, 'policies', 'fame'),
            glob('policies/fame/*')),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maxconway',
    maintainer_email='maxlconway@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'walking_node = h12_lowerbody_controller.scripts.walking_node:main',
            'fame_node = h12_lowerbody_controller.scripts.fame_node:main',
            'lowerbody_controller_node = '
            'h12_lowerbody_controller.scripts.lowerbody_controller_node:main',
            # MJPC (MuJoCo MPC) DDS control node — Python skeleton analog of the
            # fork's C++ h12_control_node.cc (drives the mujoco_mpc gRPC Agent).
            'mjpc_node = h12_lowerbody_controller.scripts.mjpc_node:main',
        ],
    },
)
