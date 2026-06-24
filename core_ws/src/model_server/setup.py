from setuptools import find_packages, setup

package_name = 'model_server'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # NOTE: weights/ is intentionally NOT in data_files. The servers resolve
        # checkpoints via __file__ (model_server/weights/), which under
        # `colcon build --symlink-install` points back at the source tree. Adding
        # the multi-GB .pt/.pth here would copy them into install/share on every
        # build for no benefit (see _WEIGHTS_DIR in the server modules).
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='max',
    maintainer_email='maxlconway@gmail.com',
    description='Self-contained ROS 2 service servers for Gemini, SAM3, and GraspGenX.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gemini_server = model_server.gemini_server:RunGeminiServer',
            'sam_server = model_server.sam_server:RunSamServer',
            'graspgen_server = model_server.graspgen_server:main',
        ],
    },
)
