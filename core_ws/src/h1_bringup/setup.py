import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'h1_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    scripts=['scripts/open_fridge.py'],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='max',
    maintainer_email='maxlconway@gmail.com',
    description='Launch files to bring up the H1 simulation visualization.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'push_red_cube = h1_bringup.push_red_cube:main',
            'slider_debugger = h1_bringup.slider_debugger:main',
        ],
    },
)
