import os
from glob import glob

from setuptools import setup

PACKAGE_NAME = 'ros2_object_tracking'
SHARE_DIR = os.path.join("share", PACKAGE_NAME)

setup(
    name=PACKAGE_NAME,
    version='0.0.1',
    packages=["object_tracking"],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + PACKAGE_NAME]),
        ('share/' + PACKAGE_NAME, ['package.xml']),
        (os.path.join(SHARE_DIR, "launch"), glob(os.path.join("launch", "*.launch.py"))),
        (os.path.join(SHARE_DIR, "config"), glob(os.path.join("config", "*.yaml"))),
    ],
    package_dir={'': 'src', },
    py_modules=[],
    install_requires=['setuptools'],
    author='Andreas Klintberg',
    author_email='andreas.klintberg@gmail.com',
    description='ROS2 object tracking using Kalman filter and Hungarian algorithm for Yolov3 bounding box predictions.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'object_tracking_node = object_tracking.object_tracking_node:main',
        ],
    },
)
