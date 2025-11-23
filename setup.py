from setuptools import find_packages, setup

package_name = 'polar_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='colin',
    maintainer_email='colinc131@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'polar = polar_system.position_system:main',
            'fake_polar_target = polar_system.one_shot_fake_target:main',
            'polar_teleop = polar_system.keyboard_teleop:main',
            'polar_controller_interface = polar_system.controller_interface:main',
        ],
    },
)
