from setuptools import find_packages, setup

package_name = 'mission_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/worlds', ['worlds/tb3_house.sdf.xacro'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aiden',
    maintainer_email='shepleraiden@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mission_manager = mission_manager.mission_manager:main',
            'amcl_and_nav = mission_manager.amcl_and_nav:main',
            'slam_and_explore = mission_manager.slam_and_explore:main'
        ],
    },
)
