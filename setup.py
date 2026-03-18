############################################################
#
#   - Name : setup.py
#
#                                 - KAIST FDCL, 2026.03.11
#
############################################################

from setuptools import find_packages, setup

package_name = 'pathfollowing'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/sim.yaml',
            'config/quad.yaml',
            'config/octo.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kestrel',
    maintainer_email='kestrel@inha.edu',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node_mppi = pathfollowing.node_mppi:main',
            'node_pathfollowing = pathfollowing.node_pathfollowing:main',
        ],
    },
)
