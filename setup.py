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
    # ★ MPPI CUDA 커널 소스(.cu)를 패키지에 포함한다. core_mppi._build_cuda_solver 가
    #   os.path.dirname(__file__)/mppi_kernel.cu 를 읽으므로 install 디렉터리에도 깔려야
    #   node_mppi 가 동작한다. 없으면 FileNotFoundError 로 솔버 빌드 실패(테스트로 확인).
    package_data={'pathfollowing': ['core/*.cu']},
    include_package_data=True,
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
