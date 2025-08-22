from setuptools import find_packages, setup

package_name = 'ubot_pkg'

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
    maintainer='fayas',
    maintainer_email='fayaspallikkara@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        "camera_pub = ubot_pkg.camera_pub:main",   
        "image_visual_pub = ubot_pkg.image_visual_pub:main",
        ],
    },
)
