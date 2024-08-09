import setuptools
from distutils.core import setup, Extension


def main():
    setup(
        name='pyikfastgen3_robotiq_2f_85',
        version='0.0.1',
        description='ikfast wrapper',
        author='Cyberbotics',
        author_email='support@cyberbotics.com',
        ext_modules=[Extension('pyikfastgen3_robotiq_2f_85', ['ikfast_robot.cpp', 'pyikfast.cpp'])],
        setup_requires=['wheel']
    )


if __name__ == '__main__':
    main()
