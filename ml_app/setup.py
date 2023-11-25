from setuptools import find_packages, setup

setup(
    name='ml_app',
    version='0.0.1',
    python_requires='>=3.10',
    packages=find_packages(exclude=['tests']),
    scripts=['bin/run-ml-app'],
    include_package_data=True
)
