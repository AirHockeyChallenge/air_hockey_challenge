from setuptools import setup, find_packages

from air_hockey_challenge import __version__

setup(
    name='AirHockeyChallenge',
    version=__version__,
    url='https://github.com/AirHockeyChallenge/air_hockey_challenge',
    author='Jonas Guenster',
    author_email='air-hockey-challenge@robot-learning.net',
    description='Package that provides the environments and frameworks for the 2023 Air Hockey Challenge.',
    packages=find_packages(),
    install_requires=[],
)
