from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    dependencies = f.read()

setup(name='controllable-embedding',
      packages=find_packages(),
      install_requires=["h5py"],
      description='demo project for jaynes launcher',
      author='Ge Yang',
      url='',
      author_email='ge.yang@berkeley.edu',
      version='0.0.0')
