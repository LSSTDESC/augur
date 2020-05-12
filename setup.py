import os
from setuptools import setup, find_packages

scripts = ['bin/augur']

__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "augur",
    "_version.py")
with open(pth, 'r') as fp:
    exec(fp.read())

setup(
    name='augur',
    version=__version__,
    description="DESC Cosmology Forecasting Tool",
    author="DESC Team",
    packages=find_packages(),
    include_package_data=True,
    scripts=scripts,
    install_requires=[
        'pyccl', 'click', 'numpy', 'firecrown',
        'scipy', 'pandas', 'pyyaml', 'jinja2'],
)
