from setuptools import setup, find_packages

setup(
    name="gerbil_client",
    packages=find_packages(),
    version="0.1",
    install_requires=[
        'bs4',
        'requests',
    ],
)
