from setuptools import setup, find_packages

setup(
    name="medical-triage-env",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "openenv-server=server.app:main",
        ],
    },
)