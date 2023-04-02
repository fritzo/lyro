# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages, setup

setup(
    name="lyro",
    version="0.0.0",
    description="Probabilistic programming with language models",
    packages=find_packages(include=["lyro", "lyro.*"]),
    url="https://github.com/fritzo/lyro",
    author="Pyro Contributors",
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().strip().split(),
    extras_require={"test": open("requirements-dev.txt").read().strip().split()},
    include_package_data=True,
)
