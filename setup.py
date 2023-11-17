import glob
import os
from setuptools import setup, find_packages


setup(
    name="cslvae",
    version="0.0.1",
    author="Atomwise",
    description="CSLVAE source code",
    packages=find_packages(),
    scripts=list(filter(os.path.isfile, glob.glob("bin/**", recursive=True))),
    install_requires=[
        "numpy==1.22.4",
        "pandas==1.5.2",
        "pyyaml==5.4.1",
        "tensorboard>=2.6.0",
        "tqdm==4.62.3",
    ],
    include_package_data=True,
    python_requires='>=3.8',
)
