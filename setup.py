from setuptools import (setup) # find_packages
import yaml

with open("./README.md", "r") as f:
    long_description = f.read()


# get version from build spec
with open("./build_spec.yml", "r") as f:
    build_spec = yaml.load(f, Loader=yaml.FullLoader)

build_spec
VERSION         = build_spec['__version__']
AUTHOR          = build_spec['__author__']
AUTHOR_EMAIL    = build_spec['__email__']
CREDITS         = build_spec['__credits__']


setup(
    name="forexflaggr",
    version=VERSION,
    description="A minimal package to pull and analyse financial (exchange rate) data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZachWolpe/forexflaggr",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "plotly-express>=0.4.1",
        "plotly>=5.10.0",
        "yfinance>=0.2.3",
        "pygam>=0.8.0",
        "moepy>=1.1.4",
        "setuptools>=65.6.3",
        "kaleido>=0.2.1",
        "nbformat"
        ],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.8",
)