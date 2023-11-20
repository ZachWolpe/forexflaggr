from setuptools import find_packages, setup

with open("./README.md", "r") as f:
    long_description = f.read()

VERSION         = "0.0.1"
AUTHOR          = "Zach Wolpe"
AUTHOR_EMAIL    = "zach.wolpe@mlxgo.com"

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