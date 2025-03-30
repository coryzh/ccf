from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ccf",
    version="0.1.0",
    description="A library of code for deriving RV and RV error with CCF.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yue Zhao",
    author_email="yue.zhao@soton.ac.uk",
    url="https://github.com/coryzh/ccf_development.git",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pytest>=6.0.0",
        "astropy>4.3.0",

    ],
    python_requires=">=3.11",
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    keywords="Spectroscopy astronomy radial-velocity"
)
