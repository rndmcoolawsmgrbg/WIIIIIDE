from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="w5xde",
    version="0.1.0",
    author="Mikus Sturmanis, Jordan Legg",
    author_email="ilovevisualstudiocode@gmail.com",
    description="Distributed machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rndmcoolawsmgrbg/WIIIIIDE/tree/dev_package",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "cryptography",
        "transformers",
        "dash",
        "dash-bootstrap-components",
        "plotly",
        "lz4"
    ]
)