from setuptools import setup, find_packages

setup(
    name="lepaute",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "kornia>=0.7.0",
        "opencv-python",
        "numpy"
    ],
    author="Carson Wu",
    author_email="carson.developer1125@gmail.com",
    description="A package for accessing L.E.P.A.U.T.E. (Lie Equivariant Perception Algebraic Unified Transform Embedding Framework)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dev1virtuoso/Machine-Learning/tree/main/Computer%20Vision/L.E.P.A.U.T.E.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
