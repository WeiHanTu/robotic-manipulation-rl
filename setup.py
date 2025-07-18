#!/usr/bin/env python3
"""
Setup script for Robotic Manipulation RL project.
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="robotic-manipulation-rl",
    version="1.0.0",
    author="Wei Han Tu",
    author_email="weihantu@example.com",
    description="Reinforcement learning for robotic manipulation tasks using ManiSkill",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WeiHanTu/robotic-manipulation-rl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-rl=robotic_manipulation_rl.train:main",
            "demo-rl=robotic_manipulation_rl.demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 