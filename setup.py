import os

from setuptools import find_packages, setup


# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Multi-Animal-Tracker for behavioral analysis"


setup(
    name="multi-animal-tracker",
    version="1.0.0",
    author="Rishika Mohanta",
    author_email="neurorishika@gmail.com",
    description="A multi-animal tracking system for high-contrast behavioral arenas",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/neurorishika/multi-animal-tracker",
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # Include non-Python files
    include_package_data=True,
    package_data={
        "multi_tracker": ["*.json", "*.yaml", "*.yml"],
    },
    # Minimal dependencies - let conda handle the heavy lifting
    install_requires=[
        # Only specify packages that conda can't handle well
        # Most scientific packages will be installed via conda
    ],
    # Console scripts
    entry_points={
        "console_scripts": [
            "multi-animal-tracker=multi_tracker.app.launcher:main",
            "mat=multi_tracker.app.launcher:main",
            "posekit-labeler=multi_tracker.posekit.pose_label:main",
            "pose=multi_tracker.posekit.pose_label:main",
        ],
    },
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.11",
    keywords="animal tracking, computer vision, behavioral analysis, opencv, kalman filter",
)
