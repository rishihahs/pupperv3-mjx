from setuptools import setup, find_packages

setup(
    name="pupperv3_mjx",  # Replace with your own package name
    version="0.1",  # Package version
    packages=find_packages(),  # Automatically find package modules
    install_requires=[
        # List of dependencies required by your package
        # e.g., 'numpy>=1.18.5',
    ],
    # Optional fields:
    author="Nathan Kau, Google",
    author_email="your.email@example.com",
    description="A short description of your project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # if your README is Markdown
    url="https://github.com/yourusername/my_package",  # Optional project URL
    classifiers=[
        # Trove classifiers
        # Full list at https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum version requirement of the package
    # Include any additional packages that need to be installed along with your package
    extras_require={
        "dev": ["check-manifest"],
        "test": ["coverage"],
    },
)
