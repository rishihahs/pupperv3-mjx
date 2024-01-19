from setuptools import setup, find_packages

setup(
    name="pupperv3_mjx",  # Replace with your own package name
    version="0.1",  # Package version
    packages=find_packages(),  # Automatically find package modules
    install_requires=[
        "brax",
        "jax",
        "mujoco",
        "mujoco_mjx",
        "matplotlib"
        # List of dependencies required by your package
        # e.g., 'numpy>=1.18.5',
    ],
    # Optional fields:
    author="Nathan Kau, Google",
    author_email="your.email@example.com",
    description="A short description of your project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # if your README is Markdown
    python_requires=">=3.10",  # Minimum version requirement of the package
)
