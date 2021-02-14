from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torch_struct",
    version="0.5",
    author="Alexander Rush",
    author_email="arush@cornell.edu",
    packages=[
        "torch_struct",
        "torch_struct.semirings",
    ],
    long_description=long_description,
    package_data={"torch_struct": []},
    long_description_content_type="text/markdown",
    url="https://github.com/harvardnlp/pytorch-struct",
    install_requires=["torch"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    python_requires='>=3.6',
)
