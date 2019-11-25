from setuptools import setup

setup(
    name="torch_struct",
    version="0.4",
    author="Alexander Rush",
    author_email="arush@cornell.edu",
    packages=[
        "torch_struct",
        "torch_struct.data",
        "torch_struct.networks",
        "torch_struct.semirings",
    ],
    package_data={"torch_struct": []},
    url="https://github.com/harvardnlp/pytorch_struct",
    install_requires=["torch"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
