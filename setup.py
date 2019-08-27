from setuptools import setup

setup(
    name="torch_struct",
    version="0.0.1",
    author="Alexander Rush",
    author_email="arush@cornell.edu",
    packages=["torchstruct", ],
    package_data={"torchstruct": []},
    url="https://github.com/harvardnlp/pytorch_struct",
    install_requires=["torch"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"]

)
