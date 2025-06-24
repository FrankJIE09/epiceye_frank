from setuptools import setup, find_packages
import datetime
import pathlib
import os


def get_version():
    import epiceye
    return epiceye.get_sdk_version()


def get_requirements():
    requirements_file = open("requirements.txt", 'r')
    return requirements_file.readlines()


HERE = pathlib.Path(__file__).parent
# VERSION = get_version()
VERSION = os.environ.get('VERSION')
current_date = datetime.datetime.now().strftime('%Y%m%d')
VERSION = VERSION + "." + current_date
print("version is:", VERSION)
REQUIREMENTS = get_requirements()

setup(
    name="epiceye",
    version=VERSION,
    author="TransferTech",
    description="TransferTech EpicEye SDK",
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    python_requires=">=3.6",
)
