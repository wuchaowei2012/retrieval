"""
    Setup for apache beam pipeline.
"""
import setuptools


NAME = 'create_tfrecords_training'
VERSION = '1.0'
REQUIRED_PACKAGES = [
    'protobuf==3.20.*',
    'apache-beam[gcp]==2.55.1',
    'tensorflow==2.9.0',
    'gcsfs==2022.8.2'
    ]

setuptools.setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
)
