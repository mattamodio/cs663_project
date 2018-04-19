import os
import yaml
from setuptools import setup


def get_requirements():
    """Get requirements from an environment.yml file.
    """
    with open('environment.yml') as f:
        yam = yaml.load(f)

    # Do some processing of the dependencies
    dep_list = []
    for dep in yam['dependencies']:
        if 'python' in dep:
            continue
        if isinstance(dep, dict):
            for nested_dep in dep.values():
                dep_list += nested_dep
            continue

        dep_list.append(dep)

    return dep_list


setup(
    name='cs663_project',
    version='0.1.0',
    description='Unsupervised domain mapping',
    author='Matt Amodio',
    author_email='TODO@TODO.com',
    packages=['cs663_project'],
    install_requires=get_requirements(),
    scripts=['bin/download_data', 'bin/train']
)
