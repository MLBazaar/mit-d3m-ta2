#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

install_requires = [
    'd3m==2019.1.21',
    'baytune==0.2.4'
]

development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',

    # style check
    'flake8>=3.5.0',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.1',
    'autopep8>=1.3.5',
]

setup(
    author="MIT Data To AI Lab",
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    description="MIT D3M TA2",
    extras_require={
        'dev': development_requires,
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='ta2',
    license="MIT license",
    name='ta2',
    packages=find_packages(include=['ta2', 'ta2.*']),
    python_requires='>=3.6, <3.7',
    url='https://github.com/HDI-Project/mit-d3m-ta2',
    version='0.0.1-dev',
    zip_safe=False,
)
