#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


install_requires = [
    'd3m==2019.11.10',
    'baytune==0.2.4',
    'tabulate>=0.8.3,<0.9',
    'numpy==1.17.3',
    'scikit-learn[alldeps]==0.21.0',
    'sri-d3m==1.5.5',
    'rpi_d3m_primitives==0.1.6',
    'Cython==0.29.7',
    'datamart-rest==0.2.3',
]


setup_requires = [
    'pytest-runner>=2.11.1',
]


tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
]


development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r>=0.2.0',
    'Sphinx>=1.7.1',
    'sphinx_rtd_theme>=0.2.4',

    # style check
    'flake8>=3.7.7',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.2',
    'autopep8>=1.4.3',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'tox>=2.9.1',
    'coverage>=4.5.1',
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
    description='MIT-Featuretools TA2 submission for the D3M program.',
    entry_points={
        'console_scripts': [
            'ta2=ta2.__main__:main',
        ]
    },
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='mit-d3m-ta2 ta2 d3m machine learning automl ml',
    license="MIT license",
    name='mit-d3m-ta2',
    packages=find_packages(include=['ta2', 'ta2.*']),
    python_requires='>=3.6, <3.7',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/HDI-Project/mit-d3m-ta2',
    version='0.2.0-dev',
    zip_safe=False,
)
