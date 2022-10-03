from setuptools import setup

setup(
    name='vecto',
    version='0.1.0',
    packages=['vecto'],
    install_requires=[
        'datasets',
        'numpy',
        'tqdm',
        'requests',
        'notebook',
        'pytest',
        'requests_toolbelt'
    ]
)