from setuptools import setup

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='vecto-sdk',
    version='0.2.3',
    author='Xpress AI',
    author_email='eduardo@xpress.ai',
    description='Official Python SDK for Vecto',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/XpressAI/vecto-python-sdk',
    packages=['vecto'],
    install_requires=[
        'requests',
        'requests_toolbelt'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    keywords='vector-database, vector-search',
    project_urls={
        'Documentation': 'https://docs.vecto.ai/',
        'Source': 'https://github.com/XpressAI/vecto-python-sdk',
        'Issues': 'https://github.com/XpressAI/vecto-python-sdk/issues'
    },
)

