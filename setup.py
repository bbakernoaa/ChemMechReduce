from setuptools import setup, find_packages

setup(
    name='ChemMechReduce',
    version='0.1',
    author='Barry Baker and Margaret Marvin',
    author_email='barry.baker@noaa.gov',
    description='A way to try and reduce atmospheric chemical mechanisms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/bbakernoaa/ChemMechReduce',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)