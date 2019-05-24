import setuptools


with open('requirements.txt') as file:
    requires = [line.strip() for line in file if not line.startswith('#')]

setuptools.setup(
    name='netrdexplorer',
    version='0.1.0',
    author='NetSI 2019 Collabathon Team',
    author_email='saffo.d@husky.neu.edu',
    description='Repository Demo Site of network reconstruction, distance, and simulation methods',
    url='https://github.com/netsiphd/netrdexplorer',
    packages=setuptools.find_packages(),
    install_requires=requires,
    classifiers=['Programming Language :: Python :: 3',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: OS Independent'],
    extras_require={
        'doc':  ['POT>=0.5.1'],
    }
)
