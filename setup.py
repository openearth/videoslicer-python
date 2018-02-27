from setuptools import setup, find_packages

setup(
    name='videoslicer',
    version='1.0.0rc2',
    author='Bas Hoonhout',
    author_email='bas.hoonhout@deltares.nl',
    url='http://videoslicer.readthedocs.io/',
    license='MIT',
    description='Toolbox to slice videos along arbitrary dimensions',
    long_description=open('README.rst').read(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=['video slice timestacks crs deltares tudelft'],
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'docopt',
        'six',
    ],
    python_requires='>=2.7, <4',
    tests_require=[
        'nose'
    ],
    test_suite='nose.collector',
)
