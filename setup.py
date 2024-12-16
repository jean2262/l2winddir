from setuptools import setup, find_packages

setup(
    name='l2winddir',
    version='0.0.1',
    description='A package for l2 wind direction',
    author='jean2262',
    author_email='jrenaud495@gmail.com',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'hydra-core',
        'hydra-zen',
        'pandas',
        'xarray',
        'numpy',
        'lightning',
        'omegaconf',
        'tqdm'
    ],
    python_requires='>=3.11',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
