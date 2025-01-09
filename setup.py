from setuptools import setup, find_packages

setup(
    name='l2winddir',
    use_scm_version={'write_to': 'l2winddir/_version.py'},
    setup_requires=['setuptools_scm'],
    description='A package for l2 wind direction',
    author='jean2262',
    author_email='jean-renaud.miadana@ocean-scope.com',
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'grdtiler',
        'hydra-zen',
        'omegaconf',
        'pytorch',
        'pytorch-cuda==11.8',
        'torchaudio',
        'torchvision',
        'lightning',
        'einops',
    ],
    python_requires='>=3.11',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)