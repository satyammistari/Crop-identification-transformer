from setuptools import setup, find_packages

setup(
    name='ampt-crop-classification',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Adaptive Multi-Modal Phenological Transformer for crop classification',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'pytorch-lightning>=1.4.0',
        'terratorch>=0.1.0',
        'rasterio>=1.2.0',
        'albumentations>=1.0.0',
        'wandb>=0.10.0',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)