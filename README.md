# BrainDiffusion

*Construct an anatomical connectivity matrix by solving an anisotropic diffusion process.*

The connectivity can be a complement of existing tractographic methods. It achieves successful
performance on estimating the tau propagation in Alzheimer's disease.

## Requirements

This is a GPU-based code. A suitable version of [cupy](https://docs.cupy.dev/en/stable/install.html) should be pre-installed in the system. Other requirements will be automatically downloaded.

## Install

Currently a test version can be installed by 
```
pip install -i https://test.pypi.org/simple/ BrainDiffusion
``` 
or git clone the package and 
```
pip install .
```
under current folder

## Usage

Example codes are available in BrainDiffusion/example/ to construct a connectivity matrix using the default DTI data.




