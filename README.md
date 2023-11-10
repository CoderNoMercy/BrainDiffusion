## About
`BrainDiffusion` is a Python package for creating a weighted graph representing brain white-matter fiber tract connectivity between gray matter regions. The package can be generally used in neuroimaging workflows.

`BrainDiffusion` solves an anisotropic partial differential equation (PDE) on the MR image space to construct an anatomical connectivity matrix. To find the edge weights between a  vertex (source ROI) to all the other vertices (target ROIs), the algorithm assigns a pseudo-concentration in the start ROI, solves a mass diffusion equation, and integrates the resulting concentration in the target ROIs. These per-ROI integrals are the edge weights. The method requires $O(m)$ PDE solves, where $m$ is the number of parcels.

<p align="center">
<img src="paper/paper_img.pdf" alt="BrainDiffusion"  width="800"/>
</p>

Our latest publication in the MICCAI workshop illustrates the context in which this tool is utilized. Please consult the paper titled 'A Two-Species Model for Abnormal Tau Dynamics in Alzheimer's Disease' for further details. This tool is also used in our IEEE-TMI manuscript.

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

## Contributors

Zheyu Wen, Ali Ghafouri, George Biros. 

If you want to contribute to BrainDiffusion, read the guidlines [CONTRIBUTING.md](CONTRIBUTING.md).




