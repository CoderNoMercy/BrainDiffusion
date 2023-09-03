---
title: 'BrainDiffusion: A Python package for brain anatomic connectivity matrix.'
tags:
  - Python
  - Cupy
  - Diffusion process
  - Brain connectivity
authors:
  - name: Zheyu Wen
    orcid: 0000-0001-9628-1449
    equal-contrib: true
    affiliation: 1
  - name: Ali Ghafouri
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: George Biros
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
affiliations:
  - name: The University of Texas at Austin, USA
    index: 1
date: 2 September 2023
bibliography: paper.bib
link-citations: true
---

# Summary

`BrainDiffusion` is a Python package for creating a weighted graph representing brain white-matter fiber tract connectivity between gray matter regions. Such graphs are commonly used in neuroimaging workflows. The code input is (1) a magnetic resonance (MR) T1 image; (2) an MR diffusion-tensor image (DTI); and (3) a brain parcellation, which segments the MR T1 image into regions of interest (ROIs). The code output is a directed graph with vertices as gray matter ROIs, and edge weights indicating inter-region connectivity strength. In contrast to traditional methods involving parameter tuning relating to denoising, fiber groupings, and resolving fiber intersections, `BrainDiffusion` uses a different approach. It solves an anisotropic partial differential equation (PDE) on the MR image space to construct an anatomical connectivity matrix. To find the edge weights between a  vertex (source ROI) to the all the other vertices (target ROIs), the algorithm assigns a pseudo-concentration in the start ROI, solves a mass diffusion equation, and integrates the resulting concentration in the target ROIs. These per-ROI integrals are the edge weights. The graph weights are computed by by repeating this step for each vertex.  Notably, this technique does not reconstruct explicit fiber tracks; it only relies on MR DTI and parcellation data. The method requires $O(m)$ PDE solves, where $m$ is the number of parcels. 

# Statement of need

Brain white-matter connectivity and graph-based representations have diverse applications in neuroscience and clinical neuroimaging, including mental health [@eickhoff2018imaging], oncology, functional connectivity analysis [@biswal1995functional], disease diagnosis and classification [@craddock2009disease], surgical planning [@potgieser2014role], and population studies. Brain parcellation and graph abstraction are used as a means of compression, denoising, and summarization. Furthermore a brain's adjacency matrix unveils complex ROI relationships, offering insights into behavior and cognition [@lang2012brain].

There are several software packages with comparable functionalities [@iturria2008studying; @lazar2010mapping; @tournier2019mrtrix3] to ours. But these techniques have some limitations. Streamline tractography methods construct anatomical fiber tracts, but this process may lead to curvature overshoot bias, termination bias (where streamlines halt in white matter), and connectivity density bias (resulting in inaccurate streamline count estimates) [@zhang2022quantitative]. In our approach, streamlines are not explicitly constructed. Additionally, the proposed method circumvents streamline-to-node assignment challenges, a limitation associated with streamline-based approaches. Currently, tractography is unable to robustly trace fibers between and within multiple gray matter regions, resulting in unreliable connectome quantification [@yeh2021mapping]. A limitation of 'BrainDiffusion' is its lack of explicit fiber reconstruction.

# Formulation and algorithm
![Data used in software. Our software employs key data components, depicted from left to right. We start with patient DTI data, informing diffusivity intensity across six directions per voxel. Then, we integrate Magnetic Resonance Imaging (MRI) data, segmenting the brain into white and gray matter. The brain parcellation forms a graph. Lastly, we illustrate a PDE solution with one ROI chosen as the edge-start, known as the "Result diffusion". \label{fig:img}](paper_img.pdf)

Let $\mathcal{B}\subset\mathbb{R}^3$ be the brain domain. Let $\mathbf{x}\in\mathcal{B}$ denote a point. We denote $\mathbf{c}(\mathbf{x}, t)\in\left[0, 1\right]$ as a pseudo-concentration at time $t$. We denote the  source gray matter ROI by $\mathcal{S}\subset\mathbb{R}^3$ and a target gray matter ROI as $\mathcal{T}\subset\mathbb{R}^3$. We use the concentration diffusion from $\mathcal{S}$ to $\mathcal{T}$ as a measure of the connectivity between $S$ and $T$. The diffusion process is modeled  by the following PDE:
\begin{subequations}
\begin{align}
  \partial_t \mathbf{c} & = \nabla \cdot (\mathbf{K}(\mathbf{x}) \nabla \mathbf{c})\text{,}, \quad t\in (0,T],\\ 
  \mathbf{c}(\mathbf{x}, 0) & = \begin{cases}
    1 & \text{if } \mathbf{x}\in\mathcal{S} \\ 
    0 & \text{if } \mathbf{x}\in\mathcal{T} \text{,} 
  \end{cases} \\
  \frac{\partial \mathbf{c}}{\partial n} & = 0 \text{ on } \partial\mathcal{B} \text{,}
\end{align}
\end{subequations}
where $\mathbf{K}(\mathbf{x})$ is defined as
\begin{align}
  \mathbf{K}(\mathbf{x}) = \mathbf{D}(\mathbf{x}) (\mathbf{m}_{\text{wm}} + \Tilde{\alpha} \mathbf{m}_{\text{gm}}).
\end{align}
$\mathbf{D}(\mathbf{x})$ is a pointwise diffusion tensor which is extracted by the DTI [@le2001diffusion], $\mathbf{m}_{\text{wm}}$ and $\mathbf{m}_{\text{gm}}$ are the segmentation mask of white matter and grey matter, and $\Tilde{\alpha}\in\mathbb{R}_+$ is the ratio between the diffusivity in the white matter over the gray matter. From the literature, we set $\Tilde{\alpha}=10^{-2}$ [@giese1996migration]. We solve the above PDE in the time period $t\in\left[0, T\right]$, where $T$ is the time horizon. Please note that we've non-dimensionalized the brain domain to $\mathcal{B}=\left[0, 1\right]^3$, and for population studies, all subjects need to be mapped to a template.

The PDE forward solve ends when the region farthest from the edge-start reaches a steady-state solution, and $T$ represents the associated time horizon. For the PDE solver, we use the Crank-Nicolson method for discretization in time in our solvers [@crank1996practical]. The spatial domain is discretized using a pseudo-spectral Fourier method [@gholami2016inverse]. The resulting linear system for the Crank-Nicolson method is solved using a matrix-free preconditioned Conjugate Gradient method. We present the input used in `BrainDiffusion` and the PDE output in \autoref{fig:img}. The DTI, represented as $\mathbf{D}(\mathbf{x})$, provides information about diffusivity within the domain $\mathcal{B}$. The T1 MR image defines the white matter and gray matter regions. The brain parcellation operation establishes a graph structure, while the diffusion image serves as the solution to the aforementioned PDE, with one ROI being designated as the source region.

The construction of the graph adjacency matrix involves the following steps:

1. For each source ROI, we calculate the solution to the previously proposed PDE. This procedure is iteratively performed for all brain regions.

2. The concentration integral within each target ROI is computed for the resulting PDE solution with the following form
\begin{equation}
W_{ij} = \int_{0}^T\int_{\mathcal{T}_j} \mathbf{c}(\mathbf{x}, t)\mathbf{1}_{\left\{\mathbf{c}>\mathbf{c}_{\infty}\right\}}d\mathbf{x}dt.
\end{equation}
Here, $\mathcal{T}_j$ represents the volume region of the $j_\mathrm{th}$ ROI. $W_{ij}$ corresponds to the source $\mathcal{S}$ for ROI $i$, and $\mathbf{c}_{\infty}=\frac{1}{\left|\mathcal{B}\right|}\int_{\mathcal{S}}\mathbf{c}(\mathbf{x}, 0)d\mathbf{x}$ is the steady state  ($T=\infty$) solution for each voxel value. $W_{ij}$ contributes to the off-diagonal weight in the adjacency matrix. We set the diagonal weight $W_{ii} = \sum_{j\neq i} W_{ij}$.

3. We normalize $\mathbf{W}$ row-wise by dividing each row by its diagonal entry.

Our implementation requires GPU support and lacks a CPU version, with parallelization facilitated using [Joblib](https://joblib.readthedocs.io). On a GPU card, we simultaneously solve four PDEs, enhancing computational efficiency and accelerating the graph construction process.

# Usage

`BrainDiffusion` requires GPUs and Python's 'cupy'. Other requirements like 'nibabel', 'pandas' and 'joblib' will be installed automatically while installing the package. The package is tested under NVIDIA GPU Quadro RTX 5000. It includes a main function file ('BrainDiffusion/diffusion_brain_net.py') and a utility file ('BrainDiffusion/operators.py'). An example usage ('example.py') is also provided. To effectively use the package, users should have processed DTI data and patient parcellation. A default dataset and parcellation are available in 'example.py'. Users should keep the folder structure consistent with the example. For detailed installation, instructions, and testing, please refer to the package's [GitHub](https://github.com/CoderNoMercy/BrainDiffusion/) repository. 

# References
