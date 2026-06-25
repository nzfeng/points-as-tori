# Points as Tori

Evaluation and training code for "[Points as Tori: Fast Pointwise Signed Distance for Point Clouds](https://nzfeng.github.io/research/SignedHeatMethod/index.html)" by [Nicole Feng](https://nzfeng.github.io/index.html), [Ioannis Gkioulekas](https://www.cs.cmu.edu/~igkioule/), [Keenan Crane](https://www.cs.cmu.edu/~kmcrane/), presented at SIGGRAPH 2026.

This repo implements _Points as Tori (PAT)_ for estimating signed distance from oriented point clouds. 

![teaser image](media/Teaser.png)

Paper PDF (28.4mb): [link](https://nzfeng.github.io/research/PointsAsTori/PointsAsTori.pdf)

Project page with paper, videos, and blog-style explanations: [link](https://nzfeng.github.io/research/PointsAsTori/index.html)

Have a feature or improvement in mind that you'd like to see? Parts of the repository buggy or poorly explained? Leave an issue on GitHub!

## Usage for signed distance

**TODO: install Python package; usage and demo code**

If building this project from source, it is likely that you may have to first initialize git submodules after cloning the repository, using 

```
git submodule update --init --recursive
```

## Demo

**TODO: browser-based demo**

**TODO: high-level API, k-nearest neighbors acceleration**

## Dependencies

Python bindings are implemented using [`nanobind`](https://github.com/wjakob/nanobind). [`nanoflann`](https://github.com/jlblancoc/nanoflann) is used to build KD-trees to accelerate signed distance queries. The C++ functions use OpenMP for parallelization. All dependencies should be installed automatically upon install of the Python package.

This project also contains submodule dependences on [`libigl`](https://libigl.github.io/) and [`fcpw`](https://github.com/rohan-sawhney/fcpw) for their routines for winding numbers and exact distance, respectively (these may be interesting to users for comparison).

## Training

**TODO: download training data**

The `training/` directory contains scripts for training the neural network component and generating training data, both from scratch. Pre-generated training data can be found at [TODO]() (total size TODO GB).

The training scripts use additional Python packages, which can be pip-installed:

```
pip install scipy numpy-stl matplotlib thingi10k py7zr pyvista trimesh pyfqmr noise
```

## Areas of improvement

1. *Precomputation cost:* Though signed distance evaluation is fast, there is a non-neglible precomputation cost to fit tori to each new point cloud, since it involves forward passes of a neural network. The neural network has not been extensively engineered; I suspect that performance may be significantly improved with better choice or normalization of input features, fewer attention layers, or perhaps a different architecture entirely.

Previously, I experimented with classic point set approaches for torus fitting that proved inadequate --- hence why I decided to use a small neural network. But it may still be possible to develop an effective non-neural approach to fitting tori that bypasses the need for a neural network entirely. Experimentation and suggestions welcome!

<!-- If used for optimization tasks, tori can perhaps be updated using simple gradient-based updates rather than forward passes of the neural network.  -->

2. *Robustness to corruption:* Our current predictions might not be robust for point clouds whose sampling characteristics are significantly different from those seen in training, such as point clouds with significantly different sampling density, or significant amounts of noise, outliers, or missing data.

## Citation

If this code contributes to academic work, please cite as:
```bibtex
@article{Feng:2026:PAT,
    author = {Feng, Nicole and Gkioulekas, Ioannis and Crane, Keenan},
    title = {Points as Tori: Fast Pointwise Signed Distance for Point Clouds},
    year = {2026},
    issue_date = {August 2026},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {45},
    number = {4},
    issn = {XXXX-XXXX},
    url = {https://doi.org/10.1145/3811385},
    doi = {10.1145/3811385},
    journal = {ACM Trans. Graph.},
    month = {jul},
    articleno = {53},
    numpages = {24}
}
```
