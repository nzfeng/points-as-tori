# Points as Tori

This repo implements _Points as Tori (PAT)_ for estimating signed distance from oriented point clouds. PAT computes signed distance directly from point clouds, and allows fast pointwise evaluation of signed distance at arbitrary spatial resolution --- without requiring discretization, global optimization, or explicit reconstruction.

PAT was introduced in the SIGGRAPH 2026 paper "[Points as Tori: Fast Pointwise Signed Distance for Point Clouds](https://nzfeng.github.io/research/SignedHeatMethod/index.html)" by [Nicole Feng](https://nzfeng.github.io/index.html), [Ioannis Gkioulekas](https://www.cs.cmu.edu/~igkioule/), [Keenan Crane](https://www.cs.cmu.edu/~kmcrane/).

![teaser image](media/Teaser.png)

Paper PDF (28.4mb): [link](https://nzfeng.github.io/research/PointsAsTori/PointsAsTori.pdf)

Project page with paper, videos, and blog-style explanations: [link](https://nzfeng.github.io/research/PointsAsTori/index.html)

If you have a feature or improvement in mind, or if parts of the repository buggy or poorly explained, leave an issue on GitHub.

## Build from source

The committed Pixi lock currently supports Apple Silicon Macs on macOS 15:
`osx-arm64`. [Install Pixi](https://pixi.prefix.dev/latest/installation/), then
initialize the native dependencies and install the exact locked environment:

```bash
git submodule update --init --recursive
pixi install --locked
```

Build and verify the Python/native wheel:

```bash
pixi run build
```

The build task checks the wheel contents, inspects its native linkage, and
prints the wheel's SHA-256 digest. Pixi supplies Python 3.12, CGAL, GMP, MPFR,
LLVM OpenMP, the OpenMP-backed BLAS implementation, build tools, and Python
packages from `pixi.lock`.

## Usage for signed distance

The API is implemented in `infer.py`. Usage:

```python
pat = PointsAsTori(points, normals, tori=None, tori_filepath=None)  # optionally provide or load existing tori
# `queries` is a (n_queries, 3) NumPy array containing query positions
distances = pat.signed_distance(queries)   # returns a (n_queries, ) NumPy array
gradients = pat.sdf_gradient(queries)      # returns a (n_queries, 3) NumPy array
```

In more detail: for a given point cloud, tori are first fit to the point cloud using a small pre-trained neural network (included in the repo); tori data needs to be precomputed only once per point cloud, and is stored in the `PointsAsTori` as object. Optionally, a `PointsAsTori` object can be initialized with existing pre-computed tori.

To give an idea of precomputation time: Using an NVIDIA RTX 3090, precomputation takes <10 seconds for 100k points, <60s for 1M points, 1-4 minutes for 5-10M points, and >10 minutes for 20M+ points (time scales linearly with point cloud size). A Macbook Pro laptop with an M3 Max chip, 16-core CPU, and 64 GB of RAM, takes about a minute for 50k points, eight minutes for 1M points.

## Demo & shader visualization

![shader visualization](media/Demo.png)

The `demo` environment locks Python 3.11, Pyglet, ImGui, PyVista, and its
native package build separately from the core environment. Launch the shader
demo directly:

```bash
pixi run -e demo demo
```

GUI interactions:

* mouse scroll: zoom in/out
* mouse click-and-drag: pan
* left-right mouse motion: move cutplane along axis
* tab: switch axis of cutplane
* spacebar: freeze the cutplane at its current location

## Dependencies

Python bindings are implemented using [`nanobind`](https://github.com/wjakob/nanobind). [`nanoflann`](https://github.com/jlblancoc/nanoflann) is used to build KD-trees to accelerate signed distance queries. The C++ functions use OpenMP for parallelization. Native and Python packages are resolved by Pixi; source dependencies are initialized as Git submodules.

This project additionally contains submodule dependences on [`libigl`](https://libigl.github.io/) and [`fcpw`](https://github.com/rohan-sawhney/fcpw) for their routines for winding numbers and exact distance, respectively. These are not strictly required, but they may be interesting to users as a point of comparison.

## Training

The `training/` directory contains scripts for training the neural network component and pre-processing training data. Pre-generated training data can be found at [this Google drive directory](https://drive.google.com/drive/folders/1hnG-1OCwZ0SWS47Kgmk4chPG825AOfGt?usp=sharing) (total size 6.2 GB).

Run a training script in the locked training environment:

```bash
pixi run -e training python training/train.py
```

## Areas of improvement

Areas of improvement mostly center around neural network performance and robustness.

1. *Precomputation cost:* On the one hand, training/inference of our neural network is at least an order of magnitude faster than many end-to-end neural methods (e.g. neural field fitting), and each signed distance query takes 10^{-4} - 10^{-3} seconds. Still, the precomputation cost of fitting tori to a new point cloud is not yet "instant"; using an NVIDIA RTX 3090, precomputation takes <10 seconds for 100k points, <60s for 1M points, 1-4 minutes for 5-10M points, and >10 minutes for 20M+ points (time scales linearly with point cloud size). (A Macbook Pro laptop with an M3 Max chip, 16-core CPU, and 64 GB of RAM, takes about a minute for 50k points, eight minutes for 1M points.)

    The neural network has not been extensively engineered; I suspect that performance may be significantly improved with better choice or normalization of input features, fewer attention layers, or perhaps a different architecture entirely.

    Previously, I experimented with classic point set approaches for torus fitting that proved inadequate --- hence why I decided to use a small neural network. But it may still be possible to develop an effective non-neural approach to fitting tori that bypasses the need for a neural network entirely. Experimentation and suggestions welcome!

<!-- If used for optimization tasks, tori can perhaps be updated using simple gradient-based updates rather than forward passes of the neural network.  -->

2. *Robustness to corruption:* The network's current predictions might not be robust for point clouds whose sampling characteristics are significantly different from those seen in training, such as point clouds with significantly different sampling density, or significant amounts of noise, outliers, or missing data. It may be worthwhile to train on more diverse data --- I only trained on clean point clouds with 2048 points each. To improve accuracy, it may also be useful to include some form of neighborhood size estimation or adopt a hierarchical approach, or simply preprocess input point clouds (e.g. subsample to match a target density, noise/outlier removal).

## Repository TODOs

* Implement the Laplacian of the SDF of a torus in the C++ function `TorusDistanceField::evaluate_distance_gradient_laplacian_single()`
* Implement browser-based shader using WebGL or WebGPU
* More efficient sphere tracing
* Release Python package on PyPI (and thus implement basic CI/CD)

## Citation

If this code contributes to academic work, cite it as:
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
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3811385},
    doi = {10.1145/3811385},
    journal = {ACM Trans. Graph.},
    month = {jul},
    articleno = {53},
    numpages = {24}
}
```
