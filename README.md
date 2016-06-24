# Code for upcoming CVPR 2016 paper

    @inproceedings{firman-cvpr-2016,
      author = {Michael Firman and Diego Thomas and Simon Julier and Akihiro Sugimoto},
      title = {{Structured Completion of Unobserved Voxels from a Single Depth Image}},
      booktitle = {Computer Vision and Pattern Recognition (CVPR)},
      year = {2016}
    }

## Downloading the dataset

The dataset can be downloaded from:

https://dl.dropboxusercontent.com/u/495646/voxlets_dataset.zip

This is a 395MB zip file. You will have to change some of the paths in the code to the location you have extracted the dataset to.

## Getting started with the dataset

An example iPython notebook file loading a ground truth TSDF grid and plotting on the same axes as a depth image is given in `src/examples/Voxel_data_io_example.ipynb`

## Code overview

The code is roughly divided into three areas:

1. `src/common/` is a Python module containing all the classes and functions used for manipulation of the data and running of the routines. Files included are:

    - `images.py` - classes for RGBD images and videos
    - `camera.py` - a camera class, enabling points in 3D to be projected into a virtual depth image and vice versa
    - `mesh.py` - a class for 3D mesh data, including IO and marching cubes conversion
    - `voxel_data.py` - classes for 3D voxel grids, including various manipulation routines and ability to copy data between grids at different locations in world space

2. `src/pipeline/` - Contains scripts for loading data, performing processing and saving files out. The pipeline as described in the CVPR paper.

3. `src/examples/` - iPython notebooks containing examples of use of the data and code.

## Prerequisites

I have run this code using a fairly up-to-date version of Anaconda on Ubuntu 14.04.

This probably includes everything you need, but soon I will check to see if there are any  requirements which are not included in Anaconda.
