This folder contains a small homography and image-stitching reference implementation.

The runnable demo lives in `main.py`. It loads one pair of images from `photos/`, reads the matching point correspondences from `correspondances/`, computes a homography with `compute_H`, warps the first image with `warp_image`, and blends the result into a mosaic.

For full usage notes, file descriptions, input format details, and output explanations, see `../../README.md`.
