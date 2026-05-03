# Calibration of cameras

## Overview
The goal of this component is to calibrate the cameras' position, orientation and focal length into a unified coordinate system. 
For that, we will take a single initial picture from each camera, and use feature detection to triangulate relative positions between cameras. 
Once the relative positions of the cameras are known, we can use two cameras and a single landmark to establish a global coordinate system and calculate the extrinsic parameters of all cameras.


## Algorithm
The core of the algorithm will consist on a function get_relative_pos(img_i, img_j)that takes the initial image from two cameras i and j, and computes a 3d transformation matrix M_{ij} that transforms coordinates from camera i's coordinate system to camera j's coordinate system together with two scale factors $f_i$ and $f_j$ for each image. The matrix $M_{ij}$ is going to be of the form
$$ M_{ij} = \begin{bmatrix} R_{ij} & t_{ij} \\ 0 & 1 \end{bmatrix}, $$
where $t_{ij}$ is a unit vector representing the relative position of camera j with respect to camera i, and $R_{ij}$ is a composition of yaw, pitch and roll rotations that transforms coordinates from camera i's coordinate system to camera j's coordinate system. $f_i$ and $f_j$ will determine the zoom/aperture of each camera.
The algorithm to produce this matrix will be the following:
- Extract matching points from the two images using feature matching (SIFT)
- Perform least squares estimation (scipy.optimize.lsq_linear) on the matching points to find the best matching transformation matrix. Given a set of point pairs $(p_l, q_l)_{l=1}^n$, and some parameters $t_{ij}\in \mathbb{R}^3$ (unitary), $\phi_{ij}$, $\theta_{ij}$ and $\psi_{ij}$, $f_i$, $f_j> 0$, define, for each l:
- $L_l$ the line that passes through the origin and pointing in the $(p_l, f_i)$ direction.
- $L'_l$ the line that passes through $t_{ij}$ and pointing in the $M_{ij}(q_l, f_j)$ direction.
- $d_l$ the distance between $L_l$ and $L'_l$.
- $d_l'$ the distance between the origin ant the closest point to $L_l'$ on $L_l$.
- $d_l''$ the distance between $t_{ij}$ and the closest point to $L_l$ on $L'_l$.

We want to minimize the following loss function
$$ L = \sum_{l=1}^n (d_l/d_l')^2 + (d_l/d_l'')^2,$$ 
over the parameters $t_{ij} \in \mathbb{R}^3$ (unitary), $\phi_{ij}$, $\theta_{ij}$, $\psi_{ij}$, $f_i$, $f_j> 0$.  
And we will return the matrix $M_{ij}$ and the focal lengths $f_i$ and $f_j$.  

## Data structures
We will have a class Image that contains image data as a numpy tensor and a title (the camera's id). There will be a static method from_file() that creates an instance from a file path with the title being the filename without the extension, and a from_video() that creates an instance from the first frame of a video.


We will have a ReferenceSystem class. An instance will be generated from a pair of initial images (camera i and j) and precomputes the transformation matrix $M_{ij}$. This class will have a function get_coords which takes an image from any camera k and returns a vector $v_{k,ij}$, a zoom/aperture factor $f_k$ and a rotation matrix $U_{k,ij}$ (which transforms a pixel coordinates from camera $k$ in the form $(x,y,f_k)^T$ to the corresponding 3d ray vector in camera i's coordinate system, which is unique up to a constant). get_coords will perform this the following way:
- Calculate rel_pos(img_i, img_k) and rel_pos(img_j, img_k) to get $M_{ik}$ and $M_{jk}$. Here, $U_{k,ij} = R_{ik}$ and $f_k$ is the same scale factor from rel_pos(img_i, img_k).
- Calculate the angles:
$$\alpha_{i} = \operatorname{Angle}(t_{ij}, t_{ik}),\quad \alpha_{j} = \operatorname{Angle}(-t_{ij}, t_{jk}),\quad \alpha_{k} = \operatorname{Angle}(t_{ik}, t_{jk})$$
where $\operatorname{Angle}(u,v) = \arccos(u \cdot v / (||u|| ||v||))$.
- This way, by law of sines we have $v_{k,ij} = t_{ik}d_{k,ij}$ where $d_{k,ij} =\frac{\sin(\alpha_i)}{\sin(\alpha_k)}$.

The ReferenceSystem class will also save each camera's parameters ($v_{k,ij}$ and $U_{k,ij}$) relative to the reference coordinate system from cameras i and j as a dict whose keys are the images' titles (i.e. the cameras' ids). During init, it will also add the parameters for the reference images (i and j) which are trivial to calculate after computing the transformation matrix $M_{ij}$.


## Tests
All the tests will be run using the four images in the images/4_images directory.

Reciprocity test: Verify that $M_{ji} = M_{ij}^{-1}$ for each pair of images. Here we will avoid using the pre_computed inverse when calculating M_{ji} by calling rel_pos(img_j, img_i) after altering the titles of the images in order to ensure that the matrix is not retrieved from the library. Also verify that the focal lengths match between the two calculations.

Transitivity test: for each triplet of images i, j, k, verify the following statements:

- $R_{ij}R_{jk} = R_{ik}$ (rotations should be transitive)
