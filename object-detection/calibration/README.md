# Calibration of cameras

## Overview
The goal of this component is to calibrate the cameras' position, orientation and focal length into a unified coordinate system. 
For that, we will take a single initial picture from each camera, and use feature detection to triangulate relative positions between cameras. 
Once the relative positions of the cameras are known, we can use two cameras and a single landmark to establish a global coordinate system and calculate the extrinsic parameters of all cameras.


## Algorithm
The core of the algorithm will consist on a function relative_pos(img_i, img_j)that takes the initial image from two cameras i and j, and computes a 3d transformation matrix M_{ij} that transforms coordinates from camera i's coordinate system to camera j's coordinate system. The matrix is going to be of the form
$$ M_{ij} = \begin{bmatrix} R_{ij}s_{ij} & t_{ij} \\ 0 & 1 \end{bmatrix}, $$
where $t_{ij}$ is a unit vector representing the relative position of camera j with respect to camera i, $s_{ij}$ is the scale factor between the two cameras, and R_{ij} is a composition of yaw, pitch and roll rotations that transforms coordinates from camera i's coordinate system to camera j's coordinate system.
The algorithm to produce this matrix will be the following:
- Extract matching points from the two images using feature matching (SIFT)
- Perform least squares estimation (scipy.optimize.lsq_linear) on the matching points to find the best matching transformation matrix. Given a set of point pairs $(p_i, q_i)_{i=1}^n$, regress for $t_{ij}$, $\phi_{ij}$, $\theta_{ij}$ and $\psi_{ij}$, $s_{ij}$ and $z_i, z_i'$ to minimize the following loss function:
$$ L = \sum_{i=1}^n ||Proj_{xy}((p_i, 1)z_i) - Proj_{xy}(M_{ij} (q_i, 1)z_i')||^2  \text{ s.t.} ||t_{ij}|| = 1.$$


## Data structures
We will have an Image class that contains image data as a numpy tensor and a title (the camera's id). There will be a static method from_file() that creates an instance from a file path with the title being the filename without the extension, and a from_video() that creates an instance from the first frame of a video.


We will have a ReferenceSystem class. An instance will be generated from a pair of initial images (camera i and j) and precomputes the transformation matrix $M_{ij}$. This class will have a function get_coords which takes an image from any camera k and returns a vector $v_{k,ij}$ and a rotation matrix $U_{k,ij}$ (which transforms a pixel coordinates from camera $k$ in the form $(x,y,1)^T$ to the corresponding 3d ray vector in camera i's coordinate system, which is unique up to a constant). get_coords will perform this the following way:
- Calculate rel_pos(img_i, img_k) and rel_pos(img_j, img_k) to get $M_{ik}$ and $M_{jk}$. $U_{k,ij} = R_{ik}$. 
- Calculate the angles:
$$\alpha_{i} = \operatorname{Angle}(t_{ij}, t_{ik}),\quad \alpha_{j} = \operatorname{Angle}(-t_{ij}, t_{jk}),\quad \alpha_{k} = \operatorname{Angle}(t_{ik}, t_{jk})$$
where $\operatorname{Angle}(u,v) = \arccos(u \cdot v / (||u|| ||v||))$.
- This way, by law of sines we have $v_{k,ij} = t_{ik}d_{k,ij}$ where $d_{k,ij} =\frac{\sin(\alpha_i)}{\sin(\alpha_k)}$.

The ReferenceSystem class will also save each camera's parameters ($v_{k,ij}$ and $U_{k,ij}$) relative to the reference coordinate system from cameras i and j as a dict whose keys are the images' titles (i.e. the cameras' ids). During init, it will also add the parameters for the reference images (i and j) which are trivial to calculate after computing the transformation matrix $M_{ij}$.
