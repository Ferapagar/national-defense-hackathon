# Raycasting pipeline

## Overview
The goal of this component is to cast rays from each camera's perspective and aggregate them into a 3D voxel grid to detect objects.

## Main Features:
The main component of the program is to take a series of rays and and aggregate them into a collection of 3d points with assigned confidences. 
These 3d points will then be used to identify objects in the scene, which will be reported to the user.

## Calibration: 
As each camera might have different time references (which may get amplified by processing times and network latency), we need to calculate a relative timeframe between cameras. 
For this, we will take the first 20-50 frames from each camera and fit the best dt_{ij} that maximizes the agreement between the two cameras' rays in the central unit.
We will then aggregate the dt_{ij} matrix into a single dt_i vector (relative to an arbitrary camera 0) by minimizing least squares error against the dt_{ij} values for each pair.
These dt_{i} values will be used in the object detection pipeline (step 2) to perform the object detection in the correct order and time. As some of the rays may be from different timeframes, the confidence of each object may (and will) be updated over time until all the cameras have been taken into account for that timeframe.


## Components: 
- **Calibration**: There will be two main functions:
    1. For each camera pair, it takes the first 20-50 frames from each camera and fits the best dt_{ij} that maximizes the agreement between the two cameras' rays in the central unit.
    2. Takes a set of 20-50 frames from each camera, computes the dt_{ij} matrix and normalizes it into a single dt_i vector (relative to an arbitrary camera 0) by minimizing least squares error against the dt_{ij} values for each pair.
## Data structures:
We need the following data structures for an integrated system:
1 Camera: A data structure that holds the intrinsic and extrinsic parameters of each camera, including position, rotation and time offset.
2 Ray: a data structure that holds a ray in all possible formats. It will store:
    - Camera associated to the ray
    - The ray in the camera's coordinate system (including local camera timestamp)
    - The ray in the world's coordinate system (including global timestamp)
3 Scene: A data structure that holds everything together: the world grid, the cameras, the rays, and the detected objects. Each camera will have an id.