# Object detection of aerial objects


## Overview:
A robust multi-camera object detection algorithm to detect aerial objects. Each camera will detect moving objects on their own, and a simple raycasting algorithm will be used to triangulate the object's position with high statistical significance. Our approach involves a low-bandwidth network that can be deployed in remote locations with minimal infrastructure, allowing for:
- Quick integration of new cameras to the system to improve coverage and/or accuracy
- Quick deployment in remote locations with minimal infrastructure
- Low bandwidth usage, reducing costs and improving reliability
Furthermore, the only data transferred to the central server will be the pixel coordinates of the detected objects, which means that any type of camera can be integrated into the system. This allows the use of different types of sensors, such as infrared or thermal cameras, to detect objects in different environments and conditions.

## The algorithm:
### 1. Preprocessing:
As the cameras will be static, the only pre-processing needed is to calibrate the cameras' position, orientation and focal length into a unified coordinate system. 
- For that, we will take a single initial picture from each camera, and use feature detection to triangulate relative positions between cameras. 
- Once the relative positions of the cameras are known, we can use two cameras and a single landmark to establish a global coordinate system and calculate the extrinsic parameters of all cameras.
- Extrisic parameters of a camera:
    - Position: (c_x, c_y, c_z)
    - Orientation: (c_theta, c_phi, c_psi)
    - Focal length: (f_x, f_y)
### 2. Object detection:
We will implement the following pipeline to aggregate all camera data into a set of 3D objects with different confidence levels:
- Raw data to camera objects: on each we will subtract consecutive frames to isolate moving objects. Then, we will use a clustering algorithm to group the pixels of moving objects into a single object with coordinates (p_x, p_y) in pixels as well as a timestamp p_t. **Perhaps confidence?**
- Camera objects to world rays: all these pixel coordinates will be sent to a central processing unit which will convert each pixel coordinate into a 3D ray using the camera's extrinsic parameters.
- Ray parameters:
    - Origin: (r_x, r_y, r_z)
    - Direction: (r_dx, r_dy, r_dz) **Perhaps spherical coordinates (r_theta, r_phi)?**
    - **Confidence?** 
- Ray aggregation: all these rays will be aggregated to form a set of 3D objects with different confidence levels. Possible algorithms to use:
    - Voxel grid: raycast the rays on a 3D grid and count the number of rays in each voxel. The voxels with a high probability of being an object (high number of passing rays) will be the detected objects.
    - Ray intersections: Calculate pairwise ray intersections (maybe detecting low distance between rays), and then apply a clustering algorithm to group the points with close intersections. 
### 3. Device synchronization: 
Since each camera will have different processing times, and the latency in data transmission will vary, it will be necessary to implement a synchronization algorithm to ensure that the rays are aggregated in the correct order and time. 
For this, we need to add the following pre-processing step that calculates the relative timeframes between cameras:
- For every pair of cameras i and j, we will take the first 20-50 frames and fit the best dt_{ij} that maximizes the agreement between the two cameras' rays **in the central unit**.
- We will agreggate the dt_{ij} matrix into a single dt_i vector (relative to an arbitrary camera 0) by minimizing least squares error against the dt_{ij} values for each pair.
- These dt_{i} values will be used in the object detection pipeline (step 2) to perform the object detection in the correct order and time. As some of the rays may be from different timeframes, the confidence of each object may (and will) be updated over time until all the cameras have been taken into account for that timeframe.

### The systems integration:
