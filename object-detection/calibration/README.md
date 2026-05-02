# Calibration of cameras

## Overview
The goal of this component is to calibrate the cameras' position, orientation and focal length into a unified coordinate system. 
For that, we will take a single initial picture from each camera, and use feature detection to triangulate relative positions between cameras. 
Once the relative positions of the cameras are known, we can use two cameras and a single landmark to establish a global coordinate system and calculate the extrinsic parameters of all cameras.
Extrisic parameters of a camera:
- Position: (c_x, c_y, c_z)
- Orientation: (c_theta, c_phi, c_psi)
- Focal length: (f_x, f_y)
These coordinates will be taken with respect to an arbitrary global coordinate system which will be defined around an arbitrary camera:
 - x-axis: the horizontal ray of the optical axis from the first camera
 - y-axis: the direction such that (x, y, z) form a right handed coordinate system
 - z-axis: the vertical ray of the optical axis from the first camera
 - Center of reference: the first camera will be located at (0,0,0) (with no rotation)
 - Distance measurement: every distance will be calculated relative to the distance between two given reference points.

Finally, this component will have a function that takes as input some camera info and the pixel coordinates of detected objects in each camera and outputs the 3D coordinates of the corresponding rays, which will be of the form:
- Origin: (r_x, r_y, r_z)
- Direction: (r_theta, r_phi) in spherical coordinates


## Components
- `main_calibration.py`: Main script to perform the calibration. It will take as input a set of images from different cameras and output the extrinsic parameters of each camera with respect to an arbitrary global coordinate system.
