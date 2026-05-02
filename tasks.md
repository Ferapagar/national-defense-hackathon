# Object detection of aerial objects


## Overview:
A robust multi-camera object detection algorithm to detect aerial objects. Each camera will detect moving objects on their own, and a simple raycasting algorithm will be used to triangulate the object's position with high statistical significance. Our approach involves a low-bandwidth network that can be deployed in remote locations with minimal infrastructure, allowing for:
- Quick integration of new cameras to the system to improve coverage and/or accuracy
- Quick deployment in remote locations with minimal infrastructure
- Low bandwidth usage, reducing costs and improving reliability
Furthermore, the only data transferred to the central server will be the pixel coordinates of the detected objects, which means that any type of camera can be integrated into the system. This allows the use of different types of sensors, such as infrared or thermal cameras, to detect objects in different environments and conditions.


## Tasks:

### The algorithm (Vittorio):



### The systems integration (Fernando):
1. Setting 