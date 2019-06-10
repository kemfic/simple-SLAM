# VOpy
simple visual odometry in python

<p float="left">
  <img src="resources/img.png" height="300" />
  <img src="resources/output.jpg" height="300" />
</p>

pipeline
---
 - Shi-Tomasi Corner Detection (Good Features to Track)
 - ORB Descriptor Extraction
 - Brute Force K-Nearest Neighbors Search (Feature Matching)
 - RANSAC 5-point Essential Matrix Estimation
 - Decompose Essential Matrix into a pose(Rt) matrix
 - output pose

todo
---
 - writeup
 - merge with ficiciSLAM


dependencies
---
 - opencv
 - numpy
 - [pangolin (uoip fork)](https://github.com/uoip/pangolin)
