"""Concrete physical Waymo v2 ingress without eager optional dependencies.

The package exposes no broad re-export surface.  Consumers import scalar
identity, camera, LiDAR, frame, index, decoder and reader owners from their
exact modules.  PyArrow and OpenCV load only during explicit Parquet or JPEG
reading; DDW, model and recipe policy stay outside this package.
"""
