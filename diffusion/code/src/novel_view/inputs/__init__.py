"""Concrete readers and physical records of external data.

The package owns nuPlan, EUVS, physical Waymo records, and exact
``gaussian_depth_export/v1`` decoding and may depend on
``novel_view.geometry``. The Gaussian reader does not open frame payloads.
The package defines no common reader interface and performs no eager optional
imports.
"""
