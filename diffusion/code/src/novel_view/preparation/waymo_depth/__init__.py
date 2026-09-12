"""Historical Waymo keysets, three-camera depth comparison and decisions.

The package owns the old central-key selection and FRONT/FRONT_LEFT/
FRONT_RIGHT depth preparation, MoGe/VGGT adapters and v1/v2 selection.
Candidate records, current selection records and read-only legacy selection
adaptation have separate concrete modules.
Physical Waymo readers remain in
``novel_view.inputs.waymo``; supported FRONT-only DDW preparation reuses only
the compatible raster and LiDAR geometry from this package.
"""
