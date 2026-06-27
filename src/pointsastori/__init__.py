"""
pointsastori — Points as Tori: fast signed distance from oriented point clouds.

Quickstart::

    import pointsastori as pat
    import numpy as np

    points, normals = pat.read_point_cloud('input.ply')
    sdf = pat.PointCloudSDF(points, normals, model_path='path/to/model.pkl')

    queries = np.random.uniform(-1, 1, (1000, 3))
    distances = sdf.evaluate(queries)
    gradients = sdf.gradient(queries)
"""

from .network import FundamentalFormPredictor
from .infer import PointCloudSDF, read_point_cloud
from .shape_3d import (
	PointCloud3D,
	write_point_cloud,
	TriangleMesh,
	TorusDistanceField,
	get_neighbors,
	compute_optimal_radius,
	fit_tori_from_forms,
	TorusDistanceField,
	BoundingVolumeHierarchy,
)
