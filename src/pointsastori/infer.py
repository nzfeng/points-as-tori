# TODO: High-level API

from __future__ import annotations

import os
import pickle
import numpy as np
from typing import Optional, Tuple

from .shape_3d import TorusDistanceField, PointCloud3D, fit_tori_from_forms


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'FundamentalFormPredictor.pkl')
K_NEIGHBORS_ACCELERATION = 32

def _load_model(model_path: str):
    from .fundamental_forms import FundamentalFormPredictor
    model, k_neighbors = FundamentalFormPredictor.load_saved_model(model_path)
    return model, k_neighbors


class PointCloudSDF:
    """
    Evaluates signed distance from an oriented point cloud using pre-fitted tori.

    Usage::

        sdf = PointCloudSDF(points, normals, model_path='path/to/model.pkl')
        distances = sdf.evaluate(queries)      # (Q,) array
        gradients = sdf.gradient(queries)      # (Q, 3) array
    """

    def __init__(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        model_path: Optional[str] = None,
        k_neighbors: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        points : (N, 3) array of point positions
        normals : (N, 3) array of unit normals
        model_path : path to a .pkl model file produced by training.
            Defaults to the bundled pre-trained model.
        k_neighbors : neighborhood size used by the network.
            Defaults to the value stored in the model file.
        """
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        model, k_nb = _load_model(model_path)
        if k_neighbors is not None:
            k_nb = k_neighbors

        coeffs = model.precompute_coefficients(points, normals, k_nb).block_until_ready()
        centers, axes, major_radii, minor_radii = fit_tori_from_forms(points, normals, np.array(coeffs))

        self._tdf = TorusDistanceField(points, normals, compute_areas=False, use_areas=False)
        self._tdf.set_tori(centers, axes, major_radii, minor_radii)
        self._tdf.set_k_evaluation(-1)

    def evaluate(self, queries: np.ndarray) -> np.ndarray:
        """Return signed distances at query points. Shape: (Q,)."""
        return self._tdf.evaluate_distance(queries)

    def gradient(self, queries: np.ndarray) -> np.ndarray:
        """Return gradient of the signed distance field at query points. Shape: (Q, 3)."""
        return self._tdf.evaluate_gradient(queries)

    def evaluate_and_gradient(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (distances, gradients) at query points."""
        distances, gradients, _ = self._tdf.evaluate_distance_gradient_laplacian(queries)
        return distances, gradients


def read_point_cloud(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an oriented point cloud from a PLY or OBJ file.

    Returns
    -------
    points : (N, 3) array
    normals : (N, 3) array
    """
    pc = PointCloud3D.read(filepath)
    return pc.points, pc.normals
