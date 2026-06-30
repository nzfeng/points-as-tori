from __future__ import annotations

import os
import pickle
import numpy as np
from typing import Optional, Tuple

from .shape_3d import TorusDistanceField, PointCloud3D, fit_tori_from_forms
from .network import FundamentalFormPredictor

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'FundamentalFormPredictor.pkl')
K_NEIGHBORS_ACCELERATION = 32


class PointsAsTori:
	"""
	Object that stores an oriented point cloud and its (pre-)fitted tori.

	Usage:

		pat = PointsAsTori(points, normals, model_path='path/to/model.pkl')
		distances = pat.signed_distance(queries)   # (n_queries, ) array
		gradients = pat.sdf_gradient(queries)	  # (n_queries, 3) array
	"""

	def __init__(
		self,
		points: np.ndarray,
		normals: np.ndarray,
		tori: dict = None,
		tori_filepath: str = None,
		model_path: str = None,
	) -> None:
		"""
		Args:
			points: (|P|, 3) array of point positions
			normals: (|P|, 3) array of point normals
			tori: dictionary of tori, has format
				 {centers: (|P|, 3) NumPy array, axes: (|P|, 2) NumPy array, major_radii: (|P|,) NumPy array, minor_radii: (|P|,) NumPy array}
			tori_filepath: filepath to precomputed tori
			model_path: path to a .pkl model file representing a trained neural network.
				Defaults to the included pre-trained model.
		"""
		self._tdf = TorusDistanceField(points, normals)

		if tori_filepath is not None:
			tori = self._load_tori(tori_filepath)
			self._tdf.set_tori(tori['centers'], tori['axes'], tori['major_radii'], tori['minor_radii'])

		if tori is None:
			if model_path is None:
				model_path = DEFAULT_MODEL_PATH
			if not os.path.isfile(model_path):
				raise FileNotFoundError(f'Model file not found: {model_path}')

			model, k_nb = PointsAsTori._load_model(model_path)

			coeffs = model.precompute_coefficients_in_chunks(points, normals, k_nb, chunk_size=50000)
			centers, axes, major_radii, minor_radii = fit_tori_from_forms(points, normals, np.array(coeffs))

			self._tdf.set_tori(centers, axes, major_radii, minor_radii)

		else:
			self._tdf.set_tori(tori['centers'], tori['axes'], tori['major_radii'], tori['minor_radii'])

	def signed_distance(self, queries: np.ndarray, accelerate: bool = True) -> np.ndarray:
		"""
		Evaluate signed distance at the given query points.

		Args:
			queries: query locations, shape (n_queries, 3)

		Returns:
			(n_queries, ) array of distances
		"""
		self._tdf.set_k_evaluation(K_NEIGHBORS_ACCELERATION if accelerate else -1)
		return self._tdf.evaluate_distance(queries)

	def sdf_gradient(self, queries: np.ndarray, accelerate: bool = True) -> np.ndarray:
		"""
		Evaluate the gradient of signed distance at the given query points.

		Args:
			queries: query locations, shape (n_queries, 3)

		Returns:
			(n_queries, 3) array
		"""
		self._tdf.set_k_evaluation(K_NEIGHBORS_ACCELERATION if accelerate else -1)
		return self._tdf.evaluate_gradient(queries)

	def signed_distance_and_gradient(
		self, queries: np.ndarray, accelerate: bool = True
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Return (distances, gradients) at the given query points.
		"""
		self._tdf.set_k_evaluation(K_NEIGHBORS_ACCELERATION if accelerate else -1)
		distances, gradients, _ = self._tdf.evaluate_distance_gradient_laplacian(queries)
		return distances, gradients

	@staticmethod
	def _load_model(model_path: str):
		model, k_neighbors = FundamentalFormPredictor.load_saved_model(model_path)
		return model, k_neighbors

	def save_tori(self, filepath: str = 'tori/tori.pkl') -> None:
		"""
		Save tori as binary file.

		Args:
			tori: dictionary of tori, has format
				  {centers: (|P|, 3) NumPy array, axes: (|P|, 2) NumPy array, major_radii: (|P|,) NumPy array, minor_radii: (|P|,) NumPy array}
			filepath: output filepath
		"""
		dirpath = os.path.dirname(filepath)
		if dirpath:
			os.makedirs(dirpath, exist_ok=True)
		tori = self._tdf.get_tori()
		with open(filepath, 'wb') as f:
			pickle.dump(tori, f)

	def _load_tori(self, filepath: str = 'tori/tori.pkl') -> dict:
		"""
		Load tori from a binary file saved by save_tori().

		Args:
			filepath: path to the tori file

		Returns:
			dictionary with keys 'centers', 'axes', 'major_radii', 'minor_radii'
		"""
		with open(filepath, 'rb') as f:
			tori = pickle.load(f)  # tuple
			return {'centers': tori[0], 'axes': tori[1], 'major_radii': tori[2], 'minor_radii': tori[3]}


def read_point_cloud(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Load an oriented point cloud from a PLY or OBJ file.

	Args:
		filepath: location of point cloud file

	Returns:
		points: (N, 3) array
		normals: (N, 3) array
	"""
	pc = PointCloud3D.read(filepath)
	return pc.points, pc.normals
