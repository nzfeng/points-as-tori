from __future__ import annotations

import numpy as np
from typing import Tuple
import gc

import pat_bindings as utilsb


# ========================================================================
# POINT CLOUDS
# ========================================================================


class PointCloud3D:
	def __init__(self, points: np.ndarray, normals: np.ndarray, colors: np.ndarray = None) -> None:
		self.points = points
		self.normals = normals
		self.colors = colors
		self.size = len(self.points)

	def center_and_scale(self):
		centroid = np.mean(self.points, axis=0)
		centered_points = self.points - centroid
		max_range = np.max(np.ptp(centered_points, axis=0))
		self.points = centered_points / max_range

	@staticmethod
	def read(filepath: str) -> PointCloud3D:
		extension = filepath.split('.')[-1]
		if extension == 'obj' or extension == 'pc':
			points = []
			normals = []
			colors = []
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'v':
						p = list(map(float, parts[1:4]))
						points.append(p)
					elif parts[0] == 'vn':
						n = list(map(float, parts[1:4]))
						n = np.asarray(n)
						n /= np.linalg.norm(n)
						normals.append(n)
					elif parts[0] == 'vt' or parts[0] == 'vc':
						c = list(map(float, parts[1:4]))
						colors.append(c)
			if len(colors) > 0:
				return PointCloud3D(np.array(points), np.array(normals), np.array(colors))
			if len(normals) == 0:
				raise ValueError(f'No normals found in point cloud')
			return PointCloud3D(np.array(points), np.array(normals))
		elif extension == 'ply':
			return PointCloud3D.read_PLY(filepath)
		else:
			raise ValueError(f'Unsupported file format: {extension}')

	@staticmethod
	def read_PLY(filepath: str) -> PointCloud3D:
		import open3d as o3d

		pcd = o3d.io.read_point_cloud(filepath)
		points = np.asarray(pcd.points)
		colors = np.asarray(pcd.colors)
		normals = np.asarray(pcd.normals)
		return PointCloud3D(points, normals, colors)

	def export(self, filepath: str, binary: bool = True):
		"""
		Export point cloud as PLY file (binary or ASCII).

		Args:
			points: (n_points, 3) point positions
			filepath: path to save PLY file
			normals: (n_points, 3) optional point normals
			colors: (n_points, 3) optional RGB colors in [0, 1] or [0, 255]
			binary: if True, use binary format; otherwise ASCII
		"""
		try:
			from plyfile import PlyData, PlyElement

			n_points = len(self.points)

			# Build vertex dtype based on available data
			dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

			if self.normals is not None:
				dtype_list.extend([('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])

			# Create structured array
			vertex_data = np.empty(n_points, dtype=dtype_list)
			vertex_data['x'] = self.points[:, 0]
			vertex_data['y'] = self.points[:, 1]
			vertex_data['z'] = self.points[:, 2]

			if normals is not None:
				vertex_data['nx'] = self.normals[:, 0]
				vertex_data['ny'] = self.normals[:, 1]
				vertex_data['nz'] = self.normals[:, 2]

			if self.colors is not None:
				vertex_data['red'] = self.colors[:, 0]
				vertex_data['green'] = self.colors[:, 1]
				vertex_data['blue'] = self.colors[:, 2]

			# Create PLY element
			vertex_element = PlyElement.describe(vertex_data, 'vertex')

			# Write PLY file
			ply_data = PlyData([vertex_element])

			if binary:
				ply_data.write(filepath)
			else:
				ply_data.text = True
				ply_data.write(filepath)

		except ImportError:
			print('Warning: plyfile not available, writing PLY in ASCII format')

			with open(filepath, 'w') as f:
				# Header
				f.write('ply\n')
				f.write('format ascii 1.0\n')
				f.write(f'element vertex {len(self.points)}\n')
				f.write('property float x\n')
				f.write('property float y\n')
				f.write('property float z\n')

				if normals is not None:
					f.write('property float nx\n')
					f.write('property float ny\n')
					f.write('property float nz\n')

				if colors is not None:
					f.write('property uchar red\n')
					f.write('property uchar green\n')
					f.write('property uchar blue\n')

				f.write('end_header\n')

				# Vertices
				for i, p in enumerate(self.points):
					line = f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}'

					if self.normals is not None:
						n = self.normals[i]
						line += f' {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}'

					if self.colors is not None:
						c = self.colors[i]
						if c.max() <= 1.0:
							c = (c * 255).astype(int)
						line += f' {int(c[0])} {int(c[1])} {int(c[2])}'

					f.write(line + '\n')


def write_point_cloud(points: np.ndarray, normals: np.ndarray, filepath: str) -> None:
	"""
	Write point cloud as OBJ.
	"""
	with open(filepath, 'w') as f:
		for p in points:
			f.write(f'v {p[0]} {p[1]} {p[2]}\n')
		for n in normals:
			f.write(f'vn {n[0]} {n[1]} {n[2]}\n')


# ========================================================================
# MESHES
# ========================================================================


class TriangleMesh:
	def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
		self.bound_object = utilsb.Mesh3D(np.asfortranarray(vertices.T), np.asfortranarray(faces.T))
		self.vertices = vertices
		self.faces = faces
		self.vertex_normals = self._compute_vertex_normals()

	def _compute_vertex_normals(self) -> np.ndarray:
		n_vertices = self.vertices.shape[0]
		vertex_normals = np.zeros((n_vertices, 3), dtype=np.float64)
		v0 = self.vertices[self.faces[:, 0]]
		v1 = self.vertices[self.faces[:, 1]]
		v2 = self.vertices[self.faces[:, 2]]
		e1 = v1 - v0
		e2 = v2 - v0
		face_normals = np.cross(e1, e2)
		np.add.at(vertex_normals, self.faces[:, 0], face_normals)
		np.add.at(vertex_normals, self.faces[:, 1], face_normals)
		np.add.at(vertex_normals, self.faces[:, 2], face_normals)
		norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
		norms = np.maximum(norms, 1e-10)
		return vertex_normals / norms

	@staticmethod
	def read_OBJ(filepath: str) -> TriangleMesh:
		vertices = []
		faces = []
		with open(filepath, 'r') as file:
			for line in file:
				parts = line.strip().split()
				if not parts:
					continue
				elif parts[0] == 'v':
					p = list(map(float, parts[1:4]))
					vertices.append(p)
				elif parts[0] == 'f':
					idxs = [int(p.split('/')[0]) - 1 for p in parts[1:]]
					faces.append(idxs)
		return TriangleMesh(
			np.array(vertices, dtype=np.float64),
			np.array(faces, dtype=np.int64),
		)

	def write_OBJ(self, filepath: str):
		with open(filepath, 'w') as file:
			for vertex in self.vertices:
				file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
			for face in self.faces:
				file.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n')

	def export_PLY(self, filepath: str, binary: bool = True):
		try:
			from plyfile import PlyData, PlyElement

			# Create vertex element
			vertex_data = np.array(
				[(v[0], v[1], v[2]) for v in self.vertices], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
			)
			vertex_element = PlyElement.describe(vertex_data, 'vertex')

			# Create face element
			# Handle both triangular and quad faces
			if self.faces.shape[1] == 3:
				face_data = np.array(
					[([f[0], f[1], f[2]],) for f in self.faces], dtype=[('vertex_indices', 'i4', (3,))]
				)
			elif self.faces.shape[1] == 4:
				face_data = np.array(
					[([f[0], f[1], f[2], f[3]],) for f in self.faces], dtype=[('vertex_indices', 'i4', (4,))]
				)
			else:
				raise ValueError(f'Faces must have 3 or 4 vertices, got {self.faces.shape[1]}')

			face_element = PlyElement.describe(face_data, 'face')

			# Create PLY data and write
			ply_data = PlyData([vertex_element, face_element])

			if binary:
				ply_data.write(filepath)
			else:
				ply_data.text = True
				ply_data.write(filepath)

		except ImportError:
			# Fallback: write PLY manually
			print('Warning: plyfile not available, writing PLY in ASCII')

			with open(filepath, 'w') as f:
				# Header
				f.write('ply\n')
				f.write('format ascii 1.0\n')
				f.write(f'element vertex {len(vertices)}\n')
				f.write('property float x\n')
				f.write('property float y\n')
				f.write('property float z\n')
				f.write(f'element face {len(faces)}\n')
				f.write('property list uchar int vertex_indices\n')
				f.write('end_header\n')

				# Vertices
				for v in self.vertices:
					f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')

				# Faces
				n_verts_per_face = self.faces.shape[1]
				for face in self.faces:
					f.write(f'{n_verts_per_face}')
					for idx in face:
						f.write(f' {idx}')
					f.write('\n')

	def get_vertices(self) -> np.ndarray:
		return self.bound_object.get_vertices().T

	def get_faces(self) -> np.ndarray:
		return self.bound_object.get_faces().T

	def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
		return self.bound_object.bounding_box()

	def center_and_scale(self):
		self.bound_object = self.bound_object.center_and_scale()
		self.vertices = self.get_vertices()
		self.faces = self.get_faces()

	def sample_point_cloud(
		self,
		n_cameras: int,
		sensor_size: float,
		grid_spacing: float,
		seed: int,
		sigma_p: float = 0.0,
		sigma_n: float = 0.0,
	):
		points, normals = self.bound_object.sample_point_cloud(
			n_cameras, sensor_size, grid_spacing, seed, sigma_p, sigma_n
		)
		return points.T, normals.T

	def sample_uniform_point_cloud(
		self,
		n_points: int,
		max_noise_level: float = 0.01,
		max_normals_to_flip: float = 0.1,
		seed: int = 0,
	):
		points, normals = self.bound_object.sample_uniform_point_cloud(
			n_points, max_noise_level, max_normals_to_flip, seed
		)
		return points.T, normals.T

	def sample_farthest_point_cloud(
		self,
		n_points: int,
		max_noise_level: float = 0.01,
		max_normals_to_flip: float = 0.1,
		seed: int = 0,
	):
		points, normals = self.bound_object.sample_farthest_point_cloud(
			n_points, max_noise_level, max_normals_to_flip, seed
		)
		return points.T, normals.T

	def sample_even_raycasted_point_cloud(
		self,
		n_cameras,
		n_points: int,
		sensor_size: float,
		grid_spacing_min: float = 0.01,
		grid_spacing_max: float = 0.1,
		seed: int = 0,
	):
		points, normals = self.bound_object.sample_even_raycasted_point_cloud(
			n_cameras, n_points, sensor_size, grid_spacing_min, grid_spacing_max, seed
		)
		return points.T, normals.T

	def sample_uneven_point_cloud(
		self,
		n_cameras: int,
		sensor_size: float,
		grid_spacing_min: float = 0.01,
		grid_spacing_max: float = 0.1,
		max_noise_level: float = 0.01,
		max_normals_to_flip: float = 0.1,
		max_points: int = 2048,
		seed: int = 0,
	):
		points, normals = self.bound_object.sample_uneven_point_cloud(
			n_cameras,
			sensor_size,
			grid_spacing_min,
			grid_spacing_max,
			max_noise_level,
			max_normals_to_flip,
			max_points,
			seed,
		)
		return points.T, normals.T

	def sample_points_and_normals_uniformly(
		self,
		n_points: int,
		seed: int = 0,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		sigma_p: float = 0.0,
		sigma_n: float = 0.0,
	) -> np.ndarray:
		points, normals = self.bound_object.sample_points_and_normals_uniformly(
			n_points, seed, bbox_min, bbox_max, sigma_p, sigma_n
		)
		return points.T, normals.T

	def sample_uniformly(
		self,
		n_points: int,
		seed: int = 0,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
	) -> np.ndarray:
		return self.bound_object.sample_uniformly(n_points, seed, bbox_min, bbox_max).T

	def sample_narrow_band(
		self,
		n_points: int,
		offset: float = 0.1,
		seed: int = 0,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
	) -> np.ndarray:
		return self.bound_object.sample_narrow_band(n_points, offset, seed, bbox_min, bbox_max).T

	def evaluate_signed_distance_and_gradient(
		self, q: np.ndarray, isovalue: float = 0.0
	) -> Tuple[np.ndarray, np.ndarray]:
		if q.ndim == 1:
			q = q[None, :]
		result = self.bound_object.evaluate_signed_distance_and_gradient(np.asfortranarray(q.T))
		return result[0].T, result[1].T

	def evaluate_signed_distance(self, q: np.ndarray, parallelize: bool = True) -> np.ndarray:
		if q.ndim == 1:
			q = q[None, :]
		return self.bound_object.evaluate_signed_distance(np.asfortranarray(q.T), parallelize).T

	def evaluate_signed_distance_gradient(self, q: np.ndarray) -> np.ndarray:
		_, grad = self.evaluate_signed_distance_and_gradient(q)
		return grad

	def evaluate_gwn(self, q: np.ndarray) -> np.ndarray:
		return self.bound_object.evaluate_gwn(np.asfortranarray(q.T))

	def evaluate_fast_gwn(self, q: np.ndarray) -> np.ndarray:
		return self.bound_object.evaluate_fast_gwn(np.asfortranarray(q.T))


# ========================================================================
# TORUS SDF EVALUATION
# ========================================================================


def get_neighbors(points: np.ndarray, k_neighbors: int, outliers: np.ndarray = np.array([])):
	"""Return a (|P|, k) array of k-nearest neighbor indices."""
	neighbors = utilsb.get_neighbors(np.asfortranarray(points.T), k_neighbors, outliers)
	return neighbors.T


def compute_optimal_radius(points, k_neighbors=64):
	D = utilsb.get_average_neighbor_distance(np.asfortranarray(points.T), k_neighbors)
	lam = 1e3 / D
	return 128.0 / lam


def fit_tori_from_forms(
	points: np.ndarray,
	normals: np.ndarray,
	coefficients: np.ndarray,
) -> np.ndarray:
	centers, axes, major_radii, minor_radii = utilsb.fit_tori_from_forms(
		np.asfortranarray(points.T),
		np.asfortranarray(normals.T),
		np.asfortranarray(coefficients.T),
	)
	return centers.T, axes.T, major_radii, minor_radii


class TorusDistanceField:
	def __init__(self, points: np.ndarray, normals: np.ndarray):
		self.bound_object = utilsb.TorusDistanceField(np.asfortranarray(points.T), np.asfortranarray(normals.T))

	def get_neighbors(self, k_neighbors: int, i: int):
		return self.bound_object.get_neighbors(k_neighbors, i)

	def get_all_neighbors(self, k_neighbors: int):
		return self.bound_object.get_neighbors(k_neighbors).T

	def get_all_neighbors_and_subsample(self, k_neighbors: int, n_subsample: int):
		neighbors, neighbors_subsampled = self.bound_object.get_all_neighbors_and_subsample(k_neighbors, n_subsample)
		return neighbors.T, neighbors_subsampled.T

	def evaluate_distance(self, queries: np.ndarray, isovalue: float = 0.0, parallelize: bool = True):
		if queries.ndim == 1:
			queries = queries[None, :]
		return self.bound_object.evaluate_distance(np.asfortranarray(queries.T), isovalue, parallelize)

	def evaluate_gradient(self, queries: np.ndarray, parallelize: bool = True):
		return self.bound_object.evaluate_gradient(np.asfortranarray(queries.T), parallelize).T

	def evaluate_laplacian(self, queries: np.ndarray, parallelize: bool = True):
		return self.bound_object.evaluate_laplacian(np.asfortranarray(queries.T), parallelize)

	def evaluate_distance_gradient_laplacian(self, queries: np.ndarray, isovalue: float = 0.0):
		input_ndim = queries.ndim
		if input_ndim == 1:
			queries = queries[None, :]
		distances, gradients, laplacians = self.bound_object.evaluate_distance_gradient_laplacian(
			np.asfortranarray(queries.T), isovalue
		)
		if input_ndim == 1:
			return np.squeeze(distances), np.squeeze(gradients.T, axis=0).T, np.squeeze(laplacians)
		return distances, gradients.T, laplacians

	def log_sum_exp(self, queries, parallelize=True):
		if queries.ndim == 1:
			queries = queries[None, :]
		return self.bound_object.log_sum_exp(np.asfortranarray(queries.T), parallelize)

	def signed_log_sum_exp(self, queries: np.ndarray, parallelize: bool = True) -> np.ndarray:
		if queries.ndim == 1:
			queries = queries[None, :]
		return self.bound_object.signed_log_sum_exp(np.asfortranarray(queries.T), parallelize)

	def signed_log_sum_exp_gradient(self, queries: np.ndarray, parallelize: bool = True) -> np.ndarray:
		if queries.ndim == 1:
			queries = queries[None, :]
		return self.bound_object.signed_log_sum_exp_gradient(np.asfortranarray(queries.T), parallelize).T

	def self_normalized_signed_log_sum_exp(self, queries: np.ndarray, parallelize: bool = True) -> np.ndarray:
		if queries.ndim == 1:
			queries = queries[None, :]
		return self.bound_object.self_normalized_signed_log_sum_exp(np.asfortranarray(queries.T), parallelize)

	def self_normalized_signed_log_sum_exp_gradient(self, queries: np.ndarray, parallelize: bool = True) -> np.ndarray:
		if queries.ndim == 1:
			queries = queries[None, :]
		return self.bound_object.self_normalized_signed_log_sum_exp_gradient(
			np.asfortranarray(queries.T), parallelize
		).T

	def winding_number(self, queries: np.ndarray, parallelize: bool = True) -> np.ndarray:
		return self.bound_object.winding_number(np.asfortranarray(queries.T), parallelize)

	def fast_winding_number_precompute(self) -> None:
		self.bound_object.fast_winding_number_precompute()

	def fast_winding_number(self, queries: np.ndarray) -> np.ndarray:
		if queries.ndim == 1:
			queries = queries[None, :]
		return self.bound_object.fast_winding_number(queries)

	def regularized_winding_number(self, queries: np.ndarray, epsilon: float, parallelize: bool = True) -> np.ndarray:
		return self.bound_object.regularized_winding_number(np.asfortranarray(queries.T), epsilon, parallelize)

	def fit_tori_from_forms(self, coefficients: np.ndarray) -> np.ndarray:
		self.bound_object.fit_tori_from_forms(np.asfortranarray(coefficients.T))
		centers, axes, major_radii, minor_radii = self.bound_object.get_tori()
		return centers.T, axes.T, major_radii, minor_radii

	def get_points(self):
		return self.bound_object.get_points().T

	def get_points_and_normals(self):
		points, normals = self.bound_object.get_points_and_normals()
		return points.T, normals.T

	def get_point_areas(self):
		return self.bound_object.get_point_areas()

	def set_tori(self, centers: np.ndarray, axes: np.ndarray, major_radii: np.ndarray, minor_radii: np.ndarray):
		self.bound_object.set_tori(np.asfortranarray(centers.T), np.asfortranarray(axes.T), major_radii, minor_radii)

	def set_k_evaluation(self, k):
		self.bound_object.set_k_evaluation(k)

	def set_radius_evaluation(self, r):
		self.bound_object.set_radius_evaluation(r)

	def set_lambda_scale(self, scale):
		self.bound_object.set_lambda_scale(scale)

	def set_outliers(self, indices):
		self.bound_object.set_outliers(indices)


# ========================================================================
# BVH
# ========================================================================


class BoundingVolumeHierarchy:
	def __init__(self, *args):
		if len(args) == 1:
			points = args[0]
			self.bound_object = utilsb.BVH(np.asfortranarray(points.T))
			self.primitive_type = 'point'
		elif len(args) == 2:
			vertices, faces = args
			self.bound_object = utilsb.BVH(np.asfortranarray(vertices.T), np.asfortranarray(faces.T, dtype=np.int32))
			self.primitive_type = 'triangle'
		else:
			raise ValueError('BVH constructor takes 1 argument (points) or 2 arguments (vertices, faces)')

	def find_closest_point(self, query: np.ndarray) -> Tuple[int, float]:
		return self.bound_object.find_closest_point(query)

	def find_k_nearest(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
		return self.bound_object.find_k_nearest(query, k)

	def find_points_in_radius(self, query: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
		return self.bound_object.find_points_in_radius(query, radius)

	def intersect_ray(
		self,
		ray_origin: np.ndarray,
		ray_dir: np.ndarray,
		t_min: float = 0.0,
		t_max: float = np.inf,
		sphere_radius: float = 0.01,
	) -> Tuple[bool, float, int, np.ndarray]:
		return self.bound_object.intersect_ray(ray_origin, ray_dir, t_min, t_max, sphere_radius)

	def closest_point_on_mesh(self, query: np.ndarray) -> Tuple[np.ndarray, float, int]:
		if self.primitive_type != 'triangle':
			raise ValueError('closest_point_on_mesh only works for triangle meshes')
		return self.bound_object.closest_point_on_mesh(query)

	def get_gpu_data(self) -> Tuple[np.ndarray, np.ndarray]:
		return self.bound_object.get_gpu_data()

	def get_num_nodes(self) -> int:
		return self.bound_object.get_num_nodes()

	def get_num_primitives(self) -> int:
		return self.bound_object.get_num_primitives()
