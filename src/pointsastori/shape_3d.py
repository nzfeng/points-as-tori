from __future__ import annotations

import numpy as np
from typing import Tuple
import jax
import jax.numpy as jnp

import pat_bindings as utilsb


# ========================================================================
# POINT CLOUDS
# ========================================================================

def point_cloud_bounding_box(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Args:
		positions: (|P|, 3) array

	Returns:
		(bbox_min, bbox_max) where both bbox_min, bbox_max are (3,) arrays
	"""
	bbox_min, bbox_max = utilsb.point_cloud_bounding_box(np.asfortranarray(positions.T))
	return bbox_min, bbox_max

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

			if self.normals is not None:
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

def export_PLY_with_colors(
	vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, filepath: str, binary: bool = True
):
	"""
	Export mesh with vertex colors to PLY.

	Args:
		vertices: (n_verts, 3)
		faces: (n_faces, 3) for triangles or (n_faces, 4) for quads
		colors: (n_verts, 3) RGB in [0, 255] or [0, 1]
		filepath: output path
		binary: use binary format
	"""
	try:
		from plyfile import PlyData, PlyElement

		# Ensure colors are uint8
		if colors.max() <= 1.0:
			colors = (colors * 255).astype(np.uint8)
		else:
			colors = colors.astype(np.uint8)

		# Create vertex element with colors
		vertex_data = np.empty(
			len(vertices), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
		)
		vertex_data['x'] = vertices[:, 0]
		vertex_data['y'] = vertices[:, 1]
		vertex_data['z'] = vertices[:, 2]
		vertex_data['red'] = colors[:, 0]
		vertex_data['green'] = colors[:, 1]
		vertex_data['blue'] = colors[:, 2]

		vertex_element = PlyElement.describe(vertex_data, 'vertex')

		# Create face element - handle both triangles and quads
		n_verts_per_face = faces.shape[1]

		if n_verts_per_face == 3:
			face_data = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
		elif n_verts_per_face == 4:
			face_data = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (4,))])
		else:
			raise ValueError(f'Faces must have 3 or 4 vertices, got {n_verts_per_face}')

		face_data['vertex_indices'] = faces

		face_element = PlyElement.describe(face_data, 'face')

		# Write PLY
		ply_data = PlyData([vertex_element, face_element])
		if binary:
			ply_data.write(filepath)
		else:
			ply_data.text = True
			ply_data.write(filepath)

	except ImportError:
		print('Warning: plyfile not available, writing ASCII PLY manually')

		n_verts_per_face = faces.shape[1]

		# Ensure colors are integers
		if colors.max() <= 1.0:
			colors = (colors * 255).astype(int)
		else:
			colors = colors.astype(int)

		with open(filepath, 'w') as f:
			f.write('ply\nformat ascii 1.0\n')
			f.write(f'element vertex {len(vertices)}\n')
			f.write('property float x\nproperty float y\nproperty float z\n')
			f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
			f.write(f'element face {len(faces)}\n')
			f.write('property list uchar int vertex_indices\n')
			f.write('end_header\n')

			for i, v in enumerate(vertices):
				c = colors[i]
				f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n')

			for face in faces:
				f.write(f'{n_verts_per_face}')
				for idx in face:
					f.write(f' {idx}')
				f.write('\n')

# ========================================================================
# SDFs
# ========================================================================

SDF_SHAPES = [
	'circle',
	'pie',
	'arc',
	'pill',
	'vesica',
	'box',
	'cross',
	'isosceles',
	'triangle',
	'hexagon',
	'ellipse',
	'heart',
	'pentagon',
	'moon',
	'trapezoid',
	'quad',
]

class SDF2D:
	def __init__(self, instance=utilsb.Circle2D):
		self.bound_object = instance
		self._has_gradient = self.bound_object.has_gradient()

	def evaluate_signed_distance_and_gradient(self, q: jnp.ndarray, isovalue: float = 0.0) -> tuple[float, jnp.ndarray]:
		if self._has_gradient:
			if q.ndim == 1:
				result = self.bound_object.evaluate_with_gradient(np.asfortranarray(q), isovalue)
				return result[0].T, result[1].T
			result = self.bound_object.evaluate_with_gradient_batch(np.asfortranarray(q.T), isovalue)
			return result[0].T, result[1].T

		return self.evaluate_signed_distance(q, isovalue), self.evaluate_signed_distance_gradient(q)

	def evaluate_signed_distance(self, q: jnp.ndarray, isovalue: float = 0.0) -> float:
		"""
		A positive offset value means the shape gets inflated.
		"""
		if q.ndim == 1:
			return self.bound_object.evaluate(np.asfortranarray(q), isovalue)
		return self.bound_object.evaluate_batch(np.asfortranarray(q).T, isovalue).T

	def evaluate_signed_distance_gradient_3D(self, q: np.ndarray, is_extrusion: bool, param: float):
		gradients = self.bound_object.evaluate_signed_distance_gradient(q, is_extrusion, param)
		return gradients.T

	def evaluate_signed_distance_gradient(self, q: jnp.ndarray) -> jnp.ndarray:
		if self._has_gradient:
			_, grad = self.evaluate_signed_distance_and_gradient(q)
			return grad

		if q.ndim == 1:
			q = q[None, :]
		return self.bound_object.evaluate_gradient_batch(np.asfortranarray(q).T).T

	def evaluate_signed_distance_laplacian(self, q: jnp.ndarray) -> jnp.ndarray:
		if q.ndim == 1:
			q = q[None, :]

		return self.bound_object.evaluate_laplacian_batch(np.asfortranarray(q).T)

	def evaluate_revolution(self, q: np.ndarray, o: float) -> np.ndarray:
		if q.ndim == 1:
			q = q[None, :]
		return self.bound_object.evaluate_revolution(np.asfortranarray(q.T), o)

	def evaluate_extrusion(self, q: np.ndarray, h: float) -> np.ndarray:
		if q.ndim == 1:
			q = q[None, :]
		return self.bound_object.evaluate_extrusion(np.asfortranarray(q.T), h)

	def sample_revolution(
		self,
		n_samples: int,
		o: float,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		sigma_p: float = 0.0,
		sigma_n: float = 0.0,
		seed: int = 0,
	) -> np.ndarray:
		points, normals = self.bound_object.sample_revolution(n_samples, o, bbox_min, bbox_max, sigma_p, sigma_n, seed)
		return points.T, normals.T

	def sample_extrusion(
		self,
		n_samples: int,
		h: float,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		sigma_p: float = 0.0,
		sigma_n: float = 0.0,
		seed: int = 0,
	) -> np.ndarray:
		points, normals = self.bound_object.sample_extrusion(n_samples, h, bbox_min, bbox_max, sigma_p, sigma_n, seed)
		return points.T, normals.T

	def sample_revolution_narrow_band(
		self,
		n_samples: int,
		o: float,
		offset: float,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		seed: int = 0,
	) -> np.ndarray:
		samples = self.bound_object.sample_revolution_narrow_band(n_samples, o, offset, bbox_min, bbox_max, seed)
		return samples.T

	def sample_extrusion_narrow_band(
		self,
		n_samples: int,
		h: float,
		offset: float,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		seed: int = 0,
	) -> np.ndarray:
		samples = self.bound_object.sample_extrusion_narrow_band(n_samples, h, offset, bbox_min, bbox_max, seed)
		return samples.T

	def sample_farthest_point_cloud(
		self,
		n_points: int,
		is_extrusion: bool,
		SDF3D_param: float,
		seed: int,
	):
		points, normals = self.bound_object.sample_farthest_point_cloud(n_points, is_extrusion, SDF3D_param, seed)
		return points.T, normals.T

	def ray_intersections(self, ros: jnp.ndarray, rds: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
		result = self.bound_object.ray_intersect_batch(np.asfortranarray(ros.T), np.asfortranarray(rds.T))
		return result[0].T, result[1].T

	@staticmethod
	def load_SDF(filepath):
		parts = re.split('_|/|-|\.', filepath)
		if 'circle' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'r':
						radius = float(parts[1])
						return Circle2D(radius)
		elif 'pie' in parts:
			radius = None
			t = None
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'r':
						radius = float(parts[1])
					elif parts[0] == 't':
						t = float(parts[1])
			return Pie2D(radius, t)
		elif 'arc' in parts:
			sc = None
			ra = None
			rb = None
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'sc':
						sc = jnp.array(list(map(float, parts[1:3])))
					elif parts[0] == 'ra':
						ra = float(parts[1])
					elif parts[0] == 'rb':
						rb = float(parts[1])
			return Arc2D(sc, ra, rb)
		elif 'pill' in parts:
			a = None
			b = None
			r = None
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'a':
						a = jnp.array(list(map(float, parts[1:3])))
					elif parts[0] == 'b':
						b = jnp.array(list(map(float, parts[1:3])))
					elif parts[0] == 'r':
						r = float(parts[1])
			return Pill2D(a, b, r)
		elif 'vesica' in parts:
			r = None
			d = None
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'r':
						r = float(parts[1])
					elif parts[0] == 'd':
						d = float(parts[1])
			return Vesica2D(r, d)
		elif 'box' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'b':
						b = jnp.array(list(map(float, parts[1:3])))
						return Box2D(b)
		elif 'cross' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'b':
						b = jnp.array(list(map(float, parts[1:3])))
						return Cross2D(b)
		elif 'isosceles' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'q':
						q = jnp.array(list(map(float, parts[1:3])))
						return TriangleIsosceles2D(q)
		elif 'triangle' in parts:
			v = []
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'v':
						v.append(list(map(float, parts[1:3])))
			return Triangle2D(v=jnp.array(v))
		elif 'hexagon' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'r':
						r = float(parts[1])
						return Hexagon2D(r)
		elif 'ellipse' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'ab':
						ab = jnp.array(list(map(float, parts[1:3])))
						return Ellipse2D(ab)
		elif 'heart' in parts:
			return Heart2D()
		elif 'pentagon' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'r':
						r = float(parts[1])
						return Pentagon2D(r)
		elif 'moon' in parts:
			d = None
			ra = None
			rb = None
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'd':
						d = float(parts[1])
					elif parts[0] == 'ra':
						ra = float(parts[1])
					elif parts[0] == 'rb':
						rb = float(parts[1])
			return Moon2D(d, ra, rb)
		elif 'trapezoid' in parts:
			ra = None
			rb = None
			he = None
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'ra':
						ra = float(parts[1])
					elif parts[0] == 'rb':
						rb = float(parts[1])
					elif parts[0] == 'he':
						he = float(parts[1])
			return Trapezoid2D(ra, rb, he)
		elif 'quad' in parts:
			v = []
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'v':
						v.append(list(map(float, parts[1:3])))
			return Quad2D(v=jnp.array(v))
		else:
			raise ValueError(f'Unknown SDF type in file: {filepath}')


class Circle2D(SDF2D):
	def __init__(self, radius: float):
		super().__init__(utilsb.Circle2D(radius))
		self.radius = radius


class Pie2D(SDF2D):
	def __init__(self, radius: float, t: float):
		super().__init__(utilsb.Pie2D(radius, t))
		self.radius = radius
		self.t = t


class Arc2D(SDF2D):
	def __init__(self, sc: np.ndarray, ra: float, rb: float):
		super().__init__(utilsb.Arc2D(np.asarray(sc), ra, rb))
		self.sc = sc
		self.ra = ra
		self.rb = rb


class Pill2D(SDF2D):
	def __init__(self, a: np.ndarray, b: np.ndarray, r: float):
		super().__init__(utilsb.Pill2D(np.asarray(a), np.asarray(b), r))
		self.a = a
		self.b = b
		self.r = r


class Vesica2D(SDF2D):
	def __init__(self, r: float, d: float):
		super().__init__(utilsb.Vesica2D(r, d))
		self.r = r
		self.d = d


class Box2D(SDF2D):
	def __init__(self, b: np.ndarray):
		super().__init__(utilsb.Box2D(np.asarray(b)))
		self.b = b


class Cross2D(SDF2D):
	def __init__(self, b: np.ndarray):
		super().__init__(utilsb.Cross2D(np.asarray(b)))
		self.b = b


class TriangleIsosceles2D(SDF2D):
	def __init__(self, q: np.ndarray):
		super().__init__(utilsb.TriangleIsosceles2D(np.asarray(q)))
		self.q = q


class Triangle2D(SDF2D):
	def __init__(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray):
		super().__init__(utilsb.Triangle2D(np.asarray(v0), np.asarray(v1), np.asarray(v2)))
		self.v = [v0, v1, v2]


class Hexagon2D(SDF2D):
	def __init__(self, r: float):
		super().__init__(utilsb.Hexagon2D(r))
		self.r = r


class Ellipse2D(SDF2D):
	def __init__(self, ab: np.ndarray):
		super().__init__(utilsb.Ellipse2D(np.asarray(ab)))
		self.ab = ab


class Heart2D(SDF2D):
	def __init__(self):
		super().__init__(utilsb.Heart2D())


class Pentagon2D(SDF2D):
	def __init__(self, r: float):
		super().__init__(utilsb.Pentagon2D(r))
		self.r = r


class Moon2D(SDF2D):
	def __init__(self, d: float, ra: float, rb: float):
		super().__init__(utilsb.Moon2D(d, np.asarray(ra), np.asarray(rb)))
		self.d = d
		self.ra = ra
		self.rb = rb


class Trapezoid2D(SDF2D):
	def __init__(self, ra: float, rb: float, he: float):
		super().__init__(utilsb.Trapezoid2D(ra, rb, he))
		self.ra = ra
		self.rb = rb
		self.he = he


class Quad2D(SDF2D):
	def __init__(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray):
		super().__init__(utilsb.Quad2D(np.asarray(v0), np.asarray(v1), np.asarray(v2), np.asarray(v3)))
		self.v = [v0, v1, v2, v3]


def sample_SDF(shape_name: str = 'circle', key: jax.random.PRNGKey = None) -> SDF2D:
	"""
	Given a shape, define an SDF with random parameters
	"""
	if key == None:
		key = jax.random.PRNGKey(42)

	# Some guidelines so shapes stay roughly within [-1,1]^2.
	r_min = 0.1
	r_max = 0.9
	r_ratio_min = 0.1  # minimum ratio of smaller radius to larger radius
	r_ratio_max = 0.9  # maximum ratio of smaller radius to larger radius
	t_min = jnp.pi / 6
	t_max = 5 * jnp.pi / 6

	shape = None
	if shape_name == 'circle':
		# Can only choose half-height, which controls global scaling --- equivalent to varying the sampling density per neighborhood.
		# Nevertheless, choose random radius.
		r = jax.random.uniform(key, minval=r_min, maxval=r_max)
		shape = Circle2D(r)
	elif shape_name == 'pie':
		key, subkey1, subkey2 = jax.random.split(key, 3)
		r = jax.random.uniform(subkey1, minval=r_min, maxval=r_max)
		t = jax.random.uniform(subkey2, minval=t_min, maxval=t_max)
		shape = Pie2D(r, t)
	elif shape_name == 'arc':
		key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
		ra = jax.random.uniform(subkey1, minval=r_min, maxval=r_max)  # outer radius
		rb = jax.random.uniform(subkey2, minval=r_ratio_min, maxval=r_ratio_max) * ra  # inner radius
		angle = jax.random.uniform(subkey3, minval=t_min, maxval=t_max)  # angle
		sc = np.array([np.sin(angle), np.cos(angle)])
		shape = Arc2D(sc, ra, rb)
	elif shape_name == 'pill':
		# two endpoints and radius
		key, subkey1, subkey2 = jax.random.split(key, 3)
		halflength = jax.random.uniform(subkey1, minval=r_min, maxval=r_max)
		a = jnp.array([-halflength, 0])  # first endpoint
		b = jnp.array([halflength, 0])  # second endpoint
		r = jax.random.uniform(subkey2, minval=r_ratio_min, maxval=r_ratio_max) * halflength
		shape = Pill2D(a, b, r)
	elif shape_name == 'vesica':
		# radius, center offset
		key, subkey1, subkey2 = jax.random.split(key, 3)
		r = jax.random.uniform(subkey1, minval=r_min, maxval=r_max)
		d = jax.random.uniform(subkey2, minval=r_ratio_min, maxval=r_ratio_max) * (2.0 * r) - r
		shape = Vesica2D(r, d)
	elif shape_name == 'box':
		# half-extents
		key, subkey1, subkey2 = jax.random.split(key, 3)
		bx = jax.random.uniform(subkey1, minval=r_min, maxval=r_max)
		by = jax.random.uniform(subkey2, minval=r_min, maxval=r_max)
		b = jnp.array([bx, by])
		shape = Box2D(b)
	elif shape_name == 'cross':
		# (b.x, b.y) are horizontal/vertical arm dimensions
		key, subkey1, subkey2 = jax.random.split(key, 3)
		bx = jax.random.uniform(subkey1, minval=r_min, maxval=r_max)
		by = jax.random.uniform(subkey2, minval=r_min, maxval=r_max)
		b = jnp.array([bx, by])
		shape = Cross2D(b)
	elif shape_name == 'isosceles':
		# size
		key, subkey1, subkey2 = jax.random.split(key, 3)
		qx = jax.random.uniform(subkey1, minval=r_min, maxval=r_max)  # base length; must be positive
		qy = -jax.random.uniform(subkey2, minval=r_min, maxval=r_max)  # controls "height" above origin
		q = jnp.array([qx, qy])
		shape = TriangleIsosceles2D(q)
	elif shape_name == 'triangle':
		# Start with equilateral triangle, then perturb
		key, subkey1, subkey2 = jax.random.split(key, 3)
		h = jax.random.uniform(subkey1, minval=r_min, maxval=r_max)  # height
		perturbation_amt = r_ratio_max * h
		perturbation_vecs = jax.random.uniform(subkey2, shape=(3, 2))
		perturbation_vecs /= jnp.linalg.norm(perturbation_vecs, axis=1, keepdims=True)
		perturbation = perturbation_amt * perturbation_vecs
		b = 2.0 * h / jnp.sqrt(3.0)
		v0 = jnp.array([-b, -h]) + perturbation[0]
		v1 = jnp.array([b, -h]) + perturbation[1]
		v2 = jnp.array([0, h]) + perturbation[2]
		shape = Triangle2D(v0, v1, v2)
	elif shape_name == 'hexagon':
		r = jax.random.uniform(key, minval=r_min, maxval=r_max)
		shape = Hexagon2D(r)
	elif shape_name == 'ellipse':
		# semi-axes (a, b)
		key, subkey1, subkey2 = jax.random.split(key, 3)
		a = jax.random.uniform(subkey1, minval=r_min, maxval=r_max)
		b = jax.random.uniform(subkey2, minval=r_min, maxval=r_max)
		ab = jnp.array([a, b])
		shape = Ellipse2D(ab)
	elif shape_name == 'heart':
		return Heart2D()
	elif shape_name == 'pentagon':
		r = jax.random.uniform(key, minval=r_min, maxval=r_max)
		shape = Pentagon2D(r)
	elif shape_name == 'moon':
		# d is offset, ra and rb are circle radii
		key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
		ra = jax.random.uniform(subkey1, minval=r_min, maxval=r_max)  # outer radius
		rb = jax.random.uniform(subkey2, minval=r_ratio_min, maxval=r_ratio_max) * ra  # inner radius
		d = jax.random.uniform(subkey3, minval=(ra - rb) / r_ratio_max, maxval=(ra + rb) * r_ratio_max)
		shape = Moon2D(d, ra, rb)
	elif shape_name == 'trapezoid':
		key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
		ra = jax.random.uniform(subkey1, minval=r_min, maxval=r_max)
		rb = jax.random.uniform(subkey2, minval=r_min, maxval=r_max)
		he = 2.0 * jax.random.uniform(subkey3, minval=r_min, maxval=r_max)
		shape = Trapezoid2D(ra, rb, he)
	elif shape_name == 'quad':
		# Generate as slightly perturbed rectangle for validity
		key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
		w = jax.random.uniform(subkey1, minval=r_min, maxval=r_max)  # width
		h = jax.random.uniform(subkey2, minval=r_min, maxval=r_max)  # height
		perturbation_amt = min((r_ratio_max * h), (r_ratio_max * w))
		perturbation_vecs = jax.random.uniform(subkey2, shape=(4, 2))
		perturbation_vecs /= jnp.linalg.norm(perturbation_vecs, axis=1, keepdims=True)
		perturbation = perturbation_amt * perturbation_vecs
		v0 = jnp.array([-w, -h]) + perturbation[0]
		v1 = jnp.array([w, -h]) + perturbation[1]
		v2 = jnp.array([w, h]) + perturbation[2]
		v3 = jnp.array([-w, h]) + perturbation[3]
		shape = Quad2D(v0, v1, v2, v3)

	return shape


class SDF3D:
	def __init__(self, sdf2d: SDF2D, is_extrusion: bool, param: float):
		self.SDF2D = sdf2d
		self.is_extrusion = is_extrusion
		self.param = param  # either o or h

	def evaluate_signed_distance(self, q: np.ndarray) -> np.ndarray:
		if self.is_extrusion:
			return self.SDF2D.evaluate_extrusion(q, self.param)

		return self.SDF2D.evaluate_revolution(q, self.param)

	def evaluate_signed_distance_and_gradient(self, q: np.ndarray):
		distances = self.evaluate_signed_distance(q)
		gradients = self.SDF2D.evaluate_signed_distance_gradient_3D(
			np.asfortranarray(q.T), self.is_extrusion, self.param
		)
		return distances, gradients

	def sample_farthest_point_cloud(
		self,
		n_points: int,
		seed: int,
	):
		return self.SDF2D.sample_farthest_point_cloud(n_points, self.is_extrusion, self.param, seed)

	def sample_uneven_point_cloud(
		self,
		is_extrusion: bool,
		SDF3D_param: float,
		n_cameras: int,
		sensor_size: float,
		grid_spacing_min: float,
		grid_spacing_max: float,
		max_noise_level: float,
		max_normals_to_flip: float,
		max_points: int,
		seed: int,
	):
		return self.SDF2D.sample_uneven_point_cloud(
			is_extrusion,
			SDF3D_param,
			n_cameras,
			sensor_size,
			grid_spacing_min,
			grid_spacing_max,
			max_noise_level,
			max_normals_to_flip,
			max_points,
			seed,
		)

	def sample_uniformly(
		self,
		n_points: int,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		sigma_p: float = 0.0,
		sigma_n: float = 0.0,
		seed: int = 0,
	) -> np.ndarray:
		"""
		Sample points uniformly on the surface.
		"""
		if self.is_extrusion:
			points, _ = self.SDF2D.sample_extrusion(n_points, self.param, bbox_min, bbox_max, sigma_p, sigma_n, seed)
			return points
		else:
			points, _ = self.SDF2D.sample_revolution(n_points, self.param, bbox_min, bbox_max, sigma_p, sigma_n, seed)
			return points

	def sample_points_and_normals_uniformly(self, n_points, seed):
		return self.SDF2D.sample_points_and_normals_uniformly(n_points, self.is_extrusion, self.param, seed)

	def sample_narrow_band(
		self,
		n_points: int,
		offset: float = 0.1,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		seed: int = 0,
	) -> np.ndarray:
		"""
		Sample points in a narrow band around the surface.
		"""
		if self.is_extrusion:
			return self.SDF2D.sample_extrusion_narrow_band(n_points, self.param, offset, bbox_min, bbox_max, seed)
		else:
			return self.SDF2D.sample_revolution_narrow_band(n_points, self.param, offset, bbox_min, bbox_max, seed)


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
