# Process training/test datasets.

import os
import glob
import numpy as np
import scipy as sp
import time
import pickle

import jax
import jax.numpy as jnp
from jax import jit

from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import RegularGridInterpolator

import thingi10k
import trimesh
import pyfqmr
import noise

import py7zr
import zipfile
import pyvista as pv
from stl import mesh as stl_mesh
import tempfile

from pointsastori.shape_3d import *
from pointsastori.network import *

seed = 57

k_neighbors = 64
n_cameras = 4
sensor_size = 1.0
grid_spacing_min = 0.01
grid_spacing_max = 0.1
max_noise_level = 0.02
max_normals_to_flip = 0.1

# legacy
grid_spacings = [0.01]
sigmas_p = [0.0, 0.01]  # less noise, more noise
sigmas_n = [0.0, 0.01]
train_test_split = 0.8

max_faces_per_mesh = 2048  # maximum-sized mesh to consider in each dataset
max_vertices_per_mesh = max_faces_per_mesh  # training meshes should all be closed and manifold, meaning only max_faces_per_mesh is effectively active
max_points_per_pointcloud = max_faces_per_mesh  # maximum-sized point cloud
meshes_per_training_npz = 500  # number of shapes to include in each saved NPZ file encoding training features
n_inspect = 7  # number of meshes, point clouds to save per dataset to manually inspect

N_POINTS_SCALING = [512 * (4**k) for k in range(1, 7)]


# ========================================================================
# GET THE DATA
# ========================================================================


def save_mesh_dataset(meshes: List[Dict], filepath: str, metadata: Dict = None):
	"""
	Save a list of meshes to an NPZ file.

	Args:
		meshes: List of dicts with keys 'vertices', 'faces', 'file_id', 'metadata'
		filepath: Path to save the NPZ file
		metadata: Optional global metadata dictionary
	"""
	# Prepare arrays for storage
	# Store meshes as separate arrays since they have different sizes
	data = {
		'n_meshes': len(meshes),
	}

	# Add global metadata
	if metadata is not None:
		data['metadata'] = json.dumps(metadata)

	# Store each mesh separately with index
	for i, mesh_data in enumerate(meshes):
		data[f'vertices_{i}'] = mesh_data['vertices']
		data[f'faces_{i}'] = mesh_data['faces']
		data[f'file_id_{i}'] = mesh_data['file_id']

		# Store per-mesh metadata as JSON string
		if 'metadata' in mesh_data:
			data[f'metadata_{i}'] = json.dumps(mesh_data['metadata'])

	# Save with compression
	np.savez_compressed(filepath, **data)
	print(f'Saved {len(meshes)} meshes to {filepath}')


def load_mesh_dataset(filepath: str) -> Tuple[List[Dict], Dict]:
	"""
	Load meshes from an NPZ file.

	Args:
		filepath: Path to the NPZ file

	Returns:
		(meshes_list, global_metadata) where meshes_list contains dicts with
		'vertices', 'faces', 'file_id', 'metadata' keys
	"""
	data = np.load(filepath, allow_pickle=True)
	n_meshes = int(data['n_meshes'])

	# Load global metadata if present
	global_metadata = {}
	if 'metadata' in data:
		global_metadata = json.loads(str(data['metadata']))

	meshes = []
	for i in range(n_meshes):
		mesh_dict = {
			'vertices': data[f'vertices_{i}'],
			'faces': data[f'faces_{i}'],
			'file_id': str(data[f'file_id_{i}']),
		}

		# Load per-mesh metadata if present
		if f'metadata_{i}' in data:
			mesh_dict['metadata'] = json.loads(str(data[f'metadata_{i}']))

		meshes.append(mesh_dict)

	return meshes, global_metadata


def load_as_mesh3d(mesh_data: Dict):
	"""
	Convert loaded mesh data dict to TriangleMesh object.

	Args:
		mesh_data: Dict with 'vertices' and 'faces' keys

	Returns:
		TriangleMesh object (you'll need to import from shape_3d.py)
	"""
	vertices = mesh_data['vertices']
	faces = mesh_data['faces']

	return TriangleMesh(vertices, faces)


def get_Thingi10k(save_dir: str, train_test_split: float):
	"""
	Get clean meshes from Thingi10k.

	Args:
		save_dir: Directory to save the train/test datasets
		train_test_split: Fraction of data to use for training (0.0 to 1.0)

	This function saves two files:
		- {save_dir}/thingi10k_train_meshes.npz
		- {save_dir}/thingi10k_test_meshes.npz

	"""

	# Create save directory
	os.makedirs(save_dir, exist_ok=True)

	thingi10k.init()

	# Collect valid meshes
	valid_meshes = []
	stats = {'total_processed': 0, 'valid': 0, 'errors': 0}

	print('\nProcessing meshes...')
	print('-' * 70)

	counter = 0
	min_n_faces = np.inf
	max_n_faces = 0
	mean_n_faces = 0
	for entry in thingi10k.dataset(
		closed=True, manifold=True, oriented=True, self_intersecting=False, solid=False, num_components=1
	):
		stats['total_processed'] += 1

		# Load mesh
		try:
			vertices, faces = thingi10k.load_file(entry['file_path'])
			min_n_faces = min(faces.shape[0], min_n_faces)
			max_n_faces = max(faces.shape[0], max_n_faces)
			mean_n_faces += faces.shape[0]

			# Center and scale data.
			shape = TriangleMesh(vertices, faces)
			shape.center_and_scale()
			vertices = shape.get_vertices()
			faces = shape.get_faces()

			# Check orientation
			query = np.array([2, 2, 2])  # outside bounding box and hence mesh
			mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
			from scipy.spatial import KDTree

			tree = KDTree(vertices)
			_, closest_vertex_id = tree.query(query)
			cp = vertices[closest_vertex_id]
			# Get normal from adjacent faces
			vertex_faces = mesh_trimesh.vertex_faces[closest_vertex_id]
			vertex_faces = vertex_faces[vertex_faces >= 0]  # Remove -1 entries
			n = mesh_trimesh.face_normals[vertex_faces].mean(axis=0)
			n = n / np.linalg.norm(n)
			if np.dot(query - cp, n) < 0.0:
				# Flip faces
				faces[:, [1, 2]] = faces[:, [2, 1]]
				shape = TriangleMesh(vertices, faces)

			# Save a few files to manually inspect
			if counter < n_inspect:
				shape.write_OBJ(f'Thingi10k_{counter}.obj')
			counter += 1

			shape.write_OBJ(f'Thingi10k_{counter}.obj')
			counter += 1

			# Store mesh with metadata
			mesh_data = {
				'vertices': vertices,
				'faces': faces,
				'file_id': entry['file_id'],
				'metadata': {
					'author': entry.get('author', 'unknown'),
					'license': entry.get('license', 'unknown'),
					'n_vertices': entry.get('n_vertices'),
					'n_faces': entry.get('n_faces'),
				},
			}
			valid_meshes.append(mesh_data)

		except Exception as e:
			stats['errors'] += 1

	if len(valid_meshes) == 0:
		raise ValueError('No valid meshes found in dataset!')

	print(f'Min # faces: {min_n_faces}')
	print(f'Max # faces: {max_n_faces}')
	print(f'Mean # faces: {mean_n_faces / len(valid_meshes)}')

	# Shuffle meshes for random split
	np.random.seed(seed)
	indices = np.random.permutation(len(valid_meshes))
	valid_meshes = [valid_meshes[i] for i in indices]

	# Split into train/test
	n_train = int(len(valid_meshes) * train_test_split)
	train_meshes = valid_meshes[:n_train]
	test_meshes = valid_meshes[n_train:]

	print(f'\nDataset Split:')
	print(f'  Training: {len(train_meshes)} meshes')
	print(f'  Test: {len(test_meshes)} meshes')

	# Save datasets
	train_path = os.path.join(save_dir, 'thingi10k_train_meshes.npz')
	test_path = os.path.join(save_dir, 'thingi10k_test_meshes.npz')

	global_metadata = {
		'dataset': 'Thingi10K',
		'train_test_split': train_test_split,
		'statistics': stats,
		'n_train': len(train_meshes),
		'n_test': len(test_meshes),
	}

	save_mesh_dataset(train_meshes, train_path, metadata=global_metadata)
	save_mesh_dataset(test_meshes, test_path, metadata=global_metadata)

	print(f'\nDatasets saved to:')
	print(f'  {train_path}')
	print(f'  {test_path}')


def save_train_test_meshes(valid_meshes: list, save_dir: str, name: str, train_test_split: float, seed: int):
	"""
	Given a list of dictionaries of mesh_data, split into training/test sets, and save.
	"""

	# Shuffle meshes for random split
	np.random.seed(seed)
	indices = np.random.permutation(len(valid_meshes))
	valid_meshes = [valid_meshes[i] for i in indices]

	# Split into train/test
	n_train = int(len(valid_meshes) * train_test_split)
	train_meshes = valid_meshes[:n_train]
	test_meshes = valid_meshes[n_train:]

	print(f'\nDataset Split:')
	print(f'  Training: {len(train_meshes)} meshes')
	print(f'  Test: {len(test_meshes)} meshes')

	# Save datasets
	train_path = os.path.join(save_dir, name + '_train_meshes.npz')
	test_path = os.path.join(save_dir, name + '_test_meshes.npz')

	save_mesh_dataset(train_meshes, train_path)
	save_mesh_dataset(test_meshes, test_path)

	print(f'\nDatasets saved to:')
	print(f'  {train_path}')
	print(f'  {test_path}')


def evaluate_perlin_noise_at_points(
	points: np.ndarray,
	scale: float = 1.0,
	octaves: int = 4,
	persistence: float = 0.5,
	lacunarity: float = 2.0,
	seed: Optional[int] = None,
) -> np.ndarray:
	"""
	Evaluate 3D Perlin noise at given points using the noise library.

	Args:
		points: Array of shape (n, 3) with (x, y, z) coordinates
		scale: Frequency scale for the noise (higher = more frequent/smaller features)
		octaves: Number of noise layers to combine
		persistence: How much each octave contributes (0-1)
		lacunarity: Frequency multiplier between octaves (typically 2.0)
		seed: Random seed for reproducibility (0-255)

	Returns:
		Array of noise values in range approximately [-1, 1], shape (n,)
	"""
	# The noise library's pnoise3 expects base parameter (seed) in range 0-255
	if seed is not None:
		base = seed % 256
	else:
		base = 0

	# Evaluate noise at each point
	n_points = points.shape[0]
	noise_values = np.zeros(n_points)

	for i in range(n_points):
		x, y, z = points[i]
		# pnoise3 returns values in approximately [-0.5, 0.5]
		# Scale to [-1, 1]
		noise_values[i] = (
			noise.pnoise3(
				x * scale,
				y * scale,
				z * scale,
				octaves=octaves,
				persistence=persistence,
				lacunarity=lacunarity,
				base=base,
			)
			* 2.0
		)

	return noise_values


def generate_blob_implicit_surface(
	alpha: float,
	base_radius: float = 1.0,
	grid_resolution: int = 64,
	bbox_extent: float = 2.0,
	scale: float = 1.0,
	octaves: int = 4,
	persistence: float = 0.5,
	lacunarity: float = 2.0,
	seed: Optional[int] = None,
) -> pv.ImageData:
	"""
	Generate implicit surface values for a deformed sphere as a PyVista ImageData.

	The implicit surface is defined by:
		F(x,y,z) = x^2 + y^2 + z^2 - (base_radius + alpha * f(x,y,z))^2 = 0

	Args:
		alpha: Deformation amplitude
		base_radius: Base radius of the sphere
		grid_resolution: Resolution of the marching cubes grid
		bbox_extent: Extent of bounding box for marching cubes
		scale: Perlin noise frequency scale (higher = smaller features)
		octaves: Number of Perlin noise octaves
		persistence: Perlin noise persistence
		lacunarity: Perlin noise lacunarity
		seed: Random seed

	Returns:
		PyVista ImageData with scalar field values
	"""
	# Create uniform grid
	spacing = (2 * bbox_extent) / (grid_resolution - 1)
	origin = (-bbox_extent, -bbox_extent, -bbox_extent)

	# Create PyVista ImageData (uniform grid)
	grid = pv.ImageData(
		dimensions=(grid_resolution, grid_resolution, grid_resolution),
		spacing=(spacing, spacing, spacing),
		origin=origin,
	)

	# Get coordinate arrays
	points = grid.points

	# Evaluate Perlin noise directly at all grid points
	f_values = evaluate_perlin_noise_at_points(
		points=points, scale=scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, seed=seed
	)

	# Compute implicit surface values
	# F(x,y,z) = x^2 + y^2 + z^2 - (base_radius + alpha * f(x,y,z))^2
	x, y, z = points[:, 0], points[:, 1], points[:, 2]
	radius_squared = x**2 + y**2 + z**2
	deformed_radius = base_radius + alpha * f_values
	volume = radius_squared - deformed_radius**2

	# Add scalar field to grid
	grid.point_data['values'] = volume

	return grid


def generate_blob_mesh(
	alpha: float,
	base_radius: float = 1.0,
	grid_resolution: int = 64,
	scale: float = 1.0,
	octaves: int = 4,
	persistence: float = 0.5,
	lacunarity: float = 2.0,
	seed: Optional[int] = None,
	validate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Generate a single blob mesh using Perlin noise deformation.

	Args:
		alpha: Deformation amplitude (should be < 1.0 to avoid disconnected components)
		base_radius: Base radius of the sphere
		grid_resolution: Resolution for marching cubes (e.g., 64)
		scale: Perlin noise frequency scale (higher = smaller/more frequent features)
		octaves: Number of Perlin noise octaves
		persistence: Perlin noise persistence
		lacunarity: Perlin noise lacunarity (frequency multiplier)
		seed: Random seed for reproducibility
		validate: Whether to validate the mesh is closed, watertight, and manifold

	Returns:
		(vertices, faces) arrays for the mesh

	Raises:
		ValueError: If validation fails
	"""
	# Generate implicit surface
	bbox_extent = base_radius * (1.0 + alpha) * 1.2  # Add 20% margin
	grid = generate_blob_implicit_surface(
		alpha=alpha,
		base_radius=base_radius,
		grid_resolution=grid_resolution,
		bbox_extent=bbox_extent,
		scale=scale,
		octaves=octaves,
		persistence=persistence,
		lacunarity=lacunarity,
		seed=seed,
	)

	# Extract mesh using PyVista's marching cubes (contour filter)
	try:
		# The contour method extracts isosurfaces at the specified value(s)
		mesh = grid.contour(isosurfaces=[0.0], scalars='values')

		if mesh.n_points == 0 or mesh.n_cells == 0:
			raise ValueError('Marching cubes produced empty mesh')

	except Exception as e:
		raise ValueError(f'Marching cubes failed: {e}')

	# Extract vertices and faces
	vertices = np.array(mesh.points)

	# PyVista returns faces as a flat array: [n0, v0, v1, v2, n1, v3, v4, v5, ...]
	# where ni is the number of vertices in face i
	faces_flat = mesh.faces
	faces = []
	i = 0
	while i < len(faces_flat):
		n_verts = faces_flat[i]
		if n_verts != 3:
			raise ValueError(f'Non-triangular face found with {n_verts} vertices')
		face = faces_flat[i + 1 : i + 1 + n_verts]
		faces.append(face)
		i += n_verts + 1

	faces = np.array(faces)
	faces[:, [1, 2]] = faces[:, [2, 1]]  # because PyVista is stupid

	# Simplify
	mesh_simplifier = pyfqmr.Simplify()
	mesh_simplifier.setMesh(vertices, faces)
	mesh_simplifier.simplify_mesh(
		target_count=max_faces_per_mesh,
		aggressiveness=10,  # will always strive to hit target_count, at the cost of potentially a lot of geometric error
		preserve_border=True,
		verbose=0,
	)
	vertices, faces, _ = mesh_simplifier.getMesh()

	# Validate mesh if requested
	if validate:
		mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)

		if not mesh_trimesh.is_watertight:
			raise ValueError('Generated mesh is not watertight')

		if not mesh_trimesh.is_volume:
			raise ValueError('Generated mesh does not enclose a volume')

	return vertices, faces


def generate_blob_dataset(
	n_blobs: int,
	save_dir: str,
	name: str = 'blobs',
	alpha_range: Tuple[float, float] = (0.1, 0.7),
	scale_range: Tuple[float, float] = (0.5, 3.0),
	octaves_range: Tuple[int, int] = (1, 4),
	base_radius: float = 1.0,
	grid_resolution: int = 64,
	persistence: float = 1.0,
	lacunarity: float = 2.0,
	seed: int = 42,
	train_test_split: float = 0.8,
) -> None:
	"""
	Generate a batch of blob meshes with varying parameters.

	Args:
		n_blobs: Number of blobs to generate
		save_dir: Directory to save the meshes and NPZ files
		name: Root name for saved files (default: 'blobs')
		alpha_range: (min, max) range for deformation amplitude
		scale_range: (min, max) range for Perlin noise scale (higher = smaller features)
		octaves_range: (min, max) range for number of octaves (inclusive)
		base_radius: Base radius of the sphere
		grid_resolution: Marching cubes resolution
		persistence: Perlin noise persistence (fixed at 1.0)
		lacunarity: Perlin noise lacunarity (fixed at 2.0)
		seed: Random seed for reproducibility
		train_test_split: Fraction of data for training
	"""

	# Create save directory
	os.makedirs(save_dir, exist_ok=True)

	# Set random seed
	np.random.seed(seed)

	# Pre-sample more parameters than needed to handle failures
	# We'll generate 3x as many in case some fail
	n_to_sample = n_blobs * 3
	alphas = np.random.uniform(alpha_range[0], alpha_range[1], n_to_sample)
	scales = np.random.uniform(scale_range[0], scale_range[1], n_to_sample)
	octaves_list = np.random.randint(octaves_range[0], octaves_range[1] + 1, n_to_sample)

	valid_meshes = []
	stats = {'total_attempted': 0, 'valid': 0, 'errors': 0}

	print(f'\nGenerating {n_blobs} blob meshes...')
	print(f'Parameters:')
	print(f'  alpha: [{alpha_range[0]}, {alpha_range[1]}]')
	print(f'  scale: [{scale_range[0]}, {scale_range[1]}]')
	print(f'  octaves: [{octaves_range[0]}, {octaves_range[1]}]')
	print(f'  persistence: {persistence} (fixed)')
	print(f'  lacunarity: {lacunarity} (fixed)')
	print('-' * 70)

	t0_total = time.time()

	i = 0  # Index into pre-sampled parameters
	while len(valid_meshes) < n_blobs:
		try:
			# Check if we need to sample more parameters
			if i >= len(alphas):
				print(f'  Sampling additional parameters (current valid: {len(valid_meshes)}/{n_blobs})...')
				# Sample another batch
				additional_samples = n_blobs - len(valid_meshes) + 10  # Add a few extra
				new_alphas = np.random.uniform(alpha_range[0], alpha_range[1], additional_samples)
				new_scales = np.random.uniform(scale_range[0], scale_range[1], additional_samples)
				new_octaves = np.random.randint(octaves_range[0], octaves_range[1] + 1, additional_samples)

				alphas = np.concatenate([alphas, new_alphas])
				scales = np.concatenate([scales, new_scales])
				octaves_list = np.concatenate([octaves_list, new_octaves])

			t0 = time.time()

			# Generate unique seed for this blob
			blob_seed = seed + i

			# Sample parameters
			alpha = alphas[i]
			scale = scales[i]
			octaves = int(octaves_list[i])

			stats['total_attempted'] += 1

			# Generate mesh
			vertices, faces = generate_blob_mesh(
				alpha=alpha,
				base_radius=base_radius,
				grid_resolution=grid_resolution,
				scale=scale,
				octaves=octaves,
				persistence=persistence,
				lacunarity=lacunarity,
				seed=blob_seed,
				validate=True,
			)

			t1 = time.time()

			# Center and scale to unit sphere
			mesh = TriangleMesh(vertices, faces)
			mesh.center_and_scale()
			vertices = mesh.get_vertices()
			faces = mesh.get_faces()

			# # Save individual OBJ file
			# obj_path = os.path.join(save_dir, f'{name}_{len(valid_meshes):05d}.obj')
			# mesh.write_OBJ(obj_path)

			# Store mesh data
			mesh_data = {
				'vertices': vertices,
				'faces': faces,
				'file_id': f'{name}_{len(valid_meshes):05d}',
				'metadata': {
					'alpha': float(alpha),
					'scale': float(scale),
					'octaves': int(octaves),
					'base_radius': float(base_radius),
					'grid_resolution': grid_resolution,
					'persistence': float(persistence),
					'lacunarity': float(lacunarity),
					'seed': blob_seed,
					'n_vertices': len(vertices),
					'n_faces': len(faces),
					'generation_time': t1 - t0,
				},
			}

			valid_meshes.append(mesh_data)
			stats['valid'] += 1

			if len(valid_meshes) % 10 == 0:
				print(
					f'Generated {len(valid_meshes)}/{n_blobs} valid blobs ({t1 - t0:.2f}s) - last: alpha={alpha:.2f}, scale={scale:.2f}, octaves={octaves}'
				)

		except Exception as e:
			print(f'Error generating blob {i} (attempt {stats["total_attempted"]}): {e}')
			stats['errors'] += 1

		i += 1  # Move to next parameter set

	t1_total = time.time()

	print(f'\nGeneration complete!')
	print(f'  Total time: {t1_total - t0_total:.2f}s')
	print(f'  Average time per valid blob: {(t1_total - t0_total) / stats["valid"]:.2f}s')
	print(f'  Valid blobs: {stats["valid"]}')
	print(f'  Failed attempts: {stats["errors"]}')
	print(f'  Total attempts: {stats["total_attempted"]}')
	print(f'  Success rate: {100.0 * stats["valid"] / stats["total_attempted"]:.1f}%')

	if len(valid_meshes) == 0:
		raise ValueError('No valid meshes generated!')

	# Shuffle and split into train/test
	np.random.seed(seed)
	indices = np.random.permutation(len(valid_meshes))
	valid_meshes = [valid_meshes[i] for i in indices]

	n_train = int(len(valid_meshes) * train_test_split)
	train_meshes = valid_meshes[:n_train]
	test_meshes = valid_meshes[n_train:]

	print(f'\nDataset Split:')
	print(f'  Training: {len(train_meshes)} meshes')
	print(f'  Test: {len(test_meshes)} meshes')

	# Save NPZ files
	from mesh_data import save_mesh_dataset

	train_path = os.path.join(save_dir, f'{name}_train_meshes.npz')
	test_path = os.path.join(save_dir, f'{name}_test_meshes.npz')

	global_metadata = {
		'dataset': 'Procedural Blobs (Perlin)',
		'train_test_split': train_test_split,
		'statistics': stats,
		'n_train': len(train_meshes),
		'n_test': len(test_meshes),
		'generation_parameters': {
			'alpha_range': alpha_range,
			'scale_range': scale_range,
			'octaves_range': octaves_range,
			'base_radius': base_radius,
			'grid_resolution': grid_resolution,
			'persistence': persistence,
			'lacunarity': lacunarity,
			'seed': seed,
		},
	}

	save_mesh_dataset(train_meshes, train_path, metadata=global_metadata)
	save_mesh_dataset(test_meshes, test_path, metadata=global_metadata)

	print(f'\nDatasets saved to:')
	print(f'  {train_path}')
	print(f'  {test_path}')


def get_ABC(load_dir: str, save_dir: str, train_test_split: float):
	"""
	Get meshes from ABC.

	Args:
		load_dir: Directory in which .7z file(s) containing meshes representing whole CAD objects (stl2) are stored; downloaded from here: https://archive.nyu.edu/handle/2451/43778
		save_dir: Directory to save the train/test datasets
		train_test_split: Fraction of data to use for training (0.0 to 1.0)

	This function saves two files:
		- {save_dir}/ABC_train_meshes.npz
		- {save_dir}/ABC_test_meshes.npz

	"""

	# Create save directory
	os.makedirs(save_dir, exist_ok=True)

	print('\nProcessing meshes...')
	print('-' * 70)

	# Go through each 7z file
	chunks = sorted(glob.glob(os.path.join(load_dir, '*.7z')))

	if len(chunks) == 0:
		raise ValueError(f'No .7z files found in {load_dir}')

	print(f'Found {len(chunks)} .7z archive(s)')

	# Go through each 7z file
	for chunk_idx, chunk_path in enumerate(chunks):
		chunk_name = os.path.basename(chunk_path)
		print(f'\nProcessing archive {chunk_idx + 1}/{len(chunks)}: {chunk_name}')

		valid_meshes = []
		error_meshes = 0

		try:
			# Create a temporary directory for extraction
			with tempfile.TemporaryDirectory() as temp_dir:
				# Extract the 7z archive
				print(f'  Extracting...')
				with py7zr.SevenZipFile(chunk_path, mode='r') as archive:
					archive.extractall(path=temp_dir)

				# Find all STL files in the extracted directory
				stl_files = []
				for root, dirs, files in os.walk(temp_dir):
					for file in files:
						if file.lower().endswith('.stl'):
							stl_files.append(os.path.join(root, file))

				print(f'  Found {len(stl_files)} STL files')

				# For each STL file, load and save vertices and faces
				counter = 0
				min_n_faces = np.inf
				max_n_faces = 0
				mean_n_faces = 0
				for stl_path in stl_files:
					try:
						# Load STL file using numpy-stl
						mesh_data = stl_mesh.Mesh.from_file(stl_path)

						# STL stores each triangle as 3 vertices (redundant)
						# mesh_data.vectors is shape (n_triangles, 3, 3)
						# Reshape to (n_triangles * 3, 3) to get all vertices
						vertices = mesh_data.vectors.reshape(-1, 3)

						# Deduplicate vertices and create face indices
						unique_vertices, inverse_indices = np.unique(vertices, axis=0, return_inverse=True)
						faces = inverse_indices.reshape(-1, 3)

						# Convert to proper types
						vertices = unique_vertices.astype(np.float64)
						faces = faces.astype(np.int64)

						# Center and scale data
						shape = TriangleMesh(vertices, faces)
						shape.center_and_scale()
						vertices = shape.get_vertices()
						faces = shape.get_faces()

						min_n_faces = min(min_n_faces, faces.shape[0])
						max_n_faces = max(max_n_faces, faces.shape[0])
						mean_n_faces += faces.shape[0]

						# Save a few files to manually inspect
						if counter < n_inspect:
							shape.write_OBJ(f'ABC_{counter}.obj')
						counter += 1

						# Create file ID from relative path
						rel_path = os.path.relpath(stl_path, temp_dir)
						file_id = f'{chunk_name}_{rel_path}'.replace(os.sep, '_')

						mesh_data = {
							'vertices': vertices,
							'faces': faces,
							'file_id': file_id,
							'metadata': {
								'archive': chunk_name,
								'stl_path': rel_path,
								'n_vertices': len(vertices),
								'n_faces': len(faces),
							},
						}

						valid_meshes.append(mesh_data)

						if len(valid_meshes) % 100 == 0:
							print(f'  Processed {len(valid_meshes)} meshes...')

					except Exception as e:
						print(f"  Couldn't process STL file {stl_path}")
						error_meshes += 1

			print(f'Number of valid meshes: {len(valid_meshes)}')
			print(f'Meshes that could not be loaded: {error_meshes}')

			print(f'Min # faces: {min_n_faces}')
			print(f'Max # faces: {max_n_faces}')
			print(f'Mean # faces: {mean_n_faces / len(valid_meshes)}')

			save_train_test_meshes(
				valid_meshes, save_dir=save_dir, name=f'ABC{chunk_idx}', train_test_split=train_test_split, seed=seed
			)

		except Exception as e:
			print(f'  Error processing archive {chunk_name}: {e}')


# ========================================================================
# GENERATE DATASETS
# ========================================================================


def pad_mesh(vertices, faces, max_vertices=None, max_faces=None):
	"""
	Pad mesh to fixed size for batching.

	Args:
		vertices: (n_vertices, 3) array
		faces: (n_faces, 3) array
		max_vertices: Maximum number of vertices (default: max_vertices_per_mesh)
		max_faces: Maximum number of faces (default: max_faces_per_mesh)

	Returns:
		vertices_padded: (max_vertices, 3) padded with zeros
		faces_padded: (max_faces, 3) padded with -1
		face_mask: (max_faces,) boolean mask for valid faces
	"""
	if max_vertices is None:
		max_vertices = max_vertices_per_mesh
	if max_faces is None:
		max_faces = max_faces_per_mesh

	n_verts = vertices.shape[0]
	n_faces = faces.shape[0]

	# Pad vertices with zeros
	vertices_padded = np.zeros((max_vertices, 3), dtype=np.float64)
	vertices_padded[:n_verts] = vertices

	# Pad faces with -1 (invalid indices)
	faces_padded = np.full((max_faces, 3), -1, dtype=np.int32)
	faces_padded[:n_faces] = faces

	# Create boolean mask for valid faces
	face_mask = np.zeros(max_faces, dtype=bool)
	face_mask[:n_faces] = True

	return vertices_padded, faces_padded, face_mask


def generate_input_features_with_meshes(all_data, max_points):
	"""
	Args:
		all_data: List of dictionaries with keys:
			{
				'positions': (n_points, 3),
				'normals': (n_points, 3),
				'areas': (n_points,)
				'neighbors': (n_points, k),
				'vertices': (n_vertices, 3),
				'faces': (n_faces, 3),
			}
		max_points: Maximum number of points to pad to

	Returns:
		Tuple of NumPy arrays:
			all_positions: (n_shapes, max_points, 3)
			all_normals: (n_shapes, max_points, 3)
			all_areas: (n_shapes, max_points, 1)
			all_neighbors: (n_shapes, max_points, k)
			all_masks: (n_shapes, max_points)
			all_mesh_vertices: (n_shapes, max_vertices, 3)
			all_mesh_faces: (n_shapes, max_faces, 3)
			all_face_masks: (n_shapes, max_faces)
	"""
	all_padded_data = []

	for i in range(len(all_data)):
		n_points = all_data[i]['positions'].shape[0]
		pad_size = max_points - n_points

		# Pad point cloud data
		positions = np.concatenate([all_data[i]['positions'], np.zeros((pad_size, 3))], axis=0)
		normals = np.concatenate([all_data[i]['normals'], np.zeros((pad_size, 3))], axis=0)
		areas = np.concatenate([all_data[i]['areas'], np.zeros((pad_size,))], axis=0)

		# Pad neighbors if |P| < k_neighbors
		if n_points < k_neighbors:
			pad_width = min(k_neighbors, max_points) - n_points
			all_data[i]['neighbors'] = np.pad(
				all_data[i]['neighbors'], ((0, 0), (0, pad_width)), mode='constant', constant_values=-1
			)

		neighbors = np.concatenate(
			[all_data[i]['neighbors'], -np.ones((pad_size, min(k_neighbors, max_points)), dtype=int)], axis=0
		)

		mask = np.concatenate([np.ones(n_points, dtype=bool), np.zeros(pad_size, dtype=bool)])

		# Pad mesh data
		vertices_padded, faces_padded, face_mask = pad_mesh(all_data[i]['vertices'], all_data[i]['faces'])

		all_padded_data.append(
			{
				'positions': positions,
				'normals': normals,
				'areas': areas,
				'neighbors': neighbors,
				'mask': mask,
				'vertices': vertices_padded,
				'faces': faces_padded,
				'face_mask': face_mask,
			}
		)

	# Stack into arrays
	all_positions = np.stack([d['positions'] for d in all_padded_data])
	all_normals = np.stack([d['normals'] for d in all_padded_data])
	all_areas = np.stack([d['areas'] for d in all_padded_data])
	all_neighbors = np.stack([d['neighbors'] for d in all_padded_data])
	all_masks = np.stack([d['mask'] for d in all_padded_data])
	all_mesh_vertices = np.stack([d['vertices'] for d in all_padded_data])
	all_mesh_faces = np.stack([d['faces'] for d in all_padded_data])
	all_face_masks = np.stack([d['face_mask'] for d in all_padded_data])

	return (
		all_positions,
		all_normals,
		all_areas,
		all_neighbors,
		all_masks,
		all_mesh_vertices,
		all_mesh_faces,
		all_face_masks,
	)


def save_mesh_training_data(
	filepath: str,
	all_positions,
	all_normals,
	all_areas,
	all_neighbors,
	all_masks,
	all_mesh_vertices,
	all_mesh_faces,
	all_face_masks,
):
	"""
	Save training data with mesh information (no pre-generated queries).

	Args:
		filepath: Path to save file (should end in .npz)
		all_positions: (n_shapes, max_points, 3)
		all_normals: (n_shapes, max_points, 3)
		all_areas: (n_shapes, max_points, 1)
		all_neighbors: (n_shapes, max_points, k)
		all_masks: (n_shapes, max_points)
		all_mesh_vertices: (n_shapes, max_vertices, 3)
		all_mesh_faces: (n_shapes, max_faces, 3)
		all_face_masks: (n_shapes, max_faces)
	"""
	np.savez_compressed(
		filepath,
		positions=np.array(all_positions),
		normals=np.array(all_normals),
		areas=np.array(all_areas),
		neighbors=np.array(all_neighbors, dtype=np.int32),
		masks=np.array(all_masks, dtype=np.bool_),
		mesh_vertices=np.array(all_mesh_vertices),
		mesh_faces=np.array(all_mesh_faces, dtype=np.int32),
		face_masks=np.array(all_face_masks, dtype=np.bool_),
	)

	print(f'Saved training data to {filepath}')
	print(f'  Shapes: {all_positions.shape[0]}')
	print(f'  Max points: {all_positions.shape[1]}')
	print(f'  k_neighbors: {all_neighbors.shape[2]}')
	print(f'  Max vertices: {all_mesh_vertices.shape[1]}')
	print(f'  Max faces: {all_mesh_faces.shape[1]}')


def generate_scanned_mesh_training_dataset(
	n_views: int, npz_filepath: str, save_filepath: str, meshes_per_npz: int, mode: int, n_files: int
):
	"""
	Generate training dataset from triangle meshes with mesh data (no pre-generated queries).

	Given triangle meshes stored in NPZ format, sample point clouds using 3D scanning simulation, and store:
	{
		'positions': (n_shapes, max_points, 3),
		'normals': (n_shapes, max_points, 3),
		'areas': (n_shapes, max_points, 1)
		'neighbors': (n_shapes, max_points, k),
		'masks': (n_shapes, max_points),

		'mesh_vertices': (n_shapes, max_vertices, 3),
		'mesh_faces': (n_shapes, max_faces, 3),
		'face_masks': (n_shapes, max_faces),
	}

	Split into batches of size `meshes_per_npz`.

	Args:
		grid_spacing: Grid spacing for point cloud sampling
		sigma_p: Position noise standard deviation
		sigma_n: Normal noise standard deviation
		flipped_normals: Fraction of normals to randomly flip
		npz_filepath: Path to input mesh NPZ file
		save_filepath: Path template for output NPZ files
		meshes_per_npz: Number of meshes per output file
	"""
	print('Generating mesh data from: %s' % npz_filepath)
	print('Saving data to: %s' % save_filepath)

	t0 = time.time()
	mesh_lists, _ = load_mesh_dataset(npz_filepath)

	n_npz = 0
	n_meshes_processed = 0
	all_data = []
	max_points = 0
	counter = 0
	n_training_examples = 0
	n_files_saved = 0

	# Set seed
	np.random.seed(seed)

	for i, mesh_data in enumerate(mesh_lists):
		print(f'Processing shape {i + 1}/{len(mesh_lists)}...')

		# Load mesh
		shape = load_as_mesh3d(mesh_data)
		shape.center_and_scale()
		n_faces = shape.get_faces().shape[0]
		n_vertices = shape.get_vertices().shape[0]

		# Skip meshes that are too large
		if n_faces > max_faces_per_mesh or n_vertices > max_vertices_per_mesh:
			print(
				f'  Skipping: {n_faces} faces, {n_vertices} vertices (max is {max_faces_per_mesh}, {max_vertices_per_mesh})'
			)
			# Inspect mesh
			# shape.write_OBJ(f'{save_dir}/{rootname}_{i}.obj')
			# write_point_cloud(points, normals, f'{save_dir}/{rootname}_{counter}_pointcloud.obj')
			continue

		vertices = shape.get_vertices()  # (n_vertices, 3)
		faces = shape.get_faces()  # (n_faces, 3)

		try:
			# Check orientation
			query = np.array([2, 2, 2])  # outside bounding box and hence mesh
			mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
			from scipy.spatial import KDTree

			tree = KDTree(vertices)
			_, closest_vertex_id = tree.query(query)
			cp = vertices[closest_vertex_id]
			# Get normal from adjacent faces
			vertex_faces = mesh_trimesh.vertex_faces[closest_vertex_id]
			vertex_faces = vertex_faces[vertex_faces >= 0]  # Remove -1 entries
			n = mesh_trimesh.face_normals[vertex_faces].mean(axis=0)
			n = n / np.linalg.norm(n)
			if np.dot(query - cp, n) < 0.0:
				# Flip faces
				print(f'Shape {i}: not oriented correctly')
				continue
		except Exception as e:
			print(f'  Shape {i}: could not check orientation, skipping')
			continue

		mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
		if not mesh_trimesh.is_watertight:
			print(f'  Shape {i}: not watertight')
			continue
		if not mesh_trimesh.is_volume:
			print(f'  Shape {i}: does not enclose volume, skipping')
			continue
		if not mesh_trimesh.is_winding_consistent:
			print(f'  Shape {i}: not winding consistent, skipping')
			continue

		print(f'  {vertices.shape[0]} vertices, {faces.shape[0]} faces')

		n_meshes_processed += 1

		# Sample point cloud from mesh
		points = None
		normals = None
		if mode == 0:
			# Easy: Nice samplings only: farthest point sampling of meshes, uniform sampling of meshes; no noise, no flipped normals
			# Choose at random whether to use uniform sampling or farthest point sampling (50-50)
			uniform = np.random.rand() < 0.5
			if uniform:
				points, normals = shape.sample_uniform_point_cloud(
					n_points=max_points_per_pointcloud, max_noise_level=0.0, max_normals_to_flip=0.0, seed=seed + i
				)
			else:
				points, normals = shape.sample_farthest_point_cloud(
					n_points=max_points_per_pointcloud, max_noise_level=0.0, max_normals_to_flip=0.0, seed=seed + i
				)
		elif mode == -1:
			# Still nice samplings: farthest-point sampling of meshes, uniform sampling of meshes; but possibly with holes from a scanning process
			points, normals = shape.sample_even_raycasted_point_cloud(
				n_cameras=n_views,
				n_points=max_points_per_pointcloud,
				sensor_size=sensor_size,
				grid_spacing_min=grid_spacing_min,
				grid_spacing_max=grid_spacing_max,
				seed=seed + i,
			)
		elif mode == -2:
			# Same as above, but flip some normals
			points, normals = shape.sample_even_raycasted_point_cloud(
				n_cameras=n_views,
				n_points=max_points_per_pointcloud,
				sensor_size=sensor_size,
				grid_spacing_min=grid_spacing_min,
				grid_spacing_max=grid_spacing_max,
				seed=seed + i,
			)
			points, normals = add_point_cloud_noise(
				points, normals, max_noise_level=0.0, max_normals_to_flip=max_normals_to_flip, seed=seed + i
			)

		elif mode == 1:
			# Medium: Still nice samplings: farthest-point sampling of meshes, uniform sampling of meshes; but with noise
			uniform = np.random.rand() < 0.5
			if uniform:
				points, normals = shape.sample_uniform_point_cloud(
					n_points=max_points_per_pointcloud,
					max_noise_level=max_noise_level,
					max_normals_to_flip=0.0,
					seed=seed + i,
				)
			else:
				points, normals = shape.sample_farthest_point_cloud(
					n_points=max_points_per_pointcloud,
					max_noise_level=max_noise_level,
					max_normals_to_flip=max_normals_to_flip,
					seed=seed + i,
				)

		elif mode == 2:
			# Still nice samplings: farthest-point sampling of meshes, uniform sampling of meshes; but with noise and flipped normals
			uniform = np.random.rand() < 0.5
			if uniform:
				points, normals = shape.sample_uniform_point_cloud(
					n_points=max_points_per_pointcloud,
					max_noise_level=max_noise_level,
					max_normals_to_flip=0.0,
					seed=seed + i,
				)
			else:
				points, normals = shape.sample_farthest_point_cloud(
					n_points=max_points_per_pointcloud,
					max_noise_level=max_noise_level,
					max_normals_to_flip=max_normals_to_flip,
					seed=seed + i,
				)
		elif mode == 3:
			# Hardest: Full sampling procedure (noise, flipped normals, holes, uneven sampling)
			points, normals = shape.sample_uneven_point_cloud(
				n_cameras=n_views,
				sensor_size=sensor_size,
				grid_spacing_min=grid_spacing_min,
				grid_spacing_max=grid_spacing_max,
				max_noise_level=max_noise_level,
				max_normals_to_flip=max_normals_to_flip,
				max_points=max_points_per_pointcloud,
				seed=seed + i,
			)

		print(f'  Number of points: {points.shape[0]}...', end='')

		# Skip point clouds that are too small
		if points.shape[0] < 10:
			continue

		points = np.asarray(points)
		normals = np.asarray(normals)

		# Compute neighbors for point cloud
		neighbors = get_neighbors(points, k_neighbors)  # (n_points, k)

		# Save a few shapes for inspection
		save_dir = npz_filepath.split('/')[0]
		filename = npz_filepath.split('/')[-1]
		rootname = filename.split('_')[0]

		if counter < n_inspect:
			shape.write_OBJ(f'{save_dir}/{rootname}_{i}_mode{mode}.obj')
			write_point_cloud(points, normals, f'{save_dir}/{rootname}_{i}_mode{mode}_pointcloud.obj')
			counter += 1

		n_points = points.shape[0]
		n_training_examples += n_points
		print(f'  Point cloud: {n_points} points')
		max_points = max(n_points, max_points)

		# Store data (no pre-generated queries/distances!)
		all_data.append(
			{
				'positions': points,
				'normals': normals,
				'neighbors': neighbors,
				'vertices': vertices,
				'faces': faces,
			}
		)

		# Save batch when we've processed enough meshes
		if n_meshes_processed == meshes_per_npz or i == len(mesh_lists) - 1:
			print(f'Number of meshes: {n_meshes_processed}')
			print(f'Number of training examples: {n_training_examples}')
			print(f'Max points: {max_points}')

			# Stack into arrays
			(
				all_positions,
				all_normals,
				all_areas,
				all_neighbors,
				all_masks,
				all_mesh_vertices,
				all_mesh_faces,
				all_face_masks,
			) = generate_input_features_with_meshes(all_data, max_points)

			# Save data
			save_dir = save_filepath.split('/')[0]
			filename = save_filepath.split('/')[-1]
			rootname = filename.split('_')[0]
			suffix = filename.split('_')[-1].split('.')[0]  # 'train', 'val'

			filepath = save_dir + '/' + rootname + f'_{suffix}_mode{mode}_{n_npz}.npz'
			save_mesh_training_data(
				filepath,
				all_positions,
				all_normals,
				all_areas,
				all_neighbors,
				all_masks,
				all_mesh_vertices,
				all_mesh_faces,
				all_face_masks,
			)

			# Reset counters
			n_training_examples = 0
			n_meshes_processed = 0
			n_npz += 1
			all_data = []
			max_points = 0

			n_files_saved += 1
			if n_files_saved == n_files:
				break

	t1 = time.time()
	print(f'Generation complete: {t1 - t0:.2f}s')


def generate_all_training_datasets(n_views: int, mode: int):
	"""
	Generate training data, consisting of

	{
		'positions': (n_shapes, max_points, 3),
		'normals': (n_shapes, max_points, 3),
		'areas': (n_shapes, max_points, 1),
		'neighbors': (n_shapes, max_points, k),
		'masks': (n_shapes, max_points),

		'mesh_vertices': (n_shapes, max_vertices, 3),
		'mesh_faces': (n_shapes, max_faces, 3),
		'face_masks': (n_shapes, max_faces),
	}

	This assumes that mesh data has already been retrieved and compressed into local NPZ files,
	e.g. the functions get_Thingi10k(), get_ABC(), etc. have already been run.
	"""

	load_dir = 'datasets'

	# ABC meshes
	ABC_mesh_files = sorted(glob.glob(os.path.join('datasets/final_train_meshes', 'ABC*_train_meshes.npz')))
	for filepath in ABC_mesh_files:
		filename = filepath.split('/')[-1]
		rootname = filename.split('_')[0]
		t0 = time.time()
		generate_scanned_mesh_training_dataset(
			n_views=n_views,
			npz_filepath=filepath,
			save_filepath=load_dir + f'/{rootname}_train.npz',
			meshes_per_npz=meshes_per_training_npz,
			mode=mode,
			n_files=2000 // meshes_per_training_npz,
		)
		t1 = time.time()
		print(f'ABC processing time: {t1 - t0}s')

	# Blobs
	blob_mesh_files = sorted(glob.glob(os.path.join('datasets/final_train_meshes', 'blob*train_meshes.npz')))
	for filepath in blob_mesh_files:
		filename = filepath.split('/')[-1]
		rootname = filename.split('_')[0]
		t0 = time.time()
		generate_scanned_mesh_training_dataset(
			n_views=n_views,
			npz_filepath=filepath,
			save_filepath=load_dir + f'/{rootname}_train.npz',
			meshes_per_npz=meshes_per_training_npz,
			mode=mode,
			n_files=2000 // meshes_per_training_npz,
		)
		t1 = time.time()
		print(f'Blob processing time: {t1 - t0}s')


def generate_pretraining_dataset():
	"""
	Generate a training dataset of just a single, smooth blobby shape.
	"""
	t0 = time.time()

	shape = TriangleMesh.read_OBJ('data/blob1024.obj')
	shape.center_and_scale()
	vertices = shape.get_vertices()
	faces = shape.get_faces()

	# Sample point cloud from mesh
	points, normals = shape.sample_point_cloud(
		n_cameras=n_cameras,
		sensor_size=sensor_size,
		grid_spacing=grid_spacings[0],
		seed=seed,
		sigma_p=0.0,
		sigma_n=0.0,
	)

	print(f'  Original number of points: {points.shape[0]}...', end='')

	# Subsample to target number of points.
	selected_indices = subsample_point_cloud(points, max_points_per_pointcloud)
	points = points[selected_indices]
	normals = normals[selected_indices]

	print(f'  Subsampled to: {points.shape[0]}')

	points = np.asarray(points)
	normals = np.asarray(normals)

	# Compute neighbors for point cloud
	neighbors = get_neighbors(points, k_neighbors)  # (n_points, k)

	n_points = points.shape[0]
	max_points = n_points

	# Store data
	all_data = [
		{
			'positions': points,
			'normals': normals,
			'neighbors': neighbors,
			'vertices': vertices,
			'faces': faces,
		}
	]

	# Stack into arrays
	(
		all_positions,
		all_normals,
		all_areas,
		all_neighbors,
		all_masks,
		all_mesh_vertices,
		all_mesh_faces,
		all_face_masks,
	) = generate_input_features_with_meshes(all_data, max_points)

	# Save data
	filepath = 'datasets/pretraining_blob.npz'
	save_mesh_training_data(
		filepath,
		all_positions,
		all_normals,
		all_areas,
		all_neighbors,
		all_masks,
		all_mesh_vertices,
		all_mesh_faces,
		all_face_masks,
	)

	t1 = time.time()
	print(f'Test training dataset processing time: {t1 - t0}s')


def generate_validation_datasets():
	n_validation = 100
	load_dir = 'datasets'
	ABC_mesh_files = sorted(glob.glob(os.path.join(load_dir, 'ABC*components_test_meshes.npz')))
	filepath = ABC_mesh_files[0]
	filename = filepath.split('/')[-1]
	rootname = filename.split('_')[0]
	t0 = time.time()
	generate_scanned_mesh_training_dataset(
		n_views=4,
		npz_filepath=filepath,
		save_filepath=load_dir + f'/{rootname}_validation.npz',
		meshes_per_npz=n_validation,
		only_save_one=True,
	)
	t1 = time.time()
	print(f'ABC processing time: {t1 - t0}s')


if __name__ == '__main__':
	# Generate blobby training data
	generate_blob_dataset(
		n_blobs=3000,
		save_dir='datasets',
		name='blobs',
		alpha_range=(0.1, 0.7),
		scale_range=(0.5, 2.0),
		octaves_range=(1, 4),
		base_radius=1.0,
		grid_resolution=64,
		persistence=0.5,
		lacunarity=2.0,
		seed=42,
		train_test_split=0.8,
	)

	generate_pretraining_dataset()

	generate_all_training_datasets(n_views=4, mode=0)
	generate_all_training_datasets(n_views=4, mode=-1)
