import os
import gc
import sys
import psutil
import argparse
import time
import glob
from enum import Enum

import pyglet
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.gl import *
import ctypes

import imgui
from imgui.integrations.pyglet import create_renderer

from pointsastori.shape_3d import *
from pointsastori.network import *
from shaders_3d import *

K_NEIGHBORS_ACCELERATION = 32
SCREEN_WIDTH, SCREEN_HEIGHT = 3024 // 4, 1964 // 4
FPS = 30

OBJ_FILEPATHS = sorted(glob.glob('../data/*.obj'))
PLY_FILEPATHS = sorted(glob.glob('../data/*.ply'))

ShaderQuantity = Enum(
	'ShaderQuantity',
	[
		('true', 0),
		('PAT', 1),
		('SSPD', 2),
		('SHC', 3),
		('GWN', 4),
		('LogSumExp', 5),
		('PAT_eikonality', 6),
		('SSPD_eikonality', 7),
		('SHC_eikonality', 8),
	],
)

SHADER_QUANTITIES = [
	'true SDF',  # 0
	'points as tori',  # 1
	'(smoothed) signed planar distance',  # 2 (a.k.a. naive self-normalized convolutional distance, smoothed pseudonormal distance)
	'signed Hopf-Cole',  # 3
	'winding number',  # 4
	'unsigned LogSumExp',  # 5
	'1-|grad(PAT)|',  # 6
	'1-|grad(SSPD)|',  # 7
	'1-|grad(SHC)|',  # 8
	'1-|grad(true)|',  # 9
]

# ../src/pointsastori/models/
MODEL_NAMES = [
	'FundamentalFormPredictor.pkl',
]


def log_memory(label):
	process = psutil.Process(os.getpid())
	mem_mb = process.memory_info().rss / 1024**2
	print(f'{label}: {mem_mb:.1f} MB')


def mesh_isocontour(
	TDF: TorusDistanceField,
	isoval: float = 0.0,
	grid_resolution: int = 64,
	eval_method: int = ShaderQuantity.PAT.value,
	epsilon=0.01,
	method: str = 'flying_edges',
):
	"""
	Create mesh of contour of the torus SDF by contouring the SDF on a regular grid of the given resolution.

	Args:
		isovalue: The isosurface value to extract (default 0.0)
		grid_resolution: Resolution of the grid in each dimension (default 64)
		method: Contouring method, 'flying_edges' or 'marching_cubes'
	"""

	# Get bounding box of point cloud
	bbox_min, bbox_max = point_cloud_bounding_box(TDF.get_points())
	bbox_center = (bbox_min + bbox_max) / 2.0
	max_extent = np.max(np.abs(bbox_max - bbox_min))

	# Pad by isovalue to ensure isosurface doesn't touch boundary
	padding = np.abs(isoval) + max_extent * 0.1  # Add 10% extra margin
	grid_half_size = (max_extent / 2.0) + padding

	# Create regular grid centered at bounding box center
	grid_min = bbox_center - grid_half_size
	grid_max = bbox_center + grid_half_size

	x = np.linspace(grid_min[0], grid_max[0], grid_resolution)
	y = np.linspace(grid_min[1], grid_max[1], grid_resolution)
	z = np.linspace(grid_min[2], grid_max[2], grid_resolution)

	xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

	# Flatten grid points for query
	grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)  # (N^3, 3)

	# Evaluate distances on all points at once
	n_points = len(grid_points)
	print(f'\tEvaluating {SHADER_QUANTITIES[eval_method]} method on {n_points} grid points...')
	distances = None
	isovalue = 0.0
	if eval_method == ShaderQuantity.PAT.value:
		distances = TDF.evaluate_distance(grid_points, isovalue)
		isovalue = isoval
	elif eval_method == ShaderQuantity.GWN.value:
		TDF.fast_winding_number_precompute()
		distances = TDF.fast_winding_number(grid_points)
		isovalue = 0.5
	elif eval_method == ShaderQuantity.SHC.value:
		distances = TDF.signed_log_sum_exp(grid_points)
		isovalue = isoval
	elif eval_method == ShaderQuantity.SSPD.value:
		distances = TDF.self_normalized_signed_log_sum_exp(grid_points)
		isovalue = isoval
	elif eval_method == ShaderQuantity.LogSumExp.value:
		distances = TDF.log_sum_exp(grid_points)
		isovalue = isoval

	# Reshape to grid
	distance_grid = distances.reshape(grid_resolution, grid_resolution, grid_resolution)

	# Use PyVista for contouring
	print(f'Running {method}...')

	try:
		import pyvista as pv

		# Create structured grid (ImageData)
		grid = pv.ImageData()
		grid.dimensions = [grid_resolution, grid_resolution, grid_resolution]
		grid.origin = grid_min
		grid.spacing = [
			(grid_max[0] - grid_min[0]) / (grid_resolution - 1),
			(grid_max[1] - grid_min[1]) / (grid_resolution - 1),
			(grid_max[2] - grid_min[2]) / (grid_resolution - 1),
		]

		grid.point_data['values'] = distance_grid.ravel(order='F')

		# Extract isosurface using selected method
		if method.lower() == 'flying_edges':
			mesh = grid.contour(isosurfaces=[isovalue], scalars='values', method='flying_edges')
		elif method.lower() == 'marching_cubes':
			mesh = grid.contour(isosurfaces=[isovalue], scalars='values', method='marching_cubes')
		else:
			raise ValueError(f"Unknown method '{method}'. Use 'flying_edges' or 'marching_cubes'")

		# Extract vertices and faces
		vertices = np.array(mesh.points)

		# PyVista faces are stored as [n, v0, v1, v2, n, v3, v4, v5, ...]
		# where n is the number of vertices in each face
		faces_raw = mesh.faces

		# Reshape to get individual faces
		n_faces = mesh.n_faces_strict
		faces = []
		idx = 0
		for _ in range(n_faces):
			n_verts = faces_raw[idx]
			face = faces_raw[idx + 1 : idx + 1 + n_verts]
			faces.append(face)
			idx += 1 + n_verts

		faces = np.array(faces)

		faces[:, [1, 2]] = faces[:, [2, 1]]  # because PyVista is annoying

		print(f'Extracted mesh: {len(vertices)} vertices, {len(faces)} faces')

		# Optional: Clean the mesh (remove duplicate vertices, etc.)
		mesh_cleaned = mesh.clean()
		if mesh_cleaned.n_points < mesh.n_points:
			print(f'Cleaned mesh: {mesh.n_points} -> {mesh_cleaned.n_points} vertices')
			vertices = np.array(mesh_cleaned.points)

			faces_raw = mesh_cleaned.faces
			n_faces = mesh_cleaned.n_faces_strict
			faces = []
			idx = 0
			for _ in range(n_faces):
				n_verts = faces_raw[idx]
				face = faces_raw[idx + 1 : idx + 1 + n_verts]
				faces.append(face)
				idx += 1 + n_verts
			faces = np.array(faces)

	except ImportError:
		raise ImportError('PyVista is not available. Please install it: pip install pyvista')

	return vertices, faces


class RenderPass:
	"""
	Represents a single-pass shader with its own framebuffer (where to store any textures that it writes).
	"""

	def __init__(self, vertex_source, fragment_source, output_size=(1, 1), num_outputs=1):
		self.vertex_shader = Shader(vertex_source, 'vertex')
		self.fragment_shader = Shader(fragment_source, 'fragment')
		self.shader_program = ShaderProgram(self.vertex_shader, self.fragment_shader)

		# Create a full-screen quad, triangulated by two triangles
		vertices = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
		indices = [0, 1, 2, 0, 2, 3]
		self.vertex_list = self.shader_program.vertex_list_indexed(
			4, GL_TRIANGLES, indices, position=('f', vertices)
		)  # 4 unique vertices

		# Create framebuffer and texture
		self.width, self.height = output_size
		self.num_outputs = num_outputs
		self.create_framebuffer()

	def create_framebuffer(self):
		# Create multiple textures
		self.textures = []
		for i in range(self.num_outputs):
			texture = GLuint()
			glGenTextures(1, ctypes.byref(texture))
			glBindTexture(GL_TEXTURE_2D, texture)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.width, self.height, 0, GL_RGBA, GL_FLOAT, None)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
			glBindTexture(GL_TEXTURE_2D, 0)
			self.textures.append(texture)

		# Create framebuffer
		self.framebuffer = GLuint()
		glGenFramebuffers(1, ctypes.byref(self.framebuffer))
		glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)
		# glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture, 0)

		# Attach multiple textures to different color attachments
		draw_buffers = []
		for i, texture in enumerate(self.textures):
			attachment = GL_COLOR_ATTACHMENT0 + i
			glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture, 0)
			draw_buffers.append(attachment)

		# Set which color attachments to draw to
		glDrawBuffers(len(draw_buffers), (GLenum * len(draw_buffers))(*draw_buffers))

		# Check completeness
		if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
			print(f'Framebuffer not complete for pass')

		glBindFramebuffer(GL_FRAMEBUFFER, 0)

	def read_output(self, output_index=0):
		"""
		Read back data from a specific output texture.
		"""
		glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)
		glReadBuffer(GL_COLOR_ATTACHMENT0 + output_index)
		result = (ctypes.c_float * 4)()
		glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, result)
		return result

	def read_full_output(self, output_index=0) -> np.ndarray:
		glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)
		glReadBuffer(GL_COLOR_ATTACHMENT0 + output_index)
		data = np.zeros((self.height, self.width, 4), dtype=np.float32)
		glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_FLOAT, data.ctypes.data)
		return data

	def render(self, uniforms=None):
		glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)
		glViewport(0, 0, self.width, self.height)
		glClear(GL_COLOR_BUFFER_BIT)
		self.shader_program.use()

		# Set shader uniforms if provided
		if uniforms:
			for name, value in uniforms.items():
				self.shader_program[name] = value

		self.vertex_list.draw(GL_TRIANGLE_STRIP)  # Warning: Very slow on the first frame

	def cleanup(self):
		glDeleteFramebuffers(1, ctypes.byref(self.framebuffer))
		for texture in self.textures:
			glDeleteTextures(1, ctypes.byref(texture))


class ShaderWindow(pyglet.window.Window):
	def __init__(self) -> None:
		super().__init__(resizable=True, style=pyglet.window.Window.WINDOW_STYLE_DEFAULT)
		self.set_caption('Point cloud SDF shader')

		# Center window on screen
		screen = self.display.get_default_screen()
		screen_width = screen.width
		screen_height = screen.height
		window_width = screen_width // 2
		window_height = screen_height // 2
		self.set_size(window_width, window_height)
		self.set_location((screen_width - window_width) // 2, (screen_height - window_height) // 2)

		self.seed = 42
		self.key = jax.random.PRNGKey(self.seed)

		# Algorithm parameters
		self.use_point_areas = True
		self.lambda_scale = 1.0
		self.max_exp_arg = 64.0
		self.max_neighborhood_size = 128
		self.neighborhood_size = -1
		self.neighborhood_radius = 0.0
		self.epsilon = 0.5
		self.model_idx = 0

		self.n_points = 2048

		# Set up ShaderToy-inspired functionalities.
		self.time = 0.0
		self.frame = 0
		pyglet.clock.schedule_interval(self.update, 1.0 / FPS)

		# GUI parameters
		self.vis_cutplane = True
		self.vis_shape = False
		self.vis_points = True
		self.vis_normals = False
		self.vis_contour = False
		self.point_radius = 0.005
		self.vis_tori = False

		self.colormap_range = 2.0
		self.isoband_frequency = 60.0
		self.isovalue = 0.0
		self.zoom = 1.0
		self.mouse_x = 0
		self.mouse_y = 0

		# Camera controls
		self.zoom_sensitivity = 0.02
		self.pan_sensitivity = 0.005
		self.camera_shift = np.array([0.0, 0.0])
		self.mouse_dragging = False
		self.last_mouse_x = 0
		self.last_mouse_y = 0

		self.floor_height = -1.0
		self.freeze_cut_plane = True
		self.cut_plane_depth = 0.0
		self.cut_plane_axis = 2  # controls cut plane normal
		self.cut_plane_normal = [0.0, 0.0, 1.0]
		self.cut_plane_rotation_x = 0.0
		self.cut_plane_rotation_y = 0.0
		self.cut_plane_diag = [1.0, 1.0, 1.0]
		self.camera_y_rotation = 5.4  # rotation about the vertical axis
		self.camera_radius = 10.0
		self.camera_x_rotation = 0.0
		self.initial_camera_position = np.array(
			[
				self.camera_radius * np.cos(self.camera_y_rotation),
				2.4,
				self.camera_radius * np.sin(self.camera_y_rotation),
			]
		)
		self.camera_position = self.initial_camera_position
		self.light_position = [0.0, 10.0, 2.0]  # diffuse lighting
		self.light_rotation = 0.0  # rotation about y-axis
		self.light_radius = 2.0
		self.light_intensity = 1.2
		self.specular_light_position = [-0.5, 1.0, 0.0]
		self.specular_light_rotation = 0.0  # rotation about y-axis

		self.grid_resolution = 256

		# Texture overhead
		max_size = GLint()
		size = GLint()
		max_frag_units = GLint()
		max_vert_units = GLint()
		max_combined_units = GLint()
		glGetIntegerv(GL_MAX_TEXTURE_SIZE, ctypes.byref(max_size))  # 16384 on Mac, 32768 on NVIDIA GPU
		glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE, ctypes.byref(size))  # 268435456 on Mac, 134217728 on NVIDIA GPU
		glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, ctypes.byref(max_frag_units))  # 16 on Mac
		glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, ctypes.byref(max_vert_units))  # 16 on Mac
		glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, ctypes.byref(max_combined_units))  # 80 on Mac
		self.texture_size = max_size.value

		# For shaders
		self.render_passes = []

		self.available_shapes = OBJ_FILEPATHS + PLY_FILEPATHS + SDF_SHAPES
		self.current_shape_index = 0
		self.quantity = SHADER_QUANTITIES[ShaderQuantity.PAT.value]
		self.quantity_idx = SHADER_QUANTITIES.index(self.quantity)
		self.shape_filepath = self.available_shapes[self.current_shape_index]
		log_memory(f'\nBefore load_shape()')
		self.load_shape()

		t0 = time.time()
		self.compile_shader()
		t1 = time.time()
		print('Compile shader: %f s' % (t1 - t0))

		# Create imgui menu
		imgui.create_context()
		self.renderer = create_renderer(self)

		io = imgui.get_io()
		io.display_size = self.get_size()
		io.display_fb_scale = 1.0, 1.0

	def load_shape(self) -> None:
		"""
		Load an input point cloud file.
		"""
		filename_components = self.shape_filepath.split('.')
		extension = filename_components[-1]
		point_colors = np.zeros((1,))
		self.shape = None
		self.SDF3D = None
		self.isocontour = None

		if len(filename_components) == 1:
			# Sample a revolved/extruded SDF
			self.key, subkey = jax.random.split(self.key)
			is_extrusion = False
			SDF3D_param = 0.0
			n_samples = 1024
			shape = sample_SDF(extension, subkey)
			if jax.random.uniform(subkey, minval=0.0, maxval=1.0) < 0.5:
				self.key, subkey = jax.random.split(self.key)
				SDF3D_param = jax.random.uniform(subkey, minval=0.1, maxval=0.3)
				self.SDF3D = SDF3D(sdf2d=shape, is_extrusion=False, param=SDF3D_param)
				points, normals = shape.sample_revolution(n_samples, o=SDF3D_param)
				self.pointcloud = PointCloud3D(points, normals)
				self.pointcloud_size = self.pointcloud.size
			else:
				is_extrusion = True
				self.key, subkey = jax.random.split(self.key)
				SDF3D_param = jax.random.uniform(subkey, minval=0.1, maxval=0.3)
				self.SDF3D = SDF3D(sdf2d=shape, is_extrusion=True, param=SDF3D_param)
				points, normals = shape.sample_extrusion(n_samples, h=SDF3D_param)
				self.pointcloud = PointCloud3D(points, normals)
				self.pointcloud_size = self.pointcloud.size

		else:
			import open3d as o3d

			o3d_mesh = o3d.io.read_triangle_mesh(self.shape_filepath)
			n_faces = len(np.asarray(o3d_mesh.triangles))

			if n_faces > 0:
				t0 = time.time()
				vertices = np.asarray(o3d_mesh.vertices, dtype=np.float64)
				faces = np.asarray(o3d_mesh.triangles, dtype=np.int64)
				self.shape = TriangleMesh(vertices, faces)
				self.shape.center_and_scale()
				self.shape_size = self.shape.faces.shape[0]

				# If a mesh was passed in, sample a point cloud
				points, normals = self.shape.sample_uniform_point_cloud(
					n_points=self.n_points, max_noise_level=0.0, max_normals_to_flip=0.0, seed=self.seed
				)
				# points, normals = self.shape.sample_farthest_point_cloud(n_points=self.n_points, max_noise_level=0.0, max_normals_to_flip=0.0, seed=self.seed)

				self.pointcloud = PointCloud3D(points, normals)
				self.pointcloud_size = self.pointcloud.size
				t1 = time.time()
				print('Point cloud generation: %f s' % (t1 - t0))
				log_memory(f'\nAfter PointCloud3D()')

				t0 = time.time()
				self.create_triangle_mesh_texture()
				t1 = time.time()
				print('Triangle mesh texture creation (precompute): %f s' % (t1 - t0))
				log_memory(f'\nAfter create_triangle_mesh_texture()')
			else:
				t0 = time.time()
				self.pointcloud = PointCloud3D.read(self.shape_filepath)
				self.pointcloud_size = self.pointcloud.size
				# self.pointcloud.center_and_scale()
				t1 = time.time()
				print('Load point cloud: %f s' % (t1 - t0))

				# Look for color/texture data
				point_colors = np.zeros((self.pointcloud.size, 3))
				if self.pointcloud.colors is not None and len(self.pointcloud.colors) > 0:
					point_colors = self.pointcloud.colors

		print('Size of point cloud: %d' % (self.pointcloud_size))

		# Initialize point colors
		zeros = np.zeros(point_colors.shape[0])
		data = np.c_[point_colors, zeros]
		self.pointcolors_texture = GLuint()
		self.create_texture(self.pointcolors_texture, data)
		log_memory(f'\nAfter point colors()')

		self.TDF = TorusDistanceField(self.pointcloud.points, self.pointcloud.normals)

		t0 = time.time()
		self.create_learned_tori_texture()
		t1 = time.time()
		print('Neural net evaluation of fitted tori (precompute): %f s' % (t1 - t0))
		log_memory(f'\nAfter create_learned_tori_texture()')

		t0 = time.time()
		self.create_pointcloud_texture()
		t1 = time.time()
		print('Point cloud texture creation (precompute): %f s' % (t1 - t0))
		log_memory(f'\nAfter create_pointcloud_texture()')

		self.create_cubemap_texture()

		# Set cut plane bounds
		bbox_min = jnp.min(self.pointcloud.points, axis=0)
		bbox_max = jnp.max(self.pointcloud.points, axis=0)
		diag = bbox_max - bbox_min  # (3,)
		self.cut_plane_diag = diag  # "radius" of cut plane
		self.floor_height = bbox_min[2] - 1.0

	def bind_textures(self) -> None:
		if self.shape != None:
			glActiveTexture(GL_TEXTURE0)
			glBindTexture(GL_TEXTURE_2D, self.shape_texture)

			glActiveTexture(GL_TEXTURE7)
			glBindTexture(GL_TEXTURE_2D, self.mesh_bvh_nodes_texture)

			glActiveTexture(GL_TEXTURE8)
			glBindTexture(GL_TEXTURE_2D, self.mesh_bvh_indices_texture)

		if self.isocontour != None:
			glActiveTexture(GL_TEXTURE9)
			glBindTexture(GL_TEXTURE_2D, self.isocontour_texture)

			glActiveTexture(GL_TEXTURE10)
			glBindTexture(GL_TEXTURE_2D, self.isocontour_bvh_nodes_texture)

			glActiveTexture(GL_TEXTURE11)
			glBindTexture(GL_TEXTURE_2D, self.isocontour_bvh_indices_texture)

		glActiveTexture(GL_TEXTURE1)
		glBindTexture(GL_TEXTURE_2D, self.pc_texture)

		glActiveTexture(GL_TEXTURE2)
		glBindTexture(GL_TEXTURE_2D, self.learned_tori_texture)

		glActiveTexture(GL_TEXTURE3)
		glBindTexture(GL_TEXTURE_CUBE_MAP, self.cubemap_texture)

		glActiveTexture(GL_TEXTURE4)
		glBindTexture(GL_TEXTURE_2D, self.pc_bvh_nodes_texture)

		glActiveTexture(GL_TEXTURE5)
		glBindTexture(GL_TEXTURE_2D, self.pc_bvh_indices_texture)

		glActiveTexture(GL_TEXTURE6)
		glBindTexture(GL_TEXTURE_2D, self.pointcolors_texture)

	def load_model(self) -> None:
		"""
		Change which trained neural network is used to infer fitted circles.
		"""
		print('Model name: %s' % MODEL_NAMES[self.model_idx])
		t0 = time.time()
		self.create_learned_tori_texture()
		t1 = time.time()
		print('Neural net evaluation of fitted tori (precompute): %f s' % (t1 - t0))

	def create_cubemap_texture(self) -> None:
		"""
		https://learnopengl.com/Advanced-OpenGL/Cubemaps
		https://shadertoyunofficial.wordpress.com/2019/07/23/shadertoy-media-files/
		"""
		# Generate and bind the cubemap texture
		self.cubemap_texture = GLuint()
		glGenTextures(1, ctypes.byref(self.cubemap_texture))
		glBindTexture(GL_TEXTURE_CUBE_MAP, self.cubemap_texture)

		faces = [
			GL_TEXTURE_CUBE_MAP_POSITIVE_X,
			GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
			GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
			GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
			GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
			GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
		]

		# Load each face of the cubemap
		for i in range(6):
			filepath = f'./assets/ForestBlurred{i}.png'

			# Load image using pyglet
			image = pyglet.image.load(filepath)
			image_data = image.get_image_data()

			# Get format and data
			format_str = image_data.format
			width = image_data.width
			height = image_data.height
			pitch = image_data.pitch
			data = image_data.get_data(format_str, pitch)

			# Determine OpenGL format
			if format_str == 'RGB':
				gl_format = GL_RGB
			elif format_str == 'RGBA':
				gl_format = GL_RGBA
			else:
				gl_format = GL_RGB

			# Upload texture data to the cubemap face
			glTexImage2D(faces[i], 0, gl_format, width, height, 0, gl_format, GL_UNSIGNED_BYTE, data)

		# Set texture parameters
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

		# Unbind the texture
		glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

	def create_texture(self, texture: GLuint, data: np.ndarray) -> None:
		"""
		Boilerplate that gets called in the creation of every texture
		"""
		n_elements = data.shape[0]
		n_rows = (n_elements + self.texture_size - 1) // self.texture_size
		texture_data = np.zeros((n_rows, self.texture_size, 4), dtype=np.float32)
		texture_data.flat[: n_elements * 4] = data.flat
		texture_data = np.ascontiguousarray(texture_data)
		glGenTextures(1, ctypes.byref(texture))  # 1 = Number of texture names to generate (1 texture)
		glBindTexture(
			GL_TEXTURE_2D, texture
		)  # bind texture (activate so that all subsequent texture operations act on this texture)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.texture_size, n_rows, 0, GL_RGBA, GL_FLOAT, None)
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.texture_size, n_rows, GL_RGBA, GL_FLOAT, texture_data.ctypes.data)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)  # sets filtering when texture is scaled down
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)  # sets filtering when texture is scaled up
		glBindTexture(GL_TEXTURE_2D, 0)  # unbind texture (deactivate)

	def pack_bvh_nodes_flat(self, node_data):
		"""Pack BVH nodes: 10 floats -> 3 RGBA rows."""
		n_nodes = node_data.shape[0]
		flat_data = np.zeros((n_nodes * 3, 4), dtype=np.float32)
		for i in range(n_nodes):
			base = i * 3
			flat_data[base] = [node_data[i, 0], node_data[i, 1], node_data[i, 2], node_data[i, 3]]
			flat_data[base + 1] = [node_data[i, 4], node_data[i, 5], node_data[i, 6], node_data[i, 7]]
			flat_data[base + 2] = [node_data[i, 8], node_data[i, 9], 0.0, 0.0]
		return flat_data

	def pack_bvh_indices_flat(self, prim_indices):
		flat_data = np.zeros((len(prim_indices), 4), dtype=np.float32)
		flat_data[:, 0] = prim_indices.astype(np.float32)
		return flat_data

	def export_cutplane(self, resolution, filename='cutplane.ply'):
		"""
		Export cut plane mesh by rendering to texture and reading back.
		"""
		# Determine grid resolution
		if self.cut_plane_axis == 0:
			width = 2.0 * self.cut_plane_diag[1]
			height = 2.0 * self.cut_plane_diag[2]
		elif self.cut_plane_axis == 1:
			width = 2.0 * self.cut_plane_diag[0]
			height = 2.0 * self.cut_plane_diag[2]
		else:
			width = 2.0 * self.cut_plane_diag[0]
			height = 2.0 * self.cut_plane_diag[1]

		if width < height:
			n_width = resolution
			n_height = int(np.ceil(resolution * height / width))
		else:
			n_height = resolution
			n_width = int(np.ceil(resolution * width / height))

		# Resize framebuffer if needed
		if self.cutplane_export_pass.width != n_width or self.cutplane_export_pass.height != n_height:
			self.cutplane_export_pass.width = n_width
			self.cutplane_export_pass.height = n_height
			self.cutplane_export_pass.create_framebuffer()

		# Bind textures and render
		self.bind_textures()

		# Get common uniforms and add cutplane-specific ones
		screen_width, screen_height = self.get_size()
		uniforms = {
			'u_resolution': (float(n_width), float(n_height)),  # pixel dimensions of buffer
			'cutplane_resolution': (float(n_width), float(n_height)),
			'screenWidth': float(screen_width),
			'cutplaneDepth': self.cut_plane_depth,
			'cutplaneAxis': self.cut_plane_axis,
			'freezeCutplane': self.freeze_cut_plane,
			'lambdaScale': self.lambda_scale,
			'maxExpArg': self.max_exp_arg,
			'neighborhood_size': self.neighborhood_size,
			'neighborhood_radius': self.neighborhood_radius,
			'mouseCoord': (self.mouse_x, self.mouse_y),
			'cutplaneDiag': (self.cut_plane_diag[0], self.cut_plane_diag[1], self.cut_plane_diag[2]),
			'cutplaneRotationX': self.cut_plane_rotation_x,
			'cutplaneRotationY': self.cut_plane_rotation_y,
			'minDist': -self.colormap_range,
			'maxDist': self.colormap_range,
			'isobandFrequency': self.isoband_frequency,
			'shape': 0,
			'pointcloud': 1,
			'learnedtori': 2,
			'n_points': self.pointcloud_size,
			'texture_size': self.texture_size,
			'pc_bvh_nodes': 4,
			'pc_bvh_prim_indices': 5,
			'n_pc_bvh_nodes': self.pc_bvh.get_num_nodes() if hasattr(self, 'pc_bvh') else 0,
			'mesh_bvh_nodes': 7,
			'mesh_bvh_prim_indices': 8,
			'n_mesh_bvh_nodes': self.mesh_bvh.get_num_nodes() if hasattr(self, 'mesh_bvh') and self.shape else 0,
			'quantity': self.quantity_idx,
		}

		if self.shape is not None:
			uniforms['n_faces'] = self.shape_size

		self.cutplane_export_pass.render(uniforms)

		# Read back data
		positions = self.cutplane_export_pass.read_full_output(0)[:, :, :3]  # (height, width, 3)
		scalars = self.cutplane_export_pass.read_full_output(1)[:, :, 0]  # (height, width)
		colors = self.cutplane_export_pass.read_full_output(2)[:, :, :3]  # (height, width, 3)

		# Flatten to vertex arrays
		vertices = positions.reshape(-1, 3)
		vertex_colors = (np.clip(colors.reshape(-1, 3), 0, 1) * 255).astype(np.uint8)

		# Generate triangle faces (quad grid)
		faces = []
		for j in range(n_height - 1):
			for i in range(n_width - 1):
				v0 = j * n_width + i
				v1 = j * n_width + (i + 1)
				v2 = (j + 1) * n_width + (i + 1)
				v3 = (j + 1) * n_width + i

				faces.append([v0, v1, v2])
				faces.append([v0, v2, v3])

		faces = np.array(faces)

		# Export
		export_PLY_with_colors(vertices, faces, vertex_colors, filename, binary=True)
		print(f'Exported cut plane to {filename}: {len(vertices)} vertices, {len(faces)} faces')

	def create_isocontour_mesh_texture(self, shape) -> None:
		zeros = np.zeros(shape.faces.shape[0])
		data_v0 = np.c_[shape.vertices[shape.faces[:, 0]], zeros]  # (|F|, 4)
		data_v1 = np.c_[shape.vertices[shape.faces[:, 1]], zeros]  # (|F|, 4)
		data_v2 = np.c_[shape.vertices[shape.faces[:, 2]], zeros]  # (|F|, 4)
		data = np.ravel(np.column_stack((data_v0, data_v1, data_v2))).reshape((-1, 4))  # (3|F|, 4)
		self.isocontour_texture = GLuint()
		self.create_texture(self.isocontour_texture, data)

		t0 = time.time()
		self.isomesh_bvh = BoundingVolumeHierarchy(shape.vertices, shape.faces)
		node_data, prim_indices = self.isomesh_bvh.get_gpu_data()
		t1 = time.time()
		print(f'  Isomesh BVH: {self.isomesh_bvh.get_num_nodes()} nodes in {t1 - t0:.3f}s')

		bvh_nodes = self.pack_bvh_nodes_flat(node_data)
		bvh_indices = self.pack_bvh_indices_flat(prim_indices)

		self.isocontour_bvh_nodes_texture = GLuint()
		self.create_texture(self.isocontour_bvh_nodes_texture, bvh_nodes)

		self.isocontour_bvh_indices_texture = GLuint()
		self.create_texture(self.isocontour_bvh_indices_texture, bvh_indices)

	def create_triangle_mesh_texture(self) -> None:
		"""
		Create a 2D texture that stores the original mesh from which the point cloud is sampled.
		"""
		# |V| ~ 3|F| => storing 3D vertex positions & face indices takes 3|V| + 3|F| ~ 12|F| entries >= 3|F| texels
		# Compare to storing three 3D vertex positions per face (repeating vertex positions), which takes 9|F| entries
		# So pack as RGBA, three texels for each face: [v0_x, v0_y, v0_z, 0], [v1_x, v1_y, v1_z, 0], [v2_x, v2_y, v2_z, 0]
		# Could use SSBO instead
		zeros = np.zeros(self.shape.faces.shape[0])
		data_v0 = np.c_[self.shape.vertices[self.shape.faces[:, 0]], zeros]  # (|F|, 4)
		data_v1 = np.c_[self.shape.vertices[self.shape.faces[:, 1]], zeros]  # (|F|, 4)
		data_v2 = np.c_[self.shape.vertices[self.shape.faces[:, 2]], zeros]  # (|F|, 4)
		data = np.ravel(np.column_stack((data_v0, data_v1, data_v2))).reshape((-1, 4))  # (3|F|, 4)
		self.shape_texture = GLuint()
		self.create_texture(self.shape_texture, data)

		t0 = time.time()
		self.mesh_bvh = BoundingVolumeHierarchy(self.shape.vertices, self.shape.faces)
		node_data, prim_indices = self.mesh_bvh.get_gpu_data()
		t1 = time.time()
		print(f'  Mesh BVH: {self.mesh_bvh.get_num_nodes()} nodes in {t1 - t0:.3f}s')

		bvh_nodes = self.pack_bvh_nodes_flat(node_data)
		bvh_indices = self.pack_bvh_indices_flat(prim_indices)

		self.mesh_bvh_nodes_texture = GLuint()
		self.create_texture(self.mesh_bvh_nodes_texture, bvh_nodes)

		self.mesh_bvh_indices_texture = GLuint()
		self.create_texture(self.mesh_bvh_indices_texture, bvh_indices)

	def create_pointcloud_texture(self) -> None:
		"""
		Create a 2D texture that will be read by shaders, that stores the input point cloud.
		"""

		# Pack as RGBA, two texels per point: [point_x, point_y, point_z, is_outlier], [normal_x, normal_y, normal_z, 0]

		# Store point areas
		# This unfortunately can take a super long time/run out of memory on large point clouds because it uses libigl
		point_attributes = self.TDF.get_point_areas()
		data_points = np.c_[self.pointcloud.points, point_attributes]  # (|P|, 4)
		data_normals = np.c_[self.pointcloud.normals, np.zeros(self.pointcloud.normals.shape[0])]  # (|P|, 4)
		data = np.ravel(np.column_stack((data_points, data_normals))).reshape((-1, 4))  # (2|P|, 4)
		self.pc_texture = GLuint()
		self.create_texture(self.pc_texture, data)

		t0 = time.time()
		self.pc_bvh = BoundingVolumeHierarchy(self.pointcloud.points)
		node_data, prim_indices = self.pc_bvh.get_gpu_data()
		t1 = time.time()
		print(f'  Point cloud BVH: {self.pc_bvh.get_num_nodes()} nodes in {t1 - t0:.3f}s')

		bvh_nodes = self.pack_bvh_nodes_flat(node_data)
		bvh_indices = self.pack_bvh_indices_flat(prim_indices)

		self.pc_bvh_nodes_texture = GLuint()
		self.create_texture(self.pc_bvh_nodes_texture, bvh_nodes)

		self.pc_bvh_indices_texture = GLuint()
		self.create_texture(self.pc_bvh_indices_texture, bvh_indices)

	def load_torus_model_and_predict_tori(self):
		model_name = MODEL_NAMES[self.model_idx]
		model_name_full = '../src/pointsastori/' + MODELS_DIR + model_name
		print(f'Loading {model_name_full}...')

		model, k_neighbors = FundamentalFormPredictor.load_saved_model(model_name_full)
		log_memory(f'\nAfter load_saved_model()')

		t00 = time.time()
		t0 = time.time()
		coefficients = model.precompute_coefficients_in_chunks(
			self.pointcloud.points,
			self.pointcloud.normals,
			k_neighbors,
			chunk_size=50000,
		)
		t1 = time.time()
		print('  Form coefficient inference: %f s' % (t1 - t0))

		jax.clear_caches()

		log_memory(f'\nAfter precompute_coefficients()')

		t0 = time.time()
		centers, axes, major_radii, minor_radii = fit_tori_from_forms(
			self.pointcloud.points, self.pointcloud.normals, coefficients
		)
		t1 = time.time()
		print('  Torus-fitting from forms: %f s' % (t1 - t0))
		t11 = time.time()
		print('Total precompute: %f s' % (t11 - t00))

		jax.clear_caches()
		gc.collect()

		centers = np.asarray(centers)
		axes = np.asarray(axes)
		major_radii = np.asarray(major_radii)
		minor_radii = np.asarray(minor_radii)

		return centers, axes, major_radii, minor_radii

	def create_learned_tori_texture(self) -> None:
		"""
		Create a 2D texture that stores precomputed torus parameters learned via neural network.
		"""
		t0 = time.time()
		self.centers, self.axes, self.major_radii, self.minor_radii = (
			self.load_torus_model_and_predict_tori()
		)  # (|P|, 7)
		t1 = time.time()
		# print(f'Min radii: {np.min(self.major_radii)} {np.min(self.minor_radii)}')
		# print(f'Max radii: {np.max(self.major_radii)} {np.max(self.minor_radii)}')
		print('Total torus-fitting time: %f s' % (t1 - t0))
		zeros = np.zeros(self.centers.shape[0])
		data_centers = np.c_[self.centers, zeros]  # (|P|, 4)
		data_axes_radii = np.c_[self.axes, self.major_radii, self.minor_radii]  # (|P|, 4)
		data = np.ravel(np.column_stack((data_centers, data_axes_radii))).reshape((-1, 4))  # (2|P|, 4)
		self.learned_tori_texture = GLuint()
		self.create_texture(self.learned_tori_texture, data)

	def mesh_isocontour(self) -> None:
		self.TDF.set_k_evaluation(self.neighborhood_size)
		self.TDF.set_tori(self.centers, self.axes, self.major_radii, self.minor_radii)
		self.TDF.set_lambda_scale(self.lambda_scale)
		vertices, faces = mesh_isocontour(
			self.TDF,
			isoval=self.isovalue,
			grid_resolution=self.grid_resolution,
			eval_method=self.quantity_idx,
			epsilon=self.epsilon,
			method='flying_edges',
		)
		self.isocontour = TriangleMesh(vertices, faces)
		self.create_isocontour_mesh_texture(self.isocontour)

	def compile_shader(self) -> None:
		"""
		(Re)build shader(s) with current parameters.

		This function re-compiles all the buffers in a possibly multi-pass shader.
		"""
		for i in range(len(self.render_passes)):
			self.render_passes[i].cleanup()
		self.render_passes.clear()

		# Fill in common functionalities with interactive parameters.
		fragment_common = COMMON_TEMPLATE.format(COMMON_SHADER=COMMON_SHADER)

		# The pass that computes quantities that depend on external/interactive parameters like mouse interaction
		fragment_interactive = INTERACTION_SHADER_TEMPLATE.format(COMMON_DYNAMIC=fragment_common)
		interactive_pass = RenderPass(VERTEX_SHADER, fragment_interactive, output_size=(1, 1), num_outputs=3)
		self.render_passes.append(interactive_pass)

		# Construct the main shader.
		fragment_main = MAIN_FRAGMENT_SHADER_TEMPLATE.format(COMMON_DYNAMIC=fragment_common)
		width, height = self.get_size()
		main_pass = RenderPass(VERTEX_SHADER, fragment_main, output_size=(width, height))
		self.render_passes.append(main_pass)

		# Cut plane export pass (compiled but not rendered every frame)
		fragment_cutplane = CUTPLANE_EXPORT_SHADER.format(COMMON_DYNAMIC=fragment_common)
		self.cutplane_export_pass = RenderPass(VERTEX_SHADER, fragment_cutplane, output_size=(1, 1), num_outputs=3)

	def save_screenshot(self) -> None:
		# Find next available screenshot number
		existing = glob.glob('screenshot_*.png')
		numbers = [
			int(f.replace('screenshot_', '').replace('.png', ''))
			for f in existing
			if f.replace('screenshot_', '').replace('.png', '').isdigit()
		]
		next_num = max(numbers) + 1 if numbers else 0
		filename = f'screenshot_{next_num}.png'
		# pyglet.image.get_buffer_manager().get_color_buffer().save(filename)
		from PIL import Image

		width, height = self.get_size()
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
		glReadBuffer(GL_BACK)  # read from back buffer
		buff = (ctypes.c_ubyte * (width * height * 4))()  # allocate buffer
		glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buff)
		image = Image.frombytes('RGBA', (width, height), bytes(buff))
		image = image.transpose(Image.FLIP_TOP_BOTTOM)
		image.save(filename)

		print(f'Screenshot saved as {filename}.')

	def update_camera(self):
		radius = self.camera_radius

		# First rotate around X axis (tilt up/down)
		cos_x = np.cos(self.camera_x_rotation)
		sin_x = np.sin(self.camera_x_rotation)
		Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])

		# Then rotate around Y axis (orbit left/right)
		cos_z = np.cos(self.camera_y_rotation)
		sin_z = np.sin(self.camera_y_rotation)
		Ry = np.array([[cos_z, 0, sin_z], [0, 1, 0], [-sin_z, 0, cos_z]])

		# Apply rotations: first X, then Y
		self.camera_position = Ry @ (Rx @ self.initial_camera_position)

	def create_menu(self) -> None:
		imgui.begin('Menu', flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
		io = imgui.get_io()
		io.font_global_scale = 1.5  # 150% of normal size

		# Select shape
		shape_names = [os.path.basename(path) for path in self.available_shapes]
		changed, self.current_shape_index = imgui.combo(
			'##shape_combo',  # ## prefix hides label
			self.current_shape_index,
			shape_names,
		)
		if changed:
			self.shape_filepath = self.available_shapes[self.current_shape_index]
			self.load_shape()

		# Select model used to fit primitives
		changed, self.model_idx = imgui.combo(
			'##model_combo',
			self.model_idx,
			MODEL_NAMES,
		)
		if changed:
			self.load_model()

		# Display current shape name
		imgui.text(f'Current: {shape_names[self.current_shape_index]}')

		imgui.separator()

		if self.shape is not None:
			changed, self.n_points = imgui.input_int('# points to sample', self.n_points)
			if changed:
				self.load_shape()

		if imgui.tree_node('Export', imgui.TREE_NODE_DEFAULT_OPEN):
			if imgui.button('Screenshot'):
				self.save_screenshot()

			if imgui.button('Export points'):
				self.pointcloud.export(filepath='points.ply', binary=True)

			if imgui.button('Export tori'):
				export_tori(self.centers, self.axes, self.major_radii, self.minor_radii, save_filepath='tori.ply')

			if imgui.button('Export cut plane'):
				self.export_cutplane(resolution=self.grid_resolution)

			if self.shape is not None and imgui.button('Export mesh'):
				self.shape.export_PLY(filepath='mesh.ply', binary=True)

			if imgui.button('Export isocontour'):
				if self.isocontour == None:
					self.mesh_isocontour()

				filename = f'isocontour_{self.isovalue:.3f}.ply'
				self.isocontour.export_PLY(filename, binary=True)
				print(f'Exported isosurface to {filename}')

			changed, self.grid_resolution = imgui.input_int('Grid resolution', self.grid_resolution)

			imgui.tree_pop()

		imgui.separator()

		if imgui.tree_node('Quantity to visualize', imgui.TREE_NODE_DEFAULT_OPEN):
			# Quick-select which quantity to visualize
			quick_picks = [
				SHADER_QUANTITIES[ShaderQuantity.PAT.value],
				SHADER_QUANTITIES[ShaderQuantity.SSPD.value],
				SHADER_QUANTITIES[ShaderQuantity.SHC.value],
				SHADER_QUANTITIES[ShaderQuantity.GWN.value],
				SHADER_QUANTITIES[ShaderQuantity.LogSumExp.value],
			]
			if self.shape != None:
				quick_picks.extend([SHADER_QUANTITIES[ShaderQuantity.true.value]])
			prev_quantity = self.quantity
			for quantity in quick_picks:
				if imgui.radio_button(quantity, self.quantity == quantity):
					self.quantity = quantity
					self.quantity_idx = SHADER_QUANTITIES.index(self.quantity)

			# Full, extended list of quantities to display
			changed, self.quantity_idx = imgui.combo('##quantity_combo', self.quantity_idx, SHADER_QUANTITIES)
			if changed:
				self.quantity = SHADER_QUANTITIES[self.quantity_idx]

			imgui.tree_pop()

		imgui.separator()

		# Algorithm stuff
		if imgui.tree_node('Optional parameters', imgui.TREE_NODE_DEFAULT_OPEN):
			changed, self.lambda_scale = imgui.slider_float(
				'lambda_scale (default is 1)', self.lambda_scale, 0, 100, '%.3f'
			)
			changed, self.neighborhood_size = imgui.slider_int(
				'Neighborhood size (-1 => no acceleration)',
				self.neighborhood_size,
				-1,
				self.max_neighborhood_size,
				'%.3f',
			)
			if changed:
				self.TDF.set_k_evaluation(self.neighborhood_size)

			imgui.tree_pop()

		imgui.separator()

		# Visualization stuff
		if imgui.tree_node('Visualization', imgui.TREE_NODE_DEFAULT_OPEN):
			changed, self.vis_cutplane = imgui.checkbox('Display cut plane', self.vis_cutplane)
			changed, self.colormap_range = imgui.slider_float('Colormap range', self.colormap_range, 0, 10.0, '%0.2f')
			if self.shape != None:
				changed, self.vis_shape = imgui.checkbox('Display input mesh', self.vis_shape)
			changed, self.vis_points = imgui.checkbox('Display points', self.vis_points)
			changed, self.vis_normals = imgui.checkbox('Display normals', self.vis_normals)
			if self.quantity_idx == ShaderQuantity.PAT.value:
				changed, self.vis_tori = imgui.checkbox('Display tori', self.vis_tori)
			changed, self.vis_contour = imgui.checkbox('Display contour', self.vis_contour)

			changed, self.isovalue = imgui.input_float('Isovalue', self.isovalue)
			if self.vis_points:
				changed, self.point_radius = imgui.slider_float('Point radius', self.point_radius, 0.0, 0.1, '%.4f')
			changed, self.floor_height = imgui.slider_float('Floor height', self.floor_height, -20.0, 0.0, '%.4f')
			changed, self.light_rotation = imgui.slider_float(
				'Lighting angle', self.light_rotation, 0, 2 * np.pi, '%.2f'
			)
			if changed:
				self.light_position = [
					self.light_radius * np.cos(self.light_rotation),
					self.light_position[1],
					self.light_radius * np.sin(self.light_rotation),
				]

			# Rotate camera
			changed, self.camera_y_rotation = imgui.slider_float(
				'Camera rotation (z)', self.camera_y_rotation, 0, 2 * np.pi, '%.2f'
			)
			if changed:
				self.update_camera()
			changed, self.camera_x_rotation = imgui.slider_float(
				'Camera rotation (x)', self.camera_x_rotation, 0, 2 * np.pi, '%.2f'
			)
			if changed:
				self.update_camera()

			# Rotate shape
			changed, self.cut_plane_rotation_y = imgui.slider_float(
				'Cut plane rotation (first)', self.cut_plane_rotation_y, 0, 2 * np.pi, '%.2f'
			)
			changed, self.cut_plane_rotation_x = imgui.slider_float(
				'Cut plane rotation (second)', self.cut_plane_rotation_x, 0, 2 * np.pi, '%.2f'
			)
			changed, self.isoband_frequency = imgui.slider_float(
				'Isoband freq.', self.isoband_frequency, 0.0, 360.0, '%.2f'
			)
			imgui.tree_pop()

		imgui.separator()

		if imgui.tree_node('Output values', imgui.TREE_NODE_DEFAULT_OPEN):
			imgui.text(f'Screen: ({self.mouse_x}, {self.mouse_y})')
			if self.shape != None:
				imgui.text(f'# vertices: {self.shape.vertices.shape[0]}\t#faces: {self.shape.faces.shape[0]}')
			imgui.text(f'# points: {self.pointcloud.size}')

			# Display values where mouse is hovering
			imgui.text(
				f'value: {self.mouse_sdf_approx:.4f}\tSDF_true - value: {self.mouse_sdf_true - self.mouse_sdf_approx:.4f}'
			)
			imgui.text(
				f'grad(value): ({self.mouse_grad_approx[0]:.4f}, {self.mouse_grad_approx[1]:.4f}, {self.mouse_grad_approx[2]:.4f})'
			)
			imgui.text(
				f'|grad(value)|: {np.linalg.norm(self.mouse_grad_approx):.4f}\t1-|grad(value)|: {1.0 - np.linalg.norm(self.mouse_grad_approx):.4f}'
			)

			imgui.tree_pop()

		imgui.end()

	def update(self, dt) -> None:
		self.time += dt
		self.frame += 1

	def get_common_uniforms(self):
		"""
		Return the common uniforms dict used by all shaders.
		"""
		width, height = self.get_size()
		return {
			'u_resolution': (float(width), float(height)),
			'screenWidth': float(width),
			'cameraShift': (self.camera_shift[0], self.camera_shift[1]),
			'cameraPosition': (self.camera_position[0], self.camera_position[1], self.camera_position[2]),
			'zoom': self.zoom,
			'cutplaneDepth': self.cut_plane_depth,
			'cutplaneAxis': self.cut_plane_axis,
			'freezeCutplane': self.freeze_cut_plane,
			'lambdaScale': self.lambda_scale,
			'maxExpArg': self.max_exp_arg,
			'usePointAreas': self.use_point_areas,
			'neighborhood_size': self.neighborhood_size,
			'minDist': -self.colormap_range,
			'maxDist': self.colormap_range,
			'mouseCoord': (self.mouse_x, self.mouse_y),
			'cutplaneDiag': (self.cut_plane_diag[0], self.cut_plane_diag[1], self.cut_plane_diag[2]),
			'cutplaneNormal': (self.cut_plane_normal[0], self.cut_plane_normal[1], self.cut_plane_normal[2]),
			'cutplaneRotationX': self.cut_plane_rotation_x,
			'cutplaneRotationY': self.cut_plane_rotation_y,
			'isovalue': self.isovalue,
			'shape': 0,
			'pointcloud': 1,
			'learnedtori': 2,
			'n_points': self.pointcloud_size,
			'texture_size': self.texture_size,
			'pc_bvh_nodes': 4,
			'pc_bvh_prim_indices': 5,
			'pointcolors': 6,
			'isocontour': 9,
			'isomesh_bvh_nodes': 10,
			'isomesh_bvh_prim_indices': 11,
			'n_isomesh_bvh_nodes': self.isomesh_bvh.get_num_nodes()
			if hasattr(self, 'isomesh_bvh') and self.isocontour
			else 0,
			'n_pc_bvh_nodes': self.pc_bvh.get_num_nodes() if hasattr(self, 'pc_bvh') else 0,
			'mesh_bvh_nodes': 7,
			'mesh_bvh_prim_indices': 8,
			'n_mesh_bvh_nodes': self.mesh_bvh.get_num_nodes() if hasattr(self, 'mesh_bvh') and self.shape else 0,
			'quantity': self.quantity_idx,
		}

	# Attach event handlers to the window.
	# Since these methods have the same name as default events, they will replace the default events.
	def on_draw(self):
		self.clear()

		self.bind_textures()

		width, height = self.get_size()
		io = imgui.get_io()
		io.display_size = width, height

		common_uniforms = self.get_common_uniforms()

		if self.shape != None:
			common_uniforms |= {
				'n_faces': self.shape_size,
			}

		# Second-to-last pass (fragment_interactive)
		self.render_passes[-2].render(common_uniforms)
		# Read back SDF value (that the mouse is hovering over)
		glBindFramebuffer(GL_FRAMEBUFFER, self.render_passes[-2].framebuffer)
		glReadBuffer(GL_COLOR_ATTACHMENT0)
		result = (ctypes.c_float * 4)()
		glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, result)
		output0 = self.render_passes[-2].read_output(0)
		output1 = self.render_passes[-2].read_output(1)
		output2 = self.render_passes[-2].read_output(2)

		self.mouse_sdf_true = output0[0]
		self.mouse_grad_true = np.array([output0[1], output0[2], output0[3]])

		self.mouse_sdf_approx = output1[0]
		self.mouse_grad_approx = np.array([output1[1], output1[2], output1[3]])

		self.mouse_world_pos = np.array([output2[0], output2[1], output2[2]])

		# Main (final) rendering pass.
		uniforms = common_uniforms | {
			'pointRadius': self.point_radius,
			'isobandFrequency': self.isoband_frequency,
			'vis_shape': self.vis_shape,
			'vis_cutplane': self.vis_cutplane,
			'vis_points': self.vis_points,
			'vis_tori': self.vis_tori,
			'vis_contour': self.vis_contour,
			'vis_normals': self.vis_normals,
			'floorHeight': self.floor_height,
			'lightPosition': (self.light_position[0], self.light_position[1], self.light_position[2]),
			'specularLightPosition': (
				self.specular_light_position[0],
				self.specular_light_position[1],
				self.specular_light_position[2],
			),
			'cubemap': 3,
		}
		self.render_passes[-1].render(uniforms)  # This can take a long time on the first frame!

		# Final pass renders to screen; copy from framebuffer to screen
		if self.render_passes:
			final_pass = self.render_passes[-1]
			glBindFramebuffer(GL_READ_FRAMEBUFFER, final_pass.framebuffer)
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
			glBlitFramebuffer(
				0, 0, final_pass.width, final_pass.height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR
			)

		# Draw imgui menu.
		imgui.new_frame()
		self.create_menu()
		imgui.render()
		self.renderer.render(imgui.get_draw_data())

	def cleanup_textures(self):
		texture_names = [
			'pc_texture',
			'learned_tori_texture',
			'pc_bvh_nodes_texture',
			'pc_bvh_indices_texture',
			'mesh_bvh_nodes_texture',
			'mesh_bvh_indices_texture',
		]
		for name in texture_names:
			if hasattr(self, name):
				texture = getattr(self, name)
				glDeleteTextures(1, ctypes.byref(texture))

	def on_close(self):
		# self.cleanup_textures()

		for render_pass in self.render_passes:
			render_pass.cleanup()

		# Clean up ImGui
		self.renderer.shutdown()

		super().on_close()

	def on_resize(self, width, height):
		super().on_resize(width, height)

	"""
	GUI interactions
	* mouse scroll: zoom in/out
	* mouse click-and-drag: pan
	* left-right mouse motion: move cutplane along axis
	* tab: switch axis of cutplane
	* spacebar: freeze the cutplane at its current location
	"""

	def on_key_press(self, symbol, modifiers):
		# self.keys_held.add(symbol)
		if symbol == pyglet.window.key.SPACE:
			self.freeze_cut_plane = not self.freeze_cut_plane
		elif symbol == pyglet.window.key.TAB:
			self.cut_plane_axis = (self.cut_plane_axis + 1) % 3
			self.cut_plane_normal = [0.0, 0.0, 0.0]
			self.cut_plane_normal[self.cut_plane_axis] = 1.0

	def on_mouse_press(self, x, y, button, modifiers):
		"""
		The x and y parameters give the position of the mouse relative to the window's lower-left corner.
		"""
		io = imgui.get_io()
		self.renderer.on_mouse_press(x, y, button, modifiers)
		# Only handle camera controls if ImGui didn't consume the event
		if not io.want_capture_mouse:
			if button == pyglet.window.mouse.LEFT:
				self.mouse_dragging = True
				self.last_mouse_x = x
				self.last_mouse_y = y

	def on_mouse_release(self, x, y, button, modifiers):
		# Let ImGui handle the event first
		self.renderer.on_mouse_release(x, y, button, modifiers)

		if button == pyglet.window.mouse.LEFT:
			self.mouse_dragging = False

	def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
		# Let ImGui handle the event first
		io = imgui.get_io()
		self.renderer.on_mouse_drag(x, y, dx, dy, buttons, modifiers)

		if not io.want_capture_mouse and self.mouse_dragging:
			self.camera_shift[0] += dx * self.pan_sensitivity / self.zoom
			self.camera_shift[1] += dy * self.pan_sensitivity / self.zoom

	def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
		"""
		Let scroll control the camera's zoom.
		"""
		# Let ImGui handle the event first
		io = imgui.get_io()
		self.renderer.on_mouse_scroll(x, y, scroll_x, scroll_y)

		# Only handle camera controls if ImGui didn't consume the event
		if not io.want_capture_mouse:
			zoom_factor = 1.0 + (-scroll_y * self.zoom_sensitivity)
			new_zoom = self.zoom * zoom_factor
			self.zoom = max(0.1, min(100.0, new_zoom))

	def on_mouse_motion(self, x, y, dx, dy):
		"""
		Track mouse position for SDF value display
		"""
		# Let ImGui handle the event first
		io = imgui.get_io()
		self.renderer.on_mouse_motion(x, y, dx, dy)

		if not io.want_capture_mouse:
			self.mouse_x = x
			self.mouse_y = y
			if not self.freeze_cut_plane:
				self.cut_plane_depth = self.mouse_x


def main():
	# parser = argparse.ArgumentParser()
	# parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
	# args = parser.parse_args()

	window = ShaderWindow()

	# Enter pyglet's default event loop.
	# This method will return only when all application windows have been closed.
	pyglet.app.run()


if __name__ == '__main__':
	main()
