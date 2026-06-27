import os
import gc
import subprocess
import glob
import numpy as np
import scipy as sp
import time
import pickle
import random

import jax
import jax.numpy as jnp
from jax import jit
from flax import nnx
from functools import partial
import optax

import matplotlib.pyplot as plt

from shape_3d import *
from mesh_sampling_jax import *

jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_debug_nans', False)

NAN_EPSILON = 1e-10

MODELS_DIR = 'models/'

# ========================================================================
# HELPERS
# ========================================================================


@jit
def orthonormal_basis_with_rotation(n: jnp.ndarray, rotation: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
	"""
	Compute an orthonormal basis (s, t) perpendicular to normal vector n, with a specified rotation of the axes.

	Args:
		n: (..., 3) array of normal vectors (will be normalized)
		rotation: scalar rotation angle about the normal

	Returns:
		s_hat: (..., 3) first basis vector
		t_hat: (..., 3) second basis vector
	"""
	# Normalize the normal vector
	n_norm = jnp.linalg.norm(n, axis=-1, keepdims=True)
	n_hat = n / jnp.maximum(n_norm, NAN_EPSILON)

	# Choose initial vector u based on which component of n is smallest
	# If |n.x| > 0.9, use (0,1,0), otherwise use (1,0,0)
	abs_n_x = jnp.abs(n_hat[..., 0:1])
	u = jnp.where(
		abs_n_x > 0.9,
		jnp.concatenate([jnp.zeros_like(abs_n_x), jnp.ones_like(abs_n_x), jnp.zeros_like(abs_n_x)], axis=-1),
		jnp.concatenate([jnp.ones_like(abs_n_x), jnp.zeros_like(abs_n_x), jnp.zeros_like(abs_n_x)], axis=-1),
	)

	# Apply rotation about n_hat using Rodrigues' formula
	# R = I + sin(θ) * K + (1 - cos(θ)) * K^2
	# where K is the skew-symmetric matrix for n_hat
	cos_rot = jnp.cos(rotation)
	sin_rot = jnp.sin(rotation)

	# Compute u_rotated = cos(θ)*u + sin(θ)*(n × u) + (1-cos(θ))*(n·u)*n
	n_cross_u = jnp.cross(n_hat, u)
	n_dot_u = jnp.sum(n_hat * u, axis=-1, keepdims=True)
	u_rotated = cos_rot * u + sin_rot * n_cross_u + (1.0 - cos_rot) * n_dot_u * n_hat

	# Compute s = n × u (perpendicular to both)
	s = jnp.cross(n_hat, u_rotated)
	s_norm = jnp.linalg.norm(s, axis=-1, keepdims=True)
	s_hat = s / jnp.maximum(s_norm, NAN_EPSILON)

	# Compute t = n × s (completes orthonormal basis)
	t_hat = jnp.cross(n_hat, s_hat)

	return s_hat, t_hat


# ========================================================================
# (AUTO-DIFFERENTIABLE) DISTANCE FUNCTIONS
# ========================================================================


@jit
def cartesian_to_rotation_jax(u: jnp.ndarray) -> jnp.ndarray:
	"""
	Convert 3D unit vector to spherical coordinates (theta, phi).

	Args:
		u: (..., 3) array of 3D vectors

	Returns:
		axes: (..., 2) array of (theta, phi) in radians
			theta: azimuthal angle in [0, 2π]
			phi: polar angle in [0, π]
	"""
	# Normalize input
	u_norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
	u_hat = u / jnp.maximum(u_norm, NAN_EPSILON)

	x = u_hat[..., 0]
	y = u_hat[..., 1]
	z = u_hat[..., 2]

	phi = jnp.arccos(jnp.clip(z, -1.0, 1.0))
	safe_x = jnp.sign(x) * jnp.maximum(jnp.abs(x), NAN_EPSILON)
	theta = jnp.arctan2(y, safe_x)
	return jnp.stack([theta, phi], axis=-1)


@jit
def rotation_to_cartesian_jax(axes: jnp.ndarray) -> jnp.ndarray:
	"""
	Convert spherical coordinates (theta, phi) to 3D unit vector.

	Args:
		axes: (..., 2) array of (theta, phi) in radians

	Returns:
		u: (..., 3) array of 3D unit vectors
	"""
	theta = axes[..., 0]
	phi = axes[..., 1]

	# Spherical to Cartesian conversion
	sin_phi = jnp.sin(phi)
	cos_phi = jnp.cos(phi)
	sin_theta = jnp.sin(theta)
	cos_theta = jnp.cos(theta)

	x = sin_phi * cos_theta
	y = sin_phi * sin_theta
	z = cos_phi

	return jnp.stack([x, y, z], axis=-1)


def _compute_shifts_from_distances(distances: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
	"""
	Compute per-query shift from a pre-computed distance matrix.

	Args:
		distances: (batch_size, n_queries, max_points) distances from each query to each point
		mask: (max_points,) valid-point mask

	Returns:
		shifts: (batch_size, n_queries)
	"""
	k = 64
	max_points = distances.shape[-1]
	k_actual = min(k, max_points)

	m = mask[None, None, :]  # (1, 1, max_points)

	# top_k on negated distances: invalid points → -inf so they're never selected.
	neg_distances = jnp.where(m, -distances, -jnp.inf)
	neg_knn, _ = jax.lax.top_k(neg_distances, k_actual)  # (batch, n_queries, k_actual)
	knn_dists = -neg_knn  # positive, ascending order (nearest first)

	# Shift = 0.5 * max distance among k nearest valid neighbors
	finite_knn = jnp.where(jnp.isfinite(knn_dists), knn_dists, 0.0)
	return 0.5 * jnp.max(finite_knn, axis=-1)  # (batch, n_queries)


@jit
def compute_lambdas_jax(shifts: jnp.ndarray) -> jnp.ndarray:
	"""
	Compute lambda parameter from shift.

	Args:
		shift: (...,) shift values

	Returns:
		lam: (...,) lambda values
	"""
	return 64.0 / shifts


@jit
def torus_precompute_masked(
	mask: jnp.ndarray,
	points: jnp.ndarray,
	normals: jnp.ndarray,
	coeffs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	"""
	Only compute torii for valid points (where mask is True).

	Args:
		mask: (max_points,) - boolean mask indicating valid points
		points: (max_points, 3) - may contain padding
		normals: (max_points, 3) - may contain padding
		coeffs: (max_points, 6) - may contain padding

	Returns:
		centers: (max_points, 3)
		axes: (max_points, 2)
		major_radii: (max_points,)
		minor_radii: (max_points,)
	"""

	max_points = points.shape[0]

	# Fit torus at a single point.
	@jit
	def solve_single(i):
		# Center point info
		p_i = points[i]
		n_i = normals[i]

		# Compute orthonormal bases
		s_hat, t_hat = orthonormal_basis_with_rotation(n_i, 0.0)

		# Extract polynomial coefficients
		a00 = coeffs[i, 0]
		a01 = coeffs[i, 1]
		a10 = coeffs[i, 2]
		a11 = coeffs[i, 3]
		a02 = coeffs[i, 4]
		a20 = coeffs[i, 5]

		# Compute principal curvatures from polynomial fit
		A = jnp.sqrt(1.0 + a01**2 + a10**2)
		# A = jnp.maximum(A, NAN_EPSILON)

		# Compute discriminant for eigenvector calculation
		v1 = a20 * a01**2 - a02 * a10**2 - a02 + a20
		v2 = a11 * (a10**2 + 1.0) - 2.0 * a01 * a10 * a20
		term = a02 * (a10**2 + 1.0) - a01 * a10 * a11 + a20 * (a01**2 + 1.0)
		discriminant = (1.0 + a01**2 + a10**2) * (a11**2 - 4.0 * a02 * a20) + term**2
		discriminant = jnp.sqrt(jnp.maximum(discriminant, NAN_EPSILON))  # sqrt not differentiable at 0

		ev_pos_0 = v1 + discriminant
		ev_pos_0 = jnp.sign(ev_pos_0) * jnp.maximum(jnp.abs(ev_pos_0), NAN_EPSILON)
		ev_pos = jnp.array([ev_pos_0, v2])

		ev_neg_1 = v2
		ev_neg_1 = jnp.sign(ev_neg_1) * jnp.maximum(jnp.abs(ev_neg_1), NAN_EPSILON)
		ev_neg = jnp.array([v1 - discriminant, ev_neg_1])

		# Even though normalization isn't necessary because we take a Rayleigh quotient, it helps numerically.
		ev_pos /= jnp.linalg.norm(ev_pos)
		ev_neg /= jnp.linalg.norm(ev_neg)
		ev_pos_x = ev_pos[0]
		ev_pos_y = ev_pos[1]
		ev_neg_x = ev_neg[0]
		ev_neg_y = ev_neg[1]

		# Convert to 3D eigenvectors
		dQstards = s_hat + a10 * n_i  # (3, )
		dQstardt = t_hat + a01 * n_i  # (3, )

		v_pos = ev_pos_x * dQstards + ev_pos_y * dQstardt
		v_neg = ev_neg_x * dQstards + ev_neg_y * dQstardt

		# Compute normal at fitted point
		normal = (n_i - a10 * s_hat - a01 * t_hat) / A

		# Compute curvatures using Rayleigh quotient
		I_11 = 1.0 + a10**2
		I_12 = a10 * a01
		I_22 = 1.0 + a01**2

		II_11 = 2.0 * a20 / A
		II_12 = a11 / A
		II_22 = 2.0 * a02 / A

		# Compute kappa = (ev^T @ II @ ev) / (ev^T @ I @ ev)
		# For positive eigenvector
		II_ev_pos_x = II_11 * ev_pos_x + II_12 * ev_pos_y
		II_ev_pos_y = II_12 * ev_pos_x + II_22 * ev_pos_y
		I_ev_pos_x = I_11 * ev_pos_x + I_12 * ev_pos_y
		I_ev_pos_y = I_12 * ev_pos_x + I_22 * ev_pos_y

		numer_pos = ev_pos_x * II_ev_pos_x + ev_pos_y * II_ev_pos_y
		denom_pos = ev_pos_x * I_ev_pos_x + ev_pos_y * I_ev_pos_y
		denom_pos = jnp.sign(denom_pos) * jnp.maximum(jnp.abs(denom_pos), NAN_EPSILON)
		kappa_pos = numer_pos / denom_pos
		# kappa_pos = jax.lax.cond(numer_pos == 0, lambda: NAN_EPSILON, lambda: numer_pos / denom_pos)

		# For negative eigenvector
		II_ev_neg_x = II_11 * ev_neg_x + II_12 * ev_neg_y
		II_ev_neg_y = II_12 * ev_neg_x + II_22 * ev_neg_y
		I_ev_neg_x = I_11 * ev_neg_x + I_12 * ev_neg_y
		I_ev_neg_y = I_12 * ev_neg_x + I_22 * ev_neg_y

		numer_neg = ev_neg_x * II_ev_neg_x + ev_neg_y * II_ev_neg_y
		denom_neg = ev_neg_x * I_ev_neg_x + ev_neg_y * I_ev_neg_y
		denom_neg = jnp.sign(denom_neg) * jnp.maximum(jnp.abs(denom_neg), NAN_EPSILON)
		kappa_neg = numer_neg / denom_neg
		# kappa_neg = jax.lax.cond(numer_neg == 0, lambda: NAN_EPSILON, lambda: numer_neg / denom_neg)

		# Identify min/max curvatures
		abs_kappa_pos = jnp.abs(kappa_pos)
		abs_kappa_neg = jnp.abs(kappa_neg)

		use_pos_as_min = abs_kappa_pos < abs_kappa_neg
		kappa_min = jnp.where(use_pos_as_min, kappa_pos, kappa_neg)
		kappa_max = jnp.where(use_pos_as_min, kappa_neg, kappa_pos)
		v_min = jnp.where(use_pos_as_min, v_pos, v_neg)

		abs_kappa_max = jnp.maximum(jnp.abs(kappa_max), NAN_EPSILON)
		abs_kappa_min = jnp.maximum(jnp.abs(kappa_min), NAN_EPSILON)

		# Compute torus parameters
		r = 1.0 / abs_kappa_max
		R = 1.0 / abs_kappa_min - jnp.sign(kappa_max * kappa_min) * r
		# s_R = jax.lax.cond(R >= 0, lambda: 1.0, lambda: -1.0)
		# major_radius = s_R * jnp.maximum(jnp.abs(R), NAN_EPSILON)

		# Sign convention from C++
		s = jnp.where(((kappa_max < 0) & (kappa_min < 0)) | ((kappa_max < 0) & (kappa_min > 0)), 1.0, -1.0)

		# Torus center
		center = p_i + a00 * n_i - (s / abs_kappa_min) * normal

		# Torus axis (u = normal × v_min, normalized)
		u = jnp.cross(normal, v_min)
		u_norm = jnp.linalg.norm(u)
		u = u / jnp.maximum(u_norm, NAN_EPSILON)

		# Convert to rotation representation
		axis = cartesian_to_rotation_jax(u)  # (2, )

		major_radius = R
		minor_radius = s * r

		return center, axis, major_radius, minor_radius

	# Compute torii for all indices (including invalid ones)
	centers, axes, major_radii, minor_radii = jax.vmap(lambda i: solve_single(i), in_axes=0)(jnp.arange(max_points))
	return centers, axes, major_radii, minor_radii


@jit
def torus_signed_distance(q, center, axis, major_radius, minor_radius):
	"""
	Compute the signed distance at q to a single (signed) torus.
	"""
	axis3d = rotation_to_cartesian_jax(axis)

	# Distance from query to torus center
	r_vec = q - center

	# v = |r_vec × axis|
	cross_prod = jnp.cross(r_vec, axis3d)
	v = jnp.linalg.norm(cross_prod, axis=-1)

	# Distance components
	d1 = v - major_radius
	d2 = jnp.sum(r_vec * axis3d, axis=-1)

	# Combined distance to torus surface
	d = jnp.sqrt(d1**2 + d2**2)

	s = jnp.sign(minor_radius)  # (1, 1, max_points)
	g_z = s * (d - jnp.abs(minor_radius))  # (batch, n_queries, max_points)

	return g_z


@jit
def torus_distance_gradient_masked(
	mask: jnp.ndarray,
	queries: jnp.ndarray,
	points: jnp.ndarray,
	centers: jnp.ndarray,
	axes: jnp.ndarray,
	major_radii: jnp.ndarray,
	minor_radii: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	"""
	Only sum over torii at valid points (where mask is True).

	Args:
		mask: (max_points,) - boolean mask indicating valid points
		queries: (batch_size, n_queries, 3) array of query points
		points: (max_points, 3) - may contain padding
		centers: (max_points, 3) - torii centers
		axes: (max_points, 2) - torii axes
		major_radii: (max_points,)
		minor_radii: (max_points,)

	Returns:
		distances: (batch_size, n_queries,) array of signed distances
		gradients: (batch_size, n_queries, 3) array of signed distance gradients
	"""
	batch_size, n_queries, _ = queries.shape

	# Create validity mask for torii without NaNs --- in case there were still NaNs that escaped torus_precompute_masked()
	valid_torus_mask = (
		mask
		& jnp.isfinite(centers).all(axis=-1)
		& jnp.isfinite(axes).all(axis=-1)
		& jnp.isfinite(major_radii)
		& jnp.isfinite(minor_radii)
	)  # (max_points,)
	# valid_torus_mask = mask  # (max_points, )

	axis3d = rotation_to_cartesian_jax(axes)  # (max_points, 3)

	# Compute distance and gradient for each query-torus pair

	# Reshape for broadcasting: queries (batch, n_queries, 1, 3)
	q = queries[:, :, None, :]  # (batch, n_queries, 1, 3)

	# Broadcast points, centers, etc. for all queries
	# (1, 1, max_points, 3)
	p_i = points[None, None, :, :]

	# Compute query-to-point distances once and reuse for both shift and weight computation.
	# This avoids materializing the (batch, n_queries, max_points) distance tensor twice.
	rVec = q - p_i  # (batch, n_queries, max_points, 3)
	dist_to_point = jnp.linalg.norm(rVec, axis=-1)  # (batch, n_queries, max_points)

	shifts = _compute_shifts_from_distances(dist_to_point, valid_torus_mask)  # (batch, n_queries)
	lam = compute_lambdas_jax(shifts)  # (batch_size, n_queries)
	c_i = centers[None, None, :, :]
	axis_i = axis3d[None, None, :, :]
	R_i = major_radii[None, None, :]
	r_i = minor_radii[None, None, :]
	mask_i = valid_torus_mask[None, None, :]  # (1, 1, max_points)

	# Distance from query to torus center
	r_vec = q - c_i  # (batch, n_queries, max_points, 3)

	# v = |r_vec × axis|
	cross_prod = jnp.cross(r_vec, axis_i)  # (batch, n_queries, max_points, 3)
	v = jnp.linalg.norm(cross_prod, axis=-1)  # (batch, n_queries, max_points)

	# Distance components
	d1 = v - R_i  # (batch, n_queries, max_points)
	d2 = jnp.sum(r_vec * axis_i, axis=-1)  # (batch, n_queries, max_points)

	# Combined distance to torus surface
	d = jnp.sqrt(d1**2 + d2**2)  # (batch, n_queries, max_points)

	s = jnp.sign(r_i)  # (1, 1, max_points)
	g_z = s * (d - jnp.abs(r_i))  # (batch, n_queries, max_points)
	g_z = jnp.where(mask_i, g_z, 0.0)  # apply mask

	# Gradient computation (matrix G from C++ code)
	# G is a 2x3 matrix, w is a 2-vector, grad = s * (w^T @ G)^T
	v_safe = jnp.maximum(v, NAN_EPSILON)

	G_11 = (
		axis_i[:, :, :, 0] * axis_i[:, :, :, 1] * (c_i[:, :, :, 1] - q[:, :, :, 1])
		+ axis_i[:, :, :, 0] * axis_i[:, :, :, 2] * (c_i[:, :, :, 2] - q[:, :, :, 2])
		- (axis_i[:, :, :, 1] ** 2 + axis_i[:, :, :, 2] ** 2) * (c_i[:, :, :, 0] - q[:, :, :, 0])
	) / v_safe

	G_12 = (
		axis_i[:, :, :, 0] * axis_i[:, :, :, 1] * (c_i[:, :, :, 0] - q[:, :, :, 0])
		+ axis_i[:, :, :, 1] * axis_i[:, :, :, 2] * (c_i[:, :, :, 2] - q[:, :, :, 2])
		- (axis_i[:, :, :, 0] ** 2 + axis_i[:, :, :, 2] ** 2) * (c_i[:, :, :, 1] - q[:, :, :, 1])
	) / v_safe

	G_13 = (
		axis_i[:, :, :, 0] * axis_i[:, :, :, 2] * (c_i[:, :, :, 0] - q[:, :, :, 0])
		+ axis_i[:, :, :, 1] * axis_i[:, :, :, 2] * (c_i[:, :, :, 1] - q[:, :, :, 1])
		- (axis_i[:, :, :, 0] ** 2 + axis_i[:, :, :, 1] ** 2) * (c_i[:, :, :, 2] - q[:, :, :, 2])
	) / v_safe

	G_21 = axis_i[:, :, :, 0]
	G_22 = axis_i[:, :, :, 1]
	G_23 = axis_i[:, :, :, 2]

	# Weight vector w = (d1, d2) / ||(d1, d2)||
	d_safe = jnp.maximum(d, NAN_EPSILON)
	w_x = d1 / d_safe
	w_y = d2 / d_safe

	# Gradient: grad = s * (w^T @ G)^T = s * G^T @ w
	grad_x = s * (w_x * G_11 + w_y * G_21)
	grad_y = s * (w_x * G_12 + w_y * G_22)
	grad_z = s * (w_x * G_13 + w_y * G_23)

	# Apply mask to gradients
	grad_x = jnp.where(mask_i, grad_x, 0.0)
	grad_y = jnp.where(mask_i, grad_y, 0.0)
	grad_z = jnp.where(mask_i, grad_z, 0.0)

	grad_g = jnp.stack([grad_x, grad_y, grad_z], axis=-1)  # (batch, n_queries, max_points, 3)

	# Compute exponential weights
	rHat = rVec / jnp.maximum(dist_to_point, NAN_EPSILON)[:, :, :, None]

	# Exponential weight
	lam_expanded = lam[:, :, None]  # (batch, n_queries, 1)
	shift_expanded = shifts[:, :, None]  # (batch, n_queries, 1)
	weight = jnp.exp(-lam_expanded * (dist_to_point - shift_expanded))  # (batch, n_queries, max_points)

	# Apply mask to weights
	weight = jnp.where(mask_i, weight, 0.0)

	# Accumulate: grad_g_minus_lambda_n_g, g, n
	grad_g_minus_lambda_n_g = jnp.sum(
		weight[:, :, :, None] * (grad_g - lam_expanded[:, :, :, None] * rHat * g_z[:, :, :, None]), axis=2
	)  # (batch, n_queries, 3)

	g = jnp.sum(weight * g_z, axis=2)  # (batch, n_queries)
	n = jnp.sum(weight[:, :, :, None] * rHat, axis=2)  # (batch, n_queries, 3)

	normalization = jnp.sum(weight, axis=2, keepdims=True)  # (batch, n_queries, 1)
	normalization_safe = jnp.maximum(normalization, NAN_EPSILON)

	# Compute final distance and gradient

	A = grad_g_minus_lambda_n_g / normalization_safe  # (batch, n_queries, 3)
	B = lam_expanded * n / normalization_safe  # (batch, n_queries, 3)
	dist = g / normalization_safe[:, :, 0]
	grad = A + dist[:, :, None] * B  # (batch, n_queries, 3)

	return dist, grad


# ========================================================================
# EXPORT FOR VISUALIZATION
# ========================================================================


def generate_torus_mesh(center: np.ndarray, axis: np.ndarray, major_radius: float, minor_radius: float):
	"""
	Generate quad mesh for a single torus.

	Args:
		center: (3,) center position
		axis: (2,) axis in spherical coordinates (theta, phi)
		major_radius: major radius R
		minor_radius: minor radius r (can be negative)

	Returns:
		vertices: (major_res * minor_res, 3)
		faces: (major_res * minor_res, 4) quad indices
	"""
	# Handle degenerate cases
	if np.abs(minor_radius) < 1e-10 or np.abs(major_radius) < 1e-10:
		# Degenerate torus - return empty mesh
		return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int32)

	# major_res: number of segments around major circle
	# minor_res: number of segments around minor circle
	max_spacing = 0.1
	min_res = 16  # for small point clouds (a few thousand points), use 16
	max_res = 128
	major_res = min(max(int(major_radius / max_spacing), min_res), max_res)
	minor_res = min(max(int(minor_radius / max_spacing), min_res), max_res)

	# Get torus axis as 3D vector
	axis_3d = rotation_to_cartesian_jax(axis)

	# Create two orthogonal vectors perpendicular to axis
	v1, v2 = orthonormal_basis_with_rotation(axis_3d, 0.0)

	# Generate vertices
	vertices = []

	for i in range(major_res):
		# Angle around major circle
		theta = 2 * np.pi * i / major_res

		# Point on major circle
		major_circle_point = center + major_radius * (np.cos(theta) * v1 + np.sin(theta) * v2)

		# Normal direction at this point on major circle (pointing away from center)
		major_normal = np.cos(theta) * v1 + np.sin(theta) * v2

		# Create local frame for minor circle
		# tangent to major circle
		major_tangent = -np.sin(theta) * v1 + np.cos(theta) * v2

		for j in range(minor_res):
			# Angle around minor circle
			phi = 2 * np.pi * j / minor_res

			# Offset from major circle
			# Note: minor_radius can be negative (changes which side of major circle)
			offset = minor_radius * (np.cos(phi) * major_normal + np.sin(phi) * axis_3d)

			vertex = major_circle_point + offset
			vertices.append(vertex)

	vertices = np.array(vertices)

	# Generate quad faces
	faces = []

	for i in range(major_res):
		for j in range(minor_res):
			v0 = i * minor_res + j
			v1 = ((i + 1) % major_res) * minor_res + j
			v2 = ((i + 1) % major_res) * minor_res + (j + 1) % minor_res
			v3 = i * minor_res + (j + 1) % minor_res

			faces.append([v1, v2, v3, v0])

	faces = np.array(faces, dtype=np.int32)

	return vertices, faces


def export_torus(center: np.ndarray, axis: np.ndarray, major_radius: float, minor_radius: float, save_filepath: str):
	"""
	Export a single torus.

	Args:
		center: (3,)
		axis: (2,)
		major_radius: scalar
		minor_radius: scalar
	"""
	vertices, faces = generate_torus_mesh(center, axis, major_radius, minor_radius)
	# export_OBJ(vertices, faces, filepath=save_filepath)
	export_PLY(vertices, faces, filepath=save_filepath, binary=True)


def hsv_to_rgb(h, s, v):
	"""
	Convert HSV to RGB (matching the GLSL hsv2rgb function).

	Args:
		h, s, v: Hue [0,1], Saturation [0,1], Value [0,1]

	Returns:
		(r, g, b) tuple in [0, 1]
	"""
	c = v * s
	x = c * (1 - abs((h * 6) % 2 - 1))
	m = v - c

	if h < 1 / 6:
		r, g, b = c, x, 0
	elif h < 2 / 6:
		r, g, b = x, c, 0
	elif h < 3 / 6:
		r, g, b = 0, c, x
	elif h < 4 / 6:
		r, g, b = 0, x, c
	elif h < 5 / 6:
		r, g, b = x, 0, c
	else:
		r, g, b = c, 0, x

	return (r + m, g + m, b + m)


def torus_color(i):
	"""
	Compute the color of the i-th torus (matching shader logic).

	Args:
		i: Torus index

	Returns:
		(r, g, b) in [0, 255] as uint8
	"""
	incr = (1.0 + np.sqrt(5.0)) / 20.0
	h = (i * incr) % 1.0
	s = 0.5
	v = 1.0

	r, g, b = hsv_to_rgb(h, s, v)
	return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def export_tori(centers: np.ndarray, axes: np.ndarray, major_radii: float, minor_radii: float, save_filepath: str):
	"""
	Export a collection of torii in a single file.

	Args:
		center: (n_torii, 3)
		axis: (n_torii, 2)
		major_radius: (n_torii, )
		minor_radius: (n_torii, )
	"""
	n_torii = len(centers)

	all_vertices = []
	all_faces = []
	all_colors = []
	vertex_offset = 0

	n_valid = 0

	for i in range(n_torii):
		# Check if torus is valid
		is_valid = (
			np.isfinite(centers[i]).all()
			and np.isfinite(axes[i]).all()
			and np.isfinite(major_radii[i])
			and np.isfinite(minor_radii[i])
			and np.abs(minor_radii[i]) > 1e-10
			and np.abs(major_radii[i]) > 1e-10
		)

		if not is_valid:
			continue

		# Generate mesh for this torus
		print(f'Generating torus mesh {i}...')
		vertices, faces = generate_torus_mesh(centers[i], axes[i], major_radii[i], minor_radii[i])

		if len(vertices) == 0:
			continue

		# Compute color for this torus (same color for all vertices of this torus)
		color = torus_color(i)
		vertex_colors = np.tile(color, (len(vertices), 1))

		# Add to combined mesh with offset
		all_vertices.append(vertices)
		all_faces.append(faces + vertex_offset)
		all_colors.append(vertex_colors)

		vertex_offset += len(vertices)
		n_valid += 1

	# Combine all meshes
	print('Combining all torus mehses...')
	if len(all_vertices) > 0:
		all_vertices = np.vstack(all_vertices)
		all_faces = np.vstack(all_faces)
		all_colors = np.vstack(all_colors)
		export_PLY_with_colors(all_vertices, all_faces, all_colors, save_filepath, binary=True)
		print(f'Exported {n_valid}/{n_torii} torii to {save_filepath}')
	else:
		print(f'Warning: No valid torii found, empty mesh exported to {save_filepath}')
		with open(save_filepath, 'w') as f:
			f.write('# Empty mesh (no valid torii)\n')


def _export_tori_single(
	centers: np.ndarray,
	axes: np.ndarray,
	major_radii: np.ndarray,
	minor_radii: np.ndarray,
	save_filepath: str,
	global_index_offset: int = 0,
	verbose: bool = True,
) -> int:
	"""
	Export a collection of torii to a single file.

	Args:
		centers: (n_torii, 3)
		axes: (n_torii, 2)
		major_radii: (n_torii,)
		minor_radii: (n_torii,)
		save_filepath: output filepath
		global_index_offset: offset for global indexing (for consistent colors across chunks)
		verbose: print progress

	Returns:
		Number of valid torii exported
	"""
	n_torii = len(centers)

	all_vertices = []
	all_faces = []
	all_colors = []
	vertex_offset = 0
	n_valid = 0

	for i in range(n_torii):
		# Check if torus is valid
		is_valid = (
			np.isfinite(centers[i]).all()
			and np.isfinite(axes[i]).all()
			and np.isfinite(major_radii[i])
			and np.isfinite(minor_radii[i])
			and np.abs(minor_radii[i]) > 1e-10
			and np.abs(major_radii[i]) > 1e-10
		)

		if not is_valid:
			continue

		# Generate mesh for this torus
		vertices, faces = generate_torus_mesh(centers[i], axes[i], major_radii[i], minor_radii[i])

		if len(vertices) == 0:
			continue

		# Compute color using global index for consistency across chunks
		global_index = global_index_offset + i
		color = torus_color(global_index)
		vertex_colors = np.tile(color, (len(vertices), 1))

		# Add to combined mesh with offset
		all_vertices.append(vertices)
		all_faces.append(faces + vertex_offset)
		all_colors.append(vertex_colors)

		vertex_offset += len(vertices)
		n_valid += 1

	# Combine and export
	if len(all_vertices) > 0:
		all_vertices = np.vstack(all_vertices)
		all_faces = np.vstack(all_faces)
		all_colors = np.vstack(all_colors)
		export_PLY_with_colors(all_vertices, all_faces, all_colors, save_filepath, binary=True)
		if verbose:
			print(f'Exported {n_valid}/{n_torii} torii to {save_filepath}')
	else:
		if verbose:
			print(f'Warning: No valid torii found, skipping {save_filepath}')

	return n_valid


def export_tori_chunked(
	centers: np.ndarray,
	axes: np.ndarray,
	major_radii: np.ndarray,
	minor_radii: np.ndarray,
	save_filepath: str,
	chunk_size: int = 1_000_000,
	verbose: bool = True,
) -> List[str]:
	"""
	Export a collection of torii in chunks, with each chunk saved to a separate file.

	Args:
		centers: (n_torii, 3) torus centers
		axes: (n_torii, 2) torus axes in spherical coordinates
		major_radii: (n_torii,) torus major radii
		minor_radii: (n_torii,) torus minor radii
		save_filepath: base filepath for output (e.g., "output/torii.ply")
					   chunks will be saved as "output/torii_chunk0.ply", etc.
		chunk_size: number of torii per chunk (default 1M)
		verbose: print progress information

	Returns:
		List of output filepaths that were created
	"""
	n_torii = len(centers)
	print(f'# torii: {n_torii}')

	# Parse base filepath
	if save_filepath.endswith('.ply'):
		base_path = save_filepath[:-4]
	else:
		base_path = save_filepath

	# # If small enough, export in one file
	# if n_torii <= chunk_size:
	# 	output_files = []
	# 	filepath = f"{base_path}.ply"
	# 	n_valid = _export_tori_single(
	# 		centers, axes, major_radii, minor_radii,
	# 		filepath, global_index_offset=0, verbose=verbose
	# 	)
	# 	if n_valid > 0:
	# 		output_files.append(filepath)
	# 	return output_files

	# Process in chunks
	n_chunks = (n_torii + chunk_size - 1) // chunk_size

	if verbose:
		print(f'Exporting {n_torii:,} torii in {n_chunks} chunks of up to {chunk_size:,} each...')

	output_files = []
	total_valid = 0

	for chunk_idx in range(n_chunks):
		start = chunk_idx * chunk_size
		end = min(start + chunk_size, n_torii)

		if verbose:
			progress = (chunk_idx + 1) / n_chunks * 100
			print(f'  Chunk {chunk_idx}/{n_chunks - 1} (torii {start:,}-{end - 1:,}, {progress:.1f}%)...')

		# Extract chunk data
		centers_chunk = centers[start:end]
		axes_chunk = axes[start:end]
		major_chunk = major_radii[start:end]
		minor_chunk = minor_radii[start:end]

		# Export chunk with global index offset for consistent coloring
		filepath = f'{base_path}_chunk{chunk_idx}.ply'
		n_valid = _export_tori_single(
			centers_chunk, axes_chunk, major_chunk, minor_chunk, filepath, global_index_offset=start, verbose=False
		)

		if n_valid > 0:
			output_files.append(filepath)
			total_valid += n_valid
			if verbose:
				print(f'	Exported {n_valid:,} valid torii to {filepath}')
		else:
			if verbose:
				print(f'	No valid torii in chunk {chunk_idx}, skipping file creation')

		# Free memory
		del centers_chunk, axes_chunk, major_chunk, minor_chunk
		gc.collect()

	if verbose:
		print(f'  Done. Exported {total_valid:,}/{n_torii:,} valid torii to {len(output_files)} files.')

	return output_files


# ========================================================================
# NEURAL NETWORK ARCHITECTURE
# ========================================================================


class TransformerBlock(nnx.Module):
	"""
	Transformer block using Flax built-in modules.
	"""

	def __init__(self, embed_dim: int, n_heads: int, mlp_dim: int, dropout_rate: float = 0.0, *, rngs: nnx.Rngs):
		self.attention = nnx.MultiHeadAttention(
			num_heads=n_heads,
			in_features=embed_dim,
			qkv_features=embed_dim,
			out_features=embed_dim,
			decode=False,
			rngs=rngs,
		)

		# feed-forward network
		self.ffn = nnx.Sequential(
			nnx.Linear(embed_dim, mlp_dim, rngs=rngs),
			nnx.gelu,
			nnx.Dropout(rate=dropout_rate, rngs=rngs),
			nnx.Linear(mlp_dim, embed_dim, rngs=rngs),
			nnx.Dropout(rate=dropout_rate, rngs=rngs),
		)

		# layer normalization
		self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
		self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
		self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

	def __call__(self, x: jax.Array) -> jax.Array:
		# Self-attention block
		residual = x
		x = self.norm1(x)
		x = self.attention(x)
		x = self.dropout(x)
		x = residual + x

		# Feed-forward block
		residual = x
		x = self.norm2(x)
		x = self.ffn(x)
		x = residual + x

		return x


class InputFeatures(nnx.Module):
	"""
	Turn a neighborhood (point positions & their normals) to the input features to be fed into the neural network.
	"""

	def __init__(self, output_dim: int, n_freq: int = 0, *, rngs: nnx.Rngs):
		# Input features: local coordinates (3), normals in local coordinates (3)
		input_dim = 6

		self.n_freq = n_freq
		if n_freq > 0:
			pos_encoding_dim = 3 * (2 * n_freq + 1)
			input_dim = pos_encoding_dim + 3

		# A linear projection simply to match transformer dimension
		self.mlp = nnx.Sequential(
			nnx.Linear(input_dim, output_dim, rngs=rngs),
			nnx.LayerNorm(output_dim, rngs=rngs),  # LayerNorm seems to really help here
		)

		self.key = jax.random.PRNGKey(57)

	def __call__(
		self,
		positions: jax.Array,  # [batch, k, 3]
		normals: jax.Array,  # [batch, k, 3]
	) -> jax.Array:
		"""
		Compute geometric features relative to first point (center).

		Returns:
			[batch, k, output_dim] geometric encodings
		"""
		batch_size, k, _ = positions.shape

		norms = jnp.linalg.norm(normals, axis=-1, keepdims=True)  # [batch, k, 1]
		normalized_normals = normals / jnp.maximum(norms, NAN_EPSILON)  # [batch, k, 3]

		# Center point (index 0)
		center_normal = normalized_normals[:, 0:1, :]  # [batch, 1, 3]

		# Regularization: scale neighborhood according to median_j |p_i - p_j|
		diff_vectors = positions - positions[:, 0:1, :]  # [batch, k, 3]
		distances = jnp.sqrt(jnp.sum(diff_vectors * diff_vectors, axis=-1))  # [batch, k]
		median = jnp.median(distances, axis=-1)  # [batch, ]
		normalized_positions = positions / median[:, None, None]  # [batch, k, 3]

		# The following block of code is copied from local_coordinates() above, because we also need the orthonormal basis.
		center_point = normalized_positions[:, 0:1, :]  # [batch, 1, 3]
		s_hat, t_hat = orthonormal_basis_with_rotation(center_normal, 0.0)

		# Encode relative positions in local frame as (s, t, Q(s, t)) tuples.
		diff_vectors = normalized_positions - center_point  # [batch, k, 3]
		heights = jnp.sum(diff_vectors * center_normal, axis=-1, keepdims=True)  # [batch, k, 1]
		S = jnp.sum(diff_vectors * s_hat, axis=-1, keepdims=True)  # [batch, k, 1]
		T = jnp.sum(diff_vectors * t_hat, axis=-1, keepdims=True)  # [batch, k, 1]
		coordinates = jnp.concatenate([S, T, heights], axis=-1)  # [batch, k, 3]

		# Encode normals in local frame.
		S = jnp.sum(normalized_normals * s_hat, axis=-1, keepdims=True)  # [batch, k, 1]
		T = jnp.sum(normalized_normals * t_hat, axis=-1, keepdims=True)  # [batch, k, 1]
		N = jnp.sum(normalized_normals * center_normal, axis=-1, keepdims=True)  # [batch, k, 1]
		local_normals = jnp.concatenate([S, T, N], axis=-1)  # [batch, k, 3]

		# [local coordinates (3), normals in local coordinates (3)]
		geometric_features = jnp.concatenate([coordinates, local_normals], axis=-1)  # [batch, k, input_dim]

		return self.mlp(geometric_features), median


class FundamentalFormPredictor(nnx.Module):
	def __init__(
		self,
		embed_dim: int = 128,
		n_freq: int = 0,
		n_layers: int = 4,
		n_heads: int = 4,
		mlp_dim: int = 256,
		dropout_rate: float = 0.0,
		*,
		rngs: nnx.Rngs,
	):
		"""
		Args: Parameters for each TransformerBlock.
			n_layers: Number of TransformerBlocks.
		"""

		self.input_encoder = InputFeatures(embed_dim, n_freq=n_freq, rngs=rngs)

		self.blocks = nnx.List(
			[TransformerBlock(embed_dim, n_heads, mlp_dim, dropout_rate, rngs=rngs) for _ in range(n_layers)]
		)

		self.final_norm = nnx.LayerNorm(embed_dim, rngs=rngs)

		# Projection
		# (a00, a01, a10, a11, a02, a20)
		self.mlp = nnx.Sequential(
			nnx.Linear(
				embed_dim,
				6,
				kernel_init=nnx.initializers.zeros_init(),
				# bias_init=nnx.initializers.ones_init(),
				bias_init=nnx.initializers.constant(jnp.array([0.0, 0.0, 0.0, 0.0, -0.5, -0.5])),
				rngs=rngs,
			),
		)

	@nnx.jit
	def __call__(self, positions: jax.Array, normals: jax.Array) -> jax.Array:
		"""
		Args:
			positions, normals: [batch, k, 3] representing a neighborhood of size k; the first point (index 0) is the center

		Returns:
			coefficients: [batch, 6] (a00, a01, a10, a11, a02, a20)
		"""
		batch_size, k, d = positions.shape

		# input_features: [batch, k, embed_dim]
		# scale: [batch, ]
		input_features, scale = self.input_encoder(positions, normals)
		x = input_features

		# Apply transformer blocks
		for block in self.blocks:
			x = block(x)

		x = self.final_norm(x)  # [batch, k, embed_dim]

		# Extract center point
		x = x[:, 0, :]  # [batch, embed_dim]

		# Predict form coefficients
		coeffs = self.mlp(x)  # [batch, 6]

		# Scale back to original space
		coeffs = jnp.stack(
			[
				coeffs[:, 0] * scale,  # a00
				coeffs[:, 1],  # a01
				coeffs[:, 2],  # a10
				coeffs[:, 3] / scale,  # a11
				coeffs[:, 4] / scale,  # a02
				coeffs[:, 5] / scale,  # a20
			],
			axis=-1,
		)

		return coeffs

	def precompute_coefficients(self, points, normals, k_neighbors, outliers=np.array([])):
		nnx.eval_mode(self)
		# Generate neighborhoods
		indices = get_neighbors(points, k_neighbors, outliers)  # (|P|, k)

		# Run forward evaluation.
		predicted_coefficients = self(points[indices], normals[indices])  # (|P|, 6)

		return predicted_coefficients

	def precompute_coefficients_in_chunks(self, points, normals, k_neighbors, chunk_size=50000, outliers=np.array([])):
		"""
		Process in chunks to avoid GPU memory issues
		"""
		nnx.eval_mode(self)

		n_points = points.shape[0]
		print(f'Processing {n_points} points in chunks of {chunk_size}...')

		# Generate patches
		t0 = time.time()
		indices = get_neighbors(points, k_neighbors, outliers)  # (|P|, k)
		t1 = time.time()
		print(f'  get_neighbors: {t1 - t0:.2f}s')

		# Allocate output array
		all_coefficients = []

		# Process in chunks
		n_chunks = (n_points + chunk_size - 1) // chunk_size
		t0 = time.time()

		for chunk_idx in range(n_chunks):
			start_idx = chunk_idx * chunk_size
			end_idx = min(start_idx + chunk_size, n_points)

			# Extract chunk
			chunk_indices = indices[start_idx:end_idx]  # (chunk_size, k)
			chunk_points = points[chunk_indices]  # (chunk_size, k, 3)
			chunk_normals = normals[chunk_indices]  # (chunk_size, k, 3)

			# Forward pass for this chunk
			chunk_coeffs = self(chunk_points, chunk_normals)

			# Wait for GPU to finish and transfer to CPU
			chunk_coeffs_cpu = np.array(chunk_coeffs.block_until_ready())
			all_coefficients.append(chunk_coeffs_cpu)

			# Clear JAX cache for this chunk
			del chunk_coeffs

			if (chunk_idx + 1) % 10 == 0:
				print(f'	{chunk_idx + 1}/{n_chunks} chunks processed')

		t1 = time.time()
		print(f'  Coefficient prediction (chunked): {t1 - t0:.2f}s')

		# Concatenate all chunks
		predicted_coefficients = np.concatenate(all_coefficients, axis=0)

		return predicted_coefficients

	@staticmethod
	def load_saved_model(filepath: str):
		"""
		Given the filepath where weights are saved, return the model and its parameters.
		"""
		with open(filepath, 'rb') as f:
			saved_data = pickle.load(f)

		n_freq = saved_data['n_freq'] if 'n_freq' in saved_data else 0

		loaded_model = FundamentalFormPredictor(
			embed_dim=saved_data['embed_dim'],
			n_freq=n_freq,
			n_heads=saved_data['n_heads'],
			n_layers=saved_data['n_layers'],
			mlp_dim=saved_data['mlp_dim'],
			dropout_rate=saved_data['dropout_rate'],  # not used in eval mode
			rngs=nnx.Rngs(0),  # not used in eval mode
		)

		# Load saved parameters into the model
		nnx.update(loaded_model, saved_data['params'])

		# Put in eval mode
		nnx.eval_mode(loaded_model)

		print(f'Loaded FundamentalFormPredictor model:')
		print(f'  embed_dim: {saved_data["embed_dim"]}')
		print(f'  n_freq: {n_freq}')
		print(f'  n_heads: {saved_data["n_heads"]}')
		print(f'  n_layers: {saved_data["n_layers"]}')
		print(f'  mlp_dim: {saved_data["mlp_dim"]}')
		print(f'  dropout_rate: {saved_data["dropout_rate"]}')

		return loaded_model, saved_data['k_neighbors']


# ========================================================================
# TRAINING
# ========================================================================


@jit
def SDF_loss(
	mask,
	batch_mask,
	positions,
	normals,
	coefficients,
	queries,
	true_distances,
):
	"""
	Compute a loss based on the end-to-end SDF process.
	Returns loss that is averaged over all neighborhoods in the batch.

	Args:
		mask: (max_points, ) boolean-valued array indicating which points are actually valid and belong to the input point cloud
		batch_mask: (batch_size, ) boolean-valued array indicating which query points in the batch are valid
		positions, normals: (max_points, 2) the entire input point cloud of a single shape
		coefficients: (max_points, 6) all point coefficients
		queries: (batch_size, n_queries, 2)
		true_distances: (batch_size, n_queries,)
	"""

	centers, axes, major_radii, minor_radii = torus_precompute_masked(mask, positions, normals, coefficients)

	predicted_distances, predicted_gradients = torus_distance_gradient_masked(
		mask, queries, positions, centers, axes, major_radii, minor_radii
	)

	n_queries_per_neighborhood = queries.shape[1]
	num_valid = jnp.sum(batch_mask)  # number of valid points in the neighborhood
	distance_error = jnp.abs((predicted_distances - true_distances))  # (batch_size, n_queries, )
	distance_valid_mask = ~jnp.isnan(distance_error) & batch_mask[:, None]
	distance_error_masked = jnp.sum(distance_error * distance_valid_mask, axis=-1)  # (batch_size,)
	num_valid_distance_queries = jnp.sum(distance_valid_mask)

	eikonality_error = jnp.abs(
		1.0 - jnp.sqrt(jnp.sum(predicted_gradients * predicted_gradients, axis=-1))
	)  # (batch_size, n_queries,)
	eikonality_valid_mask = ~jnp.isnan(eikonality_error) & batch_mask[:, None]  # (batch_size, n_queries)
	eikonality_error_masked = jnp.sum(
		jnp.where(eikonality_valid_mask, eikonality_error, 0.0), axis=-1
	)  # (batch_size, )
	num_valid_eikonality_queries = jnp.sum(eikonality_valid_mask)

	# Take mean only over valid points in the neighborhood.
	distance_loss = jax.lax.cond(
		num_valid_distance_queries > 0, lambda: jnp.sum(distance_error_masked) / num_valid_distance_queries, lambda: 0.0
	)
	eikonality_loss = jax.lax.cond(
		num_valid_eikonality_queries > 0,
		lambda: jnp.sum(eikonality_error_masked) / num_valid_eikonality_queries,
		lambda: 0.0,
	)

	loss = distance_loss + eikonality_loss

	# Return individual components for logging
	loss_dict = {
		'distance_loss': distance_loss,
		'eikonality_loss': eikonality_loss,
	}

	return loss, loss_dict


@jit
def validation_loss(
	mask,
	batch_mask,
	positions,
	normals,
	coefficients,
	queries,
	true_distances,
):
	centers, axes, major_radii, minor_radii = torus_precompute_masked(mask, positions, normals, coefficients)

	predicted_distances, predicted_gradients = torus_distance_gradient_masked(
		mask, queries, positions, centers, axes, major_radii, minor_radii
	)

	n_queries_per_neighborhood = queries.shape[1]
	num_valid = jnp.sum(batch_mask)  # number of valid points in the neighborhood
	distance_error = jnp.abs((predicted_distances - true_distances))  # (batch_size, n_queries, )
	distance_valid_mask = ~jnp.isnan(distance_error) & batch_mask[:, None]
	distance_error_masked = jnp.sum(distance_error * distance_valid_mask, axis=-1)  # (batch_size,)
	num_valid_distance_queries = jnp.sum(distance_valid_mask)

	eikonality_error = jnp.abs(
		1.0 - jnp.sqrt(jnp.sum(predicted_gradients * predicted_gradients, axis=-1))
	)  # (batch_size, n_queries,)
	eikonality_valid_mask = ~jnp.isnan(eikonality_error) & batch_mask[:, None]  # (batch_size, n_queries)
	eikonality_error_masked = jnp.sum(
		jnp.where(eikonality_valid_mask, eikonality_error, 0.0), axis=-1
	)  # (batch_size, )
	num_valid_eikonality_queries = jnp.sum(eikonality_valid_mask)

	# Take mean only over valid points in the neighborhood.
	distance_loss = jax.lax.cond(
		num_valid_distance_queries > 0, lambda: jnp.sum(distance_error_masked) / num_valid_distance_queries, lambda: 0.0
	)
	eikonality_loss = jax.lax.cond(
		num_valid_eikonality_queries > 0,
		lambda: jnp.sum(eikonality_error_masked) / num_valid_eikonality_queries,
		lambda: 0.0,
	)

	return distance_loss, eikonality_loss


def FundamentalFormPredictorTrainer(
	embed_dim=128,
	n_freq=4,
	n_layers=4,
	n_heads=4,
	mlp_dim=256,
	dropout_rate=0.0,
	rngs=nnx.Rngs(0),
	learning_rate=1e-3,
	weight_decay=0.0,
	warmup_steps=1000,
	total_steps=100000,
	seed=42,
):
	"""
	Set up network (model) with the given parameters.

	Returns: (model, optimizer, train_step) where train_step() is a function that trains the model for a single step.
	"""
	model = FundamentalFormPredictor(
		embed_dim=embed_dim,
		n_freq=n_freq,
		n_layers=n_layers,
		n_heads=n_heads,
		mlp_dim=mlp_dim,
		dropout_rate=dropout_rate,
		rngs=rngs,
	)

	# Learning rate schedule
	schedule = optax.warmup_cosine_decay_schedule(
		init_value=0.0,  # Start from 0
		peak_value=learning_rate,  # Warmup to this
		warmup_steps=warmup_steps,  # Linear warmup
		decay_steps=total_steps,  # Cosine decay over this many steps
		end_value=learning_rate * 0.01,
	)

	# Optimizer
	optimizer = nnx.Optimizer(
		model,
		optax.chain(
			# optax.clip_by_global_norm(1.0),
			optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
		),
		wrt=nnx.Param,
	)

	@nnx.jit
	def train_step(
		model: FundamentalFormPredictor,
		optimizer: nnx.Optimizer,
		mask,
		batch_mask,
		positions,
		normals,
		neighbors,
		queries,
		true_distances,
	):
		"""
		Single training step on a single batch (neighborhood) of one shape.

		Args:
			mask: (max_points,)
			batch_mask: (batch_size, )
			positions, normals: (|P|, 3)
			neighbors: (|P|, k)
			queries: (batch_size * n_queries, 3)
			true_distances: (batch_size * n_queries)

		Returns:
			Loss averaged over batch.
		"""

		def loss_fn(model):
			coefficients = model(positions[neighbors], normals[neighbors])

			loss, loss_dict = SDF_loss(mask, batch_mask, positions, normals, coefficients, queries, true_distances)

			return loss, loss_dict

		(batch_loss, loss_dict), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

		def zero_out_nans(x):
			if x is None:
				return x
			return jnp.where(jnp.isnan(x), 0.0, x)

		# Apply to all gradients in the tree
		grads = jax.tree_util.tree_map(zero_out_nans, grads)

		# Update with this batch's gradients
		optimizer.update(model, grads)

		return batch_loss, loss_dict

	return model, optimizer, train_step


def TrainFundamentalFormPredictor(
	training_data_filenames,
	num_epochs,
	learning_rate,
	batch_size,
	warmup_steps,
	n_epochs_to_save,
	n_shapes_to_save,
	load_model_path=None,
	final_save_filepath=None,
):
	"""
	Args:
		load_model_path: Optional path to a saved model file (.pkl) to use as warm start
	"""
	LABEL_WIDTH = 18  # Width for "distance loss = ", etc.
	VALUE_WIDTH = 12  # Width for the numeric values

	print(jax.default_backend())
	print(jax.devices())

	seed = 42
	key = jax.random.PRNGKey(seed)
	rngs = nnx.Rngs(seed)

	# NOTE: Could probably train on much larger meshes and point clouds using chunked versions of torus_precompute_masked() and precompute_coefficients(), to avoid GPU memory constraints (at the cost of increasing compute time).
	# The current (naive) versions of these functions are called on all points at once (usually in a vectorized manner).

	# Architecture parameters
	embed_dim = 128
	n_freq = 0
	n_layers = 8
	n_heads = 8
	mlp_dim = embed_dim * 4

	# Sampling params
	n_queries_per_neighborhood = 120
	narrow_band_offset = 0.2
	bbox_extension = 1.0

	# Learning parameters
	dropout_rate = 0.0
	weight_decay = 0.0
	total_steps = 1e8

	model, optimizer, train_step = FundamentalFormPredictorTrainer(
		embed_dim=embed_dim,
		n_freq=n_freq,
		n_layers=n_layers,
		n_heads=n_heads,
		mlp_dim=mlp_dim,
		dropout_rate=dropout_rate,
		rngs=rngs,
		learning_rate=learning_rate,
		weight_decay=weight_decay,
		warmup_steps=warmup_steps,
		total_steps=total_steps,
		seed=seed,
	)

	# Load pretrained model if specified
	if load_model_path is not None:
		print(f'Loading pretrained model from {load_model_path}')
		with open(load_model_path, 'rb') as f:
			saved_data = pickle.load(f)

		# Verify architecture matches
		assert saved_data['embed_dim'] == embed_dim, 'embed_dim mismatch'
		assert saved_data['n_freq'] == n_freq, 'n_freq mismatch'
		assert saved_data['n_heads'] == n_heads, 'n_heads mismatch'
		assert saved_data['n_layers'] == n_layers, 'n_layers mismatch'
		assert saved_data['mlp_dim'] == mlp_dim, 'mlp_dim mismatch'

		# Load parameters into model
		nnx.update(model, saved_data['params'])
		print(f'Successfully loaded pretrained weights from {load_model_path}')

	# Load mesh training data from multiple files.
	all_training_data = []
	t0 = time.time()
	k_neighbors = None
	for training_data_filename in training_data_filenames:
		if not os.path.exists(training_data_filename):
			raise ValueError(f"The path '{training_data_filename}' does not exist")

		print(f"Loading training data from '{training_data_filename}'...")

		data = load_mesh_training_data(training_data_filename)

		(all_positions, all_normals, all_neighbors, all_masks, all_mesh_vertices, all_mesh_faces, all_face_masks) = data

		k_neighbors = all_neighbors.shape[2]
		n_shapes = len(all_positions)
		max_points = all_positions.shape[1]
		print(f'  Loaded {n_shapes} shapes with max {max_points} points each')
		print(f'  k_neighbors: {k_neighbors}')

		all_training_data.append(data)

	t1 = time.time()
	print(f'Load data: {t1 - t0:.6f} s')

	# Create combined list of (dataset_idx, shape_idx) for all shapes across all datasets
	# Note that the below code still assumes multiple datasets... just leave it for now
	all_shape_indices = []
	for dataset_idx, data in enumerate(all_training_data):
		n_shapes = len(data[0])  # data[0] is all_positions
		for shape_idx in range(n_shapes):
			all_shape_indices.append((dataset_idx, shape_idx))

	total_shapes = len(all_shape_indices)
	print(f'Total shapes across all datasets: {total_shapes}')

	# Load validation data
	val_data_filename = 'datasets/ABC17_validation_3.npz'
	# val_data_filename = 'datasets/ABC_components/ABC17_validation_0.npz'
	val_positions = None
	val_normals = None
	val_neighbors = None
	val_masks = None
	val_mesh_vertices = None
	val_mesh_faces = None
	val_face_masks = None

	if os.path.exists(val_data_filename):
		try:
			(
				val_positions,
				val_normals,
				val_neighbors,
				val_masks,
				val_mesh_vertices,
				val_mesh_faces,
				val_face_masks,
			) = load_mesh_training_data(val_data_filename)
			print(f'Loaded validation data: {len(val_positions)} shapes')
		except Exception as e:
			print(f'Warning: Could not load validation data: {e}')
			print('Continuing without validation...')
	else:
		print(f'Validation file not found: {val_data_filename}')

	def save_model(curr_model, epoch, save_filepath: str = None):
		# Save weights as pkl file
		nnx.eval_mode(curr_model)
		params = nnx.state(curr_model, nnx.Param)
		saved_data = {
			'params': params,
			'embed_dim': embed_dim,
			'n_freq': n_freq,
			'n_heads': n_heads,
			'n_layers': n_layers,
			'mlp_dim': mlp_dim,
			'k_neighbors': k_neighbors,
			'dropout_rate': dropout_rate,
		}

		model_name = f'FundamentalFormPredictor_{epoch}' if save_filepath is None else save_filepath
		filepath = MODELS_DIR + model_name + '.pkl'
		with open(filepath, 'wb') as f:
			pickle.dump(saved_data, f)

		print(f'Model saved to {filepath}')

	# Training loop
	nnx.train_mode(model)

	t0 = time.time()
	all_train_losses = []
	all_distance_losses = []
	all_eikonality_losses = []

	validation_losses = []
	validation_epochs = []

	print('\nBeginning training...')
	print(f'  Batch size: {batch_size}')

	num_training_datasets = len(all_training_data)
	np.random.seed(seed)

	for epoch in range(num_epochs):
		epoch_key = jax.random.fold_in(key, epoch)  # deterministic key per epoch

		epoch_losses = []
		epoch_distance_losses = []
		epoch_eikonality_losses = []
		total_num_batches = 0

		# Shuffle all shapes across all datasets together
		shuffled_indices = np.random.permutation(total_shapes)

		for i, global_idx in enumerate(shuffled_indices):
			dataset_idx, shape_idx = all_shape_indices[global_idx]

			# Get data arrays for this dataset
			(
				all_positions,
				all_normals,
				all_neighbors,
				all_masks,
				all_mesh_vertices,
				all_mesh_faces,
				all_face_masks,
			) = all_training_data[dataset_idx]

			max_points = all_positions.shape[1]
			n_batches = (max_points + batch_size - 1) // batch_size

			shape_key = jax.random.fold_in(epoch_key, global_idx)

			# Convert single shape to JAX arrays (GPU)
			positions = jnp.array(all_positions[shape_idx])
			normals = jnp.array(all_normals[shape_idx])
			neighbors = jnp.array(all_neighbors[shape_idx])
			mask = jnp.array(all_masks[shape_idx])
			mesh_vertices = jnp.array(all_mesh_vertices[shape_idx])
			mesh_faces = jnp.array(all_mesh_faces[shape_idx])
			face_mask = jnp.array(all_face_masks[shape_idx])

			# Generate query points and ground-truth distances
			shape_queries, shape_distances = generate_shape_features(
				positions,
				neighbors,
				mesh_vertices,
				mesh_faces,
				face_mask,
				shape_key,
				n_queries=n_queries_per_neighborhood,
				narrow_band_offset=narrow_band_offset,
				bbox_extension=bbox_extension,
			)

			batch_losses_list = []
			batch_distance_losses_list = []
			batch_eikonality_losses_list = []

			# Loop over batches for this shape
			for batch_idx in range(n_batches):
				batch_key = jax.random.fold_in(shape_key, batch_idx)
				start_idx = batch_idx * batch_size

				batch_mask = jax.lax.dynamic_slice_in_dim(mask, start_idx, batch_size)
				batch_neighbors = jax.lax.dynamic_slice_in_dim(neighbors, start_idx, batch_size)

				batch_queries = jax.lax.dynamic_slice_in_dim(shape_queries, start_idx, batch_size)
				batch_distances = jax.lax.dynamic_slice_in_dim(shape_distances, start_idx, batch_size)

				# Train step
				loss, loss_dict = train_step(
					model,
					optimizer,
					mask,
					batch_mask,
					positions,
					normals,
					neighbors,
					batch_queries,
					batch_distances,
				)

				batch_losses_list.append(loss)
				batch_distance_losses_list.append(loss_dict['distance_loss'])
				batch_eikonality_losses_list.append(loss_dict['eikonality_loss'])

			# Accumulate losses for this shape
			epoch_losses.extend(batch_losses_list)
			epoch_distance_losses.extend(batch_distance_losses_list)
			epoch_eikonality_losses.extend(batch_eikonality_losses_list)
			total_num_batches += n_batches

			# Clean up GPU memory for this shape
			del positions, normals, neighbors, mask
			del mesh_vertices, mesh_faces, face_mask

			if i % n_shapes_to_save == 0 and i > 0:
				epoch_train_loss = np.sum(epoch_losses) / total_num_batches
				epoch_distance_loss = np.sum(epoch_distance_losses) / total_num_batches
				epoch_eikonality_loss = np.sum(epoch_eikonality_losses) / total_num_batches

				print(
					f'  Epoch {epoch}, shape {i}/{total_shapes}: '
					f'loss = {epoch_train_loss:.6f}  '
					f'distance = {epoch_distance_loss:.6f}  '
					f'eikonality = {epoch_eikonality_loss:.6f}'
				)

		# Compute epoch averages
		epoch_train_loss = np.sum(epoch_losses) / total_num_batches
		epoch_distance_loss = np.sum(epoch_distance_losses) / total_num_batches
		epoch_eikonality_loss = np.sum(epoch_eikonality_losses) / total_num_batches

		all_train_losses.append(epoch_train_loss)
		all_distance_losses.append(epoch_distance_loss)
		all_eikonality_losses.append(epoch_eikonality_loss)

		if epoch % n_epochs_to_save == 0:
			print(
				f'Epoch {epoch}:\n'
				f'	{"total loss:":<{LABEL_WIDTH}}{epoch_train_loss:<{VALUE_WIDTH}.6f}'
				f'{"distance:":<{LABEL_WIDTH}}{epoch_distance_loss:<{VALUE_WIDTH}.6f}'
				f'{"eikonality:":<{LABEL_WIDTH}}{epoch_eikonality_loss:<{VALUE_WIDTH}.6f}'
			)

			# Save model checkpoint
			save_model(model, epoch)

			# Validation loss
			if val_positions is not None:
				nnx.eval_mode(model)
				n_val_shapes = len(val_positions)
				val_shape_losses = []
				val_shape_eikonality_losses = []

				for val_shape_idx in range(n_val_shapes):
					# Generate validation key (deterministic per epoch)
					val_key = jax.random.fold_in(epoch_key, 10000 + val_shape_idx)

					val_pos = val_positions[val_shape_idx]
					val_norm = val_normals[val_shape_idx]
					val_neigh = val_neighbors[val_shape_idx]
					val_mask = val_masks[val_shape_idx]
					val_verts = val_mesh_vertices[val_shape_idx]
					val_faces = val_mesh_faces[val_shape_idx]
					val_fmask = val_face_masks[val_shape_idx]

					# Generate validation queries on-the-fly
					val_queries, val_distances = generate_shape_features(
						val_pos,
						val_neigh,
						val_verts,
						val_faces,
						val_fmask,
						val_key,
						n_queries=n_queries_per_neighborhood,
					)

					# Compute weights for this validation shape
					val_weights = model(val_pos[val_neigh], val_norm[val_neigh])

					# Compute loss per batch
					n_val_batches = (val_pos.shape[0] + batch_size - 1) // batch_size
					batch_val_losses = []
					batch_val_eikonality_losses = []

					for batch_idx in range(n_val_batches):
						start_idx = batch_idx * batch_size
						batch_mask = jax.lax.dynamic_slice_in_dim(val_mask, start_idx, batch_size)
						batch_queries = jax.lax.dynamic_slice_in_dim(val_queries, start_idx, batch_size)
						batch_distances = jax.lax.dynamic_slice_in_dim(val_distances, start_idx, batch_size)

						# Only compute if batch has valid points
						if jnp.sum(batch_mask) > 0:
							val_loss, val_eikonality_loss = validation_loss(
								val_mask,
								batch_mask,
								val_pos,
								val_norm,
								val_weights,
								batch_queries,
								batch_distances,
							)
							batch_val_losses.append(val_loss)
							batch_val_eikonality_losses.append(val_eikonality_loss)

					# Average over batches for this shape
					if len(batch_val_losses) > 0:
						val_shape_losses.append(np.mean(batch_val_losses))
						val_shape_eikonality_losses.append(np.mean(batch_val_eikonality_losses))

				# Average over all validation shapes
				val_loss = np.mean(val_shape_losses) if len(val_shape_losses) > 0 else 0.0
				val_eikonality_loss = (
					np.mean(val_shape_eikonality_losses) if len(val_shape_eikonality_losses) > 0 else 0.0
				)
				nnx.train_mode(model)

				validation_losses.append(val_loss)
				validation_epochs.append(epoch)

				print(
					f'	{"validation:":<{LABEL_WIDTH}}{val_loss:<{VALUE_WIDTH}.6f}'
					f'{"eikonality:":<{LABEL_WIDTH}}{val_eikonality_loss:<{VALUE_WIDTH}.6f}'
				)

	t1 = time.time()
	print(f'Total training time: {t1 - t0:.2f} s')

	train_losses = np.array(all_train_losses)
	distance_losses = np.array(all_distance_losses)
	eikonality_losses = np.array(all_eikonality_losses)

	# Save final model
	save_model(model, num_epochs - 1, final_save_filepath)

	# Plot training history

	fig, axes = plt.subplots(2, 2, figsize=(15, 10))

	# Total training loss
	axes[0, 0].plot(train_losses, label='Total Training Loss', color='blue')
	axes[0, 0].set_xlabel('Epoch')
	axes[0, 0].set_ylabel('Loss')
	axes[0, 0].set_title('Total Training Loss')
	axes[0, 0].legend()
	axes[0, 0].grid(True, alpha=0.3)

	# Distance loss
	axes[0, 1].plot(distance_losses, label='Distance Error', color='green')
	axes[0, 1].set_xlabel('Epoch')
	axes[0, 1].set_ylabel('Loss')
	axes[0, 1].set_title('Distance Error')
	axes[0, 1].legend()
	axes[0, 1].grid(True, alpha=0.3)

	# Eikonality loss
	axes[1, 0].plot(eikonality_losses, label='Eikonality Error', color='orange')
	axes[1, 0].set_xlabel('Epoch')
	axes[1, 0].set_ylabel('Loss')
	axes[1, 0].set_title('Eikonality Error')
	axes[1, 0].legend()
	axes[1, 0].grid(True, alpha=0.3)

	# Validation loss
	if len(validation_losses) > 0:
		axes[1, 1].plot(validation_epochs, validation_losses, label='Validation Loss', color='red', marker='o')
		axes[1, 1].set_xlabel('Epoch')
		axes[1, 1].set_ylabel('Loss')
		axes[1, 1].set_title('Validation Loss')
		axes[1, 1].legend()
		axes[1, 1].grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig('PlotTrainingHistory.png', dpi=150)
	print('Training history plot saved to PlotTrainingHistory.png')


if __name__ == '__main__':
	dataset_dir = 'datasets/nice'

	# Pretraining
	training_data_filenames = ['datasets/pretraining_blob.npz']
	TrainFundamentalFormPredictor(
		training_data_filenames,
		num_epochs=64,
		learning_rate=1e-3,
		warmup_steps=0,
		batch_size=32,
		n_epochs_to_save=1,
		n_shapes_to_save=1,
		load_model_path=None,
		final_save_filepath='FundamentalFormPredictor_pretrained',
	)

	# Phase 0: nicely sampled blobs
	training_data_filenames = sorted(glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz')))
	TrainFundamentalFormPredictor(
		training_data_filenames,
		num_epochs=24,
		learning_rate=1e-5,
		warmup_steps=1e2,
		batch_size=1024,
		n_epochs_to_save=1,
		n_shapes_to_save=1,
		load_model_path='models/FundamentalFormPredictor_pretrained.pkl',
		final_save_filepath='FundamentalFormPredictor_phase0',
	)

	# Phase 1: nicely sampled blobs and ABC meshes
	blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
	ABC_filenames = glob.glob(os.path.join(dataset_dir, 'ABC17_train_mode0_*.npz'))
	lists = [blob_filenames, ABC_filenames]
	training_data_filenames = [val for tup in zip(*lists) for val in tup]  # interleave lists
	TrainFundamentalFormPredictor(
		training_data_filenames,
		num_epochs=8,
		learning_rate=1e-5,
		warmup_steps=1e3,
		batch_size=1024,
		n_epochs_to_save=1,
		n_shapes_to_save=100,
		load_model_path='models/FundamentalFormPredictor_phase0.pkl',
		final_save_filepath='FundamentalFormPredictor_phase1',
	)

	# Phase 2: nicely sampled blobs and more ABC meshes
	blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
	ABC17_filenames = glob.glob(os.path.join(dataset_dir, 'ABC17_train_mode0_*.npz'))
	ABC2_filenames = glob.glob(os.path.join(dataset_dir, 'ABC2_train_mode0_*.npz'))
	training_data_filenames = blob_filenames + ABC17_filenames + ABC2_filenames
	TrainFundamentalFormPredictor(
		training_data_filenames,
		num_epochs=8,
		learning_rate=1e-6,
		warmup_steps=1e3,
		batch_size=1024,
		n_epochs_to_save=1,
		n_shapes_to_save=100,
		load_model_path='models/FundamentalFormPredictor_phase1.pkl',
		final_save_filepath='FundamentalFormPredictor_phase2',
	)

	# Phase 3: nicely sampled blobs and even more ABC meshes
	blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
	ABC_filenames = glob.glob(os.path.join(dataset_dir, 'ABC*_train_mode0_*.npz'))
	training_data_filenames = blob_filenames + ABC_filenames
	TrainFundamentalFormPredictor(
		training_data_filenames,
		num_epochs=5,
		learning_rate=1e-6,
		warmup_steps=1e3,
		batch_size=1024,
		n_epochs_to_save=1,
		n_shapes_to_save=100,
		load_model_path='models/FundamentalFormPredictor_phase2.pkl',
		final_save_filepath='FundamentalFormPredictor_phase3',
	)

	# Phase 4: everything from before, plus unevenly sampled data possibly with holes
	blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
	ABC_filenames = glob.glob(os.path.join(dataset_dir, 'ABC*_train_mode0_*.npz'))
	blob_filenames_holey = glob.glob(os.path.join('datasets/holey', 'blobs_train_mode*.npz'))
	ABC_filenames_holey = glob.glob(os.path.join('datasets/holey', 'ABC*_train_mode*.npz'))
	training_data_filenames = blob_filenames + ABC_filenames + blob_filenames_holey + ABC_filenames_holey
	TrainFundamentalFormPredictor(
		training_data_filenames,
		num_epochs=2,
		learning_rate=1e-7,
		warmup_steps=1e3,
		batch_size=1024,
		n_epochs_to_save=1,
		n_shapes_to_save=100,
		load_model_path='models/FundamentalFormPredictor_phase3.pkl',
		final_save_filepath='FundamentalFormPredictor_phase4',
	)
