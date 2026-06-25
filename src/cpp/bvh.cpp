#include "bvh.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <sstream>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;
using namespace nb::literals;

// ============================================================================
// CONSTRUCTION
// ============================================================================

BVH::BVH(const Eigen::Matrix<double, 3, Eigen::Dynamic>& points) : primitive_type(BVHPrimitiveType::Point) {
  n_primitives = points.cols();
  primitives = points;
  centroids = points;

  // Initialize primitive indices
  primitive_indices.resize(n_primitives);
  for (int i = 0; i < n_primitives; ++i) {
    primitive_indices(i) = i;
  }

  // Build tree
  nodes.clear();
  nodes.reserve(2 * n_primitives); // worst case
  build_recursive(0, n_primitives);
}

BVH::BVH(const Eigen::Matrix<double, 3, Eigen::Dynamic>& vertices, const Eigen::Matrix<int, 3, Eigen::Dynamic>& faces)
    : primitive_type(BVHPrimitiveType::Triangle) {

  n_primitives = faces.cols();
  primitives.resize(3, n_primitives * 3);

  // Compute centroids
  centroids.resize(3, n_primitives);
  for (int i = 0; i < n_primitives; ++i) {
    centroids.col(i) = (vertices.col(faces(0, i)) + vertices.col(faces(1, i)) + vertices.col(faces(2, i))) / 3.0;
    primitives.col(3 * i) = vertices.col(faces(0, i));
    primitives.col(3 * i + 1) = vertices.col(faces(1, i));
    primitives.col(3 * i + 2) = vertices.col(faces(2, i));
  }

  // Initialize primitive indices
  primitive_indices.resize(n_primitives);
  for (int i = 0; i < n_primitives; ++i) {
    primitive_indices(i) = i;
  }

  // Build tree
  nodes.clear();
  nodes.reserve(2 * n_primitives);
  build_recursive(0, n_primitives);
}

// ============================================================================
// Begin code generated mostly by Claude
// ============================================================================

int BVH::build_recursive(int start, int end) {
  int node_idx = static_cast<int>(nodes.size());
  BVHNode node;
  nodes.push_back(node);

  // Compute AABB for this node
  compute_aabb(start, end, nodes[node_idx].aabb_min, nodes[node_idx].aabb_max);

  int n_prims = end - start;

  // Create leaf if small enough
  if (n_prims <= 4) {
    nodes[node_idx].left_child = -1;
    nodes[node_idx].right_child = -1;
    nodes[node_idx].prim_start = start;
    nodes[node_idx].prim_count = n_prims;
    return node_idx;
  }

  // Find best split using SAH
  Eigen::Vector3d centroid_min, centroid_max;
  compute_centroid_aabb(start, end, centroid_min, centroid_max);

  // Check if all centroids are at same point
  if ((centroid_max - centroid_min).norm() < 1e-10) {
    nodes[node_idx].left_child = -1;
    nodes[node_idx].right_child = -1;
    nodes[node_idx].prim_start = start;
    nodes[node_idx].prim_count = n_prims;
    return node_idx;
  }

  auto [axis, mid] = find_best_split(start, end, centroid_min, centroid_max);

  // Ensure valid split
  mid = std::max(start + 1, std::min(mid, end - 1));

  // Partition primitives
  double split_value = centroids(axis, primitive_indices(mid));
  mid = partition(start, end, axis, split_value);

  // Ensure we actually split
  if (mid <= start || mid >= end) {
    mid = (start + end) / 2;
  }

  // Build children
  nodes[node_idx].prim_start = 0;
  nodes[node_idx].prim_count = 0;
  nodes[node_idx].left_child = build_recursive(start, mid);
  nodes[node_idx].right_child = build_recursive(mid, end);

  return node_idx;
}

void BVH::compute_aabb(int start, int end, Eigen::Vector3d& aabb_min, Eigen::Vector3d& aabb_max) const {
  aabb_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  aabb_max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());

  if (primitive_type == BVHPrimitiveType::Point) {
    for (int i = start; i < end; ++i) {
      int prim_idx = primitive_indices(i);
      const Eigen::Vector3d& p = primitives.col(prim_idx);
      aabb_min = aabb_min.cwiseMin(p);
      aabb_max = aabb_max.cwiseMax(p);
    }
  } else { // Triangle
    for (int i = start; i < end; ++i) {
      int prim_idx = primitive_indices(i);
      for (int j = 0; j < 3; ++j) {
        const Eigen::Vector3d& v = primitives.col(3 * prim_idx + j);
        aabb_min = aabb_min.cwiseMin(v);
        aabb_max = aabb_max.cwiseMax(v);
      }
    }
  }
}

void BVH::compute_centroid_aabb(int start, int end, Eigen::Vector3d& centroid_min,
                                Eigen::Vector3d& centroid_max) const {
  centroid_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  centroid_max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());

  for (int i = start; i < end; ++i) {
    int prim_idx = primitive_indices(i);
    const Eigen::Vector3d& c = centroids.col(prim_idx);
    centroid_min = centroid_min.cwiseMin(c);
    centroid_max = centroid_max.cwiseMax(c);
  }
}

double BVH::surface_area(const Eigen::Vector3d& aabb_min, const Eigen::Vector3d& aabb_max) const {
  Eigen::Vector3d d = aabb_max - aabb_min;
  return 2.0 * (d.x() * d.y() + d.y() * d.z() + d.z() * d.x());
}

std::pair<int, int> BVH::find_best_split(int start, int end, const Eigen::Vector3d& centroid_min,
                                         const Eigen::Vector3d& centroid_max) const {
  const int n_bins = 12;
  double best_cost = std::numeric_limits<double>::infinity();
  int best_axis = 0;
  int best_split = start + 1;

  // Try each axis
  for (int axis = 0; axis < 3; ++axis) {
    if (centroid_max(axis) - centroid_min(axis) < 1e-10) {
      continue;
    }

    // Bin primitives
    std::vector<int> bin_counts(n_bins, 0);
    std::vector<Eigen::Vector3d> bin_aabb_min(n_bins,
                                              Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity()));
    std::vector<Eigen::Vector3d> bin_aabb_max(n_bins,
                                              Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity()));

    double range = centroid_max(axis) - centroid_min(axis);
    for (int i = start; i < end; ++i) {
      int prim_idx = primitive_indices(i);
      double centroid_val = centroids(axis, prim_idx);

      int bin_idx = static_cast<int>((centroid_val - centroid_min(axis)) / range * n_bins);
      bin_idx = std::min(bin_idx, n_bins - 1);

      bin_counts[bin_idx]++;

      if (primitive_type == BVHPrimitiveType::Point) {
        const Eigen::Vector3d& p = primitives.col(prim_idx);
        bin_aabb_min[bin_idx] = bin_aabb_min[bin_idx].cwiseMin(p);
        bin_aabb_max[bin_idx] = bin_aabb_max[bin_idx].cwiseMax(p);
      } else {
        for (int j = 0; j < 3; ++j) {
          const Eigen::Vector3d& v = primitives.col(3 * prim_idx + j);
          bin_aabb_min[bin_idx] = bin_aabb_min[bin_idx].cwiseMin(v);
          bin_aabb_max[bin_idx] = bin_aabb_max[bin_idx].cwiseMax(v);
        }
      }
    }

    // Evaluate SAH for each split
    for (int split_bin = 1; split_bin < n_bins; ++split_bin) {
      // Left side: bins [0, split_bin)
      int left_count = 0;
      Eigen::Vector3d left_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
      Eigen::Vector3d left_max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());

      for (int i = 0; i < split_bin; ++i) {
        left_count += bin_counts[i];
        if (bin_counts[i] > 0) {
          left_min = left_min.cwiseMin(bin_aabb_min[i]);
          left_max = left_max.cwiseMax(bin_aabb_max[i]);
        }
      }

      // Right side: bins [split_bin, n_bins)
      int right_count = 0;
      Eigen::Vector3d right_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
      Eigen::Vector3d right_max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());

      for (int i = split_bin; i < n_bins; ++i) {
        right_count += bin_counts[i];
        if (bin_counts[i] > 0) {
          right_min = right_min.cwiseMin(bin_aabb_min[i]);
          right_max = right_max.cwiseMax(bin_aabb_max[i]);
        }
      }

      if (left_count > 0 && right_count > 0) {
        double left_sa = surface_area(left_min, left_max);
        double right_sa = surface_area(right_min, right_max);
        double cost = 0.5 + (left_count * left_sa + right_count * right_sa);

        if (cost < best_cost) {
          best_cost = cost;
          best_axis = axis;

          // Find actual split point
          double split_value = centroid_min(axis) + (split_bin / static_cast<double>(n_bins)) * range;
          int count_left = 0;
          for (int i = start; i < end; ++i) {
            if (centroids(best_axis, primitive_indices(i)) < split_value) {
              count_left++;
            }
          }
          best_split = start + count_left;
        }
      }
    }
  }

  return {best_axis, best_split};
}

int BVH::partition(int start, int end, int axis, double split_value) {
  int left = start;
  int right = end - 1;

  while (left < right) {
    while (left < end && centroids(axis, primitive_indices(left)) < split_value) {
      left++;
    }
    while (right >= start && centroids(axis, primitive_indices(right)) >= split_value) {
      right--;
    }

    if (left < right) {
      std::swap(primitive_indices(left), primitive_indices(right));
      left++;
      right--;
    }
  }

  return left;
}

// ============================================================================
// QUERIES
// ============================================================================

std::pair<int, double> BVH::find_closest_point(const Eigen::Vector3d& query) const {
  if (primitive_type != BVHPrimitiveType::Point) {
    throw std::runtime_error("find_closest_point only works for point BVHs");
  }

  double min_dist = std::numeric_limits<double>::infinity();
  int closest_idx = -1;

  std::vector<int> stack;
  stack.reserve(64);
  stack.push_back(0);

  while (!stack.empty()) {
    int node_idx = stack.back();
    stack.pop_back();

    const BVHNode& node = nodes[node_idx];

    // Check if AABB is closer than current best
    double node_dist_sq = point_aabb_distance_sq(query, node.aabb_min, node.aabb_max);
    if (node_dist_sq > min_dist * min_dist) {
      continue;
    }

    if (node.prim_count > 0) {
      // Leaf: check all points
      for (int i = 0; i < node.prim_count; ++i) {
        int prim_idx = primitive_indices(node.prim_start + i);
        const Eigen::Vector3d& p = primitives.col(prim_idx);
        double dist = (query - p).norm();
        if (dist < min_dist) {
          min_dist = dist;
          closest_idx = prim_idx;
        }
      }
    } else {
      // Internal: add children (closer first for better pruning)
      const BVHNode& left = nodes[node.left_child];
      const BVHNode& right = nodes[node.right_child];

      double left_dist_sq = point_aabb_distance_sq(query, left.aabb_min, left.aabb_max);
      double right_dist_sq = point_aabb_distance_sq(query, right.aabb_min, right.aabb_max);

      if (left_dist_sq < right_dist_sq) {
        if (right_dist_sq <= min_dist * min_dist) stack.push_back(node.right_child);
        if (left_dist_sq <= min_dist * min_dist) stack.push_back(node.left_child);
      } else {
        if (left_dist_sq <= min_dist * min_dist) stack.push_back(node.left_child);
        if (right_dist_sq <= min_dist * min_dist) stack.push_back(node.right_child);
      }
    }
  }

  return {closest_idx, min_dist};
}

std::pair<Eigen::VectorXi, Eigen::VectorXd> BVH::find_k_nearest(const Eigen::Vector3d& query, int k) const {
  if (primitive_type != BVHPrimitiveType::Point) {
    throw std::runtime_error("find_k_nearest only works for point BVHs");
  }

  // Priority queue: (distance, index)
  using Pair = std::pair<double, int>;
  std::priority_queue<Pair> heap; // max heap

  std::vector<int> stack;
  stack.reserve(64);
  stack.push_back(0);

  double max_dist = std::numeric_limits<double>::infinity();

  while (!stack.empty()) {
    int node_idx = stack.back();
    stack.pop_back();

    const BVHNode& node = nodes[node_idx];

    double node_dist_sq = point_aabb_distance_sq(query, node.aabb_min, node.aabb_max);
    if (node_dist_sq > max_dist * max_dist) {
      continue;
    }

    if (node.prim_count > 0) {
      // Leaf: check all points
      for (int i = 0; i < node.prim_count; ++i) {
        int prim_idx = primitive_indices(node.prim_start + i);
        const Eigen::Vector3d& p = primitives.col(prim_idx);
        double dist = (query - p).norm();

        if (static_cast<int>(heap.size()) < k) {
          heap.push({dist, prim_idx});
          max_dist = heap.top().first;
        } else if (dist < max_dist) {
          heap.pop();
          heap.push({dist, prim_idx});
          max_dist = heap.top().first;
        }
      }
    } else {
      stack.push_back(node.left_child);
      stack.push_back(node.right_child);
    }
  }

  // Extract results
  int count = static_cast<int>(heap.size());
  Eigen::VectorXi indices(count);
  Eigen::VectorXd distances(count);

  for (int i = count - 1; i >= 0; --i) {
    indices(i) = heap.top().second;
    distances(i) = heap.top().first;
    heap.pop();
  }

  return {indices, distances};
}

std::pair<Eigen::VectorXi, Eigen::VectorXd> BVH::find_points_in_radius(const Eigen::Vector3d& query,
                                                                       double radius) const {
  if (primitive_type != BVHPrimitiveType::Point) {
    throw std::runtime_error("find_points_in_radius only works for point BVHs");
  }

  std::vector<int> result_indices;
  std::vector<double> result_distances;
  double radius_sq = radius * radius;

  std::vector<int> stack;
  stack.reserve(64);
  stack.push_back(0);

  while (!stack.empty()) {
    int node_idx = stack.back();
    stack.pop_back();

    const BVHNode& node = nodes[node_idx];

    double node_dist_sq = point_aabb_distance_sq(query, node.aabb_min, node.aabb_max);
    if (node_dist_sq > radius_sq) {
      continue;
    }

    if (node.prim_count > 0) {
      // Leaf: check all points
      for (int i = 0; i < node.prim_count; ++i) {
        int prim_idx = primitive_indices(node.prim_start + i);
        const Eigen::Vector3d& p = primitives.col(prim_idx);
        double dist_sq = (query - p).squaredNorm();
        if (dist_sq <= radius_sq) {
          result_indices.push_back(prim_idx);
          result_distances.push_back(std::sqrt(dist_sq));
        }
      }
    } else {
      stack.push_back(node.left_child);
      stack.push_back(node.right_child);
    }
  }

  // Convert to Eigen
  int count = static_cast<int>(result_indices.size());
  Eigen::VectorXi indices(count);
  Eigen::VectorXd distances(count);
  for (int i = 0; i < count; ++i) {
    indices(i) = result_indices[i];
    distances(i) = result_distances[i];
  }

  return {indices, distances};
}

std::tuple<bool, double, int, Eigen::Vector2d> BVH::intersect_ray(const Eigen::Vector3d& ray_origin,
                                                                  const Eigen::Vector3d& ray_dir, double t_min,
                                                                  double t_max, double sphere_radius) const {

  bool found_hit = false;
  double hit_t = t_max;
  int hit_idx = -1;
  Eigen::Vector2d hit_bary(0, 0);

  std::vector<int> stack;
  stack.reserve(64);
  stack.push_back(0);

  while (!stack.empty()) {
    int node_idx = stack.back();
    stack.pop_back();

    const BVHNode& node = nodes[node_idx];

    double t_near, t_far;
    if (!intersect_aabb(ray_origin, ray_dir, node.aabb_min, node.aabb_max, t_near, t_far)) {
      continue;
    }

    if (t_near > hit_t || t_far < t_min) {
      continue;
    }

    if (node.prim_count > 0) {
      // Leaf: test all primitives
      for (int i = 0; i < node.prim_count; ++i) {
        int prim_idx = primitive_indices(node.prim_start + i);

        if (primitive_type == BVHPrimitiveType::Point) {
          // Ray-sphere intersection for points
          const Eigen::Vector3d& center = primitives.col(prim_idx);
          double t;
          Eigen::Vector3d normal;

          if (intersect_sphere(ray_origin, ray_dir, center, sphere_radius, t, normal)) {
            if (t >= t_min && t < hit_t) {
              hit_t = t;
              hit_idx = prim_idx;
              hit_bary = Eigen::Vector2d(0, 0); // unused for spheres
              found_hit = true;
            }
          }
        } else {
          // Ray-triangle intersection
          Eigen::Vector3d v0, v1, v2;
          get_triangle_vertices(prim_idx, v0, v1, v2);

          double t;
          Eigen::Vector2d bary;
          if (intersect_triangle(ray_origin, ray_dir, v0, v1, v2, t, bary)) {
            if (t >= t_min && t < hit_t) {
              hit_t = t;
              hit_idx = prim_idx;
              hit_bary = bary;
              found_hit = true;
            }
          }
        }
      }
    } else {
      // Internal: add children
      stack.push_back(node.left_child);
      stack.push_back(node.right_child);
    }
  }

  return {found_hit, hit_t, hit_idx, hit_bary};
}

std::tuple<Eigen::Vector3d, double, int> BVH::closest_point_on_mesh(const Eigen::Vector3d& query) const {
  if (primitive_type != BVHPrimitiveType::Triangle) {
    throw std::runtime_error("closest_point_on_mesh only works for triangle BVHs");
  }

  double min_dist = std::numeric_limits<double>::infinity();
  Eigen::Vector3d closest_point;
  int closest_tri_idx = -1;

  std::vector<int> stack;
  stack.reserve(64);
  stack.push_back(0);

  while (!stack.empty()) {
    int node_idx = stack.back();
    stack.pop_back();

    const BVHNode& node = nodes[node_idx];

    double node_dist_sq = point_aabb_distance_sq(query, node.aabb_min, node.aabb_max);
    if (node_dist_sq > min_dist * min_dist) {
      continue;
    }

    if (node.prim_count > 0) {
      // Leaf: check all triangles
      for (int i = 0; i < node.prim_count; ++i) {
        int tri_idx = primitive_indices(node.prim_start + i);
        Eigen::Vector3d v0, v1, v2;
        get_triangle_vertices(tri_idx, v0, v1, v2);

        Eigen::Vector3d pt = closest_point_on_triangle(query, v0, v1, v2);
        double dist = (query - pt).norm();

        if (dist < min_dist) {
          min_dist = dist;
          closest_point = pt;
          closest_tri_idx = tri_idx;
        }
      }
    } else {
      // Internal: add children (closer first)
      const BVHNode& left = nodes[node.left_child];
      const BVHNode& right = nodes[node.right_child];

      double left_dist_sq = point_aabb_distance_sq(query, left.aabb_min, left.aabb_max);
      double right_dist_sq = point_aabb_distance_sq(query, right.aabb_min, right.aabb_max);

      if (left_dist_sq < right_dist_sq) {
        if (right_dist_sq <= min_dist * min_dist) stack.push_back(node.right_child);
        if (left_dist_sq <= min_dist * min_dist) stack.push_back(node.left_child);
      } else {
        if (left_dist_sq <= min_dist * min_dist) stack.push_back(node.left_child);
        if (right_dist_sq <= min_dist * min_dist) stack.push_back(node.right_child);
      }
    }
  }

  return {closest_point, min_dist, closest_tri_idx};
}

std::vector<int> BVH::find_primitives_in_box(const Eigen::Vector3d& box_min, const Eigen::Vector3d& box_max) const {
  std::vector<int> result;
  if (nodes.empty()) {
    return result;
  }

  find_primitives_in_box_recursive(0, box_min, box_max, result);
  return result;
}

void BVH::find_primitives_in_box_recursive(int node_idx, const Eigen::Vector3d& box_min, const Eigen::Vector3d& box_max,
                                           std::vector<int>& result) const {
  const BVHNode& node = nodes[node_idx];

  // Early exit if node's AABB doesn't intersect query box
  if (!aabb_intersects_aabb(node.aabb_min, node.aabb_max, box_min, box_max)) {
    return;
  }

  // If leaf node, add all primitives
  if (node.left_child == -1 && node.right_child == -1) {
    for (int i = 0; i < node.prim_count; i++) {
      int prim_idx = primitive_indices[node.prim_start + i];
      result.push_back(prim_idx);
    }
    return;
  }

  // Recurse on children
  if (node.left_child != -1) {
    find_primitives_in_box_recursive(node.left_child, box_min, box_max, result);
  }
  if (node.right_child != -1) {
    find_primitives_in_box_recursive(node.right_child, box_min, box_max, result);
  }
}

bool BVH::aabb_intersects_aabb(const Eigen::Vector3d& min1, const Eigen::Vector3d& max1, const Eigen::Vector3d& min2,
                               const Eigen::Vector3d& max2) const {
  return (min1.x() <= max2.x() && max1.x() >= min2.x() && min1.y() <= max2.y() && max1.y() >= min2.y() &&
          min1.z() <= max2.z() && max1.z() >= min2.z());
}

// ============================================================================
// GPU EXPORT
// ============================================================================

std::pair<Eigen::Matrix<double, Eigen::Dynamic, 10>, Eigen::VectorXi> BVH::get_gpu_data() const {
  int n_nodes = static_cast<int>(nodes.size());

  // Each node: 10 values
  Eigen::MatrixXd node_data(n_nodes, 10);

  for (int i = 0; i < n_nodes; ++i) {
    const BVHNode& node = nodes[i];
    node_data(i, 0) = node.aabb_min.x();
    node_data(i, 1) = node.aabb_min.y();
    node_data(i, 2) = node.aabb_min.z();
    node_data(i, 3) = node.aabb_max.x();
    node_data(i, 4) = node.aabb_max.y();
    node_data(i, 5) = node.aabb_max.z();
    node_data(i, 6) = static_cast<double>(node.left_child); // TODO: I don't really like this...
    node_data(i, 7) = static_cast<double>(node.right_child);
    node_data(i, 8) = static_cast<double>(node.prim_start);
    node_data(i, 9) = static_cast<double>(node.prim_count);
  }

  return {node_data, primitive_indices};
}

// ============================================================================
// HELPER METHODS
// ============================================================================

bool BVH::intersect_aabb(const Eigen::Vector3d& ray_origin, const Eigen::Vector3d& ray_dir,
                         const Eigen::Vector3d& aabb_min, const Eigen::Vector3d& aabb_max, double& t_near,
                         double& t_far) const {
  Eigen::Vector3d inv_dir = ray_dir.cwiseInverse();
  Eigen::Vector3d t0 = (aabb_min - ray_origin).cwiseProduct(inv_dir);
  Eigen::Vector3d t1 = (aabb_max - ray_origin).cwiseProduct(inv_dir);

  Eigen::Vector3d t_min = t0.cwiseMin(t1);
  Eigen::Vector3d t_max = t0.cwiseMax(t1);

  t_near = std::max({t_min.x(), t_min.y(), t_min.z()});
  t_far = std::min({t_max.x(), t_max.y(), t_max.z()});

  return t_near <= t_far && t_far >= 0.0;
}

double BVH::point_aabb_distance_sq(const Eigen::Vector3d& point, const Eigen::Vector3d& aabb_min,
                                   const Eigen::Vector3d& aabb_max) const {
  Eigen::Vector3d closest = point.cwiseMax(aabb_min).cwiseMin(aabb_max);
  return (point - closest).squaredNorm();
}

bool BVH::intersect_sphere(const Eigen::Vector3d& ray_origin, const Eigen::Vector3d& ray_dir,
                           const Eigen::Vector3d& sphere_center, double radius, double& t,
                           Eigen::Vector3d& normal) const {

  Eigen::Vector3d rd = ray_dir.normalized();
  t = rd.dot(sphere_center - ray_origin) / rd.dot(rd);
  double d = (ray_origin + t * rd - sphere_center).norm();
  if (d > radius) return false;

  double s = std::sqrt(radius * radius - d * d);
  if (t - s >= 0.) {
    t -= s;
  } else {
    t += s;
  }
  normal = ray_origin + t * rd - sphere_center;
  normal /= normal.norm();
  return t > 0.;

  // Eigen::Vector3d oc = ray_origin - sphere_center;

  // // Normalize ray direction for proper distance calculation
  // Eigen::Vector3d dir = ray_dir.normalized();

  // double a = dir.dot(dir);
  // double b = 2.0 * oc.dot(dir);
  // double c = oc.dot(oc) - radius * radius;

  // double discriminant = b * b - 4 * a * c;

  // if (discriminant < 0) {
  //   return false;
  // }

  // double sqrt_disc = std::sqrt(discriminant);
  // double t0 = (-b - sqrt_disc) / (2.0 * a);
  // double t1 = (-b + sqrt_disc) / (2.0 * a);

  // // Use the nearest positive intersection
  // if (t0 > 1e-6) {
  //   t = t0;
  // } else if (t1 > 1e-6) {
  //   t = t1;
  // } else {
  //   return false;
  // }

  // // Compute normal at intersection point
  // Eigen::Vector3d hit_point = ray_origin + t * dir;
  // normal = (hit_point - sphere_center).normalized();

  // return true;
}

bool BVH::intersect_triangle(const Eigen::Vector3d& ray_origin, const Eigen::Vector3d& ray_dir,
                             const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, double& t,
                             Eigen::Vector2d& bary) const {
  // Möller-Trumbore algorithm
  const double EPSILON = 1e-8;

  Eigen::Vector3d e1 = v1 - v0;
  Eigen::Vector3d e2 = v2 - v0;
  Eigen::Vector3d h = ray_dir.cross(e2);
  double a = e1.dot(h);

  if (std::abs(a) < EPSILON) {
    return false;
  }

  double f = 1.0 / a;
  Eigen::Vector3d s = ray_origin - v0;
  double u = f * s.dot(h);

  if (u < 0.0 || u > 1.0) {
    return false;
  }

  Eigen::Vector3d q = s.cross(e1);
  double v = f * ray_dir.dot(q);

  if (v < 0.0 || u + v > 1.0) {
    return false;
  }

  t = f * e2.dot(q);
  bary = Eigen::Vector2d(u, v);

  return t > EPSILON;
}

Eigen::Vector3d BVH::closest_point_on_triangle(const Eigen::Vector3d& query, const Eigen::Vector3d& v0,
                                               const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) const {
  Eigen::Vector3d e0 = v1 - v0;
  Eigen::Vector3d e1 = v2 - v1;
  Eigen::Vector3d e2 = v0 - v2;

  Eigen::Vector3d n = e0.cross(-e2);
  double n_len = n.norm();
  if (n_len < 1e-10) {
    return v0; // degenerate
  }
  n /= n_len;

  // Project onto plane
  Eigen::Vector3d v0p = query - v0;
  double dist_to_plane = v0p.dot(n);
  Eigen::Vector3d proj = query - dist_to_plane * n;

  // Check if inside using barycentric coordinates
  Eigen::Vector3d c0 = (v1 - v0).cross(proj - v0);
  Eigen::Vector3d c1 = (v2 - v1).cross(proj - v1);
  Eigen::Vector3d c2 = (v0 - v2).cross(proj - v2);

  bool inside = c0.dot(n) >= 0.0 && c1.dot(n) >= 0.0 && c2.dot(n) >= 0.0;

  if (inside) {
    return proj;
  }

  // Find closest point on edges
  Eigen::Vector3d closest = v0;
  double min_dist_sq = (query - v0).squaredNorm();

  auto check_edge = [&](const Eigen::Vector3d& va, const Eigen::Vector3d& vb) {
    Eigen::Vector3d edge = vb - va;
    double t = (query - va).dot(edge) / edge.squaredNorm();
    t = std::clamp(t, 0.0, 1.0);
    Eigen::Vector3d pt = va + t * edge;
    double dist_sq = (query - pt).squaredNorm();
    if (dist_sq < min_dist_sq) {
      min_dist_sq = dist_sq;
      closest = pt;
    }
  };

  check_edge(v0, v1);
  check_edge(v1, v2);
  check_edge(v2, v0);

  return closest;
}

void BVH::get_triangle_vertices(int tri_idx, Eigen::Vector3d& v0, Eigen::Vector3d& v1, Eigen::Vector3d& v2) const {
  v0 = primitives.col(3 * tri_idx);
  v1 = primitives.col(3 * tri_idx + 1);
  v2 = primitives.col(3 * tri_idx + 2);
}

// ============================================================================
// BINDING CODE
// ============================================================================

void bind_bvh(nb::module_& m) {
  nb::enum_<BVHPrimitiveType>(m, "BVHPrimitiveType")
      .value("Point", BVHPrimitiveType::Point)
      .value("Triangle", BVHPrimitiveType::Triangle);

  nb::class_<BVH>(m, "BVH")

      .def(nb::init<const Eigen::Matrix<double, 3, Eigen::Dynamic>&>(), "points"_a, "Construct BVH from points")
      .def(nb::init<const Eigen::Matrix<double, 3, Eigen::Dynamic>&, const Eigen::Matrix<int, 3, Eigen::Dynamic>&>(),
           "vertices"_a, "faces"_a, "Construct BVH from triangles")

      .def("find_closest_point", &BVH::find_closest_point, "query"_a,
           "Find closest point in point cloud. Returns (index, distance)")
      .def("find_k_nearest", &BVH::find_k_nearest, "query"_a, "k"_a,
           "Find k nearest neighbors. Returns (indices, distances)")
      .def("find_points_in_radius", &BVH::find_points_in_radius, "query"_a, "radius"_a,
           "Find all points within radius. Returns (indices, distances)")
      .def("intersect_ray", &BVH::intersect_ray, "ray_origin"_a, "ray_dir"_a, "t_min"_a = 0.0,
           "t_max"_a = std::numeric_limits<double>::infinity(), "sphere_radius"_a,
           "Ray-mesh intersection. Returns (hit, t, triangle_idx, barycentric)")
      .def("closest_point_on_mesh", &BVH::closest_point_on_mesh, "query"_a,
           "Closest point on triangle mesh. Returns (point, distance, triangle_idx)")

      .def("get_gpu_data", &BVH::get_gpu_data, "Get flattened BVH data for GPU upload. Returns (node_data, indices)")
      .def("get_num_nodes", &BVH::get_num_nodes, "Get number of BVH nodes")
      .def("get_num_primitives", &BVH::get_num_primitives, "Get number of primitives")
      .def("get_primitive_type", &BVH::get_primitive_type, "Get primitive type");
}