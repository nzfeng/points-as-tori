#pragma once

#include <Eigen/Dense>
#include <limits>
#include <memory>
#include <vector>

/**
 * Bounding Volume Hierarchy (BVH) for accelerating spatial queries.
 *
 * Supports point clouds and triangle meshes:
 * - Closest point queries
 * - K-nearest neighbor queries
 * - Radius queries (all points/triangles within distance)
 * - Ray intersection queries
 *
 * The BVH is later linearized into a flattened array structure for GPU upload via textures
 * (so radius queries can be done in the shader)
 */

enum class BVHPrimitiveType { Point, Triangle };

struct BVHNode {
  Eigen::Vector3d aabb_min;
  Eigen::Vector3d aabb_max;
  int left_child;  // -1 if leaf
  int right_child; // -1 if leaf
  int prim_start;  // start index in primitive array (for leaves)
  int prim_count;  // number of primitives (0 if internal node)
};

class BVH {
public:
  // Construct BVH from point cloud.
  BVH(const Eigen::Matrix<double, 3, Eigen::Dynamic>& points);

  // Construct BVH from triangle mesh
  BVH(const Eigen::Matrix<double, 3, Eigen::Dynamic>& vertices, const Eigen::Matrix<int, 3, Eigen::Dynamic>& faces);

  // ===== QUERIES

  // Return (index of closest point, distance)
  std::pair<int, double> find_closest_point(const Eigen::Vector3d& query) const;

  // Return indices and distances of neighbors
  std::pair<Eigen::VectorXi, Eigen::VectorXd> find_k_nearest(const Eigen::Vector3d& query, int k) const;

  // Return indices and distances of points within radius
  std::pair<Eigen::VectorXi, Eigen::VectorXd> find_points_in_radius(const Eigen::Vector3d& query, double radius) const;

  // Return (hit, t, triangle_idx, barycentric_coords)
  std::tuple<bool, double, int, Eigen::Vector2d> intersect_ray(const Eigen::Vector3d& ray_origin,
                                                               const Eigen::Vector3d& ray_dir, double t_min = 0.0,
                                                               double t_max = std::numeric_limits<double>::infinity(),
                                                               double sphere_radius = 0.01) const;

  // Return (closest_point, distance, triangle_idx)
  std::tuple<Eigen::Vector3d, double, int> closest_point_on_mesh(const Eigen::Vector3d& query) const;

  /**
   * Find all primitives (points or triangles) whose bounding boxes intersect
   * the given axis-aligned bounding box.
   *
   * Returns indices of primitives that potentially intersect the query box.
   * For point clouds: returns points inside or near the box
   * For triangle meshes: returns triangles whose AABBs intersect the box
   */
  std::vector<int> find_primitives_in_box(const Eigen::Vector3d& box_min, const Eigen::Vector3d& box_max) const;

  // ===== GPU export

  /*
   * Get flattened BVH data for GPU upload.
   * Return (node_data, primitive_indices)
   *
   * node_data: (n_nodes, 10) matrix where each row is:
   *   [aabb_min.x, aabb_min.y, aabb_min.z, aabb_max.x, aabb_max.y, aabb_max.z,
   *    left_child, right_child, prim_start, prim_count]
   *
   * primitive_indices: (n_primitives,) reordered primitive indices
   */
  std::pair<Eigen::Matrix<double, Eigen::Dynamic, 10>, Eigen::VectorXi> get_gpu_data() const;

  // ===== Getters

  int get_num_nodes() const { return static_cast<int>(nodes.size()); }
  int get_num_primitives() const { return n_primitives; }
  BVHPrimitiveType get_primitive_type() const { return primitive_type; }
  const Eigen::Matrix<double, 3, Eigen::Dynamic>& get_primitives() const { return primitives; }

private:
  BVHPrimitiveType primitive_type;
  Eigen::Matrix<double, 3, Eigen::Dynamic> primitives; // (3, n_points) or (3, 3*n_tris)
  Eigen::Matrix<double, 3, Eigen::Dynamic> centroids;  // (3, n_primitives) centroid of each primitive
  int n_primitives;

  std::vector<BVHNode> nodes;
  Eigen::VectorXi primitive_indices;

  // ===== Construction

  int build_recursive(int start, int end);

  // ===== Helpers

  void compute_aabb(int start, int end, Eigen::Vector3d& aabb_min, Eigen::Vector3d& aabb_max) const;
  void compute_centroid_aabb(int start, int end, Eigen::Vector3d& centroid_min, Eigen::Vector3d& centroid_max) const;

  double surface_area(const Eigen::Vector3d& aabb_min, const Eigen::Vector3d& aabb_max) const;

  std::pair<int, int> find_best_split(int start, int end, const Eigen::Vector3d& centroid_min,
                                      const Eigen::Vector3d& centroid_max) const;

  int partition(int start, int end, int axis, double split_value);

  // ===== Query helpers

  bool intersect_aabb(const Eigen::Vector3d& ray_origin, const Eigen::Vector3d& ray_dir,
                      const Eigen::Vector3d& aabb_min, const Eigen::Vector3d& aabb_max, double& t_near,
                      double& t_far) const;

  double point_aabb_distance_sq(const Eigen::Vector3d& point, const Eigen::Vector3d& aabb_min,
                                const Eigen::Vector3d& aabb_max) const;

  bool intersect_sphere(const Eigen::Vector3d& ray_origin, const Eigen::Vector3d& ray_dir,
                        const Eigen::Vector3d& sphere_center, double radius, double& t, Eigen::Vector3d& normal) const;

  bool intersect_triangle(const Eigen::Vector3d& ray_origin, const Eigen::Vector3d& ray_dir, const Eigen::Vector3d& v0,
                          const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, double& t, Eigen::Vector2d& bary) const;

  Eigen::Vector3d closest_point_on_triangle(const Eigen::Vector3d& query, const Eigen::Vector3d& v0,
                                            const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) const;

  void get_triangle_vertices(int tri_idx, Eigen::Vector3d& v0, Eigen::Vector3d& v1, Eigen::Vector3d& v2) const;

  void find_primitives_in_box_recursive(int node_idx, const Eigen::Vector3d& box_min, const Eigen::Vector3d& box_max,
                                        std::vector<int>& result) const;

  bool aabb_intersects_aabb(const Eigen::Vector3d& min1, const Eigen::Vector3d& max1, const Eigen::Vector3d& min2,
                            const Eigen::Vector3d& max2) const;
};
