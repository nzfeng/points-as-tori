#pragma once

#include <Eigen/Core>
#include <fcpw/fcpw.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "bvh.h"
#include "shape_2d.h"

#include "igl/fast_winding_number.h"

#include <chrono>
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// ============================================================================
// SAMPLING
// ============================================================================

Eigen::Matrix<double, 3, Eigen::Dynamic> sample_bounding_box(int n_points, const Eigen::Vector3d& bbox_min,
                                                             const Eigen::Vector3d& bbox_max, int seed);

// ============================================================================
// POINT CLOUD PROCESSING
// ============================================================================

std::pair<Eigen::Vector3d, Eigen::Vector3d>
point_cloud_bounding_box(const Eigen::Matrix<double, 3, Eigen::Dynamic>& positions);

Eigen::Matrix<double, 3, Eigen::Dynamic> sample_bounding_box(int n_points, const Eigen::Vector3d& bbox_min,
                                                             const Eigen::Vector3d& bbox_max, int seed);

// ============================================================================
// MESH
// ============================================================================

class Mesh3D {
  public:
    Mesh3D(const Eigen::Matrix<double, 3, Eigen::Dynamic>& vertices, const Eigen::Matrix<int, 3, Eigen::Dynamic>& faces)
        : vertices_(vertices), faces_(faces) {

        mesh_bvh.reset(new BVH(vertices_, faces_));
        build_FCPW_scene();

        auto [V_clean, F_clean] = remove_degenerate_faces(vertices_, faces_);
        cleaned_vertices = V_clean;
        cleaned_faces = F_clean;
    }

    // Center and scale mesh to [-1,1]^3 box
    std::pair<Eigen::Vector3d, Eigen::Vector3d> bounding_box() const;
    Mesh3D center_and_scale() const;

    Eigen::Matrix<double, 3, Eigen::Dynamic> get_vertices() const;

    Eigen::Matrix<int, 3, Eigen::Dynamic> get_faces() const;

    // Ray intersections using FCPW - returns (points, normals) for hits
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    ray_intersections(const Eigen::Matrix<double, 3, Eigen::Dynamic>& ros,
                      const Eigen::Matrix<double, 3, Eigen::Dynamic>& rds) const;

    // Sample uniformly
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    sample_points_and_normals_uniformly(int n_points, unsigned int seed, const Eigen::Vector3d& bbox_min = {1, 1, 1},
                                        const Eigen::Vector3d& bbox_max = {-1, -1, -1}, const double& sigma_p = 0.0,
                                        const double& sigma_n = 0.0) const;

    Eigen::Matrix<double, 3, Eigen::Dynamic> sample_uniformly(int n_points, unsigned int seed = 0,
                                                              const Eigen::Vector3d& bbox_min = {1, 1, 1},
                                                              const Eigen::Vector3d& bbox_max = {-1, -1, -1}) const;

    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    sample_points_and_normals_uniformly_by_casting_rays(int n_points, unsigned int seed,
                                                        const Eigen::Vector3d& bbox_min,
                                                        const Eigen::Vector3d& bbox_max, const double& sigma_p = 0.0,
                                                        const double& sigma_n = 0.0) const;

    // Point cloud generation
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    sample_uniform_point_cloud(int n_points, const double& max_noise_level, const double& max_normals_to_flip,
                               int seed) const;

    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    sample_farthest_point_cloud(int n_points, const double& max_noise_level, const double& max_normals_to_flip,
                                int seed) const;

    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    sample_even_raycasted_point_cloud(int n_cameras, const int& n_points, const double& sensor_size,
                                      const double& grid_spacing_min, const double& grid_spacing_max, int seed) const;

    // Generate a point cloud by sampling the positions of orthographic cameras on a bounding sphere, and shooting rays.
    // Each camera is assumed to shoot rays arranged in a square grid, with grid spacing controlled by `grid_spacing`.
    // The size of the grid(side length of the square, `sensor_size`) is determined as a percentage of the object's
    // diameter.
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    sample_point_cloud(int n_cameras, const double& sensor_size, const double& grid_spacing, unsigned int seed,
                       const double& sigma_p = 0.0, const double& sigma_n = 0.0) const;
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    sample_uneven_point_cloud(int n_cameras, const double& sensor_size, const double& grid_spacing_min,
                              const double& grid_spacing_max, const double& max_noise_level,
                              const double& max_normals_to_flip, const int& max_points, unsigned int seed) const;

    // Sample narrow band around boundary
    Eigen::Matrix<double, 3, Eigen::Dynamic> sample_narrow_band(int n_points, double offset = 0.1,
                                                                unsigned int seed = 0,
                                                                const Eigen::Vector3d& bbox_min = {1, 1, 1},
                                                                const Eigen::Vector3d& bbox_max = {-1, -1, -1}) const;

    // Evaluate signed distance (using FCPW for closest point queries)
    double evaluate_signed_distance_single(const Eigen::Vector3d& query) const;
    Eigen::VectorXd evaluate_signed_distance(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                             bool parallelize = true) const;

    // Evaluate signed distance and gradient
    std::pair<Eigen::VectorXd, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    evaluate_signed_distance_and_gradient(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries) const;

    // Evaluate generalized winding number
    Eigen::VectorXd evaluate_gwn(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries) const;
    Eigen::VectorXd evaluate_fast_gwn(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries) const;

  private:
    Eigen::Matrix<double, 3, Eigen::Dynamic> vertices_;
    Eigen::Matrix<int, 3, Eigen::Dynamic> faces_;

    Eigen::Matrix<double, Eigen::Dynamic, 3> cleaned_vertices;
    Eigen::Matrix<int, Eigen::Dynamic, 3> cleaned_faces;

    std::unique_ptr<BVH> mesh_bvh;
    fcpw::Scene<3> scene;
    void build_FCPW_scene();

    double evaluate_gwn_single(const Eigen::Vector3d& query) const;
    std::pair<Eigen::MatrixXd, Eigen::MatrixXi>
    remove_degenerate_faces(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double area_threshold = 1e-10) const;
};
