#include "shape_3d.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;
using namespace nb::literals;

// ============================================================================
// POINT CLOUD PROCESSING
// ============================================================================

std::pair<Eigen::Vector3d, Eigen::Vector3d>
point_cloud_bounding_box(const Eigen::Matrix<double, 3, Eigen::Dynamic>& positions) {

    const double inf = std::numeric_limits<double>::infinity();
    Eigen::Vector3d bbox_min = {inf, inf, inf};
    Eigen::Vector3d bbox_max = {-inf, -inf, -inf};
    int n_points = positions.cols();
#pragma omp parallel
    {
        Eigen::Vector3d local_min = {inf, inf, inf};
        Eigen::Vector3d local_max = {-inf, -inf, -inf};

#pragma omp for schedule(static) nowait
        for (int i = 0; i < n_points; i++) {
            for (int j = 0; j < 3; j++) {
                local_min[j] = std::min(local_min[j], positions(j, i));
                local_max[j] = std::max(local_max[j], positions(j, i));
            }
        }

#pragma omp critical
        {
            for (int j = 0; j < 3; j++) {
                bbox_min[j] = std::min(bbox_min[j], local_min[j]);
                bbox_max[j] = std::max(bbox_max[j], local_max[j]);
            }
        }
    }
    return {bbox_min, bbox_max};
}

// ============================================================================
// MESH
// ============================================================================

Mesh3D Mesh3D::center_and_scale() const {
    Eigen::Vector3d centroid = vertices_.rowwise().mean();
    Eigen::Matrix<double, 3, Eigen::Dynamic> centered = vertices_.colwise() - centroid;
    double max_extent = centered.cwiseAbs().maxCoeff();
    double scale_factor = (max_extent > 1e-10) ? (1.0 / max_extent) : 1.0;
    Eigen::Matrix<double, 3, Eigen::Dynamic> scaled = centered * scale_factor;
    return Mesh3D(scaled, faces_);
}

Eigen::Matrix<double, 3, Eigen::Dynamic> Mesh3D::get_vertices() const {
    return vertices_;
}

Eigen::Matrix<int, 3, Eigen::Dynamic> Mesh3D::get_faces() const {
    return faces_;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> Mesh3D::bounding_box() const {
    const double inf = std::numeric_limits<double>::infinity();
    Eigen::Vector3d bboxMin = {inf, inf, inf};
    Eigen::Vector3d bboxMax = {-inf, -inf, -inf};
    size_t V = vertices_.cols();
    for (size_t vIdx = 0; vIdx < V; vIdx++) {
        const Eigen::Vector3d& pos = vertices_.col(vIdx);
        for (int i = 0; i < 3; i++) {
            bboxMin(i) = std::min(pos(i), bboxMin(i));
            bboxMax(i) = std::max(pos(i), bboxMax(i));
        }
    }
    return std::make_pair(bboxMin, bboxMax);
}

std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
Mesh3D::ray_intersections(const Eigen::Matrix<double, 3, Eigen::Dynamic>& ros,
                          const Eigen::Matrix<double, 3, Eigen::Dynamic>& rds) const {

    // Use FCPW
    std::pair<Eigen::Vector3d, Eigen::Vector3d> bbox = bounding_box();
    int n_rays = ros.cols();
    double diameter = (bbox.first - bbox.second).norm();
    std::vector<fcpw::Ray<3>> rays;
    std::vector<fcpw::Interaction<3>> interactions;

    for (int i = 0; i < n_rays; ++i) {
        rays.emplace_back(ros.col(i).cast<float>(), rds.col(i).cast<float>(), std::numeric_limits<float>::infinity());
    }

    scene.intersect(rays, interactions, false);

    // Extract valid hits
    std::vector<Eigen::Vector3d> points, normals;
    for (int i = 0; i < interactions.size(); ++i) {
        if (interactions[i].primitiveIndex != -1) {
            points.push_back(Eigen::Vector3d(interactions[i].p[0], interactions[i].p[1], interactions[i].p[2]));
            // Make sure the sign of the normal aligns with the camera ray (relevant for inconsistently-oriented meshes)
            Eigen::Vector3d n(interactions[i].n[0], interactions[i].n[1], interactions[i].n[2]);
            Eigen::Vector3d d(rays[i].d(0), rays[i].d(1), rays[i].d(2));
            double s = n.dot(d) < 0.0 ? 1.0 : -1.0;
            n *= s;
            normals.push_back(n);
        }
    }

    // Convert to matrices
    Eigen::Matrix<double, 3, Eigen::Dynamic> points_mat(3, points.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> normals_mat(3, normals.size());
    const Eigen::Index n = static_cast<Eigen::Index>(points.size());
#pragma omp parallel for schedule(static)
    for (Eigen::Index i = 0; i < n; ++i) {
        points_mat.col(i) = points[i];
        normals_mat.col(i) = normals[i];
    }

    return {points_mat, normals_mat};
}

std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
Mesh3D::sample_uniform_point_cloud(int n_points, const double& max_noise_level, const double& max_normals_to_flip,
                                   int seed) const {

    // Sample noise parameters
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> sigma_p_dist(0.0, max_noise_level);
    std::uniform_real_distribution<double> sigma_n_dist(0.0, max_noise_level);
    std::uniform_real_distribution<double> flipping_dist(0.0, max_normals_to_flip);
    double sigma_p = sigma_p_dist(rng);
    double sigma_n = sigma_n_dist(rng);

    Eigen::Vector3d bbox_min = {1, 1, 1};
    Eigen::Vector3d bbox_max = {-1, -1, -1};
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> result =
        sample_points_and_normals_uniformly(n_points, seed, bbox_min, bbox_max, sigma_p, sigma_n);

    // Flip normals
    double normals_to_flip = flipping_dist(rng);
    int n_normals_to_flip = int(normals_to_flip * n_points);
    std::cerr << "\t Flipping " << n_normals_to_flip << " normals" << std::endl;

    std::vector<int> indices(n_points);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    for (int i = 0; i < n_normals_to_flip; i++) {
        result.second.col(indices[i]) *= -1;
    }

    return result;
}


std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
Mesh3D::sample_farthest_point_cloud(int n_points, const double& max_noise_level, const double& max_normals_to_flip,
                                    int seed) const {

    // Sample noise parameters
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> sigma_p_dist(0.0, max_noise_level);
    std::uniform_real_distribution<double> sigma_n_dist(0.0, max_noise_level);
    std::uniform_real_distribution<double> flipping_dist(0.0, max_normals_to_flip);
    double sigma_p = sigma_p_dist(rng);
    double sigma_n = sigma_n_dist(rng);

    // Sample way more points then we need, then subsample down using farthest point sampling.
    Eigen::Vector3d bbox_min = {1, 1, 1};
    Eigen::Vector3d bbox_max = {-1, -1, -1};
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> result =
        sample_points_and_normals_uniformly(n_points * 4, seed, bbox_min, bbox_max, sigma_p, sigma_n);

    // Subsample
    Eigen::VectorXi indices = subsample_point_cloud(result.first, n_points);
    Eigen::MatrixXd samples(3, n_points);
    Eigen::MatrixXd normals(3, n_points);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_points; i++) {
        samples.col(i) = result.first.col(indices(i));
        normals.col(i) = result.second.col(indices(i));
    }

    // Flip normals
    double normals_to_flip = flipping_dist(rng);
    int n_normals_to_flip = int(normals_to_flip * n_points);
    std::cerr << "\t Flipping " << n_normals_to_flip << " normals" << std::endl;

    std::vector<int> indices_to_flip(n_points);
    std::iota(indices_to_flip.begin(), indices_to_flip.end(), 0);
    std::shuffle(indices_to_flip.begin(), indices_to_flip.end(), rng);
    for (int i = 0; i < n_normals_to_flip; i++) {
        normals.col(indices_to_flip[i]) *= -1;
    }

    return {samples, normals};
}

std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
Mesh3D::sample_even_raycasted_point_cloud(int n_cameras, const int& max_points, const double& sensor_size,
                                          const double& grid_spacing_min, const double& grid_spacing_max,
                                          int seed) const {

    std::mt19937 rng(seed);

    // Get bounding sphere.
    Eigen::Vector3d centroid = vertices_.rowwise().mean();
    std::pair<Eigen::Vector3d, Eigen::Vector3d> bbox = bounding_box();
    // double diameter = (bbox.first - bbox.second).norm();
    double diameter = 2.0 * std::sqrt(3.0); // use bounding sphere of unit cube
    double radius = 0.5 * diameter;

    // Generate camera positions on sphere (looking inward)
    Eigen::Matrix<double, 3, Eigen::Dynamic> camera_positions = sample_sphere_positions(n_cameras, rng);
    Eigen::Matrix<double, 3, Eigen::Dynamic> camera_directions = -camera_positions;

    // Compute plane of camera sensor
    Eigen::Matrix<double, 3, Eigen::Dynamic> s_axes(3, n_cameras);
    Eigen::Matrix<double, 3, Eigen::Dynamic> t_axes(3, n_cameras);
    for (int i = 0; i < n_cameras; i++) {
        Eigen::Vector3d d = camera_directions.col(i);
        std::pair<Eigen::Vector3d, Eigen::Vector3d> basis = orthonormal_basis_random(d, seed);
        s_axes.col(i) = basis.first;
        t_axes.col(i) = basis.second;
    }

    // Starting positions (offset backward along view direction)
    Eigen::Matrix<double, 3, Eigen::Dynamic> start_positions = camera_positions * 0.5 * diameter;
    for (int i = 0; i < n_cameras; i++) start_positions.col(i) += centroid;

    // Generate all ray origins and directions.
    double sensor_side = sensor_size * diameter;
    std::uniform_real_distribution<double> grid_spacing_dist(grid_spacing_min, grid_spacing_max);
    std::vector<Eigen::Vector3d> origins;
    std::vector<Eigen::Vector3d> directions;
    for (int cam = 0; cam < n_cameras; ++cam) {
        double grid_spacing = grid_spacing_dist(rng);
        int n_samples_per_row = static_cast<int>(std::ceil(sensor_side / grid_spacing));
        int n_samples = n_samples_per_row * n_samples_per_row;
        double half_grid_length = 0.5 * (n_samples_per_row - 1) * grid_spacing;
        // Create 2D grid for each camera
        std::vector<Eigen::Vector2d> grid_coords(n_samples);
        for (int i = 0; i < n_samples_per_row; i++) {
            double x = (i + 0.5) * grid_spacing - half_grid_length;
            for (int j = 0; j < n_samples_per_row; j++) {
                double y = (j + 0.5) * grid_spacing - half_grid_length;
                grid_coords[i * n_samples_per_row + j] = {x, y};
            }
        }

        // Now determine ray parameters
        for (int s = 0; s < n_samples; ++s) {
            Eigen::Vector2d grid_coord = grid_coords[s];
            Eigen::Vector3d grid_position = grid_coord[0] * s_axes.col(cam) + grid_coord[1] * t_axes.col(cam);
            origins.push_back(start_positions.col(cam) + grid_position);
            directions.push_back(camera_directions.col(cam));
        }
    }

    int n_rays = origins.size();
    Eigen::Matrix<double, 3, Eigen::Dynamic> ray_origins(3, n_rays);
    Eigen::Matrix<double, 3, Eigen::Dynamic> ray_directions(3, n_rays);
    for (int i = 0; i < n_rays; i++) {
        ray_origins.col(i) = origins[i];
        ray_directions.col(i) = directions[i];
    }

    // Cast all rays.
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> intersections =
        ray_intersections(ray_origins, ray_directions);

    int n_intersections = intersections.first.cols();
    int n_final = std::min(n_intersections, max_points);
    std::uniform_real_distribution<double> subsample_method(0.0, 1.0);
    Eigen::VectorXi indices(n_final);
    if (subsample_method(rng) < 0.5) {
        // Subsample using farthest point sampling
        indices = subsample_point_cloud(intersections.first, n_final);
    } else {
        // Subsample using uniform sampling
        std::vector<int> indices_vec(n_intersections);
        std::iota(indices_vec.begin(), indices_vec.end(), 0);
        std::shuffle(indices_vec.begin(), indices_vec.end(), rng);
        for (int i = 0; i < n_final; i++) indices(i) = indices_vec[i];
    }

    Eigen::MatrixXd samples(3, n_final);
    Eigen::MatrixXd normals(3, n_final);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_final; i++) {
        samples.col(i) = intersections.first.col(indices(i));
        normals.col(i) = intersections.second.col(indices(i));
    }

    return {samples, normals};
}


std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
Mesh3D::sample_uneven_point_cloud(int n_cameras, const double& sensor_size, const double& grid_spacing_min,
                                  const double& grid_spacing_max, const double& max_noise_level,
                                  const double& max_normals_to_flip, const int& max_points, unsigned int seed) const {

    std::mt19937 rng(seed);

    // Get bounding sphere.
    Eigen::Vector3d centroid = vertices_.rowwise().mean();
    std::pair<Eigen::Vector3d, Eigen::Vector3d> bbox = bounding_box();
    double diameter = 2.0 * std::sqrt(3.0); // use bounding sphere of unit cube
    double radius = 0.5 * diameter;

    // Generate camera positions on sphere (looking inward)
    Eigen::Matrix<double, 3, Eigen::Dynamic> camera_positions = sample_sphere_positions(n_cameras, rng);
    Eigen::Matrix<double, 3, Eigen::Dynamic> camera_directions = -camera_positions;

    // Compute plane of camera sensor
    Eigen::Matrix<double, 3, Eigen::Dynamic> s_axes(3, n_cameras);
    Eigen::Matrix<double, 3, Eigen::Dynamic> t_axes(3, n_cameras);
    for (int i = 0; i < n_cameras; i++) {
        Eigen::Vector3d d = camera_directions.col(i);
        std::pair<Eigen::Vector3d, Eigen::Vector3d> basis = orthonormal_basis_random(d, seed);
        s_axes.col(i) = basis.first;
        t_axes.col(i) = basis.second;
    }

    // Starting positions (offset backward along view direction)
    Eigen::Matrix<double, 3, Eigen::Dynamic> start_positions = camera_positions * 0.5 * diameter;
    for (int i = 0; i < n_cameras; i++) start_positions.col(i) += centroid;

    // Generate all ray origins and directions.
    double sensor_side = sensor_size * diameter;
    std::uniform_real_distribution<double> grid_spacing_dist(grid_spacing_min, grid_spacing_max);
    std::vector<Eigen::Vector3d> origins;
    std::vector<Eigen::Vector3d> directions;
    for (int cam = 0; cam < n_cameras; ++cam) {
        double grid_spacing = grid_spacing_dist(rng);
        int n_samples_per_row = static_cast<int>(std::ceil(sensor_side / grid_spacing));
        int n_samples = n_samples_per_row * n_samples_per_row;
        double half_grid_length = 0.5 * (n_samples_per_row - 1) * grid_spacing;
        // Create 2D grid for each camera
        std::vector<Eigen::Vector2d> grid_coords(n_samples);
        for (int i = 0; i < n_samples_per_row; i++) {
            double x = (i + 0.5) * grid_spacing - half_grid_length;
            for (int j = 0; j < n_samples_per_row; j++) {
                double y = (j + 0.5) * grid_spacing - half_grid_length;
                grid_coords[i * n_samples_per_row + j] = {x, y};
            }
        }

        // Now determine ray parameters
        for (int s = 0; s < n_samples; ++s) {
            Eigen::Vector2d grid_coord = grid_coords[s];
            Eigen::Vector3d grid_position = grid_coord[0] * s_axes.col(cam) + grid_coord[1] * t_axes.col(cam);
            origins.push_back(start_positions.col(cam) + grid_position);
            directions.push_back(camera_directions.col(cam));
        }
    }

    int n_rays = origins.size();
    Eigen::Matrix<double, 3, Eigen::Dynamic> ray_origins(3, n_rays);
    Eigen::Matrix<double, 3, Eigen::Dynamic> ray_directions(3, n_rays);
    for (int i = 0; i < n_rays; i++) {
        ray_origins.col(i) = origins[i];
        ray_directions.col(i) = directions[i];
    }

    // Cast all rays.
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> intersections =
        ray_intersections(ray_origins, ray_directions);

    int n_intersections = intersections.first.cols();
    std::cerr << "\tOriginal # intersections: " << n_intersections << std::endl;

    // Uniformly subsample down to target size
    int n_final = std::min(n_intersections, max_points);
    std::vector<int> indices(n_intersections);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    Eigen::MatrixXd int_positions(3, n_final);
    Eigen::MatrixXd int_normals(3, n_final);

    // Add noise.
    Eigen::Matrix<double, 3, Eigen::Dynamic>& samples = intersections.first;
    Eigen::Matrix<double, 3, Eigen::Dynamic>& normals = intersections.second;
    std::uniform_real_distribution<double> sigma_p_dist(0.0, max_noise_level);
    std::uniform_real_distribution<double> sigma_n_dist(0.0, max_noise_level);
    std::uniform_real_distribution<double> flipping_dist(0.0, max_normals_to_flip);
    double sigma_p = sigma_p_dist(rng);
    double sigma_n = sigma_n_dist(rng);
    double normals_to_flip = flipping_dist(rng);
    int n_normals_to_flip = int(normals_to_flip * n_final);
    std::cerr << "\t Flipping " << n_normals_to_flip << " normals" << std::endl;

    std::normal_distribution position_perturbation{0.0, sigma_p};
    std::normal_distribution normal_perturbation{0.0, sigma_n};

    // rng not thread-safe
    for (int i = 0; i < n_final; i++) {
        Eigen::Vector3d position_offset(position_perturbation(rng), position_perturbation(rng),
                                        position_perturbation(rng));
        Eigen::Vector3d normal_offset(normal_perturbation(rng), normal_perturbation(rng), normal_perturbation(rng));
        int_positions.col(i) = samples.col(indices[i]) + position_offset;
        Eigen::Vector3d perturbed_normal = (normals.col(indices[i]) + normal_offset).normalized();
        int_normals.col(i) = perturbed_normal;
    }

    // Flip  normals
    std::vector<int> indices_to_flip(n_final);
    std::iota(indices_to_flip.begin(), indices_to_flip.end(), 0);
    std::shuffle(indices_to_flip.begin(), indices_to_flip.end(), rng);
    for (int i = 0; i < n_normals_to_flip; i++) {
        int_normals.col(indices_to_flip[i]) *= -1;
    }

    return {int_positions, int_normals};
}

std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
Mesh3D::sample_point_cloud(int n_cameras, const double& sensor_size, const double& grid_spacing, unsigned int seed,
                           const double& sigma_p, const double& sigma_n) const {

    std::mt19937 rng(seed);

    // Get bounding sphere.
    Eigen::Vector3d centroid = vertices_.rowwise().mean();
    std::pair<Eigen::Vector3d, Eigen::Vector3d> bbox = bounding_box();
    // double diameter = (bbox.first - bbox.second).norm();
    double diameter = 2.0 * std::sqrt(3.0); // use bounding sphere of unit cube
    double radius = 0.5 * diameter;

    // Generate camera positions on sphere (looking inward)
    Eigen::Matrix<double, 3, Eigen::Dynamic> camera_positions = sample_sphere_positions(n_cameras, rng);
    Eigen::Matrix<double, 3, Eigen::Dynamic> camera_directions = -camera_positions;

    // Compute plane of camera sensor
    Eigen::Matrix<double, 3, Eigen::Dynamic> s_axes(3, n_cameras);
    Eigen::Matrix<double, 3, Eigen::Dynamic> t_axes(3, n_cameras);
    for (int i = 0; i < n_cameras; i++) {
        Eigen::Vector3d d = camera_directions.col(i);
        std::pair<Eigen::Vector3d, Eigen::Vector3d> basis = orthonormal_basis_random(d, seed);
        s_axes.col(i) = basis.first;
        t_axes.col(i) = basis.second;
    }

    // Create 2D grid for each camera
    double sensor_side = sensor_size * diameter;
    int n_samples_per_row = static_cast<int>(std::ceil(sensor_side / grid_spacing));
    int n_samples = n_samples_per_row * n_samples_per_row;
    double half_grid_length = 0.5 * (n_samples_per_row - 1) * grid_spacing;
    std::vector<Eigen::Vector2d> grid_coords(n_samples);
    for (int i = 0; i < n_samples_per_row; i++) {
        double x = (i + 0.5) * grid_spacing - half_grid_length;
        for (int j = 0; j < n_samples_per_row; j++) {
            double y = (j + 0.5) * grid_spacing - half_grid_length;
            grid_coords[i * n_samples_per_row + j] = {x, y};
        }
    }

    // Starting positions (offset backward along view direction)
    Eigen::Matrix<double, 3, Eigen::Dynamic> start_positions = camera_positions * 0.5 * diameter;
    for (int i = 0; i < n_cameras; i++) start_positions.col(i) += centroid;

    // Generate all ray origins and directions.
    int n_rays = n_cameras * n_samples;
    Eigen::Matrix<double, 3, Eigen::Dynamic> ray_origins(3, n_rays);
    Eigen::Matrix<double, 3, Eigen::Dynamic> ray_directions(3, n_rays);
    int ray_idx = 0;
    for (int cam = 0; cam < n_cameras; ++cam) {
        for (int s = 0; s < n_samples; ++s) {
            Eigen::Vector2d grid_coord = grid_coords[s];
            Eigen::Vector3d grid_position = grid_coord[0] * s_axes.col(cam) + grid_coord[1] * t_axes.col(cam);
            ray_origins.col(ray_idx) = start_positions.col(cam) + grid_position;
            ray_directions.col(ray_idx) = camera_directions.col(cam);
            ++ray_idx;
        }
    }

    // Cast all rays.
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> intersections =
        ray_intersections(ray_origins, ray_directions);

    int n_intersections = intersections.first.cols();

    // Add noise.
    Eigen::Matrix<double, 3, Eigen::Dynamic>& samples = intersections.first;
    Eigen::Matrix<double, 3, Eigen::Dynamic>& normals = intersections.second;
    std::normal_distribution position_perturbation{0.0, sigma_p};
    std::normal_distribution normal_perturbation{0.0, sigma_n};

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_intersections; i++) {
        Eigen::Vector3d position_offset(position_perturbation(rng), position_perturbation(rng),
                                        position_perturbation(rng));
        Eigen::Vector3d normal_offset(normal_perturbation(rng), normal_perturbation(rng), normal_perturbation(rng));
        samples.col(i) += position_offset;
        normals.col(i) += (normals.col(i) + normal_offset).normalized();
    }

    return intersections;
}

std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
Mesh3D::sample_points_and_normals_uniformly_by_casting_rays(int n_points, unsigned int seed,
                                                            const Eigen::Vector3d& bbox_min,
                                                            const Eigen::Vector3d& bbox_max, const double& sigma_p,
                                                            const double& sigma_n) const {

    Eigen::Vector3d bbox_center = 0.5 * (bbox_min + bbox_max);
    Eigen::Vector3d bbox_extent = bbox_max - bbox_min;
    double diameter = bbox_extent.norm();

    // Random generator
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform_01(0.0, 1.0);
    std::vector<Eigen::Vector3d> int_positions, int_normals;
    int n_rays = n_points * 4;

    while (int_positions.size() < n_points) {
        // Cast a batch of rays
        Eigen::Matrix<double, 3, Eigen::Dynamic> ray_origins(3, n_rays);
        Eigen::Matrix<double, 3, Eigen::Dynamic> ray_directions(3, n_rays);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_rays; i++) {

            // Sample direction uniformly on the sphere
            double theta = 2.0 * M_PI * uniform_01(rng);
            double cos_phi = 2.0 * uniform_01(rng) - 1.0;
            double cos_theta = std::cos(theta);
            double sin_theta = std::sin(theta);
            double sin_phi = std::sqrt(1.0 - cos_phi * cos_phi);
            Eigen::Vector3d ray_dir(cos_theta * sin_phi, sin_theta * sin_phi, cos_phi);

            // Sample an offset uniformly from a disk perpendicular to ray_dir
            double theta_disk = 2.0 * M_PI * uniform_01(rng);
            double r = std::sqrt(uniform_01(rng));
            auto [onb_u, onb_v] = orthonormal_basis(ray_dir);
            Eigen::Vector3d ray_origin =
                bbox_center - ray_dir * 0.6 * diameter +
                r * diameter * 0.5 * (std::cos(theta_disk) * onb_u + std::sin(theta_disk) * onb_v);

            ray_origins.col(i) = ray_origin;
            ray_directions.col(i) = ray_dir;
        }

        // Cast rays
        std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> intersections =
            ray_intersections(ray_origins, ray_directions);

        // Filter intersections to only include those within bounding box
        int n_intersections = intersections.first.cols();
        for (int i = 0; i < n_intersections; i++) {
            Eigen::Vector3d p = intersections.first.col(i);
            Eigen::Vector3d n = intersections.second.col(i);
            if (p.x() >= bbox_min.x() && p.x() <= bbox_max.x() && p.y() >= bbox_min.y() && p.y() <= bbox_max.y() &&
                p.z() >= bbox_min.z() && p.z() <= bbox_max.z()) {
                int_positions.push_back(p);
                int_normals.push_back(n);
            }
        }
    }

    // Subsample if we have too many
    int_positions.resize(n_points);
    int_normals.resize(n_points);

    // Add noise and convert to Eigen matrices
    std::normal_distribution<double> position_perturbation(0.0, sigma_p);
    std::normal_distribution<double> normal_perturbation(0.0, sigma_n);

    Eigen::Matrix<double, 3, Eigen::Dynamic> sample_locations(3, n_points);
    Eigen::Matrix<double, 3, Eigen::Dynamic> sample_normals(3, n_points);

    for (int i = 0; i < n_points; i++) {
        Eigen::Vector3d p = int_positions[i];
        Eigen::Vector3d n = int_normals[i];

        Eigen::Vector3d p_perturbation(position_perturbation(rng), position_perturbation(rng),
                                       position_perturbation(rng));
        Eigen::Vector3d n_perturbation(normal_perturbation(rng), normal_perturbation(rng), normal_perturbation(rng));
        sample_locations.col(i) = p + p_perturbation;
        sample_normals.col(i) = (n + n_perturbation).normalized();
    }

    return {sample_locations, sample_normals};
}

std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
Mesh3D::sample_points_and_normals_uniformly(int n_points, unsigned int seed, const Eigen::Vector3d& bbox_min,
                                            const Eigen::Vector3d& bbox_max, const double& sigma_p,
                                            const double& sigma_n) const {


    Eigen::Vector3d extent = bbox_max - bbox_min;
    bool bbox_is_invalid = (extent[0] < 0) || (extent[1] < 0) || (extent[2] < 0);

    std::mt19937 rng(seed);

    // Compute face areas.
    int n_faces;
    std::vector<double> area_weights;
    Eigen::Matrix<double, 3, Eigen::Dynamic> face_normals(3, faces_.cols()); // normalized
    Eigen::VectorXi face_indices(faces_.cols());
    if (bbox_is_invalid) {
        // No bounding box was provided; sample from entire mesh.
        n_faces = faces_.cols();
        area_weights.resize(n_faces);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_faces; ++i) {
            const Eigen::Vector3d& v0 = vertices_.col(faces_(0, i));
            const Eigen::Vector3d& v1 = vertices_.col(faces_(1, i));
            const Eigen::Vector3d& v2 = vertices_.col(faces_(2, i));
            Eigen::Vector3d area_normal = (v1 - v0).cross(v2 - v0);
            double double_area = area_normal.norm();
            area_weights[i] = 0.5 * double_area;
            face_normals.col(i) = area_normal / double_area;
            face_indices(i) = i;
        }
    } else {
        // Identify faces within bounding box using BVH
        std::vector<int> candidate_faces = mesh_bvh->find_primitives_in_box(bbox_min, bbox_max);
        n_faces = candidate_faces.size();
        area_weights.resize(n_faces);
        for (int i = 0; i < n_faces; i++) {
            int global_face_idx = candidate_faces[i];
            const Eigen::Vector3d& v0 = vertices_.col(faces_(0, global_face_idx));
            const Eigen::Vector3d& v1 = vertices_.col(faces_(1, global_face_idx));
            const Eigen::Vector3d& v2 = vertices_.col(faces_(2, global_face_idx));
            Eigen::Vector3d area_normal = (v1 - v0).cross(v2 - v0);
            double double_area = area_normal.norm();
            area_weights[i] = 0.5 * double_area;
            face_normals.col(i) = area_normal / double_area;
            face_indices(i) = global_face_idx;
        }
    }

    // Sample faces proportionally to area
    std::discrete_distribution<> face_idx_dist(area_weights.begin(), area_weights.end());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Eigen::Matrix<double, 3, Eigen::Dynamic> samples(3, n_points);
    Eigen::Matrix<double, 3, Eigen::Dynamic> normals(3, n_points);

    for (int i = 0; i < n_points; ++i) {
        // Randomly pick face.
        int face_idx = face_idx_dist(rng);
        int global_face_idx = face_indices(face_idx);

        // Sample within that face.
        const Eigen::Vector3d& v0 = vertices_.col(faces_(0, global_face_idx));
        const Eigen::Vector3d& v1 = vertices_.col(faces_(1, global_face_idx));
        const Eigen::Vector3d& v2 = vertices_.col(faces_(2, global_face_idx));
        double u = dist(rng);
        double v = dist(rng);
        double sqrt_u = std::sqrt(u);
        double alpha = sqrt_u * (1. - v);
        double beta = v * sqrt_u;
        double gamma = 1. - alpha - beta;
        Eigen::Vector3d p = alpha * v0 + beta * v1 + gamma * v2;

        samples.col(i) = p;
        normals.col(i) = face_normals.col(face_idx);
    }

    // Optionally add noise.
    if (sigma_n > 0.0 || sigma_p > 0.0) {
        std::normal_distribution position_perturbation{0.0, sigma_p};
        std::normal_distribution normal_perturbation{0.0, sigma_n};
        // rng not thread-safe
        for (int i = 0; i < n_points; i++) {
            Eigen::Vector3d position_offset(position_perturbation(rng), position_perturbation(rng),
                                            position_perturbation(rng));
            Eigen::Vector3d normal_offset(normal_perturbation(rng), normal_perturbation(rng), normal_perturbation(rng));
            samples.col(i) += position_offset;
            normals.col(i) += (normals.col(i) + normal_offset).normalized();
        }
    }

    return std::make_pair(samples, normals);
}

Eigen::Matrix<double, 3, Eigen::Dynamic> Mesh3D::sample_uniformly(int n_points, unsigned int seed,
                                                                  const Eigen::Vector3d& bbox_min,
                                                                  const Eigen::Vector3d& bbox_max) const {

    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> result =
        sample_points_and_normals_uniformly(n_points, seed, bbox_min, bbox_max);
    return result.first;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> Mesh3D::sample_narrow_band(int n_points, double offset, unsigned int seed,
                                                                    const Eigen::Vector3d& bbox_min,
                                                                    const Eigen::Vector3d& bbox_max) const {

    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> result =
        sample_points_and_normals_uniformly(n_points, seed, bbox_min, bbox_max);

    std::mt19937 rng(seed + 1);
    std::uniform_real_distribution<double> offset_dist(-offset, offset);
    for (int i = 0; i < n_points; ++i) {
        Eigen::Vector3d offset_vec = offset_dist(rng) * result.second.col(i);
        result.first.col(i) += offset_vec;
    }

    return result.first;
}

void Mesh3D::build_FCPW_scene() {
    scene.setObjectCount(1);
    Eigen::MatrixX3f vertices_f = vertices_.cast<float>().transpose();
    Eigen::MatrixX3i faces_t = faces_.transpose();
    scene.setObjectVertices(vertices_f, 0);
    scene.setObjectTriangles(faces_t, 0);
    scene.build(fcpw::AggregateType::Bvh_SurfaceArea, true);
}

double Mesh3D::evaluate_signed_distance_single(const Eigen::Vector3d& query) const {
    double gwn = evaluate_gwn_single(query);
    double sgn = gwn > 2.0 * M_PI ? -1. : 1.;
    fcpw::Interaction<3> interaction;
    bool found = scene.findClosestPoint(query.cast<float>(), interaction);
    return interaction.d * sgn;
}

Eigen::VectorXd Mesh3D::evaluate_signed_distance(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                 bool parallelize) const {

    int n_queries = queries.cols();

    std::chrono::time_point<high_resolution_clock> t1, t2;
    std::chrono::duration<double, std::milli> ms_fp;
    t1 = high_resolution_clock::now();

    // Compute GWN for sign determination

    Eigen::MatrixXd V = vertices_.transpose();
    Eigen::MatrixXi F = faces_.transpose();
    Eigen::MatrixXd O = queries.transpose();

    Eigen::VectorXd W;
    igl::fast_winding_number(V, F, O, W);


    t2 = high_resolution_clock::now();
    ms_fp = t2 - t1;
    // std::cerr << "GWN time (s): " << ms_fp.count() / 1000. << std::endl;

    int mp = parallelize;
    Eigen::VectorXd distances(n_queries);

    t1 = high_resolution_clock::now();
    // Use FCPW
    if (mp) {
        std::vector<fcpw::Interaction<3>> interactions(n_queries);
        std::vector<fcpw::BoundingSphere<3>> boundingSpheres;
        for (int i = 0; i < n_queries; ++i) {
            boundingSpheres.emplace_back(
                fcpw::BoundingSphere<3>(queries.col(i).cast<float>(), std::numeric_limits<float>::infinity()));
        }
        scene.findClosestPoints(boundingSpheres, interactions, true);

#pragma omp parallel for schedule(dynamic) if (mp)
        for (int i = 0; i < n_queries; ++i) {
            double sgn = W(i) > 0.5 ? -1. : 1.;
            distances(i) = interactions[i].d * sgn;
        }
    } else {
        for (int i = 0; i < n_queries; ++i) {
            fcpw::Interaction<3> interaction;
            double sgn = W(i) > 0.5 ? -1. : 1.;
            Eigen::Vector3d query = queries.col(i);
            bool found = scene.findClosestPoint(query.cast<float>(), interaction);
            distances(i) = interaction.d * sgn;
        }
    }
    t2 = high_resolution_clock::now();
    ms_fp = t2 - t1;
    // std::cerr << "FCPW (s): " << ms_fp.count() / 1000. << std::endl;
    return distances;
}

// Filter out degenerate (zero-area) faces so fast_winding_number behaves
std::pair<Eigen::MatrixXd, Eigen::MatrixXi>
Mesh3D::remove_degenerate_faces(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double area_threshold) const {
    std::vector<int> valid_face_indices;
    valid_face_indices.reserve(F.rows());

    for (int i = 0; i < F.rows(); ++i) {
        int v0 = F(i, 0), v1 = F(i, 1), v2 = F(i, 2);

        // Skip faces with duplicate vertices
        if (v0 == v1 || v1 == v2 || v0 == v2) continue;

        // Skip zero-area faces
        Eigen::Vector3d e1 = V.row(v1) - V.row(v0);
        Eigen::Vector3d e2 = V.row(v2) - V.row(v0);
        double area = 0.5 * e1.cross(e2).norm();
        if (area < area_threshold) continue;

        valid_face_indices.push_back(i);
    }

    Eigen::MatrixXi F_clean(valid_face_indices.size(), 3);
    for (size_t i = 0; i < valid_face_indices.size(); ++i) {
        F_clean.row(i) = F.row(valid_face_indices[i]);
    }

    return {V, F_clean}; // V unchanged, just filtered F
}

std::pair<Eigen::VectorXd, Eigen::Matrix<double, 3, Eigen::Dynamic>>
Mesh3D::evaluate_signed_distance_and_gradient(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries) const {

    // Use FCPW to evaluate distance and gradient

    Eigen::MatrixXd V = vertices_.transpose(); // Force contiguous #V x 3
    Eigen::MatrixXi F = faces_.transpose();    // Force contiguous #F x 3
    Eigen::MatrixXd O = queries.transpose();   // Force contiguous #O x 3

    std::cerr << "Evaluating GWN..." << std::endl;

    Eigen::VectorXd W;
    igl::fast_winding_number(V, F, O, W);

    std::cerr << "Evaluating closest points..." << std::endl;
    int n_queries = queries.cols();
    std::vector<fcpw::Interaction<3>> interactions(n_queries);
    std::vector<fcpw::BoundingSphere<3>> boundingSpheres;
    for (int i = 0; i < n_queries; ++i) {
        boundingSpheres.emplace_back(
            fcpw::BoundingSphere<3>(queries.col(i).cast<float>(), std::numeric_limits<float>::infinity()));
    }
    scene.findClosestPoints(boundingSpheres, interactions, true);
    std::cerr << "Signing distances and computing gradients..." << std::endl;

    // Sign distances and get gradients
    Eigen::VectorXd distances(n_queries);
    Eigen::MatrixXd gradients(3, n_queries);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_queries; ++i) {
        double sgn = (W(i) > 0.5) ? -1. : 1.;
        distances(i) = interactions[i].d * sgn;
        Eigen::Vector3d r_vec = queries.col(i) - interactions[i].p.cast<double>();
        gradients.col(i) = sgn * r_vec.normalized(); // warning: this will be undefined for points on the mesh
    }

    return {distances, gradients};
}

double Mesh3D::evaluate_gwn_single(const Eigen::Vector3d& q) const {

    double sum = 0.0;
    int n_faces = faces_.cols();
    for (int face_idx = 0; face_idx < n_faces; ++face_idx) {
        const Eigen::Vector3d& v0 = vertices_.col(faces_(0, face_idx));
        const Eigen::Vector3d& v1 = vertices_.col(faces_(1, face_idx));
        const Eigen::Vector3d& v2 = vertices_.col(faces_(2, face_idx));

        Eigen::Vector3d a_vec = v0 - q;
        Eigen::Vector3d b_vec = v1 - q;
        Eigen::Vector3d c_vec = v2 - q;
        double a = a_vec.norm();
        double b = b_vec.norm();
        double c = c_vec.norm();

        // -a_z b_y c_x + a_y b_z c_x + a_z b_x c_y - a_x b_z c_y - a_y b_x c_z + a_x b_y c_z
        double numer = -a_vec.z() * b_vec.y() * c_vec.x() + a_vec.y() * b_vec.z() * c_vec.x() +
                       a_vec.z() * b_vec.x() * c_vec.y() - a_vec.x() * b_vec.z() * c_vec.y() -
                       a_vec.y() * b_vec.x() * c_vec.z() + a_vec.x() * b_vec.y() * c_vec.z();
        double denom = a * b * c + (a_vec.dot(b_vec) * c) + (b_vec.dot(c_vec) * a) + (c_vec.dot(a_vec) * b);

        sum += 2.0 * std::atan2(numer, denom);
    }
    return sum;
}

Eigen::VectorXd Mesh3D::evaluate_gwn(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries) const {

    int n_queries = queries.cols();
    Eigen::VectorXd gwn = Eigen::VectorXd::Zero(n_queries);

#pragma omp parallel for schedule(static)
    for (int q_idx = 0; q_idx < n_queries; ++q_idx) {
        const Eigen::Vector3d& q = queries.col(q_idx);
        gwn(q_idx) = evaluate_gwn_single(q);
    }

    return gwn;
}

Eigen::VectorXd Mesh3D::evaluate_fast_gwn(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries) const {

    int n_queries = queries.cols();
    Eigen::VectorXd gwn = Eigen::VectorXd::Zero(n_queries);

    Eigen::MatrixXd V = vertices_.transpose();
    Eigen::MatrixXi F = faces_.transpose();
    Eigen::MatrixXd O = queries.transpose();

    Eigen::VectorXd W;
    igl::fast_winding_number(V, F, O, W);

    return W;
}


// ============================================================================
// BINDING CODE
// ============================================================================

void bind_shape_3d(nb::module_& m) {

    m.def("point_cloud_bounding_box", &point_cloud_bounding_box, "positions"_a,
          "Computing the bounding box of a point cloud.");

    // Mesh3D
    nb::class_<Mesh3D>(m, "Mesh3D")
        .def(nb::init<const Eigen::Matrix<double, 3, Eigen::Dynamic>&, const Eigen::Matrix<int, 3, Eigen::Dynamic>&>(),
             "vertices"_a, "faces"_a, "Create mesh from vertices (n_vertices, 3) and faces (n_faces, 3)")
        .def("bounding_box", &Mesh3D::bounding_box, "Get bounding box")
        .def("center_and_scale", &Mesh3D::center_and_scale, "Center and scale mesh to [-1,1]^3 box")
        .def("get_vertices", &Mesh3D::get_vertices, "Get vertex positions")
        .def("get_faces", &Mesh3D::get_faces, "Get face indices")
        .def("ray_intersections", &Mesh3D::ray_intersections, "ros"_a, "rds"_a,
             "Ray intersections. Returns (points, normals) for hits")
        .def("sample_point_cloud", &Mesh3D::sample_point_cloud, "n_cameras"_a, "sensor_size"_a, "grid_spacing"_a,
             "seed"_a, "sigma_p"_a, "sigma_n"_a, "Generate a point cloud by simulating a sensing process")
        .def("sample_uniform_point_cloud", &Mesh3D::sample_uniform_point_cloud, "n_points"_a, "max_noise_level"_a,
             "max_normals_to_flip"_a, "seed"_a, "Generate a point cloud")
        .def("sample_farthest_point_cloud", &Mesh3D::sample_farthest_point_cloud, "n_points"_a, "max_noise_level"_a,
             "max_normals_to_flip"_a, "seed"_a, "Generate a point cloud")
        .def("sample_even_raycasted_point_cloud", &Mesh3D::sample_even_raycasted_point_cloud, "n_cameras"_a,
             "n_points"_a, "sensor_size"_a, "grid_spacing_min"_a, "grid_spacing_max"_a, "seed"_a,
             "Generate a point cloud")
        .def("sample_uneven_point_cloud", &Mesh3D::sample_uneven_point_cloud, "n_cameras"_a, "sensor_size"_a,
             "grid_spacing_min"_a, "grid_spacing_max"_a, "max_noise_level"_a, "max_normals_to_flip"_a, "max_points"_a,
             "seed"_a, "Generate a point cloud by simulating a sensing process")
        .def("sample_points_and_normals_uniformly", &Mesh3D::sample_points_and_normals_uniformly, "n_points"_a,
             "seed"_a = 0, "bbox_min"_a, "bbox_max"_a, "sigma_p"_a, "sigma_n"_a,
             "Sample points and normals uniformly from boundary")
        .def("sample_uniformly", &Mesh3D::sample_uniformly, "n_points"_a, "seed"_a = 0, "bbox_min"_a, "bbox_max"_a,
             "Sample uniformly from boundary")
        .def("sample_narrow_band", &Mesh3D::sample_narrow_band, "n_points"_a, "offset"_a = 0.1, "seed"_a = 0,
             "bbox_min"_a, "bbox_max"_a, "Sample narrow band around boundary")

        .def("evaluate_signed_distance_single", &Mesh3D::evaluate_signed_distance_single, "query"_a,
             "Evaluate signed distance using FCPW")

        .def("evaluate_signed_distance", &Mesh3D::evaluate_signed_distance, "queries"_a, "parallelize"_a = true,
             "Evaluate signed distance using FCPW")
        .def("evaluate_signed_distance_and_gradient", &Mesh3D::evaluate_signed_distance_and_gradient, "queries"_a,
             "Evaluate signed distance and gradient. Returns (distances, gradients)")
        .def("evaluate_gwn", &Mesh3D::evaluate_gwn, "queries"_a, "Evaluate generalized winding number")
        .def("evaluate_fast_gwn", &Mesh3D::evaluate_fast_gwn, "queries"_a, "Evaluate generalized winding number");
}