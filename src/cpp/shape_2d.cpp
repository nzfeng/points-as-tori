#include "shape_2d.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;
using namespace nb::literals;

// `rotation` is in [0, 2π], and controls the rotation of the local x-axis around the local z-axis, using an arbitrary
// reference vector.
std::pair<Eigen::Vector3d, Eigen::Vector3d> orthonormal_basis(const Eigen::Vector3d& n, double rotation) {

    Eigen::Vector3d n_hat = n / n.norm();

    Eigen::Vector3d u = std::abs(n_hat.x()) > 0.9 ? Eigen::Vector3d(0., 1., 0.) : Eigen::Vector3d(1., 0., 0.);

    Eigen::Matrix3d m = Eigen::AngleAxisd(rotation, n_hat).toRotationMatrix();
    u = m * u;

    Eigen::Vector3d s = n_hat.cross(u);
    s /= s.norm();
    Eigen::Vector3d t = n_hat.cross(s);
    return {s, t};
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> orthonormal_basis_random(const Eigen::Vector3d& n, int seed) {

    std::mt19937 rng(seed);
    std::normal_distribution dist{0.0, 1.0};

    Eigen::Vector3d n_hat = n / n.norm();
    Eigen::Vector3d s(dist(rng), dist(rng), dist(rng));
    s -= s.dot(n_hat) * n_hat;
    s /= s.norm();
    Eigen::Vector3d t = n_hat.cross(s);

    return {s, t};
}

// Generate evenly-spaced positions on sphere by Fibonacci sampling.
// Add an arbitrary rotation of the points.
Eigen::Matrix<double, 3, Eigen::Dynamic> sample_sphere_positions(int n_samples, std::mt19937& rng) {

    Eigen::Matrix<double, 3, Eigen::Dynamic> positions(3, n_samples);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double golden_ratio = 0.5 * (std::sqrt(5) + 1.0);
    for (int i = 0; i < n_samples; i++) {
        double phi = std::acos(1. - 2.0 * i / n_samples);
        double theta = 2.0 * i * M_PI / golden_ratio;
        Eigen::Vector3d p(std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta), std::cos(phi));
        positions.col(i) = p;
    }

    // Apply random rotation
    double u1 = dist(rng);
    double u2 = dist(rng);
    double u3 = dist(rng);
    Eigen::Quaterniond q(std::sqrt(1 - u1) * std::sin(2 * M_PI * u2), std::sqrt(1 - u1) * std::cos(2 * M_PI * u2),
                         std::sqrt(u1) * std::sin(2 * M_PI * u3), std::sqrt(u1) * std::cos(2 * M_PI * u3));
    positions = q.toRotationMatrix() * positions;

    return positions;
}

// ============================================================================
// Point clouds
// ============================================================================

std::pair<Eigen::Vector2d, Eigen::Vector2d>
point_cloud_bounding_box_2d(const Eigen::Matrix<double, 2, Eigen::Dynamic>& positions) {

    const double inf = std::numeric_limits<double>::infinity();
    Eigen::Vector2d bbox_min = {inf, inf};
    Eigen::Vector2d bbox_max = {-inf, -inf};
    int n_points = positions.cols();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_points; i++) {
        for (int j = 0; j < 2; j++) {
            bbox_min[j] = std::min(bbox_min[j], positions(j, i));
            bbox_max[j] = std::max(bbox_max[j], positions(j, i));
        }
    }
    return {bbox_min, bbox_max};
}

// Subsample a point cloud down to the target number of points, using farthest-point sampling
Eigen::VectorXi subsample_point_cloud(const Eigen::Matrix<double, 3, Eigen::Dynamic>& positions, int n_points) {

    int n_input = positions.cols();

    if (n_input <= n_points) {
        Eigen::VectorXi indices(n_input);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_input; i++) {
            indices(i) = i;
        }
        return indices;
    }

    // Use farthest point sampling for even distribution
    Eigen::VectorXi selected_indices(n_points);

    Eigen::VectorXd min_distances = Eigen::VectorXd::Constant(n_input, std::numeric_limits<double>::infinity());

    // Start with a random point
    int curr_idx = 0;
    selected_indices(0) = 0;

    // Iteratively select the point farthest from all previously selected points
    for (int i = 1; i < n_points; ++i) {
        int last_selected = selected_indices(curr_idx);
        const Eigen::Vector3d& last_point = positions.col(last_selected);

// Update minimum distances to selected points
#pragma omp parallel for schedule(static)
        for (int j = 0; j < n_input; ++j) {
            double dist = (positions.col(j) - last_point).squaredNorm();
            min_distances(j) = std::min(min_distances(j), dist);
        }

        // Find the point with maximum minimum distance
        int farthest_idx = 0;
        double max_min_dist = -1.0;
        for (int j = 0; j < n_input; ++j) {
            if (min_distances(j) > max_min_dist) {
                max_min_dist = min_distances(j);
                farthest_idx = j;
            }
        }
        curr_idx += 1;
        selected_indices(curr_idx) = farthest_idx;
        min_distances(farthest_idx) = 0.0; // Mark as selected
    }

    return selected_indices;
}

// ============================================================================
// Base SDF class implementation
// ============================================================================

Eigen::VectorXd SDF2D::evaluate_batch(const Eigen::Matrix<double, 2, Eigen::Dynamic>& queries, double isovalue) const {
    int n = queries.cols();
    Eigen::VectorXd result(n);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        result(i) = evaluate(queries.col(i), isovalue);
    }
    return result;
}

std::pair<Eigen::VectorXd, Eigen::Matrix<double, 2, Eigen::Dynamic>>
SDF2D::evaluate_with_gradient_batch(const Eigen::Matrix<double, 2, Eigen::Dynamic>& queries, double isovalue) const {
    int n = queries.cols();
    Eigen::VectorXd distances(n);
    Eigen::Matrix<double, 2, Eigen::Dynamic> gradients(2, n);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        auto [d, g] = evaluate_with_gradient(queries.col(i), isovalue);
        distances(i) = d;
        gradients.col(i) = g;
    }
    return {distances, gradients};
}

std::tuple<Eigen::Vector2d, Eigen::Vector2d, bool> SDF2D::ray_intersect(const Eigen::Vector2d& ro,
                                                                        const Eigen::Vector2d& rd) const {
    double t = 0.;

    for (int step = 0; step < max_steps && t < diameter; ++step) {
        Eigen::Vector2d q = ro + t * rd;
        auto [d, grad] = evaluate_with_gradient(q, 0.0);
        double d_abs = std::abs(d);

        if (d_abs < epsilon) {
            Eigen::Vector2d normal = grad.normalized();
            return {q, normal, true};
        }

        t += d_abs;
    }

    return {Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(), false};
}

std::pair<Eigen::Matrix<double, 2, Eigen::Dynamic>, Eigen::Matrix<double, 2, Eigen::Dynamic>>
SDF2D::ray_intersect_batch(const Eigen::Matrix<double, 2, Eigen::Dynamic>& ros,
                           const Eigen::Matrix<double, 2, Eigen::Dynamic>& rds) const {
    int n_rays = ros.cols();
    std::vector<Eigen::Vector2d> hit_points;
    std::vector<Eigen::Vector2d> hit_normals;
    // hit_points.reserve(n_rays);
    // hit_normals.reserve(n_rays);

#pragma omp parallel
    {
        std::vector<Eigen::Vector2d> local_points;
        std::vector<Eigen::Vector2d> local_normals;

#pragma omp for nowait schedule(dynamic, 64)
        for (int i = 0; i < n_rays; ++i) {
            auto [point, normal, hit] = ray_intersect(ros.col(i), rds.col(i));
            if (hit) {
                local_points.push_back(point);
                local_normals.push_back(normal);
            }
        }

#pragma omp critical
        {
            hit_points.insert(hit_points.end(), local_points.begin(), local_points.end());
            hit_normals.insert(hit_normals.end(), local_normals.begin(), local_normals.end());
        }
    }

    int n_hits = hit_points.size();
    Eigen::Matrix<double, 2, Eigen::Dynamic> points(2, n_hits);
    Eigen::Matrix<double, 2, Eigen::Dynamic> normals(2, n_hits);

    for (int i = 0; i < n_hits; ++i) {
        points.col(i) = hit_points[i];
        normals.col(i) = hit_normals[i];
    }

    return {points, normals};
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> SDF2D::bbox_revolution(const double& o) const {
    auto [bbox_min_2d, bbox_max_2d] = bounding_box();

    // For revolution: rotate the 2D bbox around the y-axis at offset o
    // The rotated bbox in xz has radius = max(|bbox.x|) + o
    double max_radius = std::max(std::abs(bbox_min_2d.x()), std::abs(bbox_max_2d.x())) + o;

    Eigen::Vector3d bbox_min(-max_radius, bbox_min_2d.y(), -max_radius);
    Eigen::Vector3d bbox_max(max_radius, bbox_max_2d.y(), max_radius);

    return {bbox_min, bbox_max};
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> SDF2D::bbox_extrusion(const double& h) const {
    auto [bbox_min_2d, bbox_max_2d] = bounding_box();

    Eigen::Vector3d bbox_min(bbox_min_2d.x(), bbox_min_2d.y(), -h);
    Eigen::Vector3d bbox_max(bbox_max_2d.x(), bbox_max_2d.y(), h);

    return {bbox_min, bbox_max};
}

std::vector<Eigen::Vector3d> SDF2D::ray_intersection_revolution(const Eigen::Vector3d& ro, const Eigen::Vector3d& rd,
                                                                const double& o, const double& isovalue) const {

    auto [bbox_min, bbox_max] = bbox_revolution(o);
    double diameter = (bbox_max - bbox_min).norm();

    double t = 0.0;
    double tmax = diameter;
    double epsilon = 1e-4;
    std::vector<Eigen::Vector3d> int_locations;

    while (t < tmax) {
        Eigen::Vector3d q = ro + t * rd;
        double d = std::abs(evaluate_revolution_single(q, o) - isovalue);

        if (d < epsilon) {
            int_locations.push_back(q);
            // Ray-march with step size max(|φ|, ε) until φ > ε
            while (d < epsilon && t < tmax) {
                t += std::max(d, epsilon);
                q = ro + t * rd;
                d = std::abs(evaluate_revolution_single(q, o) - isovalue);
            }
        }
        t += d;
    }

    return int_locations;
}

std::vector<Eigen::Vector3d> SDF2D::ray_intersection_extrusion(const Eigen::Vector3d& ro, const Eigen::Vector3d& rd,
                                                               const double& h, const double& isovalue) const {

    auto [bbox_min, bbox_max] = bbox_extrusion(h);
    double diameter = (bbox_max - bbox_min).norm();

    double t = 0.0;
    double tmax = diameter;
    double epsilon = 1e-4;
    std::vector<Eigen::Vector3d> int_locations;

    while (t < tmax) {
        Eigen::Vector3d q = ro + t * rd;
        double d = std::abs(evaluate_extrusion_single(q, h) - isovalue);

        if (d < epsilon) {
            int_locations.push_back(q);
            // Ray-march with step size max(|φ|, ε) until φ > ε
            while (d < epsilon && t < tmax) {
                t += std::max(d, epsilon);
                q = ro + t * rd;
                d = std::abs(evaluate_extrusion_single(q, h) - isovalue);
            }
        }
        t += d;
    }

    return int_locations;
}

std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
SDF2D::sample_revolution(const int& n_samples, const double& o, const Eigen::Vector3d& bbox_min_,
                         const Eigen::Vector3d& bbox_max_, const double& sigma_p, const double& sigma_n,
                         const int& seed) const {

    // Determine bounding box
    Eigen::Vector3d bbox_min, bbox_max;
    Eigen::Vector3d extent_ = bbox_max_ - bbox_min_;
    if (extent_[0] < 0) {
        auto [bmin, bmax] = bbox_revolution(o);
        bbox_min = bmin;
        bbox_max = bmax;
    } else {
        bbox_min = bbox_min_;
        bbox_max = bbox_max_;
    }

    Eigen::Vector3d bbox_center = 0.5 * (bbox_min + bbox_max);
    Eigen::Vector3d bbox_extent = bbox_max - bbox_min;
    double diameter = bbox_extent.norm();

    // Random generator
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform_01(0.0, 1.0);
    std::normal_distribution<double> position_perturbation(0.0, sigma_p);
    std::normal_distribution<double> normal_perturbation(0.0, sigma_n);

    std::vector<Eigen::Vector3d> int_locations;

    const double epsilon = 1e-6;
    const int max_tries = n_samples * 10;
    int attempt = 0;

    while (int_locations.size() < static_cast<size_t>(n_samples) && attempt < max_tries) {
        // Sample a direction uniformly on the sphere
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
        Eigen::Vector3d ray_origin = bbox_center - ray_dir * 0.6 * diameter +
                                     r * diameter * 0.5 * (std::cos(theta_disk) * onb_u + std::sin(theta_disk) * onb_v);

        // Perform ray intersection
        std::vector<Eigen::Vector3d> new_intersections = ray_intersection_revolution(ray_origin, ray_dir, o, 0.0);

        // Filter intersections to only include those within bounding box
        for (const auto& p : new_intersections) {
            if (p.x() >= bbox_min.x() && p.x() <= bbox_max.x() && p.y() >= bbox_min.y() && p.y() <= bbox_max.y() &&
                p.z() >= bbox_min.z() && p.z() <= bbox_max.z()) {
                int_locations.push_back(p);
                if (int_locations.size() >= static_cast<size_t>(n_samples)) {
                    break;
                }
            }
        }

        attempt++;
    }

    // Handle case where we didn't get enough samples
    int n_final = std::min(static_cast<int>(int_locations.size()), n_samples);
    if (n_final < n_samples) {
        // Duplicate samples to reach n_samples
        for (int i = 0; i < n_samples - n_final; i++) {
            int_locations.push_back(int_locations[i % n_final]);
        }
    }

    // Prepare output matrices
    Eigen::Matrix<double, 3, Eigen::Dynamic> sample_locations(3, n_samples);
    Eigen::Matrix<double, 3, Eigen::Dynamic> sample_normals(3, n_samples);

    for (int i = 0; i < n_samples; i++) {
        Eigen::Vector3d p = int_locations[i];

        // Add position noise
        if (sigma_p > 0.0) {
            p.x() += position_perturbation(rng);
            p.y() += position_perturbation(rng);
            p.z() += position_perturbation(rng);
        }
        sample_locations.col(i) = p;

        // Compute gradient/normal
        Eigen::Vector3d grad = evaluate_revolution_gradient_single(int_locations[i], o);

        // Add normal noise
        if (sigma_n > 0.0) {
            grad.x() += normal_perturbation(rng);
            grad.y() += normal_perturbation(rng);
            grad.z() += normal_perturbation(rng);
        }

        double grad_norm = grad.norm();
        if (grad_norm > epsilon) {
            sample_normals.col(i) = grad / grad_norm;
        } else {
            sample_normals.col(i) = Eigen::Vector3d::Zero();
        }
    }

    return {sample_locations, sample_normals};
}

Eigen::Matrix<double, 3, Eigen::Dynamic>
SDF2D::evaluate_signed_distance_gradient(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries, bool is_extrusion,
                                         const double& SDF3D_param) const {

    int n_queries = queries.cols();
    Eigen::MatrixXd gradients(3, n_queries);
    if (is_extrusion) {
#pragma omp parallel for schedule(static)
        for (int q_idx = 0; q_idx < n_queries; q_idx++) {
            gradients.col(q_idx) = evaluate_extrusion_gradient_single(queries.col(q_idx), SDF3D_param);
        }
        return gradients;
    }

#pragma omp parallel for schedule(static)
    for (int q_idx = 0; q_idx < n_queries; q_idx++) {
        gradients.col(q_idx) = evaluate_revolution_gradient_single(queries.col(q_idx), SDF3D_param);
    }
    return gradients;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> SDF2D::sample_revolution_narrow_band(const int& n_samples, const double& o,
                                                                              const double& offset,
                                                                              const Eigen::Vector3d& bbox_min,
                                                                              const Eigen::Vector3d& bbox_max,
                                                                              const int& seed) const {

    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> samples =
        sample_revolution(n_samples, o, bbox_min, bbox_max, 0.0, 0.0, seed);

    // perturb in the normal direction
    std::mt19937 rng(seed);
    Eigen::Matrix<double, 3, Eigen::Dynamic> result(3, n_samples);
    std::uniform_real_distribution<double> offset_dist(-offset, offset);

    for (int i = 0; i < n_samples; ++i) {
        Eigen::Vector3d grad = samples.second.col(i);
        Eigen::Vector3d normal = grad.normalized();
        double random_offset = offset_dist(rng);
        result.col(i) = samples.first.col(i) + random_offset * normal;
    }
    return result;
}

std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
SDF2D::sample_extrusion(const int& n_samples, const double& h, const Eigen::Vector3d& bbox_min_,
                        const Eigen::Vector3d& bbox_max_, const double& sigma_p, const double& sigma_n,
                        const int& seed) const {

    // Determine bounding box
    Eigen::Vector3d bbox_min, bbox_max;
    Eigen::Vector3d extent_ = bbox_max_ - bbox_min_;
    if (extent_[0] < 0) {
        auto [bmin, bmax] = bbox_extrusion(h);
        bbox_min = bmin;
        bbox_max = bmax;
    } else {
        bbox_min = bbox_min_;
        bbox_max = bbox_max_;
    }

    Eigen::Vector3d bbox_center = 0.5 * (bbox_min + bbox_max);
    Eigen::Vector3d bbox_extent = bbox_max - bbox_min;
    double diameter = bbox_extent.norm();

    // Random generator
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform_01(0.0, 1.0);
    std::normal_distribution<double> position_perturbation(0.0, sigma_p);
    std::normal_distribution<double> normal_perturbation(0.0, sigma_n);

    std::vector<Eigen::Vector3d> int_locations;

    const double epsilon = 1e-6;
    const int max_tries = n_samples * 10;
    int attempt = 0;

    while (int_locations.size() < static_cast<size_t>(n_samples) && attempt < max_tries) {
        // Sample a direction uniformly on the sphere
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
        Eigen::Vector3d ray_origin = bbox_center - ray_dir * 0.6 * diameter +
                                     r * diameter * 0.5 * (std::cos(theta_disk) * onb_u + std::sin(theta_disk) * onb_v);

        // Perform ray intersection
        std::vector<Eigen::Vector3d> new_intersections = ray_intersection_extrusion(ray_origin, ray_dir, h, 0.0);

        // Filter intersections to only include those within bounding box
        for (const auto& p : new_intersections) {
            if (p.x() >= bbox_min.x() && p.x() <= bbox_max.x() && p.y() >= bbox_min.y() && p.y() <= bbox_max.y() &&
                p.z() >= bbox_min.z() && p.z() <= bbox_max.z()) {
                int_locations.push_back(p);
                if (int_locations.size() >= static_cast<size_t>(n_samples)) {
                    break;
                }
            }
        }

        attempt++;
    }

    // Handle case where we didn't get enough samples
    int n_final = std::min(static_cast<int>(int_locations.size()), n_samples);
    if (n_final < n_samples) {
        // Duplicate samples to reach n_samples
        for (int i = 0; i < n_samples - n_final; i++) {
            int_locations.push_back(int_locations[i % n_final]);
        }
    }

    // Prepare output matrices
    Eigen::Matrix<double, 3, Eigen::Dynamic> sample_locations(3, n_samples);
    Eigen::Matrix<double, 3, Eigen::Dynamic> sample_normals(3, n_samples);

    for (int i = 0; i < n_samples; i++) {
        Eigen::Vector3d p = int_locations[i];

        // Add position noise
        if (sigma_p > 0.0) {
            p.x() += position_perturbation(rng);
            p.y() += position_perturbation(rng);
            p.z() += position_perturbation(rng);
        }
        sample_locations.col(i) = p;

        // Compute gradient/normal
        Eigen::Vector3d grad = evaluate_extrusion_gradient_single(int_locations[i], h);

        // Add normal noise
        if (sigma_n > 0.0) {
            grad.x() += normal_perturbation(rng);
            grad.y() += normal_perturbation(rng);
            grad.z() += normal_perturbation(rng);
        }

        double grad_norm = grad.norm();
        // sample_normals.col(i) = (grad_norm > epsilon) ? (grad / grad_norm) : Eigen::Vector3d::Zero();
        if (grad_norm > epsilon) {
            sample_normals.col(i) = grad / grad_norm;
        } else {
            sample_normals.col(i) = Eigen::Vector3d::Zero();
        }
    }

    return {sample_locations, sample_normals};
}

Eigen::Matrix<double, 3, Eigen::Dynamic> SDF2D::sample_extrusion_narrow_band(const int& n_samples, const double& h,
                                                                             const double& offset,
                                                                             const Eigen::Vector3d& bbox_min,
                                                                             const Eigen::Vector3d& bbox_max,
                                                                             const int& seed) const {

    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> samples =
        sample_extrusion(n_samples, h, bbox_min, bbox_max, 0.0, 0.0, seed);

    // perturb in the normal direction
    std::mt19937 rng(seed);
    Eigen::Matrix<double, 3, Eigen::Dynamic> result(3, n_samples);
    std::uniform_real_distribution<double> offset_dist(-offset, offset);

    for (int i = 0; i < n_samples; ++i) {
        Eigen::Vector3d grad = samples.second.col(i);
        Eigen::Vector3d normal = grad.normalized();
        double random_offset = offset_dist(rng);
        result.col(i) = samples.first.col(i) + random_offset * normal;
    }
    return result;
}

std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
SDF2D::sample_points_and_normals_uniformly(int n_points, bool is_extrusion, const double& SDF3D_param, int seed) const {

    Eigen::Vector3d bbox_min = {1, 1, 1};
    Eigen::Vector3d bbox_max = {-1, -1, -1};
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> result;
    if (is_extrusion) {
        result = sample_extrusion(n_points, SDF3D_param, bbox_min, bbox_max, 0.0, 0.0, seed);
    } else {
        result = sample_revolution(n_points, SDF3D_param, bbox_min, bbox_max, 0.0, 0.0, seed);
    }
    return result;
}

// Sample revolution or extrusion
std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
SDF2D::sample_farthest_point_cloud(int n_points, bool is_extrusion, const double& SDF3D_param, int seed) const {

    Eigen::Vector3d bbox_min = {1, 1, 1};
    Eigen::Vector3d bbox_max = {-1, -1, -1};
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> result;
    if (is_extrusion) {
        result = sample_extrusion(n_points * 8, SDF3D_param, bbox_min, bbox_max, 0.0, 0.0, seed);
    } else {
        result = sample_revolution(n_points * 8, SDF3D_param, bbox_min, bbox_max, 0.0, 0.0, seed);
    }

    // Subsample
    Eigen::VectorXi indices = subsample_point_cloud(result.first, n_points);
    Eigen::MatrixXd samples(3, n_points);
    Eigen::MatrixXd normals(3, n_points);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_points; i++) {
        samples.col(i) = result.first.col(indices(i));
        normals.col(i) = result.second.col(indices(i));
    }
    return {samples, normals};
}

std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
SDF2D::sample_uneven_point_cloud(bool is_extrusion, const double& SDF3D_param, int n_cameras, const double& sensor_size,
                                 const double& grid_spacing_min, const double& grid_spacing_max,
                                 const double& max_noise_level, const double& max_normals_to_flip,
                                 const int& max_points, unsigned int seed) const {

    // Copied from Mesh3D

    std::mt19937 rng(seed);

    // Determine bounding box
    auto [bmin, bmax] = bbox_extrusion(SDF3D_param);
    Eigen::Vector3d bbox_min = bmin;
    Eigen::Vector3d bbox_max = bmax;

    Eigen::Vector3d bbox_center = 0.5 * (bbox_min + bbox_max);
    Eigen::Vector3d bbox_extent = bbox_max - bbox_min;
    double diameter = bbox_extent.norm();

    // Get bounding sphere.
    Eigen::Vector3d centroid = bbox_center;
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

    // Cast rays
    int n_rays = origins.size();
    std::vector<Eigen::Vector3d> ray_hits, ray_normals;
    for (int i = 0; i < n_rays; i++) {
        // Take first hit
        std::vector<Eigen::Vector3d> ints;
        if (is_extrusion) {
            ints = ray_intersection_revolution(origins[i], directions[i], SDF3D_param, 0.0);
        } else {
            ints = ray_intersection_extrusion(origins[i], directions[i], SDF3D_param, 0.0);
        }
        if (ints.size() > 0) {
            Eigen::Vector3d hit_position = ints[0];
            ray_hits.push_back(hit_position);
            Eigen::Vector3d hit_normal;
            if (is_extrusion) {
                hit_normal = evaluate_extrusion_gradient_single(hit_position, SDF3D_param);
            } else {
                hit_normal = evaluate_revolution_gradient_single(hit_position, SDF3D_param);
            }
            ray_normals.push_back(hit_normal);
        }
    }

    int n_valid_hits = ray_hits.size();
    Eigen::MatrixXd hit_positions(3, n_valid_hits);
    Eigen::MatrixXd hit_normals(3, n_valid_hits);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_valid_hits; i++) {
        hit_positions.col(i) = ray_hits[i];
        hit_normals.col(i) = ray_normals[i];
    }
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>> intersections = {
        hit_positions, hit_normals};

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
    int n_normals_to_flip = int(normals_to_flip * max_points);
    std::cerr << "\t Flipping " << n_normals_to_flip << " normals" << std::endl;

    std::normal_distribution position_perturbation{0.0, sigma_p};
    std::normal_distribution normal_perturbation{0.0, sigma_n};

#pragma omp parallel for schedule(static)
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

// ============================================================================
// Individual SDFs
// ============================================================================
std::pair<double, Eigen::Vector2d> Circle2D::evaluate_with_gradient(const Eigen::Vector2d& q, double isovalue) const {
    double len = q.norm();
    double phi = len - r_;
    Eigen::Vector2d grad = (len > 1e-10) ? (q / len) : Eigen::Vector2d(1.0, 0.0);
    return {phi - isovalue, grad};
}

double Circle2D::evaluate_laplacian(const Eigen::Vector2d& q) const {
    return 1. / q.norm();
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Circle2D::bounding_box(double isovalue) const {
    Eigen::Vector2d bbox_min(-r_, -r_);
    Eigen::Vector2d bbox_max(r_, r_);
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


Eigen::VectorXd Circle2D::evaluate_laplacian_batch(const Eigen::Matrix<double, 2, Eigen::Dynamic>& queries) const {

    int n_queries = queries.cols();
    Eigen::VectorXd laplacians(n_queries);
#pragma omp parallel for schedule(static)
    for (int q_idx = 0; q_idx < n_queries; ++q_idx) {
        laplacians(q_idx) = evaluate_laplacian(queries.col(q_idx));
    }
    return laplacians;
}


std::pair<double, Eigen::Vector2d> Pie2D::evaluate_with_gradient(const Eigen::Vector2d& query, double isovalue) const {
    // Updated from Inigo's formula so gradients are computed to *S*DF, not UDF then signed
    double s = sign(query.x());
    Eigen::Vector2d p(std::abs(query.x()), query.y());
    double l = p.norm();
    double n = l - r_;
    double t = clamp(p.dot(sc_), 0., r_);
    Eigen::Vector2d q = p - sc_ * t;
    double m = q.norm() * sign(sc_.y() * p.x() - sc_.x() * p.y());
    double d = m;
    Eigen::Vector2d g;
    if (n > m) {
        d = n;
        g = p / l;
    } else {
        // closest to straight edge
        if (std::abs(t) > epsilon_dist && std::abs(t - r_) > epsilon_dist) {
            Eigen::Vector2d edge_normal(-sc_.y(), sc_.x());
            g = -edge_normal;
        } else {
            g = q / m; // vertex, division is unavoidable
        }
    }
    g(0) *= s;
    return {d - isovalue, g};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Pie2D::bounding_box(double isovalue) const {
    double v = sc_.x();
    double u = sc_.y();
    Eigen::Vector2d k = (u > 0.) ? Eigen::Vector2d{r_ * v, 0.} : Eigen::Vector2d{r_, r_ * u};
    Eigen::Vector2d bbox_min(-k.x(), k.y());
    Eigen::Vector2d bbox_max(k.x(), r_);
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


std::pair<double, Eigen::Vector2d> Arc2D::evaluate_with_gradient(const Eigen::Vector2d& q, double isovalue) const {
    double s = sign(q.x());
    Eigen::Vector2d p(std::abs(q.x()), q.y());
    bool in_wedge = sc_.y() * p.x() > sc_.x() * p.y();

    double phi;
    Eigen::Vector2d grad;

    if (in_wedge) {
        // Distance to arc endpoint
        Eigen::Vector2d w1 = p - ra_ * sc_;
        double d1 = w1.norm();
        phi = d1 - rb_;
        grad = Eigen::Vector2d(s * w1.x(), w1.y()) / d1;
    } else {
        // Distance to ring
        double l2 = q.norm();
        double w2 = l2 - ra_;
        phi = std::abs(w2) - rb_;
        grad = sign(w2) * q / l2;
    }

    return {phi - isovalue, grad};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Arc2D::bounding_box(double isovalue) const {
    double outer_radius = ra_ + rb_;
    Eigen::Vector2d bbox_min(-outer_radius, ra_ * sc_.y() - rb_);
    Eigen::Vector2d bbox_max(outer_radius, std::max(sc_.y() * outer_radius, ra_ + rb_));
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


std::pair<double, Eigen::Vector2d> Pill2D::evaluate_with_gradient(const Eigen::Vector2d& q, double isovalue) const {
    Eigen::Vector2d ba = b_ - a_;
    Eigen::Vector2d pa = q - a_;
    double h = clamp(pa.dot(ba) / ba.squaredNorm(), 0.0, 1.0);
    Eigen::Vector2d q_vec = pa - h * ba;
    double d = q_vec.norm();
    double phi = d - r_;
    Eigen::Vector2d grad = q_vec / std::max(d, 1e-10);
    return {phi - isovalue, grad};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Pill2D::bounding_box(double isovalue) const {
    Eigen::Vector2d bbox_min = min(a_, b_).array() - r_;
    Eigen::Vector2d bbox_max = max(a_, b_).array() + r_;
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


std::pair<double, Eigen::Vector2d> Vesica2D::evaluate_with_gradient(const Eigen::Vector2d& q, double isovalue) const {
    Eigen::Vector2d s = sign(q);
    Eigen::Vector2d p = cwiseAbs(q);
    bool cond = (p.y() - b_) * d_ > p.x() * b_;

    double phi;
    Eigen::Vector2d grad;

    if (cond) {
        // Case 1: distance to the tip
        Eigen::Vector2d q1(p.x(), p.y() - b_);
        double l1 = q1.norm() * sign(d_);
        phi = l1;
        grad = s.cwiseProduct(q1) / l1;
    } else {
        // Case 2: distance to the circular arc
        Eigen::Vector2d q2(p.x() + d_, p.y());
        double l2 = q2.norm();
        phi = l2 - r_;
        grad = s.cwiseProduct(q2) / l2;
    }

    return {phi - isovalue, grad};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Vesica2D::bounding_box(double isovalue) const {
    // Vesica is formed by two overlapping circles of radius r_, separated by distance d_
    // b := sqrt(r^2 - d^2) = height at inner intersection
    double height = std::max(b_, r_);
    Eigen::Vector2d bbox_min(-r_ + d_, -height);
    Eigen::Vector2d bbox_max(r_ - d_, height);
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


std::pair<double, Eigen::Vector2d> Box2D::evaluate_with_gradient(const Eigen::Vector2d& q, double isovalue) const {
    Eigen::Vector2d w = cwiseAbs(q) - b_;
    Eigen::Vector2d s = sign(q);
    double g = std::max(w.x(), w.y());
    Eigen::Vector2d q_vec = cwiseMax(w, 0.0);
    double l = q_vec.norm();
    bool outside = g > 0;

    double phi = outside ? l : g;
    Eigen::Vector2d grad;

    if (outside) {
        grad = s.cwiseProduct(q_vec) / std::max(l, 1e-10);
    } else {
        // Inside: gradient points toward nearest face
        if (w.x() > w.y()) {
            grad = Eigen::Vector2d(s.x(), 0.0);
        } else {
            grad = Eigen::Vector2d(0.0, s.y());
        }
    }

    return {phi - isovalue, grad};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Box2D::bounding_box(double isovalue) const {
    Eigen::Vector2d bbox_min = -b_;
    Eigen::Vector2d bbox_max = b_;
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


std::pair<double, Eigen::Vector2d> Cross2D::evaluate_with_gradient(const Eigen::Vector2d& q, double isovalue) const {

    Eigen::Vector2d s = sign(q);
    Eigen::Vector2d p = cwiseAbs(q);
    bool swap = p.y() > p.x();
    Eigen::Vector2d p_temp = swap ? Eigen::Vector2d(p.y(), p.x()) : p;
    Eigen::Vector2d q_vec = p_temp - b_;
    double h = std::max(q_vec.x(), q_vec.y());

    Eigen::Vector2d o;
    if (h < 0) {
        o = cwiseMax(Eigen::Vector2d(b_.y() - b_.x() - q_vec.x(), -q_vec.y()), 0.0);
    } else {
        o = cwiseMax(q_vec, 0.0);
    }

    double l = o.norm();
    bool use_corner = (h >= 0) || (-q_vec.x() >= l);
    double phi = (use_corner ? l : -q_vec.x()) * sign(h);
    Eigen::Vector2d r = use_corner ? (o / l) : Eigen::Vector2d(1.0, 0.0);
    Eigen::Vector2d r_final = swap ? Eigen::Vector2d(r.y(), r.x()) : r;
    Eigen::Vector2d grad = s.cwiseProduct(r_final);

    return {phi - isovalue, grad};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Cross2D::bounding_box(double isovalue) const {
    Eigen::Vector2d bbox_min(-b_.x(), -b_.x());
    Eigen::Vector2d bbox_max(b_.x(), b_.x());
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


std::pair<double, Eigen::Vector2d> TriangleIsosceles2D::evaluate_with_gradient(const Eigen::Vector2d& query,
                                                                               double isovalue) const {
    // Updated from Inigo's formula so gradients are computed to *S*DF, not UDF then signed
    double w = sign(query.x());
    Eigen::Vector2d p = query;
    p.x() = std::abs(p.x());
    double t_a = clamp(p.dot(q_) / q_.dot(q_), 0., 1.);
    Eigen::Vector2d a = p - q_ * t_a;
    double t_b = clamp(p.x() / q_.x(), 0., 1.);
    Eigen::Vector2d b = p - q_.cwiseProduct(Eigen::Vector2d(t_b, 1.));
    double k = sign(q_.y());
    double l1 = a.dot(a);
    double l2 = b.dot(b);
    double d = std::sqrt((l1 < l2) ? l1 : l2);
    double s = std::max(k * (p.x() * q_.y() - p.y() * q_.x()), k * (p.y() - q_.y()));
    s = sign(s);
    Eigen::Vector2d g;
    if (l1 < l2) {
        if (std::abs(t_a) > epsilon_dist && std::abs(t_a - 1.) > epsilon_dist) {
            g = {-q_.y(), q_.x()};
            g /= g.norm();
            g(0) *= w;
        } else {
            g = a / a.norm();
            g(0) *= w;
            g *= s;
        }
    } else {
        if (std::abs(t_b) > epsilon_dist && std::abs(t_b - 1.) > epsilon_dist) {
            // closest to base
            g = {0, 1};
            g *= k;
        } else {
            g = b / b.norm();
            g(0) *= w;
            g *= s;
        }
    }

    return {d * s - isovalue, g};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> TriangleIsosceles2D::bounding_box(double isovalue) const {
    Eigen::Vector2d bbox_min(-q_.x(), std::min(0., q_.y()));
    Eigen::Vector2d bbox_max(q_.x(), std::max(0., q_.y()));
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


std::pair<double, Eigen::Vector2d> Triangle2D::evaluate_with_gradient(const Eigen::Vector2d& p, double isovalue) const {
    // Updated from Inigo's formula so gradients are computed to *S*DF, not UDF then signed
    double gs = cross(v_[0] - v_[2], v_[1] - v_[0]);
    double res_d;
    Eigen::Vector2d grad, e, w, q;
    double res_s, d, s, t;

    e = v_[1] - v_[0];
    w = p - v_[0];
    t = clamp(w.dot(e) / e.dot(e), 0., 1.);
    q = w - e * t;
    d = q.dot(q);
    s = gs * cross(w, e);
    res_d = d;
    res_s = s;
    if (std::abs(t) > epsilon_dist && std::abs(t - 1.) > epsilon_dist) {
        grad = {e.y(), -e.x()};
        grad /= grad.norm();
    } else {
        grad = w - e * t;
        grad = q / q.norm();
    }

    e = v_[2] - v_[1];
    w = p - v_[1];
    t = clamp(w.dot(e) / e.dot(e), 0., 1.);
    q = w - e * t;
    d = q.dot(q);
    s = gs * cross(w, e);
    if (d < res_d) {
        res_d = d;
        if (std::abs(t) > epsilon_dist && std::abs(t - 1.) > epsilon_dist) {
            grad = {e.y(), -e.x()};
            grad /= grad.norm();
        } else {
            grad = w - e * t;
            grad = q / q.norm();
        }
    }
    res_s = std::max(res_s, s);

    e = v_[0] - v_[2];
    w = p - v_[2];
    t = clamp(w.dot(e) / e.dot(e), 0., 1.);
    q = w - e * t;
    d = q.dot(q);
    s = gs * cross(w, e);
    if (d < res_d) {
        res_d = d;
        if (std::abs(t) > epsilon_dist && std::abs(t - 1.) > epsilon_dist) {
            grad = {e.y(), -e.x()};
            grad /= grad.norm();
        } else {
            grad = q / q.norm();
        }
    }
    res_s = std::max(res_s, s);

    double phi = std::sqrt(res_d) * sign(res_s);

    return {phi - isovalue, grad};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Triangle2D::bounding_box(double isovalue) const {
    Eigen::Vector2d bbox_min = min(v_[0], min(v_[1], v_[2]));
    Eigen::Vector2d bbox_max = max(v_[0], max(v_[1], v_[2]));
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


std::pair<double, Eigen::Vector2d> Hexagon2D::evaluate_with_gradient(const Eigen::Vector2d& q, double isovalue) const {
    Eigen::Vector2d s = sign(q);
    Eigen::Vector2d p = cwiseAbs(q);
    double w = p.x() * k_.x() + p.y() * k_.y();
    p -= 2.0 * std::min(w, 0.0) * Eigen::Vector2d(k_.x(), k_.y());
    double t = clamp(p.x(), -k_.z() * r_, k_.z() * r_);
    p -= Eigen::Vector2d(t, r_);
    double d = p.norm() * sign(p.y());
    Eigen::Matrix2d mat;
    mat << -k_.y(), -k_.x(), -k_.x(), k_.y();
    Eigen::Vector2d g = (w < 0.) ? mat * p : p;

    // Updated from Inigo's formula so gradients are computed to *S*DF, not UDF then signed
    Eigen::Vector2d grad;
    if (std::abs(t - (-k_.z() * r_)) < epsilon_dist || std::abs(t - (k_.z() * r_)) < epsilon_dist) {
        // vertex
        grad = s.cwiseProduct(g) / d;
    } else {
        if (w < 0.) {
            // Closest to angled edge
            Eigen::Vector2d edge_normal(-k_.x(), k_.y());
            grad = s.cwiseProduct(edge_normal);
        } else {
            // Closest to horizontal edge
            grad = Eigen::Vector2d{0., 1.} * sign(q.y());
        }
    }
    return {d - isovalue, grad};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Hexagon2D::bounding_box(double isovalue) const {
    double half_width = 2.0 * r_ / std::sqrt(3.);
    double half_height = r_;
    Eigen::Vector2d bbox_min(-half_width, -half_height);
    Eigen::Vector2d bbox_max(half_width, half_height);

    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


// std::pair<double, Eigen::Vector2d> Ellipse2D::evaluate_with_gradient(const Eigen::Vector2d& query,
//                                                                      double isovalue) const {
//   Eigen::Vector2d sp = sign(query);
//   Eigen::Vector2d p = cwiseAbs(query);
//   Eigen::Vector2d pab = p.cwiseQuotient(ab_);
//   bool s = pab.dot(pab) > 1.;
//   double w = std::atan2(p.y() * ab_.x(), p.x() * ab_.y());
//   if (!s) {
//     w = (ab_.x() * (p.x() - ab_.x()) < ab_.y() * (p.y() - ab_.y())) ? 0.5 * M_PI : 0.;
//   }
//   for (int i = 0; i < 4; i++) {
//     Eigen::Vector2d cs(std::cos(w), std::sin(w));
//     Eigen::Vector2d u = ab_.cwiseProduct(cs);
//     Eigen::Vector2d v = ab_.cwiseProduct(Eigen::Vector2d(-cs.y(), cs.x()));
//     w += (p - u).dot(v) / ((p - u).dot(u) + v.dot(v));
//   }
//   Eigen::Vector2d q = ab_.cwiseProduct(Eigen::Vector2d(std::cos(w), std::sin(w)));
//   double d = (p - q).norm();
//   double sgn = s ? 1. : -1.;
//   Eigen::Vector2d g = sgn * sp.cwiseProduct(p - q);
//   return {sgn * (d - isovalue), g / g.norm()};
// }

// Warning: not used and not verified yet
std::pair<double, Eigen::Vector2d> Ellipse2D::evaluate_with_gradient(const Eigen::Vector2d& query,
                                                                     double isovalue) const {
    const double epsilon = 1e-10;
    Eigen::Vector2d sp = sign(query);
    Eigen::Vector2d p = cwiseAbs(query);
    Eigen::Vector2d pab = p.cwiseQuotient(ab_);
    bool s = pab.dot(pab) > 1.;

    // Initial angle estimate
    double w = std::atan2(p.y() * ab_.x(), p.x() * ab_.y());
    if (!s) {
        w = (ab_.x() * (p.x() - ab_.x()) < ab_.y() * (p.y() - ab_.y())) ? 0.5 * M_PI : 0.;
    }

    // Newton iteration to find closest point parameter
    for (int i = 0; i < 4; i++) {
        Eigen::Vector2d cs(std::cos(w), std::sin(w));
        Eigen::Vector2d u = ab_.cwiseProduct(cs);
        Eigen::Vector2d v = ab_.cwiseProduct(Eigen::Vector2d(-cs.y(), cs.x()));
        double numerator = (p - u).dot(v);
        double denominator = (p - u).dot(u) + v.dot(v);
        if (std::abs(denominator) > epsilon) {
            w += numerator / denominator;
        }
    }

    // Closest point on ellipse
    Eigen::Vector2d q = ab_.cwiseProduct(Eigen::Vector2d(std::cos(w), std::sin(w)));
    double d = (p - q).norm();
    double sgn = s ? 1. : -1.;

    // Gradient: use ellipse normal at parameter w
    // For ellipse (a*cos(w), b*sin(w)), the unnormalized outward normal is (cos(w)/a, sin(w)/b)
    Eigen::Vector2d unnormalized_normal(std::cos(w) / ab_.x(), std::sin(w) / ab_.y());
    double norm = unnormalized_normal.norm();
    Eigen::Vector2d grad;
    if (norm > epsilon) {
        grad = sp.cwiseProduct(unnormalized_normal / norm);
    } else {
        // Degenerate case: use radial direction
        grad = sp.cwiseProduct((p - q).normalized());
    }

    return {sgn * (d - isovalue), grad};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Ellipse2D::bounding_box(double isovalue) const {
    Eigen::Vector2d bbox_min(-ab_.x(), -ab_.y());
    Eigen::Vector2d bbox_max(ab_.x(), ab_.y());
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


std::pair<double, Eigen::Vector2d> Heart2D::evaluate_with_gradient(const Eigen::Vector2d& q, double isovalue) const {
    double sx = sign(q.x());
    Eigen::Vector2d p(std::abs(q.x()), q.y());

    bool upper_cond = p.y() + p.x() > 1.0;
    double r = std::sqrt(2.0) / 4.0;

    double phi;
    Eigen::Vector2d grad;

    if (upper_cond) {
        // Upper part: distance to circular arc
        Eigen::Vector2d q0 = p - Eigen::Vector2d(0.25, 0.75);
        double l0 = q0.norm();
        phi = l0 - r;
        grad = Eigen::Vector2d(sx * q0.x(), q0.y()) / std::max(l0, 1e-10);
    } else {
        // Lower part: distance to point (0,1) or diagonal line
        Eigen::Vector2d q1 = p - Eigen::Vector2d(0.0, 1.0);
        double d1_sq = q1.squaredNorm();
        double h = std::max(p.x() + p.y(), 0.0);
        Eigen::Vector2d q2 = p - 0.5 * Eigen::Vector2d(h, h);
        double d2_sq = q2.squaredNorm();

        bool use_d1 = d1_sq < d2_sq;
        double d_sq = use_d1 ? d1_sq : d2_sq;
        Eigen::Vector2d q_lower = use_d1 ? q1 : q2;
        double phi_lower = std::sqrt(d_sq);
        Eigen::Vector2d grad_temp = q_lower / std::max(phi_lower, 1e-10);
        double sign_mult = p.x() > p.y() ? 1.0 : -1.0;
        phi = phi_lower * sign_mult;
        grad = Eigen::Vector2d(sx * grad_temp.x(), grad_temp.y()) * sign_mult;
    }

    return {phi - isovalue, grad};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Heart2D::bounding_box(double isovalue) const {
    double r = std::sqrt(2.0) / 4.0;
    Eigen::Vector2d bbox_min(-0.25 - r, 0.0);
    Eigen::Vector2d bbox_max(0.25 + r, 0.75 + r);
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


Pentagon2D::Pentagon2D(double r) : r_(r) {
    m_ << 0.80901699, 0.58778525, 0.72654253;
    n_ << m_.x() * m_.x() - m_.y() * m_.y(), 2.0 * m_.x() * m_.y();
}

std::pair<double, Eigen::Vector2d> Pentagon2D::evaluate_with_gradient(const Eigen::Vector2d& q, double isovalue) const {
    double s = sign(q.x());
    Eigen::Vector2d p(std::abs(q.x()), q.y());
    double w1 = p.x() * m_.x() + p.y() * m_.y();
    double w2 = p.x() * n_.x() - p.y() * n_.y();
    p -= 2.0 * std::max(w1, 0.0) * Eigen::Vector2d(m_.x(), m_.y());
    p -= 2.0 * std::min(w2, 0.0) * Eigen::Vector2d(m_.x(), -m_.y());
    double t = clamp(p.x(), -r_ * m_.z(), r_ * m_.z());
    p -= Eigen::Vector2d(t, -r_);
    double d = p.norm() * sign(-p.y());
    Eigen::Matrix2d mat1, mat2;
    mat1 << -m_.x(), -m_.y(), m_.y(), -m_.x();
    mat2 << -n_.x(), -n_.y(), -n_.y(), n_.x();

    Eigen::Vector2d g = (w2 < 0.) ? mat1 * p : (w1 > 0.) ? mat2 * p : p;
    g.x() *= s;

    // Updated from Inigo's formula so gradients are computed to *S*DF, not UDF then signed
    Eigen::Vector2d grad;
    if (std::abs(t - (-r_ * m_.z())) < epsilon_dist || std::abs(t - (r_ * m_.z())) < epsilon_dist) {
        // vertex
        grad = g / d;
    } else {
        // On an edge
        if (w2 < 0.) {
            Eigen::Vector2d edge_normal(m_.y(), m_.x());
            grad = Eigen::Vector2d(s * edge_normal.x(), edge_normal.y());
        } else if (w1 > 0.) {
            Eigen::Vector2d edge_normal(n_.y(), -n_.x());
            grad = Eigen::Vector2d(s * edge_normal.x(), edge_normal.y());
        } else {
            grad = Eigen::Vector2d(0., -1);
        }
    }
    return {d - isovalue, grad};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Pentagon2D::bounding_box(double isovalue) const {
    double t = 2.0 * r_ * std::sqrt(5. - 2 * std::sqrt(5)); // sidelength
    double R = std::sqrt(6. - 2. * std::sqrt(5)) * r_;
    double diag = std::sqrt(1.0 - 2.0 / std::sqrt(5.)) * (5. + std::sqrt(5.)) * r_;
    Eigen::Vector2d bbox_min(-0.5 * diag, -r_);
    Eigen::Vector2d bbox_max(0.5 * diag, R);
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


std::pair<double, Eigen::Vector2d> Moon2D::evaluate_with_gradient(const Eigen::Vector2d& q, double isovalue) const {
    double s = sign(q.y());
    Eigen::Vector2d p(q.x(), std::abs(q.y()));

    double a = (ra_ * ra_ - rb_ * rb_ + d_ * d_) / (2.0 * d_);
    double b = std::sqrt(std::max(ra_ * ra_ - a * a, 0.0));
    bool cond = d_ * (p.x() * b - p.y() * a) > d_ * d_ * std::max(b - p.y(), 0.0);

    double phi;
    Eigen::Vector2d grad;

    if (cond) {
        // Distance to intersection point of two circles
        Eigen::Vector2d w = p - Eigen::Vector2d(a, b);
        double d_cond = w.norm();
        phi = d_cond;
        grad = Eigen::Vector2d(w.x(), w.y() * s) / d_cond;
    } else {
        // Distance to either circle
        Eigen::Vector2d w1 = p;
        double l1 = w1.norm();
        double d1 = l1 - ra_;

        Eigen::Vector2d w2 = p - Eigen::Vector2d(d_, 0.0);
        double l2 = w2.norm();
        double d2 = rb_ - l2;

        if (d1 > d2) {
            phi = d1;
            grad = Eigen::Vector2d(w1.x(), w1.y() * s) / l1;
        } else {
            phi = d2;
            grad = Eigen::Vector2d(-w2.x(), -w2.y() * s) / l2;
        }
    }

    return {phi - isovalue, grad};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Moon2D::bounding_box(double isovalue) const {
    double a = (ra_ * ra_ - rb_ * rb_ + d_ * d_) / (2.0 * d_);
    Eigen::Vector2d bbox_min(-ra_, -ra_);
    Eigen::Vector2d bbox_max(a, ra_);
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


std::pair<double, Eigen::Vector2d> Trapezoid2D::evaluate_with_gradient(const Eigen::Vector2d& q,
                                                                       double isovalue) const {

    // Updated from Inigo's formula so gradients are computed to *S*DF, not UDF then signed
    double sx = sign(q.x());
    double sy = sign(q.y());
    Eigen::Vector2d p = q;
    p.x() = std::abs(p.x());

    double h = std::min(p.x(), (p.y() < 0. ? ra_ : rb_));
    Eigen::Vector2d c(h, sy * he_);
    Eigen::Vector2d q_vec = p - c;
    double d = q_vec.dot(q_vec);
    double s = std::abs(p.y()) - he_;
    double res_d = d;
    Eigen::Vector2d res_q = q_vec;
    Eigen::Vector2d grad = sy * Eigen::Vector2d{0, 1};
    Eigen::Vector2d v0(0.5 * rb_, 0.5 * he_);
    Eigen::Vector2d v1(-0.5 * rb_, 0.5 * he_);
    Eigen::Vector2d v2(-0.5 * ra_, -0.5 * he_);
    Eigen::Vector2d v3(0.5 * ra_, -0.5 * he_);
    double h_max = p.y() < 0. ? ra_ : rb_;
    bool closest_vertex = false;
    if (std::abs(h) < epsilon_dist || std::abs(h - h_max) < epsilon_dist) {
        closest_vertex = true;
    }
    double res_s = s;
    Eigen::Vector2d ocl = c;

    Eigen::Vector2d k(rb_ - ra_, 2.0 * he_);
    Eigen::Vector2d w = p - Eigen::Vector2d{ra_, -he_};
    h = clamp(w.dot(k) / k.dot(k), 0., 1.);
    c = Eigen::Vector2d{ra_, -he_} + h * k;
    q_vec = p - c;
    d = q_vec.dot(q_vec);
    s = w.x() * k.y() - w.y() * k.x();
    res_s = std::max(res_s, s);
    if (d < res_d) {
        ocl = c;
        res_d = d;
        res_q = q_vec;
        if (std::abs(h) > epsilon_dist && std::abs(h - 1.) > epsilon_dist) {
            // side edge
            Eigen::Vector2d e = v0 - v3;
            grad = {e.y(), -e.x()};
            grad(0) *= sx;
            grad /= grad.norm();
        } else {
            closest_vertex = true;
        }
    }

    double d_final = std::sqrt(res_d) * sign(res_s);
    if (closest_vertex) {
        grad = {sx * res_q.x(), res_q.y()};
        grad *= sign(res_s);
        grad /= grad.norm();
    }

    return {d_final - isovalue, grad};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Trapezoid2D::bounding_box(double isovalue) const {
    double max_width = std::max(ra_, rb_);
    Eigen::Vector2d bbox_min(-max_width, -he_);
    Eigen::Vector2d bbox_max(max_width, he_);
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


std::pair<double, Eigen::Vector2d> Quad2D::evaluate_with_gradient(const Eigen::Vector2d& p, double isovalue) const {
    // Global sign for winding order
    double gs = (v_[0].x() - v_[3].x()) * (v_[1].y() - v_[0].y()) - (v_[0].y() - v_[3].y()) * (v_[1].x() - v_[0].x());

    double res_d = std::numeric_limits<double>::max();
    double res_s = -std::numeric_limits<double>::max();

    // Process each edge
    // Updated from Inigo's formula so gradients are computed to *S*DF, not UDF then signed
    Eigen::Vector2d grad;
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector2d e = v_[(i + 1) % 4] - v_[i];
        Eigen::Vector2d w = p - v_[i];
        double h = clamp(w.dot(e) / e.squaredNorm(), 0.0, 1.0);
        Eigen::Vector2d qe = w - h * e;
        double d = qe.squaredNorm();
        double s_curr = gs * cross(w, e);

        if (d < res_d) {
            res_d = d;
            if (std::abs(h) > epsilon_dist && std::abs(h - 1.) > epsilon_dist) {
                grad = {e.y(), -e.x()};
                grad /= grad.norm();
            } else {
                grad = qe / qe.norm();
            }
        }
        res_s = std::max(res_s, s_curr);
    }

    double phi = std::sqrt(res_d) * sign(res_s);

    return {phi - isovalue, grad};
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Quad2D::bounding_box(double isovalue) const {
    Eigen::Vector2d bbox_min = v_[0];
    Eigen::Vector2d bbox_max = v_[0];
    for (int i = 1; i < 4; ++i) {
        bbox_min = min(bbox_min, v_[i]);
        bbox_max = max(bbox_max, v_[i]);
    }
    return {bbox_min.array() - isovalue, bbox_max.array() + isovalue};
}


// ============================================================================
// BINDING CODE
// ============================================================================

void bind_shape_2d(nb::module_& m) {

    m.def("point_cloud_bounding_box_2d", &point_cloud_bounding_box_2d, "positions"_a,
          "Compute the bounding box of a point set.");

    // ========================================================================
    // Base SDF2D class
    // ========================================================================
    nb::class_<SDF2D>(m, "SDF2D")
        .def("evaluate", &SDF2D::evaluate, "q"_a, "isovalue"_a = 0.0, "Evaluate signed distance at a single point")
        .def("has_gradient", &SDF2D::has_gradient, "Return True if SDF has analytical gradient")
        .def("evaluate_with_gradient", &SDF2D::evaluate_with_gradient, "q"_a, "isovalue"_a = 0.0,
             "Evaluate signed distance and gradient at a single point")
        .def("bounding_box", &SDF2D::bounding_box, "isovalue"_a = 0.0, "Compute the bounding box")
        .def("evaluate_batch", &SDF2D::evaluate_batch, "queries"_a, "isovalue"_a = 0.0,
             "Batch evaluate signed distance at multiple points")
        .def("evaluate_gradient", &SDF2D::evaluate_gradient, "queries"_a,
             "Evaluate signed distance gradient at a single point")
        .def("evaluate_gradient_batch", &SDF2D::evaluate_gradient_batch, "queries"_a,
             "Batch evaluate signed distance gradient at multiple points")
        .def("evaluate_with_gradient_batch", &SDF2D::evaluate_with_gradient_batch, "queries"_a, "isovalue"_a = 0.0,
             "Batch evaluate signed distance and gradient")
        .def("evaluate_gradient_autodiff", &SDF2D::evaluate_gradient_autodiff, "q"_a, "isovalue"_a = 0.0,
             "Evaluate gradient using autodiff")

        .def("evaluate_signed_distance_gradient", &SDF2D::evaluate_signed_distance_gradient, "queries"_a,
             "is_extrusion"_a, "SDF3D_param"_a, "Evaluate gradient of 3D SDF")

        .def("evaluate_revolution", &SDF2D::evaluate_revolution, "queries"_a, "o"_a, "Evaluate revolved SDF")
        .def("evaluate_extrusion", &SDF2D::evaluate_extrusion, "queries"_a, "h"_a, "Evaluate extruded SDF")
        .def("sample_revolution", &SDF2D::sample_revolution, "n_samples"_a, "o"_a, "bbox_min"_a, "bbox_max"_a,
             "sigma_p"_a, "sigma_n"_a, "seed"_a, "Uniformly sample revolved SDF")
        .def("sample_extrusion", &SDF2D::sample_extrusion, "n_samples"_a, "h"_a, "bbox_min"_a, "bbox_max"_a,
             "sigma_p"_a, "sigma_n"_a, "seed"_a, "Uniformly sample extruded SDF")
        .def("sample_revolution_narrow_band", &SDF2D::sample_revolution_narrow_band, "n_samples"_a, "o"_a, "offset"_a,
             "bbox_min"_a, "bbox_max"_a, "seed"_a, "Uniformly sample narrow band of revolved SDF")
        .def("sample_extrusion_narrow_band", &SDF2D::sample_extrusion_narrow_band, "n_samples"_a, "h"_a, "offset"_a,
             "bbox_min"_a, "bbox_max"_a, "seed"_a, "Uniformly sample narrow band of extruded SDF")

        .def("sample_points_and_normals_uniformly", &SDF2D::sample_points_and_normals_uniformly, "n_points"_a,
             "is_extrusion"_a, "SDF3D_param"_a, "seed"_a, "Sample point cloud of 3D SDF")
        .def("sample_farthest_point_cloud", &SDF2D::sample_farthest_point_cloud, "n_points"_a, "is_extrusion"_a,
             "SDF3D_param"_a, "seed"_a, "Sample point cloud of 3D SDF")
        .def("sample_uneven_point_cloud", &SDF2D::sample_uneven_point_cloud, "is_extrusion"_a, "SDF3D_param"_a,
             "n_cameras"_a, "sensor_size"_a, "grid_spacing_min"_a, "grid_spacing_max"_a, "max_noise_level"_a,
             "max_normals_to_flip"_a, "max_points"_a, "seed"_a, "Sample point cloud of 3D SDF")

        .def("evaluate_laplacian_batch", &SDF2D::evaluate_laplacian_batch, "queries"_a,
             "Batch evaluate signed distance Laplacian")
        .def("ray_intersect", &SDF2D::ray_intersect, "ro"_a, "rd"_a,
             "Ray intersection using sphere tracing. Returns (position, normal, hit)")
        .def("ray_intersect_batch", &SDF2D::ray_intersect_batch, "ros"_a, "rds"_a,
             "Batch ray intersection. Returns (positions, normals)");


    // ========================================================================
    // Individual SDF shapes
    // ========================================================================
    nb::class_<Circle2D, SDF2D>(m, "Circle2D").def(nb::init<double>(), "r"_a, "Circle with radius r");

    nb::class_<Pie2D, SDF2D>(m, "Pie2D").def(nb::init<double, double>(), "r"_a, "t"_a, "Pie/pacman shape");

    nb::class_<Arc2D, SDF2D>(m, "Arc2D")
        .def(nb::init<const Eigen::Vector2d&, double, double>(), "sc"_a, "ra"_a, "rb"_a,
             "Arc with thickness. sc = (sin(angle), cos(angle))");

    nb::class_<Pill2D, SDF2D>(m, "Pill2D")
        .def(nb::init<const Eigen::Vector2d&, const Eigen::Vector2d&, double>(), "a"_a, "b"_a, "r"_a,
             "Pill/segment shape");

    nb::class_<Vesica2D, SDF2D>(m, "Vesica2D").def(nb::init<double, double>(), "r"_a, "d"_a, "Vesica shape");

    nb::class_<Box2D, SDF2D>(m, "Box2D").def(nb::init<const Eigen::Vector2d&>(), "b"_a, "Box with half-extents b");

    nb::class_<Cross2D, SDF2D>(m, "Cross2D").def(nb::init<const Eigen::Vector2d&>(), "b"_a, "Cross shape");

    nb::class_<TriangleIsosceles2D, SDF2D>(m, "TriangleIsosceles2D")
        .def(nb::init<const Eigen::Vector2d&>(), "q"_a, "Isosceles triangle");

    nb::class_<Triangle2D, SDF2D>(m, "Triangle2D")
        .def(nb::init<const Eigen::Vector2d&, const Eigen::Vector2d&, const Eigen::Vector2d&>(), "v0"_a, "v1"_a, "v2"_a,
             "General triangle");

    nb::class_<Hexagon2D, SDF2D>(m, "Hexagon2D").def(nb::init<double>(), "r"_a, "Regular hexagon");

    nb::class_<Ellipse2D, SDF2D>(m, "Ellipse2D")
        .def(nb::init<const Eigen::Vector2d&>(), "ab"_a, "Ellipse with semi-axes ab");

    nb::class_<Heart2D, SDF2D>(m, "Heart2D").def(nb::init<>(), "Heart shape");

    nb::class_<Pentagon2D, SDF2D>(m, "Pentagon2D").def(nb::init<double>(), "r"_a, "Regular pentagon");

    nb::class_<Moon2D, SDF2D>(m, "Moon2D")
        .def(nb::init<double, double, double>(), "d"_a, "ra"_a, "rb"_a, "Moon/crescent shape");

    nb::class_<Trapezoid2D, SDF2D>(m, "Trapezoid2D")
        .def(nb::init<double, double, double>(), "ra"_a, "rb"_a, "he"_a, "Isosceles trapezoid");

    nb::class_<Quad2D, SDF2D>(m, "Quad2D")
        .def(nb::init<const Eigen::Vector2d&, const Eigen::Vector2d&, const Eigen::Vector2d&, const Eigen::Vector2d&>(),
             "v0"_a, "v1"_a, "v2"_a, "v3"_a, "Quadrilateral");
}
