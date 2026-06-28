#include "torus_distance.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;
using namespace nb::literals;

//===== HELPERS

double compute_shift(const Eigen::Vector3d& query, const Eigen::Matrix<double, 3, Eigen::Dynamic>& positions,
                     const Eigen::Matrix<double, 3, Eigen::Dynamic>& normals, const Eigen::VectorXi& neighbors) {
    double max_dist = 0.;
    int n_neighbors = neighbors.size();
    for (int i = 0; i < n_neighbors; i++) {
        Eigen::Vector3d p_i = positions.col(neighbors(i));
        max_dist = std::max(max_dist, (query - p_i).norm());
    }
    return 0.5 * max_dist;
}

Eigen::VectorXd compute_shifts(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                               const Eigen::Matrix<double, 3, Eigen::Dynamic>& positions,
                               const Eigen::Matrix<double, 3, Eigen::Dynamic>& normals,
                               const Eigen::MatrixXi& neighbors) {

    int n_queries = queries.cols();
    Eigen::VectorXd shifts(n_queries);
#pragma omp parallel for schedule(static)
    for (int q_idx = 0; q_idx < n_queries; q_idx++) {
        Eigen::Vector3d query = queries.col(q_idx);
        shifts(q_idx) = compute_shift(queries.col(q_idx), positions, normals, neighbors.col(q_idx));
    }

    return shifts;
}

double TorusDistanceField::compute_shift(const Eigen::Vector3d& query, const Eigen::VectorXi& neighbors) const {
    return ::compute_shift(query, points, normals, neighbors);
}

// Compute shift from std::vector<int> neighbors
double TorusDistanceField::compute_shift(const Eigen::Vector3d& query, const std::vector<int>& neighbors) const {
    double max_dist = 0.;
    for (int idx : neighbors) {
        Eigen::Vector3d p_i = points.col(idx);
        max_dist = std::max(max_dist, (query - p_i).norm());
    }
    return 0.5 * max_dist;
}

Eigen::VectorXd TorusDistanceField::compute_shifts(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                   const Eigen::MatrixXi& neighbors) const {
    return ::compute_shifts(queries, points, normals, neighbors);
}

// Compute shifts for radius-based neighbors
Eigen::VectorXd TorusDistanceField::compute_shifts(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                   const std::vector<std::vector<int>>& neighbors) const {
    int n_queries = queries.cols();
    Eigen::VectorXd shifts(n_queries);
#pragma omp parallel for schedule(static)
    for (int q_idx = 0; q_idx < n_queries; q_idx++) {
        shifts(q_idx) = compute_shift(queries.col(q_idx), neighbors[q_idx]);
    }
    return shifts;
}


Eigen::VectorXi TorusDistanceField::nearest_points_single(const Eigen::Vector3d& q, const int& k_neighbors) const {

    Eigen::VectorXi indices(k_neighbors);
    std::vector<size_t> ret_indexes(k_neighbors);
    std::vector<double> out_dists_sqr(k_neighbors);

    nanoflann::KNNResultSet<double> resultSet(k_neighbors);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    kd_tree->index_->findNeighbors(resultSet, &q[0]);
    for (int i = 0; i < k_neighbors; i++) indices(i) = ret_indexes[i];

    return indices;
}

Eigen::MatrixXi TorusDistanceField::nearest_points(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                   const int& k_neighbors) const {

    int n_queries = queries.cols();
    Eigen::MatrixXi all_neighbors(k_neighbors, n_queries);

#pragma omp parallel for schedule(static)
    for (int q_idx = 0; q_idx < n_queries; q_idx++) {
        Eigen::VectorXi neighbors = nearest_points_single(queries.col(q_idx), k_neighbors);
        all_neighbors.col(q_idx) = neighbors;
    }

    return all_neighbors;
}

Eigen::MatrixXi TorusDistanceField::compute_nearest_points(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                           const int& k_neighbors) const {

    int n_queries = queries.cols();
    Eigen::MatrixXi neighbors;
    if (k_neighbors < 0 || k_neighbors > n_points) {
        neighbors.resize(n_points, n_queries);
        for (int q_idx = 0; q_idx < n_queries; q_idx++) {
            for (int i = 0; i < n_points; i++) {
                neighbors(i, q_idx) = i;
            }
        }
    } else {
        neighbors = nearest_points(queries, k_neighbors);
    }
    return neighbors;
}

std::vector<int> TorusDistanceField::points_within_radius_single(const Eigen::Vector3d& q, const double& radius) const {

    std::vector<nanoflann::ResultItem<my_kd_tree_t::IndexType, double>> ret_matches;
    nanoflann::SearchParameters params;
    params.sorted = false;
    // Since L2 norms are used, search radius and all passed and returned distances are actually squared distances (why
    // would you do this in the API???)
    const double radius_sq = radius * radius;
    const size_t nMatches = kd_tree->index_->radiusSearch(&q[0], radius_sq, ret_matches, params);

    std::vector<int> indices(nMatches);
    for (size_t i = 0; i < nMatches; i++) indices[i] = ret_matches[i].first;

    return indices;
}

std::vector<std::vector<int>>
TorusDistanceField::compute_points_within_radius(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                 const double& radius) const {
    int n_queries = queries.cols();
    std::vector<std::vector<int>> all_neighbors(n_queries);
#pragma omp parallel for schedule(dynamic)
    for (int q_idx = 0; q_idx < n_queries; q_idx++) {
        std::vector<int> neighbors = points_within_radius_single(queries.col(q_idx), radius);
        all_neighbors[q_idx] = neighbors;
    }
    return all_neighbors;
}

// Compute the signed distance of the SDF of a single given torus.
double TorusDistanceField::torus_signed_distance(const Eigen::Vector3d& q, const Eigen::Vector3d& center,
                                                 const Eigen::Vector2d& axis, const double& major_radius,
                                                 const double& minor_radius) const {

    Eigen::Vector3d axis3d = rotation_to_cartesian(axis);
    Eigen::Vector3d r_vec = q - center;
    double v = r_vec.cross(axis3d).norm();
    double d1 = v - major_radius;
    double d2 = r_vec.dot(axis3d);
    Eigen::Vector2d d(d1, d2);
    double udf = d.norm() - std::abs(minor_radius);
    double s = sign(minor_radius);
    double phi = s * udf;
    return phi;
}

Eigen::VectorXd TorusDistanceField::torus_signed_distances(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                           const Eigen::Vector3d& center, const Eigen::Vector2d& axis,
                                                           const double& major_radius,
                                                           const double& minor_radius) const {
    int n_queries = queries.cols();
    Eigen::VectorXd distances(n_queries);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_queries; i++) {
        distances(i) = torus_signed_distance(queries.col(i), center, axis, major_radius, minor_radius);
    }
    return distances;
}

// Compute the signed distance and gradient of the SDF of a single given torus.
std::pair<double, Eigen::Vector3d>
TorusDistanceField::torus_signed_distance_gradient(const Eigen::Vector3d& q, const Eigen::Vector3d& center,
                                                   const Eigen::Vector2d& axis, const double& major_radius,
                                                   const double& minor_radius) const {
    Eigen::Vector3d axis3d = rotation_to_cartesian(axis);
    Eigen::Vector3d r_vec = q - center;
    double v = r_vec.cross(axis3d).norm();
    double d1 = v - major_radius;
    double d2 = r_vec.dot(axis3d);
    Eigen::Vector2d d(d1, d2);
    double udf = d.norm() - std::abs(minor_radius);
    double s = sign(minor_radius);
    double phi = s * udf;

    // Entries of a 2x3 matrix
    double G_11 = (axis3d.x() * axis3d.y() * (center.y() - q.y()) + axis3d.x() * axis3d.z() * (center.z() - q.z()) -
                   (axis3d.y() * axis3d.y() + axis3d.z() * axis3d.z()) * (center.x() - q.x())) /
                  v;
    double G_12 = (axis3d.x() * axis3d.y() * (center.x() - q.x()) + axis3d.y() * axis3d.z() * (center.z() - q.z()) -
                   (axis3d.x() * axis3d.x() + axis3d.z() * axis3d.z()) * (center.y() - q.y())) /
                  v;
    double G_13 = (axis3d.x() * axis3d.z() * (center.x() - q.x()) + axis3d.y() * axis3d.z() * (center.y() - q.y()) -
                   (axis3d.x() * axis3d.x() + axis3d.y() * axis3d.y()) * (center.z() - q.z())) /
                  v;
    double G_21 = axis3d.x();
    double G_22 = axis3d.y();
    double G_23 = axis3d.z();
    Eigen::Vector2d w = d / d.norm();
    Eigen::Vector3d grad = s * Eigen::Vector3d(w.x() * G_11 + w.y() * G_21, w.x() * G_12 + w.y() * G_22,
                                               w.x() * G_13 + w.y() * G_23); // (w^T * G)^T

    return {phi, grad};
}

std::pair<Eigen::VectorXd, Eigen::Matrix<double, 3, Eigen::Dynamic>>
TorusDistanceField::torus_signed_distances_gradients(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                     const Eigen::Vector3d& center, const Eigen::Vector2d& axis,
                                                     const double& major_radius, const double& minor_radius) const {

    int n_queries = queries.cols();
    Eigen::VectorXd distances(n_queries);
    Eigen::MatrixXd gradients(3, n_queries);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_queries; i++) {
        std::pair<double, Eigen::Vector3d> result =
            torus_signed_distance_gradient(queries.col(i), center, axis, major_radius, minor_radius);
        distances(i) = result.first;
        gradients.col(i) = result.second;
    }
    return {distances, gradients};
}


//===== SINGLE QUERIES

double TorusDistanceField::evaluate_distance_single(const Eigen::Vector3d& q, const double& isovalue,
                                                    const double& shift_, const std::vector<int>& neighbors_) const {
    double normalization = 0.;

    // Determine which points to sum over
    std::vector<int> neighbors;
    if (neighbors_.empty()) {
        // No neighbors provided, determine based on evaluation mode
        if (radius_evaluation > 0) {
            // Radius-based evaluation
            neighbors = points_within_radius_single(q, radius_evaluation);
            // Fallback to k-NN if no points found
            if (neighbors.empty()) {
                int k = (k_evaluation > 0 && k_evaluation < n_points) ? k_evaluation : std::min(10, n_points);
                Eigen::VectorXi knn = nearest_points_single(q, k);
                neighbors.resize(k);
                for (int i = 0; i < k; i++) neighbors[i] = knn(i);
            }
        } else if (k_evaluation > 0 && k_evaluation < n_points) {
            // k-NN based evaluation
            Eigen::VectorXi knn = nearest_points_single(q, k_evaluation);
            neighbors.resize(k_evaluation);
            for (int i = 0; i < k_evaluation; i++) neighbors[i] = knn(i);
        } else {
            // Use all points
            neighbors.resize(n_points);
            for (int i = 0; i < n_points; i++) neighbors[i] = i;
        }
    } else {
        neighbors = neighbors_;
    }

    double shift = std::isnan(shift_) ? compute_shift(q, neighbors) : shift_;
    double lam = compute_lambda(shift) * lambda_scale;

    // Scale lambda for radius-based truncation
    // I can't remember why I did this: I think I was trying to scale by the ratio of radius_evaluation to the radius of
    // the box [-1,1]^3. So in essence, we are always scaling the radius of evaluation so that it roughly matches the
    // "usual" radius corresponding to [-1,1]^3 (i.e. if we were to sum over the whole point cloud as usual).
    if (radius_evaluation > 0 && neighbors.size() < static_cast<size_t>(n_points)) {
        lam *= radius_evaluation / std::sqrt(3.0);
    }

    double g = 0.;
    for (int i : neighbors) {
        if (is_outlier(i) == 1) continue;
        const Eigen::Vector3d& p_i = points.col(i);
        Eigen::Vector3d rVec = q - p_i;
        double d = rVec.norm();
        Eigen::Vector3d rHat = rVec / d;
        double weight = std::exp(-lam * (d - shift));
        if (use_areas) {
            weight *= areas(i);
        }
        Eigen::Vector3d center = centers.col(i);
        double g_z = torus_signed_distance(q, center, axes.col(i), major_radii(i), minor_radii(i));
        g += weight * g_z;
        normalization += weight;
    }

    double dist = g / normalization;
    return dist - isovalue;
}

Eigen::Vector3d TorusDistanceField::evaluate_gradient_single(const Eigen::Vector3d& query, const double& shift,
                                                             const std::vector<int>& neighbors) const {
    // Call generic version
    std::tuple<double, Eigen::Vector3d, double> result =
        evaluate_distance_gradient_laplacian_single(query, 0., shift, neighbors);
    return std::get<1>(result);
}

double TorusDistanceField::evaluate_laplacian_single(const Eigen::Vector3d& query, const double& shift,
                                                     const std::vector<int>& neighbors) const {
    // Call generic version
    std::tuple<double, Eigen::Vector3d, double> result =
        evaluate_distance_gradient_laplacian_single(query, 0., shift, neighbors);
    return std::get<2>(result);
}

std::tuple<double, Eigen::Vector3d, double> TorusDistanceField::evaluate_distance_gradient_laplacian_single(
    const Eigen::Vector3d& q, const double& isovalue, const double& shift_, const std::vector<int>& neighbors_) const {

    double normalization = 0.;

    // Determine which points to sum over
    std::vector<int> neighbors;
    if (neighbors_.empty()) {
        // No neighbors provided, determine based on evaluation mode
        if (radius_evaluation > 0) {
            // Radius-based evaluation
            neighbors = points_within_radius_single(q, radius_evaluation);
            // Fallback to k-NN if no points found
            if (neighbors.empty()) {
                int k = (k_evaluation > 0 && k_evaluation < n_points) ? k_evaluation : std::min(10, n_points);
                Eigen::VectorXi knn = nearest_points_single(q, k);
                neighbors.resize(k);
                for (int i = 0; i < k; i++) neighbors[i] = knn(i);
            }
        } else if (k_evaluation > 0 && k_evaluation < n_points) {
            // k-NN based evaluation
            Eigen::VectorXi knn = nearest_points_single(q, k_evaluation);
            neighbors.resize(k_evaluation);
            for (int i = 0; i < k_evaluation; i++) neighbors[i] = knn(i);
        } else {
            // Use all points
            neighbors.resize(n_points);
            for (int i = 0; i < n_points; i++) neighbors[i] = i;
        }
    } else {
        neighbors = neighbors_;
    }

    double shift = std::isnan(shift_) ? compute_shift(q, neighbors) : shift_;
    double lam = compute_lambda(shift) * lambda_scale;

    // Scale lambda for radius-based truncation
    // I can't remember why I did this: I think I was trying to scale by the ratio of radius_evaluation to the radius of
    // the box [-1,1]^3. So in essence, we are always scaling the radius of evaluation so that it roughly matches the
    // "usual" radius corresponding to [-1,1]^3 (i.e. if we were to sum over the whole point cloud as usual).
    if (radius_evaluation > 0 && neighbors.size() < static_cast<size_t>(n_points)) {
        lam *= radius_evaluation / std::sqrt(3.0);
    }

    Eigen::Vector3d grad_g_minus_lambda_n_g(0, 0, 0);
    double g = 0.;
    Eigen::Vector3d n(0., 0., 0.);
    double numer1 = 0.; // Δg(z) - 2λ< rHat, ∇g(x) > + g(z)( λ^2 - λ(d-1)/|x-z| )
    double numer2 = 0.; // (d-1)/|x-z| - λ

    for (int i : neighbors) {
        if (is_outlier(i) == 1) continue;
        const Eigen::Vector3d& p_i = points.col(i);
        Eigen::Vector3d rVec = q - p_i;
        double d = rVec.norm();
        Eigen::Vector3d rHat = rVec / d;
        double weight = std::exp(-lam * (d - shift));
        if (use_areas) {
            weight *= areas(i);
        }
        Eigen::Vector3d center = centers.col(i);
        Eigen::Vector3d cVec = q - center;
        double d_to_center = cVec.norm();

        std::pair<double, Eigen::Vector3d> torus_distance =
            torus_signed_distance_gradient(q, center, axes.col(i), major_radii(i), minor_radii(i));
        double g_z = torus_distance.first;
        Eigen::Vector3d grad_g = torus_distance.second;
        double s = sign(minor_radii(i));

        grad_g_minus_lambda_n_g += weight * (grad_g - lam * rHat * g_z);
        g += weight * g_z;
        n += weight * rHat;
        double term1 = s * (dimension - 1) / d_to_center; // TODO: Δg(z)
        double term2 = -2. * lam * rHat.dot(grad_g);
        double term3 = lam * g_z * (lam - (dimension - 1) / d);
        numer1 += weight * (term1 + term2 + term3);
        numer2 += weight * ((dimension - 1) / d - lam);
        normalization += weight;
    }

    Eigen::Vector3d A = grad_g_minus_lambda_n_g / normalization;
    Eigen::Vector3d B = lam * n / normalization;
    double dist = g / normalization;
    Eigen::Vector3d grad = A + dist * B;
    double first_term = numer1 / normalization + A.dot(B);
    double second_term = grad.dot(B) + dist * (lam * numer2 / normalization + B.dot(B));

    return {dist - isovalue, grad, first_term + second_term};
}

//===== BATCH QUERIES

Eigen::VectorXd TorusDistanceField::evaluate_distance(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                      const double& isovalue, bool parallelize) const {

    std::chrono::time_point<high_resolution_clock> t1, t2;
    std::chrono::duration<double, std::milli> ms_fp;
    t1 = high_resolution_clock::now();

    int n_queries = queries.cols();
    Eigen::VectorXd distances(n_queries);

    int mp = parallelize;

    if (radius_evaluation > 0) {
        std::chrono::time_point<high_resolution_clock> s1, s2;
        std::chrono::duration<double, std::milli> s_ms_fp;
        s1 = high_resolution_clock::now();

        // Radius-based evaluation
        std::vector<std::vector<int>> neighbors = compute_points_within_radius(queries, radius_evaluation);
        Eigen::VectorXd shifts = compute_shifts(queries, neighbors);

#pragma omp parallel for schedule(dynamic) if (mp)
        for (int q_idx = 0; q_idx < n_queries; q_idx++) {
            // If no neighbors found, pass empty vector and NaN shift to trigger k-NN fallback
            double shift = neighbors[q_idx].empty() ? SPECIAL_VALUE : shifts(q_idx);
            double dist = evaluate_distance_single(queries.col(q_idx), isovalue, shift, neighbors[q_idx]);
            distances(q_idx) = dist;
        }
        s2 = high_resolution_clock::now();
        s_ms_fp = s2 - s1;
        std::cerr << "Tree traversal time (s): " << s_ms_fp.count() / 1000. << std::endl;
    } else {
        // k-NN based evaluation (original behavior)
        Eigen::MatrixXi neighbors = compute_nearest_points(queries, k_evaluation);
        Eigen::VectorXd shifts = compute_shifts(queries, neighbors);

#pragma omp parallel for schedule(static) if (mp)
        for (int q_idx = 0; q_idx < n_queries; q_idx++) {
            // Convert Eigen::VectorXi to std::vector<int>
            Eigen::VectorXi nbrs_eigen = neighbors.col(q_idx);
            std::vector<int> nbrs(nbrs_eigen.data(), nbrs_eigen.data() + nbrs_eigen.size());
            double dist = evaluate_distance_single(queries.col(q_idx), isovalue, shifts(q_idx), nbrs);
            distances(q_idx) = dist;
        }
    }

    t2 = high_resolution_clock::now();
    ms_fp = t2 - t1;
    std::cerr << "TorusDistanceField::evaluate_distance C++ time (s): " << ms_fp.count() / 1000. << std::endl;

    return distances;
}

Eigen::Matrix<double, 3, Eigen::Dynamic>
TorusDistanceField::evaluate_gradient(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries, bool parallelize) const {
    // Call generic version.
    std::tuple<Eigen::VectorXd, Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::VectorXd> result =
        evaluate_distance_gradient_laplacian(queries, 0., parallelize);
    return std::get<1>(result);
}

Eigen::VectorXd TorusDistanceField::evaluate_laplacian(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                       bool parallelize) const {
    // Call generic version.
    std::tuple<Eigen::VectorXd, Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::VectorXd> result =
        evaluate_distance_gradient_laplacian(queries, 0., parallelize);
    return std::get<2>(result);
}

std::tuple<Eigen::VectorXd, Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::VectorXd>
TorusDistanceField::evaluate_distance_gradient_laplacian(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                         const double& isovalue, bool parallelize) const {

    std::chrono::time_point<high_resolution_clock> t1, t2;
    std::chrono::duration<double, std::milli> ms_fp;
    t1 = high_resolution_clock::now();

    int n_queries = queries.cols();
    Eigen::VectorXd distances(n_queries);
    Eigen::MatrixXd gradients(3, n_queries);
    Eigen::VectorXd laplacians(n_queries);

    int mp = parallelize;

    if (radius_evaluation > 0) {
        // Radius-based evaluation
        std::vector<std::vector<int>> neighbors = compute_points_within_radius(queries, radius_evaluation);
        Eigen::VectorXd shifts = compute_shifts(queries, neighbors);

#pragma omp parallel for schedule(dynamic) if (mp)
        for (int q_idx = 0; q_idx < n_queries; q_idx++) {
            // If no neighbors found, pass empty vector and NaN shift to trigger k-NN fallback
            double shift = neighbors[q_idx].empty() ? SPECIAL_VALUE : shifts(q_idx);
            std::tuple<double, Eigen::Vector3d, double> result =
                evaluate_distance_gradient_laplacian_single(queries.col(q_idx), isovalue, shift, neighbors[q_idx]);
            distances(q_idx) = std::get<0>(result);
            gradients.col(q_idx) = std::get<1>(result);
            laplacians(q_idx) = std::get<2>(result);
        }
    } else {
        // k-NN based evaluation
        Eigen::MatrixXi neighbors = compute_nearest_points(queries, k_evaluation);
        Eigen::VectorXd shifts = compute_shifts(queries, neighbors);

#pragma omp parallel for schedule(static) if (mp)
        for (int q_idx = 0; q_idx < n_queries; q_idx++) {
            // Convert Eigen::VectorXi to std::vector<int>
            Eigen::VectorXi nbrs_eigen = neighbors.col(q_idx);
            std::vector<int> nbrs(nbrs_eigen.data(), nbrs_eigen.data() + nbrs_eigen.size());
            std::tuple<double, Eigen::Vector3d, double> result =
                evaluate_distance_gradient_laplacian_single(queries.col(q_idx), isovalue, shifts(q_idx), nbrs);
            distances(q_idx) = std::get<0>(result);
            gradients.col(q_idx) = std::get<1>(result);
            laplacians(q_idx) = std::get<2>(result);
        }
    }

    t2 = high_resolution_clock::now();
    ms_fp = t2 - t1;
    // std::cerr << "C++ evaluate_distance_gradient_laplacian time (s): " << ms_fp.count() / 1000. << std::endl;

    return {distances, gradients, laplacians};
}


//===== TORUS FITTING

// Return a00, kappa_min, kappa_max, v_min, normal at p_i
std::tuple<double, double, double, Eigen::Vector3d, Eigen::Vector3d>
TorusDistanceField::get_torus_parameters(int i, const Eigen::Matrix<double, 6, 1>& coeffs, const Eigen::Vector3d& s_hat,
                                         const Eigen::Vector3d& t_hat) const {

    Eigen::Vector3d p_i = points.col(i);
    Eigen::Vector3d n_i = normals.col(i);

    double a00 = coeffs(0);
    double a01 = coeffs(1);
    double a10 = coeffs(2);
    double a11 = coeffs(3);
    double a02 = coeffs(4);
    double a20 = coeffs(5);

    // Expressions from bashing Mathematica
    double A = std::sqrt(1. + a01 * a01 + a10 * a10);
    double v1 = a20 * a01 * a01 - a02 * a10 * a10 - a02 + a20;
    double v2 = a11 * (a10 * a10 + 1.0) - 2.0 * a01 * a10 * a20;
    double term = a02 * (a10 * a10 + 1.0) - a01 * a10 * a11 + a20 * (a01 * a01 + 1.0);
    double discriminant = (1.0 + a01 * a01 + a10 * a10) * (a11 * a11 - 4.0 * a02 * a20) + term * term;
    discriminant = std::sqrt(std::max(discriminant, 0.));
    Eigen::Vector2d ev_pos(v1 + discriminant, v2);
    Eigen::Vector2d ev_neg(v1 - discriminant, v2);
    Eigen::Vector3d dQstards = s_hat + a10 * n_i;
    Eigen::Vector3d dQstardt = t_hat + a01 * n_i;
    Eigen::Vector3d v_pos = ev_pos.x() * dQstards + ev_pos.y() * dQstardt;
    Eigen::Vector3d v_neg = ev_neg.x() * dQstards + ev_neg.y() * dQstardt;
    Eigen::Vector3d normal = (n_i - a10 * s_hat - a01 * t_hat) / A; // n^*(0,0)

    // Compute principal curvatures using Rayleigh quotient
    Eigen::Matrix2d I, II;
    I << 1. + a10 * a10, a10 * a01, a10 * a01, 1. + a01 * a01;
    II << 2.0 * a20, a11, a11, 2.0 * a02;
    II /= A;
    double kappa_pos = ev_pos.dot(II * ev_pos) / ev_pos.dot(I * ev_pos);
    double kappa_neg = ev_neg.dot(II * ev_neg) / ev_neg.dot(I * ev_neg);

    double kappa_min, kappa_max;
    Eigen::Vector3d v_min;
    if (std::abs(kappa_pos) < std::abs(kappa_neg)) {
        kappa_min = kappa_pos;
        kappa_max = kappa_neg;
        v_min = v_pos;
    } else {
        kappa_min = kappa_neg;
        kappa_max = kappa_pos;
        v_min = v_neg;
    }

    return {a00, kappa_min, kappa_max, v_min, normal};
}

std::tuple<double, double, Eigen::Vector3d, Eigen::Vector3d>
TorusDistanceField::get_principal_directions_at_point(int i, const Eigen::Matrix<double, 6, 1>& coeffs,
                                                      const Eigen::Vector3d& s_hat,
                                                      const Eigen::Vector3d& t_hat) const {

    Eigen::Vector3d p_i = points.col(i);
    Eigen::Vector3d n_i = normals.col(i);

    double a00 = coeffs(0);
    double a01 = coeffs(1);
    double a10 = coeffs(2);
    double a11 = coeffs(3);
    double a02 = coeffs(4);
    double a20 = coeffs(5);

    // Expressions from bashing Mathematica
    double A = std::sqrt(1. + a01 * a01 + a10 * a10);
    double v1 = a20 * a01 * a01 - a02 * a10 * a10 - a02 + a20;
    double v2 = a11 * (a10 * a10 + 1.0) - 2.0 * a01 * a10 * a20;
    double term = a02 * (a10 * a10 + 1.0) - a01 * a10 * a11 + a20 * (a01 * a01 + 1.0);
    double discriminant = (1.0 + a01 * a01 + a10 * a10) * (a11 * a11 - 4.0 * a02 * a20) + term * term;
    discriminant = std::sqrt(std::max(discriminant, 0.));
    Eigen::Vector2d ev_pos(v1 + discriminant, v2);
    Eigen::Vector2d ev_neg(v1 - discriminant, v2);
    Eigen::Vector3d dQstards = s_hat + a10 * n_i;
    Eigen::Vector3d dQstardt = t_hat + a01 * n_i;
    Eigen::Vector3d v_pos = ev_pos.x() * dQstards + ev_pos.y() * dQstardt;
    Eigen::Vector3d v_neg = ev_neg.x() * dQstards + ev_neg.y() * dQstardt;
    Eigen::Vector3d normal = (n_i - a10 * s_hat - a01 * t_hat) / A; // n^*(0,0)

    // Compute principal curvatures using Rayleigh quotient
    Eigen::Matrix2d I, II;
    I << 1. + a10 * a10, a10 * a01, a10 * a01, 1. + a01 * a01;
    II << 2.0 * a20, a11, a11, 2.0 * a02;
    II /= A;
    double kappa_pos = ev_pos.dot(II * ev_pos) / ev_pos.dot(I * ev_pos);
    double kappa_neg = ev_neg.dot(II * ev_neg) / ev_neg.dot(I * ev_neg);

    double kappa_min, kappa_max;
    Eigen::Vector3d v_min, v_max;
    if (std::abs(kappa_pos) < std::abs(kappa_neg)) {
        kappa_min = kappa_pos;
        kappa_max = kappa_neg;
        v_min = v_pos;
        v_max = v_neg;
    } else {
        kappa_min = kappa_neg;
        kappa_max = kappa_pos;
        v_min = v_neg;
        v_max = v_pos;
    }

    return {kappa_min, kappa_max, v_min, v_max};
}

// Return a00, kappa_min, kappa_max, v_min, normal at p_i
std::tuple<double, double, double, Eigen::Vector3d, Eigen::Vector3d>
TorusDistanceField::get_torus_parameters_python(int i, const Eigen::Matrix<double, 6, 1>& coeffs,
                                                const Eigen::Vector3d& s_hat, const Eigen::Vector3d& t_hat) const {

    Eigen::Vector3d p_i = points.col(i);
    Eigen::Vector3d n_i = normals.col(i);

    double a00 = coeffs(0);
    double a01 = coeffs(1);
    double a10 = coeffs(2);
    double a11 = coeffs(3);
    double a02 = coeffs(4);
    double a20 = coeffs(5);

    // Expressions from bashing Mathematica
    double A = std::sqrt(1. + a01 * a01 + a10 * a10);
    double v1 = a20 * a01 * a01 - a02 * a10 * a10 - a02 + a20;
    double v2 = a11 * (a10 * a10 + 1.0) - 2.0 * a01 * a10 * a20;
    double term = a02 * (a10 * a10 + 1.0) - a01 * a10 * a11 + a20 * (a01 * a01 + 1.0);
    double discriminant = (1.0 + a01 * a01 + a10 * a10) * (a11 * a11 - 4.0 * a02 * a20) + term * term;
    discriminant = std::sqrt(std::max(discriminant, NAN_EPSILON)); // reproduce Python function used in training

    double ev_pos_0 = v1 + discriminant;
    ev_pos_0 = sign(ev_pos_0) * std::max(std::abs(ev_pos_0), NAN_EPSILON);
    double ev_neg_1 = v2;
    ev_neg_1 = sign(ev_neg_1) * std::max(std::abs(ev_neg_1), NAN_EPSILON);

    Eigen::Vector2d ev_pos(ev_pos_0, v2);
    Eigen::Vector2d ev_neg(v1 - discriminant, ev_neg_1);

    ev_pos /= ev_pos.norm();
    ev_neg /= ev_neg.norm();

    Eigen::Vector3d dQstards = s_hat + a10 * n_i;
    Eigen::Vector3d dQstardt = t_hat + a01 * n_i;
    Eigen::Vector3d v_pos = ev_pos.x() * dQstards + ev_pos.y() * dQstardt;
    Eigen::Vector3d v_neg = ev_neg.x() * dQstards + ev_neg.y() * dQstardt;
    Eigen::Vector3d normal = (n_i - a10 * s_hat - a01 * t_hat) / A; // n^*(0,0)

    // Compute principal curvatures using Rayleigh quotient
    Eigen::Matrix2d I, II;
    I << 1. + a10 * a10, a10 * a01, a10 * a01, 1. + a01 * a01;
    II << 2.0 * a20, a11, a11, 2.0 * a02;
    II /= A;
    double denom_pos = ev_pos.dot(I * ev_pos);
    double denom_neg = ev_neg.dot(I * ev_neg);
    denom_pos = sign(denom_pos) * std::max(std::abs(denom_pos), NAN_EPSILON);
    denom_neg = sign(denom_neg) * std::max(std::abs(denom_neg), NAN_EPSILON);
    double kappa_pos = ev_pos.dot(II * ev_pos) / denom_pos;
    double kappa_neg = ev_neg.dot(II * ev_neg) / denom_neg;

    double kappa_min, kappa_max;
    Eigen::Vector3d v_min;
    if (std::abs(kappa_pos) < std::abs(kappa_neg)) {
        kappa_min = kappa_pos;
        kappa_max = kappa_neg;
        v_min = v_pos;
    } else {
        kappa_min = kappa_neg;
        kappa_max = kappa_pos;
        v_min = v_neg;
    }

    return {a00, kappa_min, kappa_max, v_min, normal};
}

// Return a00, kappa_min, kappa_max, v_min, normal at p_i
std::tuple<double, double, double, Eigen::Vector3d, Eigen::Vector3d>
TorusDistanceField::get_polynomial_curvatures(int i, const Eigen::Matrix<double, 9, 1>& coeffs,
                                              const Eigen::Vector3d& s_hat, const Eigen::Vector3d& t_hat) const {

    Eigen::Vector3d p_i = points.col(i);
    Eigen::Vector3d n_i = normals.col(i);

    double a00 = coeffs(3 * 0 + 0);
    double a01 = coeffs(3 * 0 + 1);
    double a10 = coeffs(3 * 1 + 0);
    double a11 = coeffs(3 * 1 + 1);
    double a02 = coeffs(3 * 0 + 2);
    double a20 = coeffs(3 * 2 + 0);

    // Calculation using mean/Gaussian curvature
    // double A = std::sqrt(1. + a01 * a01 + a10 * a10);
    // double A3 = A * A * A;
    // double A4 = A3 * A;
    // double H = (a02 * (1 + a10 * a10) + a20 * (1 + a01 * a01) - a11 * a10 * a01) / A3;
    // double K = (4. * a02 * a20 - a11 * a11) / A4;
    // double discr = std::sqrt(std::max(H * H - K, 0.));
    // double kappa_pos = H + discr;
    // double kappa_neg = H - discr;
    // Eigen::Vector2d ev_pos(kappa_pos * a10 * a01 * A - a11, kappa_pos * (1. + a10 * a10) * A - 2. * a20);
    // Eigen::Vector2d ev_neg(kappa_neg * a10 * a01 * A - a11, kappa_neg * (1. + a10 * a10) * A - 2. * a20);

    // Expressions from bashing Mathematica
    double A = std::sqrt(1. + a01 * a01 + a10 * a10);
    double v1 = a20 * a01 * a01 - a02 * a10 * a10 - a02 + a20;
    double v2 = a11 * (a10 * a10 + 1.0) - 2.0 * a01 * a10 * a20;
    double term = a02 * (a10 * a10 + 1.0) - a01 * a10 * a11 + a20 * (a01 * a01 + 1.0);
    double discriminant = (1.0 + a01 * a01 + a10 * a10) * (a11 * a11 - 4.0 * a02 * a20) + term * term;
    discriminant = std::sqrt(std::max(discriminant, 0.));
    Eigen::Vector2d ev_pos(v1 + discriminant, v2);
    Eigen::Vector2d ev_neg(v1 - discriminant, v2);
    Eigen::Vector3d dQstards = s_hat + a10 * n_i;
    Eigen::Vector3d dQstardt = t_hat + a01 * n_i;
    Eigen::Vector3d v_pos = ev_pos.x() * dQstards + ev_pos.y() * dQstardt;
    Eigen::Vector3d v_neg = ev_neg.x() * dQstards + ev_neg.y() * dQstardt;
    Eigen::Vector3d normal = (n_i - a10 * s_hat - a01 * t_hat) / A; // n^*(0,0)

    // Compute principal curvatures using Rayleigh quotient
    Eigen::Matrix2d I, II;
    I << 1. + a10 * a10, a10 * a01, a10 * a01, 1. + a01 * a01;
    II << 2.0 * a20, a11, a11, 2.0 * a02;
    II /= A;
    double kappa_pos = ev_pos.dot(II * ev_pos) / ev_pos.dot(I * ev_pos);
    double kappa_neg = ev_neg.dot(II * ev_neg) / ev_neg.dot(I * ev_neg);

    double kappa_min, kappa_max;
    Eigen::Vector3d v_min;
    if (std::abs(kappa_pos) < std::abs(kappa_neg)) {
        kappa_min = kappa_pos;
        kappa_max = kappa_neg;
        v_min = v_pos;
    } else {
        kappa_min = kappa_neg;
        kappa_max = kappa_pos;
        v_min = v_neg;
    }

    return {a00, kappa_min, kappa_max, v_min, normal};
}

// Fit a single torus at point i, directly from fundamental form coefficients
std::tuple<Eigen::Vector3d, Eigen::Vector2d, double, double>
TorusDistanceField::fit_torus_at_point(int i, const Eigen::Matrix<double, 6, 1>& coeffs) const {
    Eigen::Vector3d p_i = points.col(i);
    Eigen::Vector3d n_i = normals.col(i);

    std::pair<Eigen::Vector3d, Eigen::Vector3d> basis = orthonormal_basis(n_i);
    Eigen::Vector3d s_hat = basis.first;
    Eigen::Vector3d t_hat = basis.second;

    std::tuple<double, double, double, Eigen::Vector3d, Eigen::Vector3d> result =
        get_torus_parameters_python(i, coeffs, s_hat, t_hat);
    double a00 = std::get<0>(result);
    double kappa_min = std::get<1>(result);
    double kappa_max = std::get<2>(result);
    Eigen::Vector3d v_min = std::get<3>(result);
    Eigen::Vector3d normal = std::get<4>(result);

    double r = 1. / std::abs(kappa_max);
    double R = 1. / std::abs(kappa_min) - sgn(kappa_max * kappa_min) * r;
    double s = ((kappa_max < 0. && kappa_min < 0.) || (kappa_max < 0 && kappa_min > 0.)) ? 1. : -1.;

    Eigen::Vector3d center = p_i + a00 * n_i - s / std::abs(kappa_min) * normal;
    Eigen::Vector3d u = normal.cross(v_min);
    u /= std::max(u.norm(), NAN_EPSILON); // reproduce Python results
    Eigen::Vector2d axis = cartesian_to_rotation(u);

    return {center, axis, R, s * r};
}

void TorusDistanceField::fit_tori_from_forms(const Eigen::Matrix<double, 6, Eigen::Dynamic>& coefficients) {

    centers.resize(3, n_points);
    axes.resize(2, n_points);
    major_radii.resize(n_points);
    minor_radii.resize(n_points);

    // TODO: Option for GPU acceleration

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_points; ++i) {
        Eigen::VectorXd coeffs = coefficients.col(i);
        std::tuple<Eigen::Vector3d, Eigen::Vector2d, double, double> result = fit_torus_at_point(i, coeffs);
        centers.col(i) = std::get<0>(result);
        axes.col(i) = std::get<1>(result);
        major_radii(i) = std::get<2>(result);
        minor_radii(i) = std::get<3>(result);
    }
}

std::tuple<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 2, Eigen::Dynamic>, Eigen::VectorXd,
           Eigen::VectorXd>
fit_tori_from_forms(const Eigen::Matrix<double, 3, Eigen::Dynamic>& points,
                    const Eigen::Matrix<double, 3, Eigen::Dynamic>& normals,
                    const Eigen::Matrix<double, 6, Eigen::Dynamic>& coefficients) {
    TorusDistanceField DF(points, normals);
    DF.fit_tori_from_forms(coefficients);
    return DF.get_tori();
}

//===== SAMPLING

std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
TorusDistanceField::sample_isosurface(const size_t& n_samples, const double& isovalue) const {

    std::vector<Eigen::Vector3d> int_locations;
    while (int_locations.size() < n_samples) {
        // Sample a direction uniformly.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> uniform_dist(0.0, 1.0);

        double theta = 2.0 * M_PI * uniform_dist(gen);
        double cos_phi = 2.0 * uniform_dist(gen) - 1.0;
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        double sin_phi = std::sqrt(1.0 - cos_phi * cos_phi);
        Eigen::Vector3d ray_dir(cos_theta * sin_phi, sin_theta * sin_phi, cos_phi);

        // Among all lines parallel to this direction, uniformly sample an offset so the resulting line intersects the
        // bounding sphere. This means uniformly sampling from a disk of diameter `diameter`, with normal `rd`.
        double theta_disk = 2.0 * M_PI * uniform_dist(gen);
        double r = std::sqrt(uniform_dist(gen));
        std::pair<Eigen::Vector3d, Eigen::Vector3d> onb = orthonormal_basis(ray_dir, 0.);
        Eigen::Vector3d ray_origin =
            -ray_dir + r * (std::cos(theta_disk) * onb.first + std::sin(theta_disk) * onb.second);

        // Perform ray intersection
        std::vector<Eigen::Vector3d> new_intersections =
            ray_intersection_single(ray_origin, ray_dir, diameter, isovalue);
        int_locations.insert(int_locations.end(), new_intersections.begin(), new_intersections.end());
    }

    Eigen::MatrixXd sample_locations(3, n_samples);
    Eigen::Matrix<double, 3, Eigen::Dynamic> sample_normals = evaluate_gradient(sample_locations);
    for (int i = 0; i < n_samples; i++) {
        sample_locations.col(i) = int_locations[i];
        sample_normals.col(i) /= sample_normals.col(i).norm();
        if (std::isnan(sample_normals.col(i).norm())) sample_normals.col(i) = Eigen::Vector3d::Zero();
    }

    return {sample_locations, sample_normals};
}

std::vector<Eigen::Vector3d> TorusDistanceField::ray_intersection_single(const Eigen::Vector3d& ro,
                                                                         const Eigen::Vector3d& rd, const double& diam,
                                                                         const double& isovalue) const {
    double t = 0.;
    double tmax = diam;
    double epsilon = 1e-4;
    std::vector<Eigen::Vector3d> int_locations;

    while (t < tmax) {
        Eigen::Vector3d q = ro + t * rd;
        double d = std::abs(evaluate_distance_single(q, isovalue));
        if (d < epsilon) {
            int_locations.push_back(q);
            // Ray-march with step size max(|φ|, ε) until φ > ε
            while (d < epsilon) {
                t += std::max(d, epsilon);
                q = ro + t * rd;
                d = std::abs(evaluate_distance_single(q, isovalue));
            }
        }
        t += d;
    }

    return int_locations;
}

std::vector<Eigen::Vector3d> TorusDistanceField::ray_intersections(const Eigen::Matrix<double, 3, Eigen::Dynamic>& ros,
                                                                   const Eigen::Matrix<double, 3, Eigen::Dynamic>& rds,
                                                                   const double& isovalue) const {
    std::vector<Eigen::Vector3d> int_locations;
    int n_rays = ros.cols();
    // I don't think we can multithread this due to vector expansions
    for (int i = 0; i < n_rays; i++) {
        std::vector<Eigen::Vector3d> new_locations =
            ray_intersection_single(ros.col(i), rds.col(i), diameter, isovalue);
        int_locations.insert(int_locations.end(), new_locations.begin(), new_locations.end());
    }
    return int_locations;
}

//===== APPLICATIONS

Eigen::Matrix<double, 2, Eigen::Dynamic>
TorusDistanceField::principal_curvatures(const Eigen::Matrix<double, 6, Eigen::Dynamic>& coefficients) const {
    Eigen::MatrixXd curvatures(2, n_points);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_points; i++) {

        Eigen::Vector3d p_i = points.col(i);
        Eigen::Vector3d n_i = normals.col(i);

        std::pair<Eigen::Vector3d, Eigen::Vector3d> basis = orthonormal_basis(n_i);
        Eigen::Vector3d s_hat = basis.first;
        Eigen::Vector3d t_hat = basis.second;

        // a00, kappa_min, kappa_max, v_min, normal
        std::tuple<double, double, double, Eigen::Vector3d, Eigen::Vector3d> result =
            get_torus_parameters(i, coefficients.col(i), s_hat, t_hat);
        double kappa_min = std::get<1>(result);
        double kappa_max = std::get<2>(result);

        curvatures(0, i) = kappa_min;
        curvatures(1, i) = kappa_max;
    }
    return curvatures;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::Matrix<double, 3, Eigen::Dynamic>,
           Eigen::Matrix<double, 3, Eigen::Dynamic>>
TorusDistanceField::principal_directions(const Eigen::Matrix<double, 6, Eigen::Dynamic>& coefficients) const {
    Eigen::VectorXd kappa_min(n_points);
    Eigen::VectorXd kappa_max(n_points);
    Eigen::MatrixXd v_min(3, n_points);
    Eigen::MatrixXd v_max(3, n_points);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_points; i++) {

        Eigen::Vector3d p_i = points.col(i);
        Eigen::Vector3d n_i = normals.col(i);

        std::pair<Eigen::Vector3d, Eigen::Vector3d> basis = orthonormal_basis(n_i);
        Eigen::Vector3d s_hat = basis.first;
        Eigen::Vector3d t_hat = basis.second;

        std::tuple<double, double, Eigen::Vector3d, Eigen::Vector3d> result =
            get_principal_directions_at_point(i, coefficients.col(i), s_hat, t_hat);

        double k_min = std::get<0>(result);
        double k_max = std::get<1>(result);
        Eigen::Vector3d ev_min = std::get<2>(result);
        Eigen::Vector3d ev_max = std::get<3>(result);

        kappa_min(i) = k_min;
        kappa_max(i) = k_max;
        v_min.col(i) = ev_min;
        v_max.col(i) = ev_max;
    }
    return {kappa_min, kappa_max, v_min, v_max};
}

// ========================================================================
// NAIVE CONVOLUTIONAL FORMULA
// ========================================================================

double TorusDistanceField::signed_log_sum_exp_single(const Eigen::Vector3d& q, const double& shift_,
                                                     const Eigen::VectorXi& neighbors) const {

    double w = 0.;
    double shift = std::isnan(shift_) ? compute_shift(q, neighbors) : shift_;
    double lam = compute_lambda(shift) * lambda_scale;

    int k_neighbors = neighbors.size();
    for (int nbr = 0; nbr < k_neighbors; nbr++) {
        const int i = neighbors(nbr);
        const Eigen::Vector3d& p_i = points.col(i);
        const Eigen::Vector3d& n_i = normals.col(i);
        Eigen::Vector3d r_vec = q - p_i;
        double r = r_vec.norm();
        double h = (lam * r + 1.0) * r_vec.dot(n_i) / (2. * M_PI * r * r * r);
        w += areas(i) * h * std::exp(-lam * (r - shift));
    }
    return sgn(w) * (-std::log(std::abs(w)) / lam + shift);
}

Eigen::Vector3d TorusDistanceField::signed_log_sum_exp_gradient_single(const Eigen::Vector3d& q, const double& shift_,
                                                                       const Eigen::VectorXi& neighbors) const {

    Eigen::Vector3d A(0., 0., 0.);
    Eigen::Vector3d B(0., 0., 0.);
    double C = 0.;

    double shift = std::isnan(shift_) ? compute_shift(q, neighbors) : shift_;
    double lam = compute_lambda(shift) * lambda_scale;

    int k_neighbors = neighbors.size();
    for (int nbr = 0; nbr < k_neighbors; nbr++) {
        const int i = neighbors(nbr);
        const Eigen::Vector3d& p_i = points.col(i);
        const Eigen::Vector3d& n_i = normals.col(i);
        Eigen::Vector3d r_vec = q - p_i;
        double r = r_vec.norm();
        Eigen::Vector3d r_hat = r_vec / r;
        double weight = std::exp(-lam * (r - shift));
        double h = (lam * r + 1.0) * r_vec.dot(n_i) / (2. * M_PI * r * r * r);
        // Eigen::Vector3d grad_h = n_i / r_vec.dot(n_i) - r_vec / r_vec.dot(r_vec);
        Eigen::Vector3d grad_N = (lam * r + 1.0) * n_i + (lam * r_hat * r_vec.dot(n_i));
        Eigen::Vector3d term1 = grad_N / (2. * M_PI * r * r * r);
        Eigen::Vector3d term2 =
            ((lam * r + 1.0) * r_vec.dot(n_i) * 6. * M_PI * r * r_vec) / (4. * M_PI * M_PI * r * r * r * r * r * r);
        Eigen::Vector3d grad_h = term1 - term2;
        A += areas(i) * grad_h * weight;
        B += areas(i) * lam * r_hat * h * weight;
        C += areas(i) * weight * h;
    }
    return -(A - B) / (C * lam);
}

Eigen::VectorXd TorusDistanceField::signed_log_sum_exp(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                       bool parallelize) const {

    int n_queries = queries.cols();
    Eigen::VectorXd distances(n_queries);

    // Compute neighbors
    Eigen::MatrixXi neighbors = compute_nearest_points(queries, k_evaluation);

    Eigen::VectorXd shifts = compute_shifts(queries, neighbors);

    // TODO: Option for GPU acceleration
    int mp = parallelize;
#pragma omp parallel for schedule(static) if (mp)
    for (int q_idx = 0; q_idx < n_queries; q_idx++) {
        Eigen::Vector3d query = queries.col(q_idx);
        double result = signed_log_sum_exp_single(queries.col(q_idx), shifts(q_idx), neighbors.col(q_idx));
        distances(q_idx) = result;
    }
    return distances;
}

double TorusDistanceField::log_sum_exp_single(const Eigen::Vector3d& q) const {

    double w = 0.;
    // use Madan & Levin's heuristic of setting α = 100 / length of bounding box diagonal.
    // double lam = 100. / (bbox_max - bbox_min).norm();
    double lam = lambda_scale;
    double shift = 64. / lam;

    for (int i = 0; i < n_points; i++) {
        Eigen::Vector3d p_i = points.col(i);
        Eigen::Vector3d r_vec = q - p_i;
        double r = r_vec.norm();
        w += std::exp(-lam * (r - shift));
    }
    return (-std::log(std::abs(w)) / lam) + shift;
}

Eigen::VectorXd TorusDistanceField::log_sum_exp(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                bool parallelize) const {

    int n_queries = queries.cols();
    Eigen::VectorXd distances(n_queries);

    int mp = parallelize;
#pragma omp parallel for schedule(static) if (mp)
    for (int q_idx = 0; q_idx < n_queries; q_idx++) {
        Eigen::Vector3d query = queries.col(q_idx);
        double result = log_sum_exp_single(query);
        distances(q_idx) = result;
    }
    return distances;
}

Eigen::Matrix<double, 3, Eigen::Dynamic>
TorusDistanceField::signed_log_sum_exp_gradient(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                bool parallelize) const {

    int n_queries = queries.cols();
    Eigen::MatrixXd gradients(3, n_queries);

    // Compute neighbors
    Eigen::MatrixXi neighbors = compute_nearest_points(queries, k_evaluation);

    Eigen::VectorXd shifts = compute_shifts(queries, neighbors);

    // TODO: Option for GPU acceleration
    int mp = parallelize;
#pragma omp parallel for schedule(static) if (mp)
    for (int q_idx = 0; q_idx < n_queries; q_idx++) {
        Eigen::Vector3d query = queries.col(q_idx);
        Eigen::Vector3d result = signed_log_sum_exp_gradient_single(query, shifts(q_idx), neighbors.col(q_idx));
        gradients.col(q_idx) = result;
    }
    return gradients;
}

double TorusDistanceField::self_normalized_signed_log_sum_exp_single(const Eigen::Vector3d& q, const double& shift_,
                                                                     const Eigen::VectorXi& neighbors) const {

    double u = 0.;
    double normalization = 0.;
    double shift = std::isnan(shift_) ? compute_shift(q, neighbors) : shift_;
    double lam = compute_lambda(shift) * lambda_scale;

    int k_neighbors = neighbors.size();
    for (int nbr = 0; nbr < k_neighbors; nbr++) {
        const int i = neighbors(nbr);
        const Eigen::Vector3d& p_i = points.col(i);
        const Eigen::Vector3d& n_i = normals.col(i);
        Eigen::Vector3d r_vec = q - p_i;
        double r = r_vec.norm();
        double weight = std::exp(-lam * (r - shift));
        u += areas(i) * r_vec.dot(n_i) * weight;
        normalization += areas(i) * weight;
    }
    return u / normalization;
}

Eigen::Vector3d
TorusDistanceField::self_normalized_signed_log_sum_exp_gradient_single(const Eigen::Vector3d& q, const double& shift_,
                                                                       const Eigen::VectorXi& neighbors) const {

    Eigen::Vector3d A(0., 0., 0.);
    Eigen::Vector3d B(0., 0., 0.);
    double C = 0.0;
    double D = 0.0;

    double shift = std::isnan(shift_) ? compute_shift(q, neighbors) : shift_;
    double lam = compute_lambda(shift) * lambda_scale;

    int k_neighbors = neighbors.size();
    for (int nbr = 0; nbr < k_neighbors; nbr++) {
        const int i = neighbors(nbr);
        const Eigen::Vector3d& p_i = points.col(i);
        const Eigen::Vector3d& n_i = normals.col(i);
        Eigen::Vector3d r_vec = q - p_i;
        double r = r_vec.norm();
        Eigen::Vector3d r_hat = r_vec / r;
        double weight = std::exp(-lam * (r - shift));
        double h = r_vec.dot(n_i);
        Eigen::Vector3d grad_h = n_i;

        A += areas(i) * (grad_h - lam * h * r_hat) * weight;
        B += areas(i) * r_hat * weight;
        C += areas(i) * h * weight;
        D += areas(i) * weight;
    }
    return A / D + lam * (B * C) / (D * D);
}

Eigen::VectorXd
TorusDistanceField::self_normalized_signed_log_sum_exp(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                       bool parallelize) const {

    int n_queries = queries.cols();
    Eigen::VectorXd distances(n_queries);

    // Compute neighbors
    Eigen::MatrixXi neighbors = compute_nearest_points(queries, k_evaluation);

    Eigen::VectorXd shifts = compute_shifts(queries, neighbors);

    // TODO: Option for GPU acceleration
    int mp = parallelize;
#pragma omp parallel for schedule(static) if (mp)
    for (int q_idx = 0; q_idx < n_queries; q_idx++) {
        double result =
            self_normalized_signed_log_sum_exp_single(queries.col(q_idx), shifts(q_idx), neighbors.col(q_idx));
        distances(q_idx) = result;
    }
    return distances;
}

Eigen::Matrix<double, 3, Eigen::Dynamic>
TorusDistanceField::self_normalized_signed_log_sum_exp_gradient(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                                bool parallelize) const {
    int n_queries = queries.cols();
    Eigen::MatrixXd gradients(3, n_queries);

    // Compute neighbors
    Eigen::MatrixXi neighbors = compute_nearest_points(queries, k_evaluation);

    Eigen::VectorXd shifts = compute_shifts(queries, neighbors);

    // TODO: Option for GPU acceleration
    int mp = parallelize;
#pragma omp parallel for schedule(static) if (mp)
    for (int q_idx = 0; q_idx < n_queries; q_idx++) {
        Eigen::Vector3d result =
            self_normalized_signed_log_sum_exp_gradient_single(queries.col(q_idx), shifts(q_idx), neighbors.col(q_idx));
        gradients.col(q_idx) = result;
    }
    return gradients;
}

double TorusDistanceField::winding_number_single(const Eigen::Vector3d& q) const {

    double u = 0.0;
    for (int i = 0; i < n_points; i++) {
        const Eigen::Vector3d& p_i = points.col(i);
        const Eigen::Vector3d& n_i = normals.col(i);
        Eigen::Vector3d r_vec = q - p_i;
        double r = r_vec.norm();
        double P = n_i.dot(r_vec) / (r * r * r) / (4. * M_PI);
        u += areas(i) * P;
    }
    return u;
}

Eigen::VectorXd TorusDistanceField::winding_number(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                   bool parallelize) const {

    int n_queries = queries.cols();
    Eigen::VectorXd values(n_queries);

    int mp = parallelize;
#pragma omp parallel for schedule(static) if (mp)
    for (int q_idx = 0; q_idx < n_queries; q_idx++) {
        double result = winding_number_single(queries.col(q_idx));
        values(q_idx) = result;
    }
    return values;
}

void TorusDistanceField::compute_point_areas() {
    Eigen::MatrixXd P = points.transpose();
    Eigen::MatrixXd N = normals.transpose();
    igl::octree(P, point_indices, octree_CH, octree_CN, octree_W);

    {
        int k = 20; // Number of neighbors (typically 10-30)
        Eigen::MatrixXi I;
        igl::knn(P, k, point_indices, octree_CH, octree_CN, octree_W, I);

        igl::copyleft::cgal::point_areas(points.transpose(), I, N, areas);
    }
}

void TorusDistanceField::fast_winding_number_precompute() {

    // must clear if we already called compute_point_areas() in constructor
    point_indices.clear();
    octree_CH.resize(0, 0);
    octree_CN.resize(0, 0);
    octree_W.resize(0);

    Eigen::MatrixXd P = points.transpose();
    Eigen::MatrixXd N = normals.transpose();
    igl::octree(P, point_indices, octree_CH, octree_CN, octree_W);

    Eigen::VectorXd A;
    {
        int k = 21; // number of neighbors; used in CGAL fast_winding_number
        Eigen::MatrixXi I;
        igl::knn(points.transpose(), k, point_indices, octree_CH, octree_CN, octree_W, I);

        igl::copyleft::cgal::point_areas(P, I, N, A);
    }


    int expansion_order = 2; // 2 is recommended

    igl::fast_winding_number(P,               // Point positions
                             N,               // Point normals
                             A,               // Point areas
                             point_indices,   // Octree point indices
                             octree_CH,       // Octree children
                             expansion_order, // Taylor expansion order
                             cluster_centers, // Output: cluster centers
                             cluster_radii,   // Output: cluster radii
                             expansion_coeffs // Output: expansion coefficients
    );
}

Eigen::VectorXd TorusDistanceField::fast_winding_number(const Eigen::Matrix<double, Eigen::Dynamic, 3>& queries) const {
    Eigen::VectorXd W;
    Eigen::MatrixXd P = points.transpose();
    Eigen::MatrixXd N = normals.transpose();

    double beta = 2.0;                         // Accuracy parameter (typically 2.0)
    igl::fast_winding_number(P,                // Point positions
                             N,                // Point normals
                             areas,            // Point areas
                             point_indices,    // Octree point indices
                             octree_CH,        // Octree children
                             cluster_centers,  // From precomputation
                             cluster_radii,    // From precomputation
                             expansion_coeffs, // From precomputation
                             queries,          // Query points
                             beta,             // Accuracy parameter
                             W                 // Output: winding numbers
    );
    return W;
}


double TorusDistanceField::regularized_winding_number_single(const Eigen::Vector3d& q, const double& epsilon) const {

    // Must have first called fast_winding_number_precompute()

    double u = 0.0;
    for (int i = 0; i < n_points; i++) {
        const Eigen::Vector3d& p_i = points.col(i);
        const Eigen::Vector3d& n_i = normals.col(i);
        Eigen::Vector3d r_vec = p_i - q;
        double r = r_vec.norm();
        double t = r / epsilon;
        double S = std::erf(t) - 2. * t / std::sqrt(M_PI) * std::exp(-t * t);
        double P = n_i.dot(r_vec) / (r * r * r) / (4. * M_PI);
        u += areas(i) * S * P;
    }
    return u;
}

Eigen::VectorXd TorusDistanceField::regularized_winding_number(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                               const double& epsilon, bool parallelize) const {
    // Must have first called fast_winding_number_precompute()

    int n_queries = queries.cols();
    Eigen::VectorXd values(n_queries);

    int mp = parallelize;
#pragma omp parallel for schedule(static) if (mp)
    for (int q_idx = 0; q_idx < n_queries; q_idx++) {
        double result = regularized_winding_number_single(queries.col(q_idx), epsilon);
        values(q_idx) = result;
    }
    return values;
}

// ============================================================================
// BINDING CODE
// ============================================================================

void bind_torus_distance(nb::module_& m) {

    nb::class_<TorusDistanceField>(m, "TorusDistanceField")
        .def(nb::init<const Eigen::Matrix<double, 3, Eigen::Dynamic>&,
                      const Eigen::Matrix<double, 3, Eigen::Dynamic>&>(),
             "points"_a, "normals"_a, "Create an SDF parameterized by tori, from an input point cloud")

        // batch queries only in Python
        .def("evaluate_distance", &TorusDistanceField::evaluate_distance, "queries"_a, "isovalue"_a, "parallelize"_a,
             "Evaluate signed distance")
        .def("evaluate_gradient", &TorusDistanceField::evaluate_gradient, "queries"_a, "parallelize"_a,
             "Evaluate the gradient of signed distance")
        .def("evaluate_laplacian", &TorusDistanceField::evaluate_laplacian, "queries"_a, "parallelize"_a,
             "Evaluate the Laplacian of signed distance")
        .def("evaluate_distance_gradient_laplacian", &TorusDistanceField::evaluate_distance_gradient_laplacian,
             "queries"_a, "isovalue"_a, "parallelize"_a,
             "Evaluate the value, gradient, and Laplacian of signed distance")

        .def("log_sum_exp", &TorusDistanceField::log_sum_exp, "queries"_a, "parallelize"_a,
             "Compare against Madan & Levin")

        .def("signed_log_sum_exp", &TorusDistanceField::signed_log_sum_exp, "queries"_a, "parallelize"_a,
             "Naive convolutional distance")
        .def("self_normalized_signed_log_sum_exp", &TorusDistanceField::self_normalized_signed_log_sum_exp, "queries"_a,
             "parallelize"_a, "Naive convolutional distance")
        .def("winding_number", &TorusDistanceField::winding_number, "queries"_a, "parallelize"_a, "Winding number")

        .def("signed_log_sum_exp_gradient", &TorusDistanceField::signed_log_sum_exp_gradient, "queries"_a,
             "parallelize"_a, "Gradient of naive convolutional distance")
        .def("self_normalized_signed_log_sum_exp_gradient",
             &TorusDistanceField::self_normalized_signed_log_sum_exp_gradient, "queries"_a, "parallelize"_a,
             "Gradient of naive convolutional distance")

        .def("fast_winding_number_precompute", &TorusDistanceField::fast_winding_number_precompute,
             "Fast winding number (libigl)")
        .def("fast_winding_number", &TorusDistanceField::fast_winding_number, "queries"_a,
             "Fast winding number (libigl)")
        .def("regularized_winding_number", &TorusDistanceField::regularized_winding_number, "queries"_a, "epsilon"_a,
             "parallelize"_a, "Regularized winding number")

        // sampling
        .def("sample_isosurface", &TorusDistanceField::sample_isosurface, "n_samples"_a, "isovalue"_a,
             "Sample an isosurface")

        // applications
        .def("principal_curvatures", &TorusDistanceField::principal_curvatures, "coefficients"_a,
             "Compute principal curvature at each point")
        .def("principal_directions", &TorusDistanceField::principal_directions, "coefficients"_a,
             "Compute principal curvatures and directions at each point")

        // torus fitting
        .def("fit_tori_from_forms", &TorusDistanceField::fit_tori_from_forms, "coefficients"_a,
             "Fit tori from first and second fundamental forms (and offset)")
        .def("torus_signed_distance", &TorusDistanceField::torus_signed_distance, "q"_a, "center"_a, "axis"_a,
             "major_radius"_a, "minor_radius"_a, "Evaluate SDF of a single torus")
        .def("torus_signed_distances", &TorusDistanceField::torus_signed_distances, "queries"_a, "center"_a, "axis"_a,
             "major_radius"_a, "minor_radius"_a, "Evaluate signed distance to single torus at multiple query points")
        .def("torus_signed_distances_gradients", &TorusDistanceField::torus_signed_distances_gradients, "queries"_a,
             "center"_a, "axis"_a, "major_radius"_a, "minor_radius"_a,
             "Evaluate signed distance and its gradient to single torus at multiple query points")

        // other
        .def("get_points", &TorusDistanceField::get_points, "Get the original point positions")
        .def("get_points_and_normals", &TorusDistanceField::get_points_and_normals, "Get the original point cloud")
        .def("get_point_areas", &TorusDistanceField::get_point_areas, "Get point areas")
        .def("get_tori", &TorusDistanceField::get_tori, "Get the tori parameters")
        .def("set_tori", &TorusDistanceField::set_tori, "centers"_a, "axes"_a, "major_radii"_a, "minor_radii"_a,
             "Set the tori parameters")
        .def("set_k_evaluation", &TorusDistanceField::set_k_evaluation, "k"_a,
             "Set number of nearest neighbors to use during evaluation.")
        .def("set_radius_evaluation", &TorusDistanceField::set_radius_evaluation, "r"_a,
             "Set radius of points to use during evaluation.")
        .def("set_lambda_scale", &TorusDistanceField::set_lambda_scale, "scale"_a,
             "Scale the (automatically-computed) parameter λ")
        .def("set_outliers", &TorusDistanceField::set_outliers, "indices"_a, "Mark outliers")

        .def("get_neighbors", &TorusDistanceField::get_neighbors, "k_neighbors"_a, "i"_a, "Get nearest neighbors")
        .def("get_neighbors", &TorusDistanceField::get_all_neighbors, "k_neighbors"_a, "Get nearest neighbors")
        .def("get_all_neighbors_and_subsample", &TorusDistanceField::get_all_neighbors_and_subsample, "k_neighbors"_a,
             "n_subsample"_a, "Get nearest neighbors");

    // external
    m.def("get_neighbors", &get_neighbors, "points"_a, "k_neighbors"_a, "outliers"_a,
          "Compute the k nearest neighbors of each point in the given point cloud");
    m.def("get_average_neighbor_distance", &get_average_neighbor_distance, "points"_a, "k_neighbors"_a,
          "Compute average distance of each point to its k nearest neighbors, averaged over the whole point cloud");
    m.def("fit_tori_from_forms", &fit_tori_from_forms, "points"_a, "normals"_a, "coefficients"_a, "Fit tori");
    m.def("orthonormal_basis", &orthonormal_basis, "n"_a, "rotation"_a, "Compute an orthonormal basis");
}