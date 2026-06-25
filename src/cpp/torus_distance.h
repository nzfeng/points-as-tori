#pragma once

#include <algorithm>
#include <queue>

#include <Eigen/Geometry>
#include <nanoflann.hpp>

#include <chrono>
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#include "shape_3d.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <igl/copyleft/cgal/point_areas.h>
#include <igl/fast_winding_number.h>
#include <igl/knn.h>
#include <igl/octree.h>
#include <igl/per_vertex_normals.h>

inline double SPECIAL_VALUE = std::numeric_limits<double>::quiet_NaN();

inline Eigen::Vector3d rotation_to_cartesian(const Eigen::Vector2d& angles) {
    double theta = angles.x();
    double phi = angles.y();
    double sin_phi = std::sin(phi);
    return {std::cos(theta) * sin_phi, std::sin(theta) * sin_phi, std::cos(phi)};
}

inline Eigen::Vector2d cartesian_to_rotation(const Eigen::Vector3d& p) {
    double theta = std::atan2(p.y(), p.x());
    double phi = std::acos(p.z());
    return {theta, phi};
}

inline double compute_lambda(double shift) {
    return 64. / shift;
}


// MatrixType, DIM, Distance, row_major
using my_kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<double, 3, Eigen::Dynamic>, 3,
                                                         nanoflann::metric_L2_Simple, false>;

// Return (k, |P|) matrix representing the neighbors of the point cloud
// Potentially ignore outliers
Eigen::MatrixXi get_neighbors(const Eigen::Matrix<double, 3, Eigen::Dynamic>& points, const int& k_neighbors_,
                              const Eigen::VectorXi& outliers = Eigen::VectorXi::Zero(0)) {

    std::chrono::time_point<high_resolution_clock> t1, t2;
    std::chrono::duration<double, std::milli> ms_fp;

    int n_points = points.cols();
    int n_outliers = outliers.size();
    Eigen::Matrix<double, 3, Eigen::Dynamic> inlier_points;
    Eigen::VectorXi idx_map; // maps from inlier points to points in the original point cloud
    int n_inliers = n_points;
    bool exist_outliers = (n_outliers > 0);
    if (exist_outliers > 0) {
        n_inliers = n_points - n_outliers;
        inlier_points.resize(3, n_inliers);
        Eigen::VectorXi is_outlier = Eigen::VectorXi::Zero(n_points);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_outliers; i++) {
            is_outlier(outliers(i)) = 1;
        }
        int curr_idx = 0;
        idx_map.resize(n_inliers);
        for (int i = 0; i < n_points; i++) {
            if (is_outlier(i) == 1) continue;
            inlier_points.col(curr_idx) = points.col(i);
            idx_map(curr_idx) = i;
            curr_idx += 1;
        }
    } else {
        inlier_points = points;
        n_inliers = n_points;
    }

    int max_leaf = 20;
    int n_thread_build = 0;
    t1 = high_resolution_clock::now();
    my_kd_tree_t kd_tree(3, std::cref(inlier_points), max_leaf, n_thread_build);
    t2 = high_resolution_clock::now();
    ms_fp = t2 - t1;
    // std::cerr << "KD tree build time (s): " << ms_fp.count() / 1000. << std::endl;


    int k_neighbors = (k_neighbors_ > 0 && k_neighbors_ < n_inliers) ? k_neighbors_ : n_inliers;
    Eigen::MatrixXi neighbors = -Eigen::MatrixXi::Ones(k_neighbors, n_points);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_inliers; i++) {
        std::vector<size_t> ret_indexes(k_neighbors);
        std::vector<double> out_dists_sqr(k_neighbors);
        nanoflann::KNNResultSet<double> resultSet(k_neighbors);
        resultSet.init(ret_indexes.data(), out_dists_sqr.data());
        Eigen::Vector3d q = inlier_points.col(i);
        kd_tree.index_->findNeighbors(resultSet, q.data());

        int original_query_idx = exist_outliers ? idx_map(i) : i;
        for (int j = 0; j < k_neighbors; j++) {
            // Map inlier indices back to original indices
            int inlier_idx = static_cast<int>(ret_indexes[j]);
            int original_neighbor_idx = exist_outliers ? idx_map(inlier_idx) : inlier_idx;
            neighbors(j, original_query_idx) = original_neighbor_idx;
        }
    }

    return neighbors;
}

double get_average_neighbor_distance(const Eigen::Matrix<double, 3, Eigen::Dynamic>& points, const int& k_neighbors_) {

    std::chrono::time_point<high_resolution_clock> t1, t2;
    std::chrono::duration<double, std::milli> ms_fp;

    int n_points = points.cols();

    int max_leaf = 20;
    int n_thread_build = 0;
    my_kd_tree_t kd_tree(3, std::cref(points), max_leaf, n_thread_build);

    int k_neighbors = (k_neighbors_ > 0 && k_neighbors_ < n_points) ? k_neighbors_ : n_points;
    Eigen::MatrixXi neighbors = -Eigen::MatrixXi::Ones(k_neighbors, n_points);

    double total_distance = 0.;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_points; i++) {
        std::vector<size_t> ret_indexes(k_neighbors);
        std::vector<double> out_dists_sqr(k_neighbors);
        nanoflann::KNNResultSet<double> resultSet(k_neighbors);
        resultSet.init(ret_indexes.data(), out_dists_sqr.data());
        Eigen::Vector3d q = points.col(i);
        kd_tree.index_->findNeighbors(resultSet, q.data());

        for (int j = 0; j < k_neighbors; j++) {
            Eigen::Vector3d p_j = points.col(ret_indexes[j]);
            total_distance += (q - p_j).norm();
        }
    }

    return total_distance / (n_points * k_neighbors);
}

class TorusDistanceField {

  public:
    TorusDistanceField(const Eigen::Matrix<double, 3, Eigen::Dynamic>& points_,
                       const Eigen::Matrix<double, 3, Eigen::Dynamic>& normals_)
        : points(points_), normals(normals_), n_points(points_.cols()) {


        std::pair<Eigen::Vector3d, Eigen::Vector3d> bbox = point_cloud_bounding_box(points);
        bbox_min = bbox.first;
        bbox_max = bbox.second;

        is_outlier = Eigen::VectorXi::Zero(n_points);

        // Set the bounding box of the SDF as approximately double the bounding box of the input point cloud (a
        // heuristic).
        Eigen::Vector3d diag = bbox_max - bbox_min;
        bbox_min -= 0.5 * diag;
        bbox_max += 0.5 * diag;
        diameter = (bbox_min - bbox_max).norm();

        // Build kd tree using nanoflann
        int max_leaf = 20;
        int n_thread_build = 0; // let nanoflann automatically determine how many threads to use
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        kd_tree.reset(new my_kd_tree_t(3, std::cref(points), max_leaf, n_thread_build));
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        // std::cerr << "KD tree build time (s): " << ms_fp.count() / 1000. << std::endl;
    }

    // Get the k closest neighbors to a given point in the input point cloud, including the point
    Eigen::VectorXi get_neighbors(const int& k_neighbors_, const int& i) const {

        int k_neighbors = (k_neighbors_ > 0 && k_neighbors_ < n_points) ? k_neighbors_ : n_points;
        std::vector<size_t> ret_indexes(k_neighbors);
        std::vector<double> out_dists_sqr(k_neighbors);
        nanoflann::KNNResultSet<double> resultSet(k_neighbors);
        resultSet.init(ret_indexes.data(), out_dists_sqr.data());
        Eigen::Vector3d q = points.col(i);
        kd_tree->index_->findNeighbors(resultSet, q.data());

        Eigen::VectorXi indices(k_neighbors);
        for (int j = 0; j < k_neighbors; j++) {
            indices(j) = ret_indexes[j];
        }
        return indices;
    }

    Eigen::MatrixXi get_all_neighbors(const int& k_neighbors_) const {
        int k_neighbors = (k_neighbors_ > 0 && k_neighbors_ < n_points) ? k_neighbors_ : n_points;
        Eigen::MatrixXi neighbors = -Eigen::MatrixXi::Ones(k_neighbors, n_points);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_points; i++) {
            neighbors.col(i) = get_neighbors(k_neighbors, i);
        }
        return neighbors;
    }

    std::pair<Eigen::MatrixXi, Eigen::MatrixXi> get_all_neighbors_and_subsample(const int& k_neighbors_,
                                                                                const int& n_subsample_) const {
        int k_neighbors = (k_neighbors_ > 0 && k_neighbors_ < n_points) ? k_neighbors_ : n_points;
        int n_subsample = (n_subsample_ > 0 && n_subsample_ < n_points) ? n_subsample_ : n_points;
        Eigen::MatrixXi neighbors_subsampled = -Eigen::MatrixXi::Ones(n_subsample, n_points);
        Eigen::MatrixXi neighbors = -Eigen::MatrixXi::Ones(k_neighbors, n_points);
        Eigen::MatrixXd neighborhood(3, k_neighbors);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_points; i++) {
            Eigen::VectorXi indices = get_neighbors(k_neighbors, i);
            neighbors.col(i) = indices;
            // get neighborhood positions
            for (int j = 0; j < k_neighbors; j++) {
                neighborhood.col(j) = points.col(indices(j));
            }
            Eigen::VectorXi indices_subsampled = subsample_point_cloud(neighborhood, n_subsample);
            for (int j = 0; j < n_subsample; j++) {
                neighbors_subsampled(j, i) = indices(indices_subsampled(j));
            }
        }
        return {neighbors, neighbors_subsampled};
    }

    // batch queries
    Eigen::VectorXd evaluate_distance(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                      const double& isovalue = 0., bool parallelize = true) const;
    Eigen::Matrix<double, 3, Eigen::Dynamic> evaluate_gradient(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                               bool parallelize = true) const;
    Eigen::VectorXd evaluate_laplacian(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                       bool parallelize = true) const;
    std::tuple<Eigen::VectorXd, Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::VectorXd>
    evaluate_distance_gradient_laplacian(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                         const double& isovalue = 0., bool parallelize = true) const;

    // naive convolutional formula
    double log_sum_exp_single(const Eigen::Vector3d& q) const;
    Eigen::VectorXd log_sum_exp(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                bool parallelize = true) const; // Madan and Levin

    Eigen::VectorXd signed_log_sum_exp(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                       bool parallelize = true) const;
    Eigen::Matrix<double, 3, Eigen::Dynamic>
    signed_log_sum_exp_gradient(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries, bool parallelize = true) const;

    Eigen::VectorXd self_normalized_signed_log_sum_exp(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                       bool parallelize = true) const;
    Eigen::Matrix<double, 3, Eigen::Dynamic>
    self_normalized_signed_log_sum_exp_gradient(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                bool parallelize = true) const;

    void fast_winding_number_precompute();
    Eigen::VectorXd fast_winding_number(const Eigen::Matrix<double, Eigen::Dynamic, 3>& queries) const;

    Eigen::VectorXd winding_number(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                   bool parallelize = true) const;

    Eigen::VectorXd regularized_winding_number(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                               const double& epsilon, bool parallelize = true) const;

    // torus fitting --- this function modifies the member variables controlling tori parameters
    void fit_tori_from_forms(const Eigen::Matrix<double, 6, Eigen::Dynamic>& coefficients);
    double torus_signed_distance(const Eigen::Vector3d& q, const Eigen::Vector3d& center, const Eigen::Vector2d& axis,
                                 const double& major_radius, const double& minor_radius) const;

    Eigen::VectorXd torus_signed_distances(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                           const Eigen::Vector3d& center, const Eigen::Vector2d& axis,
                                           const double& major_radius, const double& minor_radius) const;
    std::pair<Eigen::VectorXd, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    torus_signed_distances_gradients(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                     const Eigen::Vector3d& center, const Eigen::Vector2d& axis,
                                     const double& major_radius, const double& minor_radius) const;

    // getters & setters
    Eigen::Matrix<double, 3, Eigen::Dynamic> get_points() const {
        return points;
    }
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    get_points_and_normals() const {
        return {points, normals};
    }
    std::tuple<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 2, Eigen::Dynamic>, Eigen::VectorXd,
               Eigen::VectorXd>
    get_tori() const {
        return {centers, axes, major_radii, minor_radii};
    }
    void set_tori(const Eigen::Matrix<double, 3, Eigen::Dynamic>& centers_,
                  const Eigen::Matrix<double, 2, Eigen::Dynamic>& axes_, const Eigen::VectorXd& major_radii_,
                  const Eigen::VectorXd& minor_radii_) {
        centers = centers_;
        axes = axes_;
        major_radii = major_radii_;
        minor_radii = minor_radii_;
    }

    void set_use_areas(bool val) {
        use_areas = val;
    }
    void set_k_evaluation(int k) {
        k_evaluation = k;
    }
    void set_radius_evaluation(float r) {
        radius_evaluation = r;
    }
    void set_outliers(const Eigen::VectorXi& indices) {
        is_outlier = Eigen::VectorXi::Zero(n_points);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < indices.size(); i++) {
            is_outlier(indices(i)) = 1;
        }
    }
    void set_lambda_scale(float scale) {
        lambda_scale = scale;
    }

    Eigen::VectorXd get_point_areas() const {
        return areas;
    }

    // applications
    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    sample_isosurface(const size_t& n_samples, const double& isovalue = 0.) const; // sample uniformly

    Eigen::Matrix<double, 2, Eigen::Dynamic> principal_curvatures(
        const Eigen::Matrix<double, 6, Eigen::Dynamic>& coefficients) const; // principal curvatures at each point

    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::Matrix<double, 3, Eigen::Dynamic>,
               Eigen::Matrix<double, 3, Eigen::Dynamic>>
    principal_directions(const Eigen::Matrix<double, 6, Eigen::Dynamic>& coefficients) const;

  private:
    // input point cloud --- copy (since references from Python will be destroyed)
    const Eigen::Matrix<double, 3, Eigen::Dynamic> points;
    const Eigen::Matrix<double, 3, Eigen::Dynamic> normals;
    Eigen::VectorXd areas; // point areas
    int n_points;
    const double NAN_EPSILON = 1e-10; // reproduce Python results
    bool use_areas = false;

    //  SDF parameters
    int dimension = 3;
    Eigen::Vector3d bbox_min, bbox_max;
    double diameter;
    int k_evaluation = -1;           // number of nearest neighbors to use when evaluating distance
    double radius_evaluation = -1.0; // radius for radius-based evaluation (-1 = disabled, use k_evaluation)
    double lambda_scale = 1.0;
    Eigen::VectorXi is_outlier;

    // torus parameters
    Eigen::Matrix<double, 3, Eigen::Dynamic> centers;
    Eigen::Matrix<double, 2, Eigen::Dynamic> axes; // two angles; axis always passes through the center
    Eigen::VectorXd major_radii, minor_radii;      // signs are encoded via minor_radii


    // libigl fast winding number stuff
    Eigen::VectorXd compute_point_areas();
    std::vector<std::vector<int>> point_indices;        // Point indices for each octree cell
    Eigen::Matrix<int, Eigen::Dynamic, 8> octree_CH;    // #octree_cells x 8 children indices
    Eigen::Matrix<double, Eigen::Dynamic, 3> octree_CN; // #octree_cells x 3 cell centers
    Eigen::VectorXd octree_W;                           // #octree_cells widths

    Eigen::MatrixXd cluster_centers;  // Cluster centers
    Eigen::VectorXd cluster_radii;    // Cluster radii
    Eigen::MatrixXd expansion_coeffs; // Expansion coefficients

    // single-queries
    double evaluate_distance_single(const Eigen::Vector3d& query, const double& isovalue = 0.,
                                    const double& shift = SPECIAL_VALUE,
                                    const std::vector<int>& neighbors = std::vector<int>()) const;
    Eigen::Vector3d evaluate_gradient_single(const Eigen::Vector3d& query, const double& shift = SPECIAL_VALUE,
                                             const std::vector<int>& neighbors = std::vector<int>()) const;
    double evaluate_laplacian_single(const Eigen::Vector3d& query, const double& shift = SPECIAL_VALUE,
                                     const std::vector<int>& neighbors = std::vector<int>()) const;
    std::tuple<double, Eigen::Vector3d, double>
    evaluate_distance_gradient_laplacian_single(const Eigen::Vector3d& query, const double& isovalue = 0.,
                                                const double& shift = SPECIAL_VALUE,
                                                const std::vector<int>& neighbors = std::vector<int>()) const;

    double signed_log_sum_exp_single(const Eigen::Vector3d& q, const double& shift = SPECIAL_VALUE,
                                     const Eigen::VectorXi& neighbors = Eigen::VectorXi::Zero(0)) const;
    Eigen::Vector3d signed_log_sum_exp_gradient_single(const Eigen::Vector3d& q, const double& shift_,
                                                       const Eigen::VectorXi& neighbors) const;

    double self_normalized_signed_log_sum_exp_single(const Eigen::Vector3d& q, const double& shift = SPECIAL_VALUE,
                                                     const Eigen::VectorXi& neighbors = Eigen::VectorXi::Zero(0)) const;
    Eigen::Vector3d self_normalized_signed_log_sum_exp_gradient_single(const Eigen::Vector3d& q, const double& shift_,
                                                                       const Eigen::VectorXi& neighbors) const;
    double winding_number_single(const Eigen::Vector3d& q) const;
    double regularized_winding_number_single(const Eigen::Vector3d& q, const double& epsilon) const;

    // acceleration
    std::unique_ptr<my_kd_tree_t> kd_tree;
    Eigen::VectorXi nearest_points_single(const Eigen::Vector3d& q, const int& k_neighbors) const;
    Eigen::MatrixXi nearest_points(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                   const int& k_neighbors) const;
    std::vector<int> points_within_radius_single(const Eigen::Vector3d& q, const double& radius) const;
    Eigen::MatrixXi compute_nearest_points(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                           const int& k_neighbors) const;
    std::vector<std::vector<int>> compute_points_within_radius(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                                               const double& radius) const;

    double compute_shift(const Eigen::Vector3d& query, const std::vector<int>& neighbors) const;
    double compute_shift(const Eigen::Vector3d& query, const Eigen::VectorXi& neighbors) const;
    Eigen::VectorXd compute_shifts(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                   const Eigen::MatrixXi& neighbors) const;
    Eigen::VectorXd compute_shifts(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                   const std::vector<std::vector<int>>& neighbors) const;

    // tori routines
    std::pair<double, Eigen::Vector3d>
    torus_signed_distance_gradient(const Eigen::Vector3d& q, const Eigen::Vector3d& center, const Eigen::Vector2d& axis,
                                   const double& major_radius, const double& minor_radius) const;
    std::tuple<Eigen::Vector3d, Eigen::Vector2d, double, double>
    fit_torus_at_point(int i, const Eigen::Matrix<double, 6, 1>& coeffs) const;
    std::tuple<double, double, double, Eigen::Vector3d, Eigen::Vector3d>
    get_torus_parameters(int i, const Eigen::Matrix<double, 6, 1>& coeffs, const Eigen::Vector3d& s_hat,
                         const Eigen::Vector3d& t_hat) const;
    std::tuple<double, double, double, Eigen::Vector3d, Eigen::Vector3d>
    get_torus_parameters_python(int i, const Eigen::Matrix<double, 6, 1>& coeffs, const Eigen::Vector3d& s_hat,
                                const Eigen::Vector3d& t_hat) const;
    std::tuple<double, double, Eigen::Vector3d, Eigen::Vector3d>
    get_principal_directions_at_point(int i, const Eigen::Matrix<double, 6, 1>& coeffs, const Eigen::Vector3d& s_hat,
                                      const Eigen::Vector3d& t_hat) const;
    std::tuple<double, double, double, Eigen::Vector3d, Eigen::Vector3d>
    get_polynomial_curvatures(int i, const Eigen::Matrix<double, 9, 1>& coeffs, const Eigen::Vector3d& s_hat,
                              const Eigen::Vector3d& t_hat) const;

    // sampling
    std::vector<Eigen::Vector3d> ray_intersection_single(const Eigen::Vector3d& ro, const Eigen::Vector3d& rd,
                                                         const double& diameter, const double& isovalue = 0.) const;
    std::vector<Eigen::Vector3d> ray_intersections(const Eigen::Matrix<double, 3, Eigen::Dynamic>& ros,
                                                   const Eigen::Matrix<double, 3, Eigen::Dynamic>& rds,
                                                   const double& isovalue = 0.) const;
};
