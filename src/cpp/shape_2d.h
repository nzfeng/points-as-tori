#pragma once

#include <Eigen/Core>
#include <fcpw/fcpw.h>
#include <unsupported/Eigen/AutoDiff>

#include <cmath>
#include <memory>
#include <mutex>
#include <random>
#include <tuple>
#include <vector>

#include "shape_2d.h"

// ============================================================================
// shared utility functions
// ============================================================================

template <typename T>
int sgn(T val) {
    return (T(0) < val) - (val < T(0)); // can output 0
}

inline double sign(double x) {
    return x >= 0.0 ? 1.0 : -1.0;
} // force to pick a side

// `rotation` is in [0, 2π], and controls the rotation of the local x-axis around the local z-axis, using an arbitrary
// reference vector.
std::pair<Eigen::Vector3d, Eigen::Vector3d> orthonormal_basis(const Eigen::Vector3d& n, double rotation = 0.);
std::pair<Eigen::Vector3d, Eigen::Vector3d> orthonormal_basis_random(const Eigen::Vector3d& n, int seed);
Eigen::Matrix<double, 3, Eigen::Dynamic> sample_sphere_positions(int n_samples, std::mt19937& rng);

// ============================================================================
// Point clouds
// ============================================================================

std::pair<Eigen::Vector2d, Eigen::Vector2d>
point_cloud_bounding_box_2d(const Eigen::Matrix<double, 2, Eigen::Dynamic>& positions);

// Subsample a point cloud down to the target number of points
Eigen::VectorXi subsample_point_cloud(const Eigen::Matrix<double, 3, Eigen::Dynamic>& positions, int n_points);

// ============================================================================
// Utility functions
// ============================================================================

inline Eigen::Vector2d sign(const Eigen::Vector2d& v) {
    return Eigen::Vector2d(sign(v.x()), sign(v.y()));
}
inline double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}
inline Eigen::Vector2d cwiseAbs(const Eigen::Vector2d& v) {
    return Eigen::Vector2d(std::abs(v.x()), std::abs(v.y()));
}
inline Eigen::Vector2d cwiseMax(const Eigen::Vector2d& v, double s) {
    return Eigen::Vector2d(std::max(v.x(), s), std::max(v.y(), s));
}
inline Eigen::Vector2d min(const Eigen::Vector2d& u, const Eigen::Vector2d& v) {
    return Eigen::Vector2d(std::min(u.x(), v.x()), std::min(u.y(), v.y()));
}
inline Eigen::Vector2d max(const Eigen::Vector2d& u, const Eigen::Vector2d& v) {
    return Eigen::Vector2d(std::max(u.x(), v.x()), std::max(u.y(), v.y()));
}
inline double cross(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}


// Type aliases for autodiff
using ADScalar = Eigen::AutoDiffScalar<Eigen::Vector2d>;
using ADVector2 = Eigen::Matrix<ADScalar, 2, 1>;
using ADScalar2 = Eigen::AutoDiffScalar<ADVector2>;
using ADVector2_2 = Eigen::Matrix<ADScalar2, 2, 1>;

// ============================================================================
// Autodiff-safe min/max helpers
// ============================================================================
template <typename Derived>
inline Eigen::AutoDiffScalar<Derived> ad_min(const Eigen::AutoDiffScalar<Derived>& a,
                                             const Eigen::AutoDiffScalar<Derived>& b) {
    return (a.value() < b.value()) ? a : b;
}

template <typename Derived>
inline Eigen::AutoDiffScalar<Derived> ad_max(const Eigen::AutoDiffScalar<Derived>& a,
                                             const Eigen::AutoDiffScalar<Derived>& b) {
    return (a.value() > b.value()) ? a : b;
}

template <typename Derived>
inline Eigen::AutoDiffScalar<Derived> ad_abs(const Eigen::AutoDiffScalar<Derived>& a) {
    // If the value is negative, return -a (which correctly computes the derivative as -a.derivatives())
    return (a.value() < 0.0) ? -a : a;
}

// ============================================================================
// Base SDF class
// ============================================================================
class SDF2D {
  public:
    virtual ~SDF2D() = default;

    // Evaluate signed distance and gradient at a single point
    virtual std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                                      double isovalue = 0.0) const = 0;

    // Evaluate signed distance at a single point
    virtual double evaluate(const Eigen::Vector2d& q, double isovalue = 0.0) const {
        return evaluate_with_gradient(q, isovalue).first;
    }

    // Evaluate signed distance at a single point
    virtual Eigen::Vector2d evaluate_gradient(const Eigen::Vector2d& q) const {
        return evaluate_with_gradient(q, 0.).second;
    }

    // Evaluate signed distance at a single point
    virtual Eigen::Matrix<double, 2, Eigen::Dynamic>
    evaluate_gradient_batch(const Eigen::Matrix<double, 2, Eigen::Dynamic>& queries) const {

        int n_queries = queries.cols();
        Eigen::Matrix<double, 2, Eigen::Dynamic> gradients(2, n_queries);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_queries; ++i) {
            gradients.col(i) = evaluate_gradient(queries.col(i));
        }
        return gradients;
    }

    Eigen::Vector2d evaluate_gradient_autodiff(const Eigen::Vector2d& q, double isovalue = 0.0) const {
        ADVector2 q_ad;
        q_ad[0] = ADScalar(q[0], 2, 0);
        q_ad[1] = ADScalar(q[1], 2, 1);

        ADScalar result = evaluate_autodiff_impl(q_ad, isovalue);
        return result.derivatives();
    }

    virtual Eigen::Matrix<double, 3, Eigen::Dynamic>
    evaluate_signed_distance_gradient(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries, bool is_extrusion,
                                      const double& SDF3D_param) const;

    // https://iquilezles.org/articles/distfunctions/
    double evaluate_revolution_single(const Eigen::Vector3d& p, const double& o) const {
        Eigen::Vector2d p_xz(p.x(), p.z());
        Eigen::Vector2d q(p_xz.norm() - o, p.y());
        return evaluate(q);
    }

    double evaluate_extrusion_single(const Eigen::Vector3d& p, const double& h) const {
        Eigen::Vector2d p_xy(p.x(), p.y());
        double d = evaluate(p_xy);
        Eigen::Vector2d w(d, std::abs(p.z()) - h);
        Eigen::Vector2d w_max(std::max(w.x(), 0.0), std::max(w.y(), 0.0));
        return std::min(std::max(w.x(), w.y()), 0.) + w_max.norm();
    }

    Eigen::Vector3d evaluate_revolution_gradient_single(const Eigen::Vector3d& p, const double& o) const {
        Eigen::Vector2d p_xz(p.x(), p.z());
        double r = p_xz.norm();

        // Query point in 2D SDF space
        Eigen::Vector2d q(r - o, p.y());

        // Get 2D gradient
        Eigen::Vector2d grad_2d = evaluate_gradient(q);

        Eigen::Vector3d grad;
        grad.x() = grad_2d.x() * p.x() / r;
        grad.y() = grad_2d.y();
        grad.z() = grad_2d.x() * p.z() / r;
        return grad;
    }

    Eigen::Vector3d evaluate_extrusion_gradient_single(const Eigen::Vector3d& p, const double& h) const {
        Eigen::Vector2d p_xy(p.x(), p.y());
        double d = evaluate(p_xy);
        Eigen::Vector2d grad_2d = evaluate_gradient(p_xy);

        double abs_pz = std::abs(p.z());
        double w_x = d;
        double w_y = abs_pz - h;

        Eigen::Vector3d grad;

        // Case analysis based on the extrusion formula
        if (w_x > 0.0 && w_y > 0.0) {
            // Outside both: gradient is from the norm
            double norm_w = std::sqrt(w_x * w_x + w_y * w_y);
            grad.x() = grad_2d.x() * w_x / norm_w;
            grad.y() = grad_2d.y() * w_x / norm_w;
            grad.z() = (p.z() > 0.0 ? 1.0 : -1.0) * w_y / norm_w;
        } else if (w_x > w_y) {
            // Closest to xy plane
            grad.x() = grad_2d.x();
            grad.y() = grad_2d.y();
            grad.z() = 0.0;
        } else {
            // Closest to z planes
            grad.x() = 0.0;
            grad.y() = 0.0;
            grad.z() = (p.z() > 0.0 ? 1.0 : -1.0);
        }

        return grad;
    }

    Eigen::VectorXd evaluate_revolution(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries,
                                        const double& o) const {
        int n_queries = queries.cols();
        Eigen::VectorXd distances(n_queries);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_queries; i++) {
            distances(i) = evaluate_revolution_single(queries.col(i), o);
        }
        return distances;
    }

    Eigen::VectorXd evaluate_extrusion(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries, const double& h) const {

        int n_queries = queries.cols();
        Eigen::VectorXd distances(n_queries);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_queries; i++) {
            distances(i) = evaluate_extrusion_single(queries.col(i), h);
        }
        return distances;
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic>
    evaluate_revolution_gradient(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries, const double& o) const {
        int n_queries = queries.cols();
        Eigen::MatrixXd gradients(3, n_queries);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_queries; i++) {
            gradients.col(i) = evaluate_revolution_gradient_single(queries.col(i), o);
        }
        return gradients;
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic>
    evaluate_extrusion_gradient(const Eigen::Matrix<double, 3, Eigen::Dynamic>& queries, const double& h) const {
        int n_queries = queries.cols();
        Eigen::MatrixXd gradients(3, n_queries);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_queries; i++) {
            gradients.col(i) = evaluate_extrusion_gradient_single(queries.col(i), h);
        }
        return gradients;
    }

    std::vector<Eigen::Vector3d> ray_intersection_revolution(const Eigen::Vector3d& ro, const Eigen::Vector3d& rd,
                                                             const double& o, const double& isovalue = 0.0) const;
    std::vector<Eigen::Vector3d> ray_intersection_extrusion(const Eigen::Vector3d& ro, const Eigen::Vector3d& rd,
                                                            const double& h, const double& isovalue = 0.0) const;

    // Uniformly sample
    std::pair<Eigen::Vector3d, Eigen::Vector3d> bbox_revolution(const double& o) const;
    std::pair<Eigen::Vector3d, Eigen::Vector3d> bbox_extrusion(const double& h) const;

    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    sample_revolution(const int& n_samples, const double& o, const Eigen::Vector3d& bbox_min = {1, 1, 1},
                      const Eigen::Vector3d& bbox_max = {-1, -1, -1}, const double& sigma_p = 0.,
                      const double& sigma_n = 0., const int& seed = 0) const;

    Eigen::Matrix<double, 3, Eigen::Dynamic>
    sample_revolution_narrow_band(const int& n_samples, const double& o, const double& offset,
                                  const Eigen::Vector3d& bbox_min = {1, 1, 1},
                                  const Eigen::Vector3d& bbox_max = {-1, -1, -1}, const int& seed = 0) const;

    std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    sample_extrusion(const int& n_samples, const double& h, const Eigen::Vector3d& bbox_min = {1, 1, 1},
                     const Eigen::Vector3d& bbox_max = {-1, -1, -1}, const double& sigma_p = 0.,
                     const double& sigma_n = 0., const int& seed = 0) const;

    Eigen::Matrix<double, 3, Eigen::Dynamic>
    sample_extrusion_narrow_band(const int& n_samples, const double& h, const double& offset,
                                 const Eigen::Vector3d& bbox_min = {1, 1, 1},
                                 const Eigen::Vector3d& bbox_max = {-1, -1, -1}, const int& seed = 0) const;

    // Sample revolution or extrusion
    virtual std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    sample_points_and_normals_uniformly(int n_points, bool is_extrusion, const double& SDF3D_param, int seed) const;

    virtual std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    sample_farthest_point_cloud(int n_points, bool is_extrusion, const double& SDF3D_param, int seed) const;

    virtual std::pair<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>>
    sample_uneven_point_cloud(bool is_extrusion, const double& SDF3D_param, int n_cameras, const double& sensor_size,
                              const double& grid_spacing_min, const double& grid_spacing_max,
                              const double& max_noise_level, const double& max_normals_to_flip, const int& max_points,
                              unsigned int seed) const;


    // Laplacian via autodiff
    double evaluate_laplacian(const Eigen::Vector2d& q, double isovalue = 0.0) const {

        // Set up second-order autodiff
        ADVector2_2 q_ad;

        // Initialize x
        q_ad[0].value() = ADScalar(q[0], 2, 0);
        q_ad[0].derivatives().resize(2);
        q_ad[0].derivatives()[0] = ADScalar(1.0, 2, 0);
        q_ad[0].derivatives()[1] = ADScalar(0.0, 2, 1);

        // Initialize y
        q_ad[1].value() = ADScalar(q[1], 2, 1);
        q_ad[1].derivatives().resize(2);
        q_ad[1].derivatives()[0] = ADScalar(0.0, 2, 0);
        q_ad[1].derivatives()[1] = ADScalar(1.0, 2, 1);

        // Evaluate with second-order autodiff
        ADScalar2 result = evaluate_autodiff_2nd_impl(q_ad, isovalue);

        // Extract Laplacian
        double d2f_dx2 = result.derivatives()[0].derivatives()[0];
        double d2f_dy2 = result.derivatives()[1].derivatives()[1];
        return d2f_dx2 + d2f_dy2;
    }

    virtual Eigen::VectorXd evaluate_laplacian_batch(const Eigen::Matrix<double, 2, Eigen::Dynamic>& queries) const {
        int n = queries.cols();
        Eigen::VectorXd laplacians(n);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            laplacians(i) = evaluate_laplacian(queries.col(i));
        }
        return laplacians;
    }

    // Bounding box of the isosurface
    // https://iquilezles.org/articles/bboxes2d/
    virtual std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const = 0;

    // Batch evaluate signed distance at multiple points (parallelized)
    Eigen::VectorXd evaluate_batch(const Eigen::Matrix<double, 2, Eigen::Dynamic>& queries,
                                   double isovalue = 0.0) const;

    // Batch evaluate signed distance and gradient (parallelized)
    std::pair<Eigen::VectorXd, Eigen::Matrix<double, 2, Eigen::Dynamic>>
    evaluate_with_gradient_batch(const Eigen::Matrix<double, 2, Eigen::Dynamic>& queries, double isovalue = 0.0) const;

    // Ray intersection using sphere tracing
    // Return position and normal at intersection point.
    std::tuple<Eigen::Vector2d, Eigen::Vector2d, bool> ray_intersect(const Eigen::Vector2d& ro,
                                                                     const Eigen::Vector2d& rd) const;

    // Batch ray intersection (parallelized)
    std::pair<Eigen::Matrix<double, 2, Eigen::Dynamic>, Eigen::Matrix<double, 2, Eigen::Dynamic>>
    ray_intersect_batch(const Eigen::Matrix<double, 2, Eigen::Dynamic>& ros,
                        const Eigen::Matrix<double, 2, Eigen::Dynamic>& rds) const;

    bool has_gradient() const {
        return has_analytical_gradient;
    }
    bool has_analytical_gradient = true;

  protected:
    const float diameter = 4.0;        // upper bound of diameter of a [-1,1]^2 box
    const double epsilon = 1e-4;       // epsilon used for ray-shape intersections
    const int max_steps = 1000;        // max steps for ray intersections
    const double epsilon_dist = 1e-10; // epsilon used for determining whether two points are close

    virtual ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const = 0;

    virtual ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const = 0;
};

// ============================================================================
// Circle SDF: distance to circle centered at origin
// ============================================================================
class Circle2D : public SDF2D {
  public:
    Circle2D(double r) : r_(r) {}
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

    double evaluate_laplacian(const Eigen::Vector2d& q) const;
    Eigen::VectorXd evaluate_laplacian_batch(const Eigen::Matrix<double, 2, Eigen::Dynamic>& queries) const override;

  private:
    double r_;

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using std::sqrt;
        Scalar len = sqrt(q[0] * q[0] + q[1] * q[1]);
        return len - Scalar(r_) - Scalar(isovalue);
    }

    // These two are repeated in every class because I can't do virtual template methods (I know it's ugly)
    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Pie SDF: pie/pacman shape (axis-aligned)
// ============================================================================
class Pie2D : public SDF2D {
  public:
    Pie2D(double r, double t) : r_(r) {
        sc_ = Eigen::Vector2d(std::sin(t), std::cos(t));
    }
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    double r_;
    Eigen::Vector2d sc_; // sin(t), cos(t)

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::abs;
        using std::sqrt;

        Scalar r_s = Scalar(r_);
        Vector2S sc_s((Scalar(sc_[0])), (Scalar(sc_[1])));

        // Apply symmetry
        Vector2S p((ad_abs(Scalar(q[0]))), (Scalar(q[1])));

        // Compute both distances
        Scalar l = sqrt(p[0] * p[0] + p[1] * p[1]);
        Scalar n = l - r_s;

        // Project onto sc direction and clamp
        Scalar dot_p_sc = p[0] * sc_s[0] + p[1] * sc_s[1];
        Scalar clamped = ad_max(Scalar(0.0), dot_p_sc);
        clamped = (clamped.value() > r_) ? r_s : clamped;

        Vector2S closest = sc_s * clamped;
        Vector2S q_vec = p - closest;
        Scalar m_dist = sqrt(q_vec[0] * q_vec[0] + q_vec[1] * q_vec[1]);

        // Determine sign
        Scalar cross_val = sc_s[1] * p[0] - sc_s[0] * p[1];
        Scalar m = (cross_val.value() >= 0.0) ? m_dist : -m_dist;

        // Return minimum
        return (n.value() > m.value()) ? n - Scalar(isovalue) : m - Scalar(isovalue);
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Arc SDF: arc with thickness
// ============================================================================
class Arc2D : public SDF2D {
  public:
    Arc2D(const Eigen::Vector2d& sc, double ra, double rb) : sc_(sc), ra_(ra), rb_(rb) {}
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    Eigen::Vector2d sc_; // (sin(angle), cos(angle))
    double ra_;          // radius
    double rb_;          // thickness

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::abs;
        using std::sqrt;

        Scalar ra_s = (Scalar(ra_));
        Scalar rb_s = (Scalar(rb_));
        Vector2S sc_s((Scalar(sc_[0])), (Scalar(sc_[1])));

        Vector2S p((ad_abs(Scalar(q[0]))), (Scalar(q[1])));

        // Check if in wedge
        bool in_wedge = (sc_[1] * p[0].value() > sc_[0] * p[1].value());

        if (in_wedge) {
            // Distance to arc endpoint
            Vector2S w1 = p - ra_s * sc_s;
            Scalar d1 = sqrt(w1[0] * w1[0] + w1[1] * w1[1]);
            return d1 - rb_s - Scalar(isovalue);
        } else {
            // Distance to ring
            Scalar l2 = sqrt(q[0] * q[0] + q[1] * q[1]);
            Scalar w2 = l2 - ra_s;
            return ad_abs(w2) - rb_s - Scalar(isovalue);
        }
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Pill/Segment SDF
// ============================================================================
class Pill2D : public SDF2D {
  public:
    Pill2D(const Eigen::Vector2d& a, const Eigen::Vector2d& b, double r) : a_(a), b_(b), r_(r) {}
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    Eigen::Vector2d a_, b_;
    double r_;

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::sqrt;

        Vector2S a_s((Scalar(a_[0])), (Scalar(a_[1])));
        Vector2S b_s((Scalar(b_[0])), (Scalar(b_[1])));
        Scalar r_s = Scalar(r_);

        Vector2S ba = b_s - a_s;
        Vector2S pa = q - a_s;
        Scalar ba_sqnorm = ba[0] * ba[0] + ba[1] * ba[1];
        Scalar dot = pa[0] * ba[0] + pa[1] * ba[1];
        Scalar h = dot / ba_sqnorm;
        h = ad_max(Scalar(0.0), ad_min(Scalar(1.0), h));

        Vector2S q_vec = pa - h * ba;
        Scalar d = sqrt(q_vec[0] * q_vec[0] + q_vec[1] * q_vec[1]);
        return d - r_s - Scalar(isovalue);
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Vesica SDF
// ============================================================================
class Vesica2D : public SDF2D {
  public:
    Vesica2D(double r, double d) : r_(r), d_(d), b_(std::sqrt(r * r - d * d)) {}
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    double r_, d_, b_;

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::abs;
        using std::sqrt;

        Scalar r_s = Scalar(r_);
        Scalar d_s = Scalar(d_);
        Scalar b_s = Scalar(b_);

        // Apply symmetry
        Vector2S p(ad_abs(q[0]), ad_abs(q[1]));

        // Check condition
        bool cond = ((p[1] - b_s) * d_s).value() > (p[0] * b_s).value();

        if (cond) {
            // Distance to tip
            Vector2S q1(p[0], p[1] - b_s);
            Scalar l1 = sqrt(q1[0] * q1[0] + q1[1] * q1[1]);
            Scalar sign_d = (d_ >= 0.0) ? Scalar(1.0) : Scalar(-1.0);
            return l1 * sign_d - Scalar(isovalue);
        } else {
            // Distance to circular arc
            Vector2S q2(p[0] + d_s, p[1]);
            Scalar l2 = sqrt(q2[0] * q2[0] + q2[1] * q2[1]);
            return l2 - r_s - Scalar(isovalue);
        }
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Box SDF
// ============================================================================
class Box2D : public SDF2D {
  public:
    Box2D(const Eigen::Vector2d& b) : b_(b) {}
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    Eigen::Vector2d b_; // half-extents

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::abs;
        using std::sqrt;

        Vector2S b_s((Scalar(b_[0])), (Scalar(b_[1])));
        Vector2S w(ad_abs(q[0]) - b_s[0], ad_abs(q[1]) - b_s[1]);
        Scalar g = max(w[0], w[1]);
        Vector2S q_vec(ad_max(w[0], Scalar(0.0)), ad_max(w[1], Scalar(0.0)));
        Scalar l = sqrt(q_vec[0] * q_vec[0] + q_vec[1] * q_vec[1]);
        Scalar phi = (g > 0.) ? l : g;
        return phi - Scalar(isovalue);
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Cross SDF
// ============================================================================
class Cross2D : public SDF2D {
  public:
    Cross2D(const Eigen::Vector2d& b) : b_(b) {}
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    Eigen::Vector2d b_;

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::abs;
        using std::sqrt;

        Vector2S b_s((Scalar(b_[0])), (Scalar(b_[1])));

        // Apply symmetry
        Vector2S p(ad_abs(q[0]), ad_abs(q[1]));

        // Swap if needed
        bool swap = p[1].value() > p[0].value();
        if (swap) {
            Scalar temp = p[0];
            p[0] = p[1];
            p[1] = temp;
        }

        Vector2S q_vec = p - b_s;
        Scalar h = ad_max(q_vec[0], q_vec[1]);

        Vector2S o;
        if (h.value() < 0.0) {
            Scalar ox = ad_max((Scalar(b_s[1] - b_s[0] - q_vec[0])), (Scalar(0.0)));
            Scalar oy = ad_max((Scalar(-q_vec[1])), (Scalar(0.0)));
            o = Vector2S(ox, oy);
        } else {
            o = Vector2S(ad_max(q_vec[0], (Scalar(0.0))), ad_max(q_vec[1], Scalar(0.0)));
        }

        Scalar l = sqrt(o[0] * o[0] + o[1] * o[1]);

        bool use_corner = (h.value() >= 0.0) || ((-q_vec[0]).value() >= l.value());
        Scalar result;
        if (use_corner) {
            Scalar sign_h = (h.value() >= 0.0) ? Scalar(1.0) : Scalar(-1.0);
            result = l * sign_h;
        } else {
            Scalar sign_h = (h.value() >= 0.0) ? Scalar(1.0) : Scalar(-1.0);
            result = -q_vec[0] * sign_h;
        }

        return result - Scalar(isovalue);
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Isosceles Triangle SDF
// ============================================================================
class TriangleIsosceles2D : public SDF2D {
  public:
    TriangleIsosceles2D(const Eigen::Vector2d& q) : q_(q) {}
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& p,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    Eigen::Vector2d q_;

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::abs;
        using std::sqrt;

        Vector2S q_s((Scalar(q_[0])), (Scalar(q_[1])));
        Vector2S p(ad_abs(q[0]), q[1]);

        // Project onto edge
        Scalar dot_pq = p[0] * q_s[0] + p[1] * q_s[1];
        Scalar dot_qq = q_s[0] * q_s[0] + q_s[1] * q_s[1];
        Scalar t = dot_pq / dot_qq;
        t = ad_max((Scalar(0.0)), ad_min((Scalar(1.0)), t));

        Vector2S a = p - q_s * t;

        // Project onto vertical edge
        Scalar tx = p[0] / q_s[0];
        tx = ad_max((Scalar(0.0)), ad_min((Scalar(1.0)), tx));
        Vector2S b = p - Vector2S(q_s[0] * tx, q_s[1]);

        Scalar l1 = a[0] * a[0] + a[1] * a[1];
        Scalar l2 = b[0] * b[0] + b[1] * b[1];

        Scalar d = sqrt((l1.value() < l2.value()) ? l1 : l2);

        // Determine sign
        Scalar k = (q_[1] >= 0.0) ? Scalar(1.0) : Scalar(-1.0);
        Scalar cross_val = p[0] * q_s[1] - p[1] * q_s[0];
        Scalar s1 = k * cross_val;
        Scalar s2 = k * (p[1] - q_s[1]);
        Scalar s = ad_max(s1, s2);
        Scalar sign_s = (s.value() >= 0.0) ? Scalar(1.0) : Scalar(-1.0);

        return d * sign_s - Scalar(isovalue);
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// General Triangle SDF
// ============================================================================
class Triangle2D : public SDF2D {
  public:
    Triangle2D(const Eigen::Vector2d& v0, const Eigen::Vector2d& v1, const Eigen::Vector2d& v2) {
        v_[0] = v0;
        v_[1] = v1;
        v_[2] = v2;
    }
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& p,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    Eigen::Vector2d v_[3];

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::sqrt;

        Vector2S v0_s((Scalar(v_[0][0])), (Scalar(v_[0][1])));
        Vector2S v1_s((Scalar(v_[1][0])), (Scalar(v_[1][1])));
        Vector2S v2_s((Scalar(v_[2][0])), (Scalar(v_[2][1])));

        // Edge vectors
        Vector2S e0 = v1_s - v0_s;
        Vector2S e1 = v2_s - v1_s;
        Vector2S e2 = v0_s - v2_s;

        // Vectors from vertices to point
        Vector2S p((Scalar(q[0])), (Scalar(q[1])));
        Vector2S v0p = p - v0_s;
        Vector2S v1p = p - v1_s;
        Vector2S v2p = p - v2_s;

        // Project onto edges
        Scalar t0 = (v0p[0] * e0[0] + v0p[1] * e0[1]) / (e0[0] * e0[0] + e0[1] * e0[1]);
        t0 = ad_max((Scalar(0.0)), ad_min((Scalar(1.0)), t0));

        Scalar t1 = (v1p[0] * e1[0] + v1p[1] * e1[1]) / (e1[0] * e1[0] + e1[1] * e1[1]);
        t1 = ad_max((Scalar(0.0)), ad_min((Scalar(1.0)), t1));

        Scalar t2 = (v2p[0] * e2[0] + v2p[1] * e2[1]) / (e2[0] * e2[0] + e2[1] * e2[1]);
        t2 = ad_max((Scalar(0.0)), ad_min((Scalar(1.0)), t2));

        // Closest points
        Vector2S c0 = v0p - e0 * t0;
        Vector2S c1 = v1p - e1 * t1;
        Vector2S c2 = v2p - e2 * t2;

        // Squared distances
        Scalar d0 = c0[0] * c0[0] + c0[1] * c0[1];
        Scalar d1 = c1[0] * c1[0] + c1[1] * c1[1];
        Scalar d2 = c2[0] * c2[0] + c2[1] * c2[1];

        Scalar res_d = ad_min(d0, ad_min(d1, d2));

        // Sign using cross products
        Scalar s0 = v0p[0] * e0[1] - v0p[1] * e0[0];
        Scalar s1 = v1p[0] * e1[1] - v1p[1] * e1[0];
        Scalar s2 = v2p[0] * e2[1] - v2p[1] * e2[0];

        Scalar res_s = ad_max(s0, ad_max(s1, s2));
        Scalar sign_s = (res_s.value() >= 0.0) ? Scalar(1.0) : Scalar(-1.0);

        return sqrt(res_d) * sign_s - Scalar(isovalue);
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Hexagon SDF
// ============================================================================
class Hexagon2D : public SDF2D {
  public:
    Hexagon2D(double r) : r_(r) {}
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    double r_;
    Eigen::Vector3d k_{-0.5 * std::sqrt(3), 0.5, 1. / std::sqrt(3)};

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::abs;
        using std::sqrt;

        Scalar r_s = Scalar(r_);
        Scalar kx = Scalar(k_[0]);
        Scalar ky = Scalar(k_[1]);
        Scalar kz = Scalar(k_[2]);

        Vector2S p(ad_abs(q[0]), ad_abs(q[1]));

        // w = p.x * k.x + p.y * k.y
        Scalar w = p[0] * kx + p[1] * ky;

        // p -= 2 * ad_min(w, 0) * (k.x, k.y)
        Scalar factor = (Scalar(2.0)) * ad_min(w, (Scalar(0.0)));
        p[0] = p[0] - factor * kx;
        p[1] = p[1] - factor * ky;

        // p -= (clamp(p.x, -k.z * r, k.z * r), r)
        Scalar clamped_x = p[0];
        clamped_x = ad_max(clamped_x, (Scalar(-kz * r_s)));
        clamped_x = ad_min(clamped_x, (Scalar(kz * r_s)));
        p[0] = p[0] - clamped_x;
        p[1] = p[1] - r_s;

        Scalar d = sqrt(p[0] * p[0] + p[1] * p[1]);
        Scalar sign_py = (p[1].value() >= 0.0) ? (Scalar(1.0)) : (Scalar(-1.0));

        return d * sign_py - Scalar(isovalue);
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Ellipse SDF
// ============================================================================
class Ellipse2D : public SDF2D {
  public:
    Ellipse2D(const Eigen::Vector2d& ab) : ab_(ab) {}
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    Eigen::Vector2d ab_; // semi-axes

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::abs;
        using std::atan2;
        using std::cos;
        using std::sin;
        using std::sqrt;

        Vector2S ab_s((Scalar(ab_[0])), (Scalar(ab_[1])));
        Vector2S p(ad_abs(q[0]), ad_abs(q[1]));

        Scalar normalized = (p[0] * p[0]) / (ab_s[0] * ab_s[0]) + (p[1] * p[1]) / (ab_s[1] * ab_s[1]);
        bool s = normalized > 1.;
        Scalar w = atan2(ab_s[0] * p[1], ab_s[1] * p[0]);
        if (!s) {
            w = (ab_s[0] * (p[0] - ab_s[0]) < ab_s[1] * (p[1] - ab_s[1])) ? 1.570796327 : 0.0;
        }

        for (int i = 0; i < 4; i++) {
            Scalar c = cos(w);
            Scalar s = sin(w);
            Vector2S u(ab_s[0] * c, ab_s[1] * s);
            Vector2S v(-ab_s[0] * s, ab_s[1] * c);
            Vector2S diff = p - u;
            Scalar numerator = diff[0] * v[0] + diff[1] * v[1];
            Scalar denominator = diff[0] * u[0] + diff[1] * u[1] + v[0] * v[0] + v[1] * v[1];
            w += numerator / denominator;
        }

        Scalar final_c = cos(w);
        Scalar final_s = sin(w);
        Vector2S closest(ab_s[0] * final_c, ab_s[1] * final_s);

        Vector2S diff = p - closest;
        Scalar dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1]);

        Scalar sgn = s ? Scalar(1.0) : Scalar(-1.0);
        return sgn * dist - Scalar(isovalue);
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Heart SDF
// ============================================================================
class Heart2D : public SDF2D {
  public:
    Heart2D() = default;
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::abs;
        using std::sqrt;

        Scalar r = Scalar(std::sqrt(2.0) / 4.0);
        Vector2S p(ad_abs(q[0]), q[1]);

        bool upper_cond = (p[1] + p[0]).value() > 1.0;

        if (upper_cond) {
            Vector2S q0 = p - Vector2S((Scalar(0.25)), (Scalar(0.75)));
            Scalar l0 = sqrt(q0[0] * q0[0] + q0[1] * q0[1]);
            return l0 - r - (Scalar(isovalue));
        } else {
            Vector2S q1 = p - Vector2S((Scalar(0.0)), (Scalar(1.0)));
            Scalar d1_sq = q1[0] * q1[0] + q1[1] * q1[1];

            Scalar h = ad_max((Scalar(p[0] + p[1])), (Scalar(0.0)));
            Vector2S q2 = p - Vector2S(h * (Scalar(0.5)), h * (Scalar(0.5)));
            Scalar d2_sq = q2[0] * q2[0] + q2[1] * q2[1];

            Scalar d_sq = (d1_sq.value() < d2_sq.value()) ? d1_sq : d2_sq;
            Scalar d = sqrt(d_sq);

            Scalar s = p[0] - p[1];
            Scalar sign_mult = (s.value() >= 0.0) ? (Scalar(1.0)) : (Scalar(-1.0));

            return d * sign_mult - (Scalar(isovalue));
        }
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Pentagon SDF
// ============================================================================
class Pentagon2D : public SDF2D {
  public:
    explicit Pentagon2D(double r);
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    double r_;
    Eigen::Vector3d m_;
    Eigen::Vector2d n_;

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::abs;
        using std::sqrt;

        Scalar r_s = Scalar(r_);
        Scalar mx = Scalar(m_[0]);
        Scalar my = Scalar(m_[1]);
        Scalar mz = Scalar(m_[2]);
        Vector2S n_s((Scalar(n_[0])), (Scalar(n_[1])));

        Vector2S p(ad_abs(q[0]), q[1]);

        Scalar w1 = p[0] * mx + p[1] * my;
        Scalar w2 = p[0] * n_s[0] - p[1] * n_s[1];

        Scalar factor1 = Scalar(2.0) * ad_max(w1, (Scalar(0.0)));
        p[0] = p[0] - factor1 * mx;
        p[1] = p[1] - factor1 * my;

        Scalar factor2 = Scalar(2.0) * ad_min(w2, (Scalar(0.0)));
        p[0] = p[0] - factor2 * mx;
        p[1] = p[1] + factor2 * my;

        Scalar clamped_x = p[0];
        clamped_x = ad_max(clamped_x, (Scalar(-r_s * mz)));
        clamped_x = ad_min(clamped_x, (Scalar(r_s * mz)));
        p[0] = p[0] - clamped_x;
        p[1] = p[1] + r_s;

        Scalar d = sqrt(p[0] * p[0] + p[1] * p[1]);
        Scalar sign_py = (p[1].value() >= 0.0) ? Scalar(-1.0) : Scalar(1.0);

        return d * sign_py - Scalar(isovalue);
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Moon SDF
// ============================================================================
class Moon2D : public SDF2D {
  public:
    Moon2D(double d, double ra, double rb) : d_(d), ra_(ra), rb_(rb) {}
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    double d_, ra_, rb_;

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::abs;
        using std::sqrt;

        Scalar d_s = Scalar(d_);
        Scalar ra_s = Scalar(ra_);
        Scalar rb_s = Scalar(rb_);

        Vector2S p(q[0], ad_abs(q[1]));

        Scalar a = (ra_s * ra_s - rb_s * rb_s + d_s * d_s) / (Scalar(2.0) * d_s);
        Scalar b = sqrt(ad_max((Scalar(ra_s * ra_s - a * a)), (Scalar(0.0))));

        Scalar lhs = d_s * (p[0] * b - p[1] * a);
        Scalar rhs = d_s * d_s * ad_max((Scalar(b - p[1])), (Scalar(0.0)));
        bool cond = lhs.value() > rhs.value();

        if (cond) {
            Vector2S w = p - Vector2S(a, b);
            Scalar d_cond = sqrt(w[0] * w[0] + w[1] * w[1]);
            return d_cond - Scalar(isovalue);
        } else {
            Vector2S w1 = p;
            Scalar l1 = sqrt(w1[0] * w1[0] + w1[1] * w1[1]);
            Scalar d1 = l1 - ra_s;

            Vector2S w2 = p - Vector2S(d_s, Scalar(0.0));
            Scalar l2 = sqrt(w2[0] * w2[0] + w2[1] * w2[1]);
            Scalar d2 = rb_s - l2;

            Scalar result = (d1.value() > d2.value()) ? d1 : d2;
            return result - Scalar(isovalue);
        }
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Isosceles Trapezoid SDF
// ============================================================================
class Trapezoid2D : public SDF2D {
  public:
    Trapezoid2D(double ra, double rb, double he) : ra_(ra), rb_(rb), he_(he) {}
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& q,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    double ra_, rb_, he_;

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::abs;
        using std::sqrt;

        Scalar ra_s = Scalar(ra_);
        Scalar rb_s = Scalar(rb_);
        Scalar he_s = Scalar(he_);

        Vector2S p(ad_abs(q[0]), q[1]);

        Scalar h = ad_min(p[0], (p[1].value() < 0.0) ? ra_s : rb_s);
        Scalar sy = (p[1].value() >= 0.0) ? Scalar(1.0) : Scalar(-1.0);
        Vector2S c(h, sy * he_s);
        Vector2S q_vec = p - c;
        Scalar d = q_vec[0] * q_vec[0] + q_vec[1] * q_vec[1];
        Scalar s = ad_abs(p[1]) - he_s;

        Scalar res_d = d;
        Scalar res_s = s;

        Vector2S k(rb_s - ra_s, Scalar(2.0) * he_s);
        Vector2S w = p - Vector2S(ra_s, -he_s);
        Scalar dot_wk = w[0] * k[0] + w[1] * k[1];
        Scalar dot_kk = k[0] * k[0] + k[1] * k[1];
        h = dot_wk / dot_kk;
        h = max(Scalar(0.0), ad_min((Scalar(1.0)), h));

        c = Vector2S(ra_s, -he_s) + k * h;
        q_vec = p - c;
        d = q_vec[0] * q_vec[0] + q_vec[1] * q_vec[1];
        s = w[0] * k[1] - w[1] * k[0];

        res_d = (d.value() < res_d.value()) ? d : res_d;
        res_s = max(res_s, s);

        Scalar d_final = sqrt(res_d);
        Scalar sign_s = (res_s.value() >= 0.0) ? Scalar(1.0) : Scalar(-1.0);

        return d_final * sign_s - Scalar(isovalue);
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};

// ============================================================================
// Quad SDF
// ============================================================================
class Quad2D : public SDF2D {
  public:
    Quad2D(const Eigen::Vector2d& v0, const Eigen::Vector2d& v1, const Eigen::Vector2d& v2, const Eigen::Vector2d& v3) {
        v_[0] = v0;
        v_[1] = v1;
        v_[2] = v2;
        v_[3] = v3;
    }
    std::pair<double, Eigen::Vector2d> evaluate_with_gradient(const Eigen::Vector2d& p,
                                                              double isovalue = 0.0) const override;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> bounding_box(double isovalue = 0.0) const override;

  private:
    Eigen::Vector2d v_[4];

  protected:
    template <typename Scalar>
    Scalar evaluate_autodiff_template(const Eigen::Matrix<Scalar, 2, 1>& q, double isovalue) const {
        using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
        using std::sqrt;

        Vector2S v0_s((Scalar(v_[0][0])), (Scalar(v_[0][1])));
        Vector2S v1_s((Scalar(v_[1][0])), (Scalar(v_[1][1])));
        Vector2S v2_s((Scalar(v_[2][0])), (Scalar(v_[2][1])));
        Vector2S v3_s((Scalar(v_[3][0])), (Scalar(v_[3][1])));

        // Global sign
        Scalar gs_x = v0_s[0] - v3_s[0];
        Scalar gs_y = v0_s[1] - v3_s[1];
        Scalar gs_dx = v1_s[1] - v0_s[1];
        Scalar gs_dy = v1_s[0] - v0_s[0];
        Scalar gs = gs_x * gs_dx - gs_y * gs_dy;

        Scalar res_d = Scalar(1e10);
        Scalar res_s = Scalar(-1e10);

        Vector2S p((Scalar(q[0])), (Scalar(q[1])));

        Vector2S vertices[4] = {v0_s, v1_s, v2_s, v3_s};
        for (int i = 0; i < 4; ++i) {
            Vector2S vi = vertices[i];
            Vector2S vj = vertices[(i + 1) % 4];
            Vector2S e = vj - vi;
            Vector2S w = p - vi;

            Scalar dot_we = w[0] * e[0] + w[1] * e[1];
            Scalar dot_ee = e[0] * e[0] + e[1] * e[1];
            Scalar h = dot_we / dot_ee;
            h = ad_max((Scalar(0.0)), ad_min((Scalar(1.0)), h));

            Vector2S qe = w - e * h;
            Scalar d = qe[0] * qe[0] + qe[1] * qe[1];
            Scalar s_curr = gs * (w[0] * e[1] - w[1] * e[0]);

            res_d = (d.value() < res_d.value()) ? d : res_d;
            res_s = max(res_s, s_curr);
        }

        Scalar phi = sqrt(res_d);
        Scalar sign_s = (res_s.value() >= 0.0) ? Scalar(1.0) : Scalar(-1.0);

        return phi * sign_s - Scalar(isovalue);
    }

    ADScalar evaluate_autodiff_impl(const ADVector2& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }

    ADScalar2 evaluate_autodiff_2nd_impl(const Eigen::Matrix<ADScalar2, 2, 1>& q, double isovalue) const override {
        return evaluate_autodiff_template(q, isovalue);
    }
};
