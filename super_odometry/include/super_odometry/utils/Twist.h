//
// Created by ubuntu on 2020/4/2.
//
// ============================================================================
// OVERVIEW
// ============================================================================
// This file defines Twist<T>, the rigid-body transform type used throughout
// SuperOdom (usually via the aliases Transformd = Twist<double> and
// Transformf = Twist<float>).
//
// Despite its name, this class is NOT a velocity twist; it is an element of
// SE(3), i.e. a 3D rotation + translation. It stores:
//   - rot: a unit quaternion representing the rotation R
//   - pos: a 3-vector representing the translation t
// together they represent the transform T = [R t; 0 1], applied to a point
// as p_out = R * p_in + t (see operator*(Tangent3)).
//
// Conventions used in the codebase: T_parent_child denotes the pose of the
// child frame expressed in the parent frame, so chaining works like
// T_world_lidar = T_world_odom * T_odom_lidar, and
// T_parent_child.inverse() == T_child_parent.
//
// The class also provides a few Lie-group utilities (se3exp, expAndTheta,
// adjoints) used by the optimization code. It is header-only and is included
// by nearly every node in the package.
// ============================================================================

#ifndef TWIST_H
#define TWIST_H

#include <cmath>
#include <iostream>

#include <Eigen/Core>
#include <super_odometry/utils/EigenTypes.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

////////////////////////////////////////////////////////////////////////////
// Forward Declarations / typedefs
////////////////////////////////////////////////////////////////////////////

typedef pcl::PointXYZI PointType;
template <typename T> struct Twist;

// Aliases used throughout the codebase; Transformd is the common choice.
typedef Twist<float> Transformf;

typedef Twist<double> Transformd;

namespace TwistConstants {
// Numeric tolerances used to switch between the exact closed-form formulas
// and their small-angle Taylor expansions (which avoid dividing by ~0).
template <typename Scalar> struct Constants {
  EIGEN_ALWAYS_INLINE static const Scalar epsilon() {
    return static_cast<Scalar>(1e-10);
  }

  EIGEN_ALWAYS_INLINE static const Scalar pi() {
    return static_cast<Scalar>(M_PI);
  }
};

template <> struct Constants<float> {
  EIGEN_ALWAYS_INLINE static float epsilon() {
    return static_cast<float>(1e-5);
  }

  EIGEN_ALWAYS_INLINE static float pi() { return static_cast<float>(M_PI); }
};
} // namespace TwistConstants

/// A rigid-body transform (element of SE(3)) stored as quaternion + vector.
/// Applying it to a point gives p_out = rot * p_in + pos.
template <typename T> struct Twist {
public:
  typedef Eigen::Matrix<T, 4, 4> Transformation;   // 4x4 homogeneous matrix [R t; 0 1]
  typedef Eigen::Matrix<T, 3, 3> Matrix3x3;
  typedef Eigen::Matrix<T, 6, 6> Adjoint;          // 6x6 adjoint matrix of the transform
  typedef Eigen::Matrix<T, 6, 1> Tangent;          // 6-vector [translation; rotation] (v, omega)
  typedef Eigen::Matrix<T, 3, 1> Tangent3;         // plain 3-vector (point or axis-angle)
  typedef Eigen::Matrix<T, 4, 1> HomogeneousPoint; // point in homogeneous coordinates [x y z w]

public:
public:
  Eigen::Quaternion<T> rot;    // rotation part R as a unit quaternion
  Eigen::Matrix<T, 3, 1> pos;  // translation part t

public:
  /// Returns the identity transform (no rotation, no translation).
  static Twist Identity() { return Twist(); }

  /// Default constructor: identity rotation and zero translation.
  Twist() : rot(T(1), T(0), T(0), T(0)), pos(Tangent3::Zero()) {}

  /// Constructs from a quaternion and a translation vector.
  Twist(const Eigen::Quaternion<T> &rot_in,
        const Eigen::Matrix<T, 3, 1> &pos_in)
      : rot(rot_in), pos(pos_in) {}

  /// Constructs from a 4x4 homogeneous transform matrix.
  Twist(const Transformation &matrix)
      : rot(matrix.template topLeftCorner<3, 3>()),
        pos(matrix.template block<3, 1>(0, 3)) {}

  /// Constructs from a 3x3 rotation matrix and a translation vector.
  Twist(const Matrix3x3 &R, const Tangent3 &t) : rot(R), pos(t) {}

  /// Constructs from an Eigen affine transform (rotation is re-normalized).
  Twist(
      const Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> &transform) {
    this->rot = Eigen::Quaternion<T>{transform.linear()}.normalized();
    this->pos = transform.translation();
  }

  /// Converts this transform to an Eigen affine transform.
  Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> transform() const {
    Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> transform;
    transform.linear() = rot.normalized().toRotationMatrix();
    transform.translation() = pos;
    return transform;
  }

  /// Returns the 4x4 homogeneous transform matrix [R t; 0 0 0 1].
  inline Transformation matrix() const {
    Transformation homogenious_matrix;
    homogenious_matrix.setIdentity();
    homogenious_matrix.block(0, 0, 3, 3) =
        this->rot.normalized().toRotationMatrix();
    homogenious_matrix.col(3).head(3) = this->pos;

    return homogenious_matrix;
  }

  /// Returns the compact 3x4 matrix [R | t] (the top three rows of matrix()).
  inline Eigen::Matrix<T, 3, 4> matrix3x4() const {
    Eigen::Matrix<T, 3, 4> matrix;
    matrix.block(0, 0, 3, 3) = this->rot.normalized().toRotationMatrix();
    matrix.col(3) = this->pos;

    return matrix;
  }

  /// Returns the rotation part as a 3x3 rotation matrix.
  inline Matrix3x3 rotationMatrix() const {
    Matrix3x3 matrix;
    matrix = this->rot.normalized().toRotationMatrix();

    return matrix;
  }

  /// Raw pointer to the underlying storage. Because rot and pos are laid out
  /// contiguously, this exposes 7 scalars (qx,qy,qz,qw, x,y,z), which is what
  /// Ceres parameter blocks expect.
  inline T *data() { return rot.coeffs().data(); }

  // Const version of data() above.
  T const *data() const { return rot.coeffs().data(); }

  /// Returns the 6x6 adjoint matrix of this transform. The adjoint maps a
  /// tangent vector expressed in one frame to the other frame: it is used to
  /// move Jacobians/covariances between frames, Adj = [R  hat(t)*R; 0  R].
  inline Adjoint SE3Adj() const {
    const Eigen::Matrix<T, 3, 3> R = this->rot.normalized().toRotationMatrix();
    Adjoint res;
    res.template block<3, 3>(0, 0) = R;
    res.template block<3, 3>(3, 3) = R;
    res.template block<3, 3>(0, 3) = hat(this->pos) * R;
    res.template block<3, 3>(3, 0) = Eigen::Matrix<T, 3, 3>::Zero(3, 3);

    return res;
  }

  /// Adjoint-like matrix for the "decoupled" parametrization used by
  /// so3Transexp(), where rotation and translation are treated independently
  /// instead of the full SE(3) coupling.
  inline Adjoint SO3TransAdj() const {
    const Eigen::Matrix<T, 3, 3> R = this->rot.normalized().toRotationMatrix();
    Adjoint res;
    res.template block<3, 3>(0, 0) = R;
    res.template block<3, 3>(3, 3) = R;
    res.template block<3, 3>(0, 3) = hat(this->pos) * R + R * hat(this->pos);
    res.template block<3, 3>(3, 0) = Eigen::Matrix<T, 3, 3>::Zero(3, 3);

    return res;
  }

  /// SE(3) exponential map: converts a 6-vector a = [v; omega] (translation
  /// part first, rotation part last) into a rigid transform. The rotation is
  /// exp(omega) and the translation is V*v, where V is the standard SE(3)
  /// "left Jacobian" that couples rotation and translation. Inverse of a log
  /// map; used to apply small optimization updates to a pose.
  inline static Twist<T> se3exp(const Tangent &a) {
    const Eigen::Matrix<T, 3, 1> &omega = a.template tail<3>();

    T theta;
    const Eigen::Quaternion<T> &so3 = expAndTheta(omega, &theta);
    const Eigen::Matrix<T, 3, 3> &Omega = hat(omega);
    const Eigen::Matrix<T, 3, 3> &Omega_sq = Omega * Omega;
    Eigen::Matrix<T, 3, 3> V;

    if (theta < TwistConstants::Constants<T>::epsilon()) {
      // Note: That is an accurate expansion!
      V = so3.matrix();
    } else {
      T theta_sq = theta * theta;
      V = (Eigen::Matrix<T, 3, 3>::Identity() +
           (static_cast<T>(1) - std::cos(theta)) / (theta_sq)*Omega +
           (theta - std::sin(theta)) / (theta_sq * theta) * Omega_sq);
    }

    return Twist<T>(so3, V * a.template head<3>());
  }

  /// "Decoupled" exponential map: rotation = exp(omega), translation = t
  /// taken directly (no V matrix coupling). Simpler than se3exp and matches
  /// the SO(3) x R^3 parametrization used elsewhere in the optimizer.
  inline static Twist<T> so3Transexp(const Tangent &a) {
    const Eigen::Matrix<T, 3, 1> &t = a.template head<3>();
    const Eigen::Matrix<T, 3, 1> &omega = a.template tail<3>();
    T theta;
    const Eigen::Quaternion<T> &so3 = expAndTheta(omega, &theta);

    return Twist<T>(so3, t);
  }

  /// Returns the inverse transform: if this is T_a_b then the result is
  /// T_b_a, with R_inv = R^T and t_inv = -R^T * t.
  Twist inverse() const {
    //        Eigen::Transform<T, 3, Eigen::TransformTraits::Affine>
    //        transform_inv = this->transform().inverse();
    Twist twist_inv;
    twist_inv.rot = this->rot.conjugate();
    twist_inv.pos = -twist_inv.rot.toRotationMatrix() * this->pos;
    return twist_inv;
  }

  /// Composes two transforms: T_a_c = T_a_b * T_b_c.
  Twist operator*(const Twist &other) const {
    Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> transform_out =
        this->transform() * other.transform();
    return Twist(transform_out);
  }

  /// Transforms a 3D point: p_out = R * p_in + t.
  Tangent3 operator*(const Tangent3 &p) const { return rot * p + pos; }

  /// Transforms a homogeneous point [x y z w]; the translation is scaled by
  /// w so that directions (w = 0) are only rotated.
  HomogeneousPoint operator*(const HomogeneousPoint &p) const {
    const Tangent3 tp = rot * p.template head<3>() + p(3) * pos;
    return HomogeneousPoint(tp(0), tp(1), tp(2), p(3));
  }

  /// Returns a copy with a different scalar type, e.g. double -> float.
  template <typename NewType> Twist<NewType> cast() const {
    Twist<NewType> twist_new{this->rot.template cast<NewType>(),
                             this->pos.template cast<NewType>()};
    return twist_new;
  }

  /// Prints as "x y z qw qx qy qz" (translation first, then quaternion).
  friend std::ostream &operator<<(std::ostream &os, const Twist &twist) {
    os << twist.pos.x() << " " << twist.pos.y() << " " << twist.pos.z() << " "
       << twist.rot.w() << " " << twist.rot.x() << " " << twist.rot.y() << " "
       << twist.rot.z();
    return os;
  }

  /// SO(3) exponential map: converts an axis-angle vector omega (direction =
  /// rotation axis, norm = rotation angle in radians) into a unit quaternion,
  /// and also returns the angle through *theta. Uses a Taylor expansion for
  /// very small angles to stay numerically stable.
  inline static Eigen::Quaternion<T> expAndTheta(const Tangent3 &omega,
                                                 T *theta) {
    const T theta_sq = omega.squaredNorm();
    *theta = std::sqrt(theta_sq);
    const T half_theta = static_cast<T>(0.5) * (*theta);

    T imag_factor;
    T real_factor;
    ;
    if ((*theta) < TwistConstants::Constants<T>::epsilon()) {
      const T theta_po4 = theta_sq * theta_sq;
      imag_factor = static_cast<T>(0.5) -
                    static_cast<T>(1.0 / 48.0) * theta_sq +
                    static_cast<T>(1.0 / 3840.0) * theta_po4;
      real_factor = static_cast<T>(1) - static_cast<T>(0.5) * theta_sq +
                    static_cast<T>(1.0 / 384.0) * theta_po4;
    } else {
      const T sin_half_theta = std::sin(half_theta);
      imag_factor = sin_half_theta / (*theta);
      real_factor = std::cos(half_theta);
    }

    return Eigen::Quaternion<T>(real_factor, imag_factor * omega.x(),
                                imag_factor * omega.y(),
                                imag_factor * omega.z());
  }

private:
  /// The "hat" operator: builds the 3x3 skew-symmetric matrix of omega, so
  /// that hat(omega) * v == omega.cross(v) for any vector v.
  inline static const Matrix3x3 hat(const Tangent3 &omega) {
    Matrix3x3 Omega;
    Omega << static_cast<T>(0), -omega(2), omega(1), omega(2),
        static_cast<T>(0), -omega(0), -omega(1), omega(0), static_cast<T>(0);
    return Omega;
  }

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}; // class Twist

#endif // TWIST_H
