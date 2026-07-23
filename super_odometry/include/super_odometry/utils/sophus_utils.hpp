//
// Created by ubuntu on 2020/5/23.
//
// ============================================================================
// OVERVIEW
// ============================================================================
// Extensions to the Sophus Lie-group library, used by the optimization code
// (e.g. the local parameterization of poses in the Ceres solvers).
//
// Background for beginners:
//  - A pose lives on the curved manifold SE(3); a rotation on SO(3). To
//    optimize over them we work in the flat "tangent space": log() maps a
//    pose to a 6-vector (or a rotation to a 3-vector phi, the axis-angle),
//    and exp() maps back.
//  - This file uses a DECOUPLED convention (logd/expd): the 6-vector is
//    [translation; rotation] where the translation is stored as-is, not
//    mixed with the rotation as in the standard SE(3) log/exp. This treats a
//    pose as the product SO(3) x R^3, which is simpler and common in SLAM.
//  - Jacobians of the exp/log maps tell us how a small change e in the
//    tangent space moves the group element. "Right" Jacobians multiply the
//    perturbation on the right side, "left" Jacobians on the left. They are
//    needed to write the chain rule for residuals that involve poses, e.g.
//    in Ceres cost functions and in IMU preintegration bias updates.
//
// Each function's one-line comment states the approximation it implements.
// ============================================================================

#ifndef SOPHUS_UTILS_HPP
#define SOPHUS_UTILS_HPP

#include "sophus/se3.hpp"
namespace Sophus {

/// Decoupled SE(3) log: returns [translation; so3-log(rotation)]. The
/// translation is copied unchanged (unlike the fully coupled SE(3) log).
template <typename Scalar>
inline static typename SE3<Scalar>::Tangent logd(const SE3<Scalar> &se3) {
    typename SE3<Scalar>::Tangent upsilon_omega;
    upsilon_omega.template tail<3>() = se3.so3().log();
    upsilon_omega.template head<3>() = se3.translation();

    return upsilon_omega;
}

/// Decoupled SE(3) exp, inverse of logd: rotation = exp(tail 3 entries),
/// translation = head 3 entries taken directly.
template <typename Derived>
inline static SE3<typename Derived::Scalar> expd(
    const Eigen::MatrixBase<Derived> &upsilon_omega) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 6);

    using Scalar = typename Derived::Scalar;

    return SE3<Scalar>(SO3<Scalar>::exp(upsilon_omega.template tail<3>()),
                       upsilon_omega.template head<3>());
}

/// Right Jacobian of SO(3): exp(phi+e) ~= exp(phi)*exp(J*e). Converts an
/// additive change of the axis-angle vector into a local (right-side)
/// rotation perturbation; used when chaining rotation derivatives in cost
/// functions. J is an output parameter (written through a const_cast).
// exp(phi+e) ~= exp(phi)*exp(J*e)
template <typename Derived1, typename Derived2>
void rightJacobianSO3(const Eigen::MatrixBase<Derived1> &phi,
                      const Eigen::MatrixBase<Derived2> &J_const) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

    using Scalar = typename Derived1::Scalar;

    Eigen::MatrixBase<Derived2> &J =
        const_cast<Eigen::MatrixBase<Derived2> &>(J_const);

    Scalar phi_norm2 = phi.squaredNorm();
    Scalar phi_norm = std::sqrt(phi_norm2);
    Scalar phi_norm3 = phi_norm2 * phi_norm;

    J.setIdentity();

    if (Sophus::Constants<Scalar>::epsilon() < phi_norm) {
        Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J -= phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
        J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
    }
}

/// Inverse of the right Jacobian: log(exp(phi)*exp(e)) ~= phi + J*e. Maps a
/// local rotation perturbation back to an additive change of the axis-angle
/// vector; used e.g. in the Jacobians of rotation-residual cost functions.
// log(exp(phi)exp(e)) ~= phi + J*e
template <typename Derived1, typename Derived2>
void rightJacobianInvSO3(const Eigen::MatrixBase<Derived1> &phi,
                         const Eigen::MatrixBase<Derived2> &J_const) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

    using Scalar = typename Derived1::Scalar;

    Eigen::MatrixBase<Derived2> &J =
        const_cast<Eigen::MatrixBase<Derived2> &>(J_const);

    Scalar phi_norm2 = phi.squaredNorm();
    Scalar phi_norm = std::sqrt(phi_norm2);

    J.setIdentity();

    if (Sophus::Constants<Scalar>::epsilon() < phi_norm) {
        Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J += phi_hat / 2;
        J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) /
            (2 * phi_norm * std::sin(phi_norm)));
    }
}

/// Left Jacobian of SO(3): exp(phi+e) ~= exp(J*e)*exp(phi). Same idea as the
/// right Jacobian but for perturbations applied on the left (world side).
// exp(phi+e) ~= exp(J*e)*exp(phi)
template <typename Derived1, typename Derived2>
void leftJacobianSO3(const Eigen::MatrixBase<Derived1> &phi,
                     const Eigen::MatrixBase<Derived2> &J_const) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

    using Scalar = typename Derived1::Scalar;

    Eigen::MatrixBase<Derived2> &J =
        const_cast<Eigen::MatrixBase<Derived2> &>(J_const);

    Scalar phi_norm2 = phi.squaredNorm();
    Scalar phi_norm = std::sqrt(phi_norm2);
    Scalar phi_norm3 = phi_norm2 * phi_norm;

    J.setIdentity();

    if (Sophus::Constants<Scalar>::epsilon() < phi_norm) {
        Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J += phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
        J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
    }
}

/// Inverse of the left Jacobian: log(exp(e)*exp(phi)) ~= phi + J*e.
// log(exp(e)*exp(phi)) ~= phi + J*e
template <typename Derived1, typename Derived2>
void leftJacobianInvSO3(const Eigen::MatrixBase<Derived1> &phi,
                        const Eigen::MatrixBase<Derived2> &J_const) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

    using Scalar = typename Derived1::Scalar;

    Eigen::MatrixBase<Derived2> &J =
        const_cast<Eigen::MatrixBase<Derived2> &>(J_const);

    Scalar phi_norm2 = phi.squaredNorm();
    Scalar phi_norm = std::sqrt(phi_norm2);

    J.setIdentity();

    if (Sophus::Constants<Scalar>::epsilon() < phi_norm) {
        Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J -= phi_hat / 2;
        J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) /
            (2 * phi_norm * std::sin(phi_norm)));
    }
}

/// 6x6 right Jacobian for the decoupled pose parametrization:
/// expd(phi+e) ~= expd(phi)*expd(J*e). The translation block is R^T and the
/// rotation block is the SO(3) right Jacobian; used by the pose local
/// parameterization in the Ceres-based solvers.
// expd(phi+e) ~= expd(phi)*expd(J*e)
template <typename Derived1, typename Derived2>
void rightJacobianSE3Decoupled(const Eigen::MatrixBase<Derived1> &phi,
                               const Eigen::MatrixBase<Derived2> &J_const) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 6);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 6, 6);

    using Scalar = typename Derived1::Scalar;

    Eigen::MatrixBase<Derived2> &J =
        const_cast<Eigen::MatrixBase<Derived2> &>(J_const);

    J.setZero();

    Eigen::Matrix<Scalar, 3, 1> omega = phi.template tail<3>();
    rightJacobianSO3(omega, J.template bottomRightCorner<3, 3>());
    J.template topLeftCorner<3, 3>() =
        Sophus::SO3<Scalar>::exp(omega).inverse().matrix();
}

/// Inverse of rightJacobianSE3Decoupled:
/// logd(expd(phi)*expd(e)) ~= phi + J*e.
// logd(expd(phi)expd(e)) ~= phi + J*e
template <typename Derived1, typename Derived2>
void rightJacobianInvSE3Decoupled(const Eigen::MatrixBase<Derived1> &phi,
                                  const Eigen::MatrixBase<Derived2> &J_const) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 6);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 6, 6);

    using Scalar = typename Derived1::Scalar;

    Eigen::MatrixBase<Derived2> &J =
        const_cast<Eigen::MatrixBase<Derived2> &>(J_const);

    J.setZero();

    Eigen::Matrix<Scalar, 3, 1> omega = phi.template tail<3>();
    rightJacobianInvSO3(omega, J.template bottomRightCorner<3, 3>());
    J.template topLeftCorner<3, 3>() = Sophus::SO3<Scalar>::exp(omega).matrix();
}

}  // namespace Sophus

#endif //SOPHUS_UTILS_HPP