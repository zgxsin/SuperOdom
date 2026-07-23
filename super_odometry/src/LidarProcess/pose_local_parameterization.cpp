//
// Created by shiboz on 2021-02-06.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// Implementation of the SE3 local parameterization declared in
// pose_local_parameterization.h.
//
// Background: the pose is stored globally as 7 numbers
// [px, py, pz, qx, qy, qz, qw], but has only 6 degrees of freedom because
// the quaternion must stay on the unit sphere. Optimizing the 7 numbers
// directly would let the quaternion drift off the sphere (no longer a valid
// rotation) and would give Ceres a redundant, degenerate direction to
// search in. Instead, Ceres works in a minimal 6-dim tangent space
// delta = [dp (3), dtheta (3)]:
//
//   - Each Levenberg-Marquardt step produces a small delta.
//   - Plus() maps that delta back onto the manifold of valid poses:
//     p_new = p + dp, q_new = q * dq(dtheta), where dq turns the small
//     rotation vector dtheta (radians, axis * angle) into a quaternion
//     approximately (1, dtheta/2), applied on the right (a body-frame
//     perturbation).
//   - ComputeJacobian() supplies d(Plus)/d(delta) at delta = 0, the 7x6
//     "lifting" matrix Ceres uses to convert cost-function Jacobians (which
//     are w.r.t. the 7 global parameters) into tangent-space Jacobians.
// ============================================================================

#include "super_odometry/LidarProcess/factor/pose_local_parameterization.h"

/// x is the current pose [p (3), q (4)], delta the 6-dim step
/// [dp (3), dtheta (3)]; writes the perturbed pose into x_plus_delta.
bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    // Map the raw arrays as Eigen objects (no copies). The quaternion part
    // starts at x+3 and Eigen stores quaternions as (x, y, z, w).
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    // deltaQ turns the small rotation vector dtheta into the quaternion
    // (1, dtheta/2) (first-order exponential map).
    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

      p = _p + dp;
      // Right multiplication = rotate by dq in the body frame; normalize to
      // stay exactly on the unit sphere despite floating-point error.
      q = (_q * dq).normalized();

    return true;
}

/// The 7x6 lifting Jacobian at delta = 0. The simple [I6; 0] form works
/// because cost functions in this codebase (e.g. SE3AbsolutatePoseFactor)
/// already write their rotation Jacobian into the qx,qy,qz columns in a way
/// that matches this convention, leaving the qw column unused.
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
