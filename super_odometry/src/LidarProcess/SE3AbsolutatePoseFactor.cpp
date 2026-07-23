//
// Created by ubuntu on 2020/7/7.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// Implementation of SE3AbsolutatePoseFactor::Evaluate (see the header for
// what the factor is used for). Given the current pose estimate
// T_world_lidar_est = (q_world_lidar_est, t_world_lidar_est) stored as
// [px, py, pz, qx, qy, qz, qw], it computes a 6-dim residual against the
// measured lidar pose T_world_lidar_meas:
//
//   r[0..2] = P_estimate - P_meas                   (position error, meters)
//   r[3..5] = 2 * vec(Q_meas^-1 * Q_estimate)       (rotation error)
//
// The rotation residual works because the quaternion Q_err = Q_meas^-1 *
// Q_estimate represents the small rotation still separating measurement and
// estimate; for a small angle its vector part is approximately axis *
// sin(angle/2) ~ angle/2 * axis, so multiplying by 2 gives the rotation
// vector (radians). Both blocks are then whitened by sqrt_information_ so
// Ceres effectively minimizes the Mahalanobis distance
// r^T * Information * r.
//
// The analytic Jacobian is the derivative of this residual w.r.t. the 7
// global pose parameters (a 6x7 matrix); Ceres later multiplies it by the
// 7x6 lifting matrix of PoseLocalParameterization to obtain the 6x6
// tangent-space Jacobian actually used in the solver.
// ============================================================================

#include "super_odometry/LidarProcess/factor/SE3AbsolutatePoseFactor.h"
#include "super_odometry/utils/sophus_utils.hpp"
#include "super_odometry/utils/utility.h"

bool SE3AbsolutatePoseFactor::Evaluate(double const *const *parameters,
                                       double *residuals,
                                       double **jacobians) const {
  // Unpack the single parameter block. Note Eigen's Quaterniond constructor
  // takes (w, x, y, z) while the block stores [.., qx, qy, qz, qw].
  Eigen::Vector3d t_world_lidar_est(parameters[0][0], parameters[0][1],
                                    parameters[0][2]);
  Eigen::Quaterniond q_world_lidar_est(
      parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

  Transformd T_world_lidar_est(q_world_lidar_est, t_world_lidar_est);

  Eigen::Map<Sophus::Vector6d> res(residuals);

  // Position residual: plain difference in the world frame (meters).
  res.segment<3>(0) =
      T_world_lidar_est.pos - T_world_lidar_meas_.pos;

  // Rotation residual: 2 * vector part of the error quaternion
  // Q_meas^-1 * Q_estimate, a small-angle rotation vector in radians.
  Eigen::Vector3d error_quat =
      2.0 * (T_world_lidar_meas_.rot.conjugate() * q_world_lidar_est).vec();

  res.segment<3>(3) = error_quat;

  // Whiten: r' = sqrt_information_ * r, so ||r'||^2 = r^T * Info * r.
  res.applyOnTheLeft(sqrt_information_);

  if (jacobians) {

    if (jacobians[0]) {
      // 6x7 Jacobian w.r.t. the global parameters [p (3), q (4)].
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_from(
          jacobians[0]);
      jacobian_from.setZero();

      // d(position residual)/d(t_world_lidar_est) = Identity; the residual does
      // not depend on the quaternion, and the rotation residual does not
      // depend on the position (those blocks stay zero).
      jacobian_from.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

      // d(rotation residual)/d(qx, qy, qz): Qleft(q) is the 4x4 matrix of
      // left-multiplication by q; the bottom-right 3x3 block is the
      // derivative of 2 * vec(Q_err * dq) w.r.t. the vector part of dq
      // (the qw column is left zero and is discarded by the local
      // parameterization's lifting matrix).
      jacobian_from.block<3, 3>(3, 3) =
          (Utility::Qleft(T_world_lidar_meas_.rot.conjugate() *
                          q_world_lidar_est))
              .bottomRightCorner<3, 3>();

      // LOG(ERROR) << "\ndR_dqi:\n" << dR_dqi;

      // Whiten the Jacobian with the same square-root information matrix.
      jacobian_from.applyOnTheLeft(sqrt_information_);
    }
  }
  return true;
}
