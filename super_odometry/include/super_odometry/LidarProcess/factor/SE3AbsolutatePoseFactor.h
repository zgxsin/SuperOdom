//
// Created by shibo zhao on 2020-09-27.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// This header declares a Ceres cost function ("factor") that penalizes the
// deviation of an optimized pose from an externally measured absolute pose
// T_world_lidar_meas (for example an IMU-preintegration prediction or another
// odometry source). The laser mapping optimization adds it as a soft prior
// so the scan-matching solution stays close to that measurement, weighted
// by how much we trust it.
//
// Template arguments of SizedCostFunction<6, 7>:
//   - 6 = residual dimension (3 position + 3 orientation errors).
//   - 7 = size of the single parameter block: the pose stored as
//     [px, py, pz, qx, qy, qz, qw] (see PoseLocalParameterization for why a
//     7-dim block has only 6 degrees of freedom).
//
// The residual is weighted by the square root of a 6x6 information matrix
// (inverse covariance): Ceres minimizes ||r||^2, so pre-multiplying r by
// sqrt_information_ makes the minimized quantity r^T * Information * r, the
// standard Mahalanobis distance. A large information entry means "trust
// this component of the measurement strongly".
// ============================================================================

#ifndef SE3ABSOLUTATEPOSEFACTOR_H
#define SE3ABSOLUTATEPOSEFACTOR_H

#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include "super_odometry/utils/EigenTypes.h"
#include "super_odometry/utils/Twist.h"
#include <Eigen/Dense>

/// Ceres factor tying one 7-dim pose parameter block to a measured absolute
/// lidar pose T_world_lidar_meas (world <- lidar), producing a 6-dim weighted residual.
class SE3AbsolutatePoseFactor : public ceres::SizedCostFunction<6, 7> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// @param T_world_lidar_meas Measured pose of the lidar in the world frame.
  /// @param information 6x6 information matrix (inverse covariance) in the
  ///                    residual order [position (3), rotation (3)].
  SE3AbsolutatePoseFactor(const Transformd &T_world_lidar_meas,
                          const Eigen::Mat66d &information)
      : T_world_lidar_meas_(T_world_lidar_meas), information_matrix(information) {
    // Cholesky-factor the information matrix once: Information = L * L^T,
    // and store L^T so that residual r can be whitened as r' = L^T * r,
    // giving ||r'||^2 = r^T * Information * r.
    sqrt_information_ =
        Eigen::LLT<Eigen::Matrix<double, 6, 6>>(information_matrix)
            .matrixL()
            .transpose();
  }
  /// Computes the 6-dim residual (and optionally the 6x7 Jacobian) for the
  /// current pose estimate; see the .cpp file for the math.
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;
public:
  const Transformd T_world_lidar_meas_; // measured absolute pose (world <- lidar)

  Eigen::Mat66d information_matrix;  // inverse covariance of the measurement
  Eigen::Mat66d sqrt_information_;   // upper-triangular square root, whitens the residual
};

#endif // SE3ABSOLUTATEPOSEFACTOR_H
