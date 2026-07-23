//
// Created by shiboz on 2021-02-06.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// This header declares the Ceres "local parameterization" for a 6-DOF pose
// used by the lidar odometry/mapping optimization. The pose is stored as a
// 7-element array [px, py, pz, qx, qy, qz, qw] (position + unit quaternion),
// but a rigid-body pose only has 6 degrees of freedom: the quaternion has 4
// numbers constrained to unit length.
//
// If Ceres updated all 7 numbers independently, the quaternion would drift
// off the unit sphere and one gradient direction would be meaningless. A
// local parameterization fixes this by telling Ceres to take optimization
// steps in a minimal 6-dim "tangent space" (3 for translation + 3 for a
// small rotation vector) and to map each step back onto the 7-dim
// representation with a custom Plus() operation. See the .cpp file for the
// details of Plus() and the lifting Jacobian.
//
// Used by laserMapping/LidarSlam wherever a pose parameter block is added
// to a Ceres problem (together with factors such as SE3AbsolutatePoseFactor
// and the lidar feature factors).
// ============================================================================

#ifndef ARISE_SLAM_MID360_POSE_LOCAL_PARAMETERIZATION_H
#define ARISE_SLAM_MID360_POSE_LOCAL_PARAMETERIZATION_H


#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../../utils/utility.h"

/// Ceres local parameterization for a pose stored as
/// [px, py, pz, qx, qy, qz, qw] (7 global parameters, 6 local DOF).
class PoseLocalParameterization : public ceres::LocalParameterization
{
    /// Applies a 6-dim tangent-space step to a 7-dim pose:
    /// position += dp, quaternion = q * dq(dtheta), renormalized.
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    /// Jacobian of Plus() w.r.t. delta at delta = 0 (7x6 "lifting" matrix).
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    /// Size of the stored (ambient) parameter vector: 3 position + 4 quaternion.
    virtual int GlobalSize() const { return 7; };
    /// Size of the tangent space: 3 translation + 3 rotation-vector DOF.
    virtual int LocalSize() const { return 6; };
};






#endif //ARISE_SLAM_MID360_POSE_LOCAL_PARAMETERIZATION_H
