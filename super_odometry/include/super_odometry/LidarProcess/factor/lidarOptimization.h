//
// Created by shibo zhao on 2020-09-27.
//
// ============================================================================
// OVERVIEW
// ============================================================================
// Ceres building blocks for the scan-to-map registration used by the
// laserMapping node. The unknown is a single rigid-body pose (the lidar pose
// in the world frame), stored as a 7-double array:
//
//   parameters = [tx, ty, tz, qx, qy, qz, qw]   (translation + unit quaternion)
//
// Two kinds of "residuals" (error terms that Ceres drives toward zero) are
// defined, one per matched feature point:
//
//   - EdgeAnalyticCostFunction: distance from a transformed edge (corner)
//     point to the 3D LINE through two map points (point-to-line residual).
//   - SurfNormAnalyticCostFunction: distance from a transformed planar point
//     to a map PLANE given in Hessian normal form (point-to-plane residual).
//
// "Analytic" means the jacobians (derivatives of the residual with respect
// to the pose) are hand-coded instead of computed by automatic
// differentiation; this is faster and is standard practice in LOAM-family
// systems.
//
// Because a unit quaternion has 4 numbers but only 3 degrees of freedom, the
// 7-dim parameter block actually moves on a 6-dim manifold (SE(3): 3 for
// rotation + 3 for translation). PoseSE3Parameterization tells Ceres how to
// apply a small 6-dim increment to the 7-dim state ("local
// parameterization"), keeping the quaternion a valid rotation.
// ============================================================================

#ifndef _LIDAR_OPTIMIZATION_ANALYTIC_H_
#define _LIDAR_OPTIMIZATION_ANALYTIC_H_

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

/// Converts an se(3) tangent vector [upsilon; omega] (translation part first,
/// rotation part last, both 3-dim) into a quaternion q and translation t via
/// the SE(3) exponential map.
void getTransformFromSe3(const Eigen::Matrix<double,6,1>& se3, Eigen::Quaterniond& q, Eigen::Vector3d& t);

/// Returns the 3x3 skew-symmetric ("hat") matrix of a vector, so that
/// skew(a) * b == a.cross(b). Used everywhere in the rotation jacobians.
Eigen::Matrix3d skew(const Eigen::Vector3d& mat_in);

/// Point-to-line residual for one edge feature. SizedCostFunction<3, 7>
/// means: the residual is 3-dim and there is one parameter block of size 7
/// (the pose). Given the current pose, the scan point is transformed into
/// the world frame and its perpendicular offset from the line through map
/// points a and b becomes the residual.
class EdgeAnalyticCostFunction : public ceres::SizedCostFunction<3, 7> {
	public:

		EdgeAnalyticCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_, Eigen::Vector3d last_point_b_);
		virtual ~EdgeAnalyticCostFunction() {}
		/// Computes the residual and, if requested, the analytic jacobian.
		virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

		Eigen::Vector3d curr_point;   // feature point in the lidar (scan) frame
		Eigen::Vector3d last_point_a; // first point on the map line (world frame)
		Eigen::Vector3d last_point_b; // second point on the map line (world frame)
};

/// Point-to-plane residual for one planar feature. The plane is stored in
/// Hessian normal form: n.dot(x) + d = 0, where n = plane_unit_norm and
/// d = negative_OA_dot_norm. The residual (1-dim) is the signed distance of
/// the transformed scan point from the plane.
class SurfNormAnalyticCostFunction : public ceres::SizedCostFunction<1, 7> {
	public:
		SurfNormAnalyticCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_, double negative_OA_dot_norm_);
		virtual ~SurfNormAnalyticCostFunction() {}
		/// Computes the signed point-to-plane distance and its jacobian.
		virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

		Eigen::Vector3d curr_point;      // feature point in the lidar (scan) frame
		Eigen::Vector3d plane_unit_norm; // unit normal n of the map plane (world frame)
		double negative_OA_dot_norm;     // plane offset d in n.dot(x) + d = 0
};

/// Tells Ceres how the 7-double pose block behaves as a 6-dim SE(3)
/// manifold: Plus() applies a small se(3) increment on the left of the
/// current pose, and ComputeJacobian() provides d(global)/d(local), chosen
/// so that the cost functions can write their jacobians directly with
/// respect to the 6-dim increment.
class PoseSE3Parameterization : public ceres::LocalParameterization {
public:
	
    PoseSE3Parameterization() {}
    virtual ~PoseSE3Parameterization() {}
    /// x_plus_delta = exp(delta) * x, where delta is a 6-dim se(3) vector.
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const;
    /// Jacobian of Plus with respect to delta at delta = 0 (identity over
    /// the first 6 rows, zero for the redundant quaternion component).
    virtual bool ComputeJacobian(const double* x, double* jacobian) const;
    virtual int GlobalSize() const { return 7; }  // doubles stored (t + quaternion)
    virtual int LocalSize() const { return 6; }   // true degrees of freedom
};



#endif // _LIDAR_OPTIMIZATION_ANALYTIC_H_
