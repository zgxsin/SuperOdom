//
// Created by shibo zhao on 2020-09-27.
//
// ============================================================================
// OVERVIEW
// ============================================================================
// Implementation of the Ceres residuals and the SE(3) parameterization
// declared in lidarOptimization.h. These are the mathematical heart of
// scan-to-map registration:
//
//   - Every matched edge point contributes a point-to-line residual.
//   - Every matched planar point contributes a point-to-plane residual.
//   - Ceres stacks thousands of these tiny residuals into one least-squares
//     problem and solves for the single pose (rotation q, translation t)
//     that minimizes their sum of squares.
//
// Convention used throughout: the pose parameter block is
// [tx, ty, tz, qx, qy, qz, qw] and maps points from the lidar (scan) frame
// into the world (map) frame: p_world = q * p_scan + t.
// ============================================================================

#include "super_odometry/LidarProcess/factor/lidarOptimization.h"

EdgeAnalyticCostFunction::EdgeAnalyticCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_, Eigen::Vector3d last_point_b_)
        : curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_){

}

// Point-to-line residual and jacobian.
//
// Geometry: the distance from point p to the line through a and b is
//   |(p - a) x (p - b)| / |a - b|
// (the cross product's magnitude is twice the area of triangle (p, a, b),
// and dividing by the base length |a - b| gives the height, i.e. the
// perpendicular distance). Instead of the scalar distance, the residual
// here is the 3-vector (p - a) x (p - b) / |a - b|, which has that distance
// as its norm; Ceres minimizes its squared norm, which is equivalent and
// keeps the derivatives simple.
bool EdgeAnalyticCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{

    // Interpret the raw parameter array as translation + quaternion.
    Eigen::Map<const Eigen::Vector3d> t_world_lidar(parameters[0]);
    Eigen::Map<const Eigen::Quaterniond> q_world_lidar(parameters[0]+3);

    // Transform the scan point into the world frame with the current pose
    // estimate, then form the cross-product numerator and the line
    // direction denominator described above.
    Eigen::Vector3d lp;
    lp = q_world_lidar * curr_point + t_world_lidar; // new world-frame point
    Eigen::Vector3d nu = (lp - last_point_a).cross(lp - last_point_b);
    Eigen::Vector3d de = last_point_a - last_point_b;

    residuals[0] = nu.x() / de.norm();
    residuals[1] = nu.y() / de.norm();
    residuals[2] = nu.z() / de.norm();

    // Analytic jacobian of the residual with respect to the pose. Ceres asks
    // for a 3x7 matrix (residual dim x global parameter dim); combined with
    // PoseSE3Parameterization only the first 6 columns (the se(3) tangent
    // directions) matter, the 7th column stays zero.
    if(jacobians != NULL)
    {
        if(jacobians[0] != NULL)
        {
            // Derivative of the transformed point p = q * x + t with respect
            // to the 6-dim increment [dt, dtheta]:
            //   dp/dt      = Identity
            //   dp/dtheta  = -R * skew(x)
            // (perturbing the rotation by a small angle dtheta moves the
            // point by -R * (x cross dtheta); skew() encodes the cross
            // product as a matrix).
            Eigen::Matrix3d skew_lp = skew(curr_point);
            Eigen::Matrix<double, 3, 6> dp_by_so3;
            (dp_by_so3.block<3,3>(0,0)).setIdentity();
            (dp_by_so3.block<3,3>(0, 3))=-q_world_lidar.toRotationMatrix()*skew_lp;
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3(jacobians[0]);
            J_se3.setZero();
            // Chain rule: d(residual)/dp = skew(b - a) / |a - b|, because
            // d/dp [(p - a) x (p - b)] = skew(b - a) (the p x p term cancels).
            // Multiply by dp/d(pose) to get the full 3x6 jacobian.
            Eigen::Vector3d re = last_point_b - last_point_a;
            Eigen::Matrix3d skew_re = skew(re);

            J_se3.block<3,6>(0,0) = skew_re * dp_by_so3/de.norm();
      
        }
    }

    return true;
 
}   


SurfNormAnalyticCostFunction::SurfNormAnalyticCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_, double negative_OA_dot_norm_) 
                                                        : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_), negative_OA_dot_norm(negative_OA_dot_norm_) {

}

// Point-to-plane residual and jacobian.
//
// The map plane was fitted (elsewhere) to nearby map points and stored in
// Hessian normal form n.dot(x) + d = 0 with |n| = 1. For a point exactly on
// the plane the expression n.dot(p) + d is zero; otherwise it equals the
// SIGNED perpendicular distance of p from the plane. That scalar is the
// residual.
bool SurfNormAnalyticCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Map<const Eigen::Vector3d> t_world_lidar(parameters[0]);
    Eigen::Map<const Eigen::Quaterniond> q_world_lidar(parameters[0]+3);
    // Transform the scan point into the world frame, then evaluate the
    // plane equation: signed distance to the plane.
    Eigen::Vector3d point_world =
        q_world_lidar * curr_point + t_world_lidar;

    residuals[0] = plane_unit_norm.dot(point_world) + negative_OA_dot_norm;

    if(jacobians != NULL)
    {
        if(jacobians[0] != NULL)
        {

            // Same dp/d(pose) block as in the edge factor: identity for the
            // translation part, -R * skew(x) for the rotation part.
            Eigen::Matrix3d skew_point_world = skew(curr_point);
            Eigen::Matrix<double, 3, 6> dp_by_so3; 
            (dp_by_so3.block<3,3>(0,0)).setIdentity();
            dp_by_so3.block<3,3>(0,3) =
                -q_world_lidar.toRotationMatrix()*skew_point_world;
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3(jacobians[0]);
            J_se3.setZero();
            // Chain rule: d(residual)/dp = n^T (the derivative of n.dot(p)+d),
            // so the 1x6 jacobian is n^T * dp/d(pose).
            J_se3.block<1,6>(0,0) = plane_unit_norm.transpose() * dp_by_so3;
   
        }
    }
    return true;

}   


// Applies a small 6-dim update delta = [upsilon; omega] to the 7-dim pose:
// the delta is converted to a rigid transform (delta_q, delta_t) via the
// SE(3) exponential map and composed on the LEFT of the current pose
// (i.e. the correction is expressed in the world frame):
//   q_new = delta_q * q,   t_new = delta_q * t + delta_t.
// Ceres calls this after each solver iteration to move the pose along the
// manifold without breaking the unit-quaternion constraint.
bool PoseSE3Parameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> trans(x);

    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_t;
    getTransformFromSe3(Eigen::Map<const Eigen::Matrix<double,6,1>>(delta), delta_q, delta_t);
    Eigen::Map<const Eigen::Quaterniond> quater(x+3);
    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> quater_plus(x_plus_delta+3);

    quater_plus = delta_q * quater;
    quater_plus.normalized();
    trans_plus = delta_q * trans + delta_t;

    return true;
}


// Jacobian of Plus with respect to delta, evaluated at delta = 0. Returning
// [Identity(6); 0] here is a common trick: it makes Ceres' chained jacobian
// equal to the first 6 columns the cost functions already wrote, so the
// cost functions effectively differentiate directly with respect to the
// 6-dim se(3) increment.
bool PoseSE3Parameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    (j.topRows(6)).setIdentity();
    (j.bottomRows(1)).setZero();

    return true;
}


// SE(3) exponential map: converts a tangent vector [upsilon; omega] into a
// rotation q and translation t. omega (rad) is the rotation part, upsilon
// the translation part. theta = |omega| is the rotation angle.
void getTransformFromSe3(const Eigen::Matrix<double,6,1>& se3, Eigen::Quaterniond& q, Eigen::Vector3d& t){
    Eigen::Vector3d omega(se3.data()+3);
    Eigen::Vector3d upsilon(se3.data());
    Eigen::Matrix3d Omega = skew(omega);

    double theta = omega.norm();
    double half_theta = 0.5*theta;

    // Quaternion for a rotation of theta about axis omega/theta is
    // [cos(theta/2), sin(theta/2) * axis]. imag_factor = sin(theta/2)/theta
    // so that multiplying by omega gives sin(theta/2) * axis directly.
    double imag_factor;
    double real_factor = cos(half_theta);
   
    if(theta<1e-10)
    {
        // Near theta = 0, sin(theta/2)/theta is 0/0; use its Taylor series
        // 1/2 - theta^2/48 + theta^4/3840 (coefficients written as decimals).
        double theta_sq = theta*theta;
        double theta_po4 = theta_sq*theta_sq;
        imag_factor = 0.5-0.0208333*theta_sq+0.000260417*theta_po4;
    }
    else
    {
        double sin_half_theta = sin(half_theta);
        imag_factor = sin_half_theta/theta;
    }

    q = Eigen::Quaterniond(real_factor, imag_factor*omega.x(), imag_factor*omega.y(), imag_factor*omega.z());


    // Translation part: t = J * upsilon, where J is the "left jacobian" of
    // SO(3). It accounts for the fact that while the frame rotates by omega,
    // the translation direction rotates along with it (for pure translation
    // J is the identity).
    Eigen::Matrix3d J;
    if (theta<1e-10)
    {
        J = q.matrix();
    }
    else
    {
        // Closed form (Rodrigues-like):
        // J = I + (1 - cos(theta))/theta^2 * Omega
        //       + (theta - sin(theta))/theta^3 * Omega^2.
        Eigen::Matrix3d Omega2 = Omega*Omega;
        J = (Eigen::Matrix3d::Identity() + (1-cos(theta))/(theta*theta)*Omega + (theta-sin(theta))/(pow(theta,3))*Omega2);
    }

    t = J*upsilon;
}

// Builds the skew-symmetric matrix [a]_x such that [a]_x * b = a.cross(b):
//   [   0  -a3   a2 ]
//   [  a3    0  -a1 ]
//   [ -a2   a1    0 ]
Eigen::Matrix<double,3,3> skew(const Eigen::Matrix<double,3,1>& mat_in){
    Eigen::Matrix<double,3,3> skew_mat;
    skew_mat.setZero();
    skew_mat(0,1) = -mat_in(2);
    skew_mat(0,2) =  mat_in(1);
    skew_mat(1,2) = -mat_in(0);
    skew_mat(1,0) =  mat_in(2);
    skew_mat(2,0) = -mat_in(1);
    skew_mat(2,1) =  mat_in(0);
    return skew_mat;
}
