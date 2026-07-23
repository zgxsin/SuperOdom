// Created by Shibo Zhao on 2025-03-31
//
// ============================================================================
// OVERVIEW
// ============================================================================
// Grab-bag of helper functions used mainly by the laserMapping node:
//
//   - Frame/transform helpers (transformAssociateToMap, transformUpdate):
//     maintain the correction between the incremental odometry frame and the
//     drift-corrected map frame. Naming follows T_frame_reference_frame_body,
//     e.g. T_world_lidar_current is the current lidar pose in the world frame
//     and T_odom_lidar_current is the same pose in the odometry frame.
//   - Point transforms (pointAssociateToMap and friends): move individual
//     lidar points between the sensor frame and the world/map frame.
//   - File I/O (readPointCloud, read/saveLocalizationPose, savePly):
//     load prior maps and save/restore start poses for localization mode.
//   - Small math utilities: PCA of point sets (used for plane/line fitting),
//     degree/radian conversion, timing (ScopedTimer).
//
// Declarations live here; non-template definitions are in
// src/utils/superodom_utils.cpp.
// ============================================================================

# pragma once
#ifndef SUPER_ODOMETRY_LASER_MAPPING_UTILS_H
#define SUPER_ODOMETRY_LASER_MAPPING_UTILS_H

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <fstream>
#include <string>
#include <vector>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include "super_odometry/utils/Twist.h"
#include <queue>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "super_odometry/sensor_data/pointcloud/point_os.h"


namespace super_odometry {
namespace utils {


/// RAII stopwatch: construct at the top of a scope and it logs (at DEBUG
/// level) how many milliseconds the scope took when it is destroyed.
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name, rclcpp::Logger logger = rclcpp::get_logger("ScopedTimer"))
        : name_(name), 
          start_(std::chrono::high_resolution_clock::now()),
          logger_(logger) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        RCLCPP_DEBUG(logger_, "%s took %ld ms", name_.c_str(), duration);
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    rclcpp::Logger logger_;
};


/// One pose sample as read from / written to the start_pose.txt file:
/// position + roll/pitch/yaw plus timing information.
struct OdometryData {
    double timestamp;
    double duration;   // seconds since the first saved pose
    double x, y, z;
    double roll, pitch, yaw;
};

// Shared pose history used by read/saveLocalizationPose (defined in the .cpp).
extern std::vector<OdometryData> odometryResults;

void transformAssociateToMap(Transformd& T_world_lidar_current,
                           const Transformd& T_world_lidar_prev,
                           const Transformd& T_odom_lidar_current,
                           const Transformd& T_odom_lidar_prev);

/// Recomputes the map-to-odometry correction after mapping has refined the
/// current pose: T_map_odom = T_world_lidar * T_odom_lidar^-1, split into a
/// quaternion and a translation.
inline void transformUpdate(const Eigen::Quaterniond &q_world_lidar, const Eigen::Vector3d &t_world_lidar,
                           const Eigen::Quaterniond &q_odom_lidar_current, const Eigen::Vector3d &t_odom_lidar_current,
                           Eigen::Quaterniond &q_map_odom, Eigen::Vector3d &t_map_odom) {
    q_map_odom = q_world_lidar * q_odom_lidar_current.inverse();
    t_map_odom = t_world_lidar - q_map_odom * t_odom_lidar_current;
}


/// Loads a PCD file into cloud_out; returns false if the file is missing or
/// unreadable. Used to load a prior map in localization mode.
bool readPointCloud(const std::string &file_path, pcl::PointCloud<PointType>::Ptr cloud_out);


/// Reads saved poses from "start_pose.txt" (in the directory of file_path)
/// into odometry_results; the first entry serves as the initial pose.
bool readLocalizationPose(const std::string &file_path, std::vector<OdometryData> &odometry_results);


/// Appends the given lidar pose to "start_pose.txt" so a later run can start
/// from it.
bool saveLocalizationPose(double timestamp, const Transformd &T_world_lidar,
                         const std::string &file_path, std::vector<OdometryData> &odometry_results);

// Transform utilities

/// Predicts the current map-frame pose from the previous map-frame pose and
/// the relative motion measured by odometry:
/// T_world_lidar_current = T_world_lidar_prev *
///     (T_odom_lidar_prev^-1 * T_odom_lidar_current).
void transformAssociateToMap(Transformd& T_world_lidar_current,
                           const Transformd& T_world_lidar_prev,
                           const Transformd& T_odom_lidar_current,
                           const Transformd& T_odom_lidar_prev);

/// Quaternion+vector variant: applies the map-to-odometry correction to an
/// odometry-frame pose to get the map-frame pose.
void transformAssociateToMap(Eigen::Quaterniond& q_world_lidar,
                           Eigen::Vector3d& t_world_lidar,
                           const Eigen::Quaterniond& q_map_odom,
                           const Eigen::Quaterniond& q_odom_lidar_current,
                           const Eigen::Vector3d& t_odom_lidar_current,
                           const Eigen::Vector3d& t_map_odom);

/// Same as the inline transformUpdate above but with output arguments first;
/// recomputes the map-to-odometry correction after a mapping update.
void transformUpdate(Eigen::Quaterniond& q_map_odom,
                    Eigen::Vector3d& t_map_odom,
                    const Eigen::Quaterniond& q_world_lidar,
                    const Eigen::Quaterniond& q_odom_lidar_current,
                    const Eigen::Vector3d& t_world_lidar,
                    const Eigen::Vector3d& t_odom_lidar_current);

/// Transforms one point from the sensor frame into the world/map frame:
/// p_out = q_world_lidar * p_in + t_world_lidar.
void pointAssociateToMap(PointType const *const pi, PointType *const po,
                        const Eigen::Quaterniond& q_world_lidar,
                        const Eigen::Vector3d& t_world_lidar);

/// Overload for XYZHSV points (used for the feature clouds that store extra
/// per-point attributes in the h/s/v channels).
void pointAssociateToMap(pcl::PointXYZHSV const *const pi, pcl::PointXYZHSV *const po,
                        const Eigen::Quaterniond& q_world_lidar,
                        const Eigen::Vector3d& t_world_lidar);

/// Inverse of pointAssociateToMap: moves a world/map-frame point back into
/// the sensor frame.
void pointAssociateTobeMapped(PointType const *const pi, PointType *const po,
                            const Eigen::Quaterniond& q_world_lidar,
                            const Eigen::Vector3d& t_world_lidar);

/// Keeps only the roll and pitch of an IMU orientation (yaw set to zero);
/// used to constrain the gravity direction while leaving heading free.
tf2::Quaternion extractRollPitch(Eigen::Quaterniond& imu_rotation);

/// Prints a transform's translation and quaternion to stdout (debugging aid).
void printTransform(const Transformd& T, const std::string& name);

/// Transforms an Ouster point (with ring/time fields) by the given transform,
/// copying x/y/z/intensity into the output point type.
void transformOusterPoints(point_os::OusterPointXYZIRT const *const pi, point_os::PointcloudXYZITR *const po, Transformd &transform);

/// Saves the accumulated cloud to PLY every 10th call (once it has at least
/// 10 points); returns true only when a file was actually written.
bool savePly(pcl::PointCloud<PointType>::Ptr pcl_to_save, rclcpp::Node::SharedPtr node);


template<typename T>
inline constexpr T Deg2Rad(const T &deg) { return deg / 180. * M_PI; }

/// Transforms a PCL point in place: p = transform * p (computed in double
/// precision, stored back as float).
template<typename PointT>
inline void TransformPoint(PointT &p, const Transformd &transform) {
    Eigen::Vector3d temp = p.getVector3fMap().template cast<double>();
    p.getVector3fMap() = (transform * temp).template cast<float>();
}

/// Non-mutating version of TransformPoint: returns the transformed copy.
template<typename PointT>
inline PointT TransformPointd(const PointT &p, const Transformd &transform) {
    PointT out(p);
    TransformPoint(out, transform);
    return out;
}

/// Comparator for std::sort that orders (value, index) pairs by descending value.
inline bool compare_pair_first(const std::pair<float, int> a, const std::pair<float, int> b) // sort from big to small
{
    return a.first > b.first;
}

/*!
* @brief Compute PCA of Nx3 data array and mean value
* @param[in] data Nx3 array (e.g. stacked 3D points)
* @param[out] mean Where to store mean value
* @return The PCA
*/
inline Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
ComputePCA(const Eigen::Matrix<double, Eigen::Dynamic, 3> &data,
            Eigen::Vector3d &mean) {
    mean = data.colwise().mean();
    Eigen::MatrixXd centered = data.rowwise() - mean.transpose();
    Eigen::Matrix3d varianceCovariance = centered.transpose() * centered;

    return Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(varianceCovariance);
}

//------------------------------------------------------------------------------
/*!
* @brief Compute PCA of Nx3 data array and mean value
* @param data Nx3 array (e.g. stacked 3D points)
* @return The PCA
*/
inline Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
ComputePCA(const Eigen::Matrix<double, Eigen::Dynamic, 3> &data) {
    Eigen::Vector3d mean;
    return ComputePCA(data, mean);
}

template<typename T>
inline constexpr T Rad2Deg(const T &rad) { return rad / M_PI * 180.; }

} // namespace utils
} // namespace super_odometry

#endif // SUPER_ODOMETRY_LASER_MAPPING_UTILS_H
