// Created by Shibo on 2025-03-29.
//
// ============================================================================
// OVERVIEW
// ============================================================================
// Implementations for the helpers declared in
// super_odometry/utils/superodom_utils.h: pose/point transforms between the
// odometry and map frames, and file I/O for prior maps and start poses.
// See the header for per-function documentation and for the T_a_b frame
// naming convention.
// ============================================================================

#include "super_odometry/utils/superodom_utils.h"
#include <iostream>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <rclcpp/rclcpp.hpp>
#include <pcl/io/ply_io.h>
#include <filesystem>

namespace super_odometry {
namespace utils {

// Pose history shared by read/saveLocalizationPose (declared in the header).
std::vector<OdometryData> odometryResults;


// Loads a PCD file into cloud_out, checking first that the file exists.
bool readPointCloud(const std::string &file_path, pcl::PointCloud<PointType>::Ptr cloud_out) {
    std::ifstream file_check(file_path.c_str());
    if (!file_check.good()) {
        std::cerr << "Error: File does not exist: " << file_path << std::endl;
        return false;
    }
    file_check.close();
    
    pcl::PCDReader reader;
    int result = reader.read(file_path, *cloud_out);
    
    if (result < 0) {
        std::cerr << "Error reading PCD file: " << file_path << std::endl;
        return false;
    }
    
    return true;
}

// Reads "start_pose.txt" (one pose per line: duration x y z roll pitch yaw)
// from the directory of file_path; the first pose is used to initialize
// localization mode.
bool readLocalizationPose(const std::string &file_path, std::vector<OdometryData> &odometry_results) {
    std::string localizationPosePath = file_path;
    
    // Replace the file name in file_path with "start_pose.txt"
    size_t lastSlashPos = file_path.find_last_of('/');
    if (lastSlashPos != std::string::npos) {
        std::string directory = file_path.substr(0, lastSlashPos + 1);
        localizationPosePath = directory + "start_pose.txt";
    }
    
    std::ifstream file(localizationPosePath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << localizationPosePath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        
        std::istringstream iss(line);
        OdometryData odom;
        if (iss >> odom.duration >> odom.x >> odom.y >> odom.z >> odom.roll >> odom.pitch >> odom.yaw) {
            std::cout << "Read odometry data: " << odom.x << " " << odom.y << " " << odom.z << std::endl;
            odometry_results.push_back(odom);
        } else {
            std::cerr << "Error reading line: " << line << std::endl;
        }
    }
    
    if (!odometry_results.empty()) {
        std::cout << "\033[1;32m Loaded the localization_pose.txt successfully \033[0m" 
                  << odometry_results[0].x << " " << odometry_results[0].y << " " 
                  << odometry_results[0].z << std::endl;
    }
    
    file.close();
    return !odometry_results.empty();
}

// Converts the lidar pose to x/y/z + roll/pitch/yaw and appends it to
// "start_pose.txt" next to file_path, so a later run can resume from it.
bool saveLocalizationPose(double timestamp, const Transformd &T_world_lidar,
                         const std::string &file_path, std::vector<OdometryData> &odometry_results) {
    std::string saveOdomPath;
    size_t lastSlashPos = file_path.find_last_of('/');
    if (lastSlashPos != std::string::npos) {
        saveOdomPath = file_path.substr(0, lastSlashPos + 1); // Include the trailing slash
    } else {
        saveOdomPath = "./";
    }

    OdometryData odom;
    {
        odom.timestamp = timestamp;
        odom.x = T_world_lidar.pos.x();
        odom.y = T_world_lidar.pos.y();
        odom.z = T_world_lidar.pos.z();
        tf2::Quaternion q_world_lidar(
            T_world_lidar.rot.x(), T_world_lidar.rot.y(),
            T_world_lidar.rot.z(), T_world_lidar.rot.w());
        tf2::Matrix3x3(q_world_lidar).getRPY(
            odom.roll, odom.pitch, odom.yaw);
    }

    odometry_results.push_back(odom);
    
    std::string OdomResultPath = saveOdomPath + "start_pose.txt";
    std::ofstream outFile(OdomResultPath, std::ios::app);
    
    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << OdomResultPath << std::endl;
        return false;
    }

    outFile << std::fixed << (odometry_results.size() > 1 ? (odom.timestamp - odometry_results[0].timestamp) : 0.0) << " "
            << odom.x << " " << odom.y << " " << odom.z << " "
            << odom.roll << " " << odom.pitch << " " << odom.yaw << std::endl;

    outFile.close();
    return true;
}

void transformAssociateToMap(Transformd& T_world_lidar_current,
                           const Transformd& T_world_lidar_prev,
                           const Transformd& T_odom_lidar_current,
                           const Transformd& T_odom_lidar_prev) {
    // Calculate relative transform between previous and current odometry
    Transformd T_lidar_prev_lidar_current = T_odom_lidar_prev.inverse() * T_odom_lidar_current;
    
    // Apply the relative transform to the previous world pose
    T_world_lidar_current = T_world_lidar_prev * T_lidar_prev_lidar_current;
}

void transformAssociateToMap(Eigen::Quaterniond& q_world_lidar,
                           Eigen::Vector3d& t_world_lidar,
                           const Eigen::Quaterniond& q_map_odom,
                           const Eigen::Quaterniond& q_odom_lidar_current,
                           const Eigen::Vector3d& t_odom_lidar_current,
                           const Eigen::Vector3d& t_map_odom) {
    // Transform from odometry frame to world frame
    q_world_lidar = q_map_odom * q_odom_lidar_current;
    t_world_lidar = q_map_odom * t_odom_lidar_current + t_map_odom;
}

void transformUpdate(Eigen::Quaterniond& q_map_odom,
                    Eigen::Vector3d& t_map_odom,
                    const Eigen::Quaterniond& q_world_lidar,
                    const Eigen::Quaterniond& q_odom_lidar_current,
                    const Eigen::Vector3d& t_world_lidar,
                    const Eigen::Vector3d& t_odom_lidar_current) {
    // Update the transform between world and odometry frames
    q_map_odom = q_world_lidar * q_odom_lidar_current.inverse();
    t_map_odom = t_world_lidar - q_map_odom * t_odom_lidar_current;
}

void pointAssociateToMap(PointType const *const pi, PointType *const po,
                        const Eigen::Quaterniond& q_world_lidar,
                        const Eigen::Vector3d& t_world_lidar) {
    // Transform point from the lidar frame to the world frame
    Eigen::Vector3d point_lidar(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_world = q_world_lidar * point_lidar + t_world_lidar;
    po->x = point_world.x();
    po->y = point_world.y();
    po->z = point_world.z();
    po->intensity = pi->intensity;
}

void pointAssociateToMap(pcl::PointXYZHSV const *const pi, pcl::PointXYZHSV *const po,
                        const Eigen::Quaterniond& q_world_lidar,
                        const Eigen::Vector3d& t_world_lidar) {
    // Transform HSV point from the lidar frame to the world frame
    Eigen::Vector3d point_lidar(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_world = q_world_lidar * point_lidar + t_world_lidar;
    po->x = point_world.x();
    po->y = point_world.y();
    po->z = point_world.z();
    po->h = pi->h;
    po->s = pi->s;
    po->v = pi->v;
}

void pointAssociateTobeMapped(PointType const *const pi, PointType *const po,
                            const Eigen::Quaterniond& q_world_lidar,
                            const Eigen::Vector3d& t_world_lidar) {
    // Transform point from the world frame to the lidar frame
    Eigen::Vector3d point_world(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_lidar = q_world_lidar.inverse() * (point_world - t_world_lidar);
    po->x = point_lidar.x();
    po->y = point_lidar.y();
    po->z = point_lidar.z();
    po->intensity = pi->intensity;
}


// Builds a quaternion that keeps only the roll and pitch of the given IMU
// orientation (yaw forced to zero). Used to inject the gravity direction
// into ICP without constraining the heading.
tf2::Quaternion extractRollPitch(Eigen::Quaterniond& imu_rotation){
    double imu_roll, imu_pitch, imu_yaw;
    tf2::Quaternion orientation(imu_rotation.x(), imu_rotation.y(), imu_rotation.z(), imu_rotation.w());
    tf2::Matrix3x3(orientation).getRPY(imu_roll, imu_pitch, imu_yaw);
    tf2::Quaternion quat ;
    quat.setRPY(imu_roll,imu_pitch, 0.0);
    RCLCPP_INFO(rclcpp::get_logger("super_odometry"), "Using IMU Roll Pitch in ICP: %f %f %f", imu_roll, imu_pitch, imu_yaw);
    return quat;
}

void printTransform(const Transformd& T, const std::string& name){
    std::cout<<name<<": "<<T.pos.transpose()<<std::endl;
    std::cout<<name<<": "<<T.rot<<std::endl;
}

// Applies a rigid transform to an Ouster point: p_out = R * p_in + t.
void transformOusterPoints(point_os::OusterPointXYZIRT const *const pi, point_os::PointcloudXYZITR *const po, Transformd &transform) {
    Eigen::Vector3d point_input(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_transformed = transform.rot * point_input + transform.pos;
    po->x = point_transformed.x();
    po->y = point_transformed.y();
    po->z = point_transformed.z();
    po->intensity = pi->intensity;
}

// Writes the accumulated cloud to <ROOT_DIR>/PLY/saved_scans.ply, but only
// every 10th call (when the point count is a multiple of 10) to limit disk I/O.
bool savePly(pcl::PointCloud<PointType>::Ptr pcl_to_save, rclcpp::Node::SharedPtr node) {
    if (pcl_to_save->size() >= 10 && pcl_to_save->size() % 10 == 0) {
        std::string las_dir = std::string(ROOT_DIR) + "PLY";
        std::filesystem::create_directories(las_dir);
        std::string all_points_dir = las_dir + "/saved_scans.ply";
        if (pcl::io::savePLYFileBinary(all_points_dir, *pcl_to_save) == 0) {
            RCLCPP_INFO(node->get_logger(), "All scans saved to %s with %zu points", all_points_dir.c_str(), pcl_to_save->size());
            return true;
        } else {
            RCLCPP_ERROR(node->get_logger(), "Failed to save scans to %s", all_points_dir.c_str());
            return false;
        }
    }
    return false;
}


} // namespace utils
} // namespace super_odometry
