// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// This header declares the global configuration shared by every SuperOdom
// node (feature extraction, laser mapping, IMU preintegration). It contains:
//
//   1. Topic names and TF frame names, loaded from the ROS parameter file by
//      readGlobalparam().
//   2. Sensor extrinsics (rigid transforms between the IMU, the lidar and
//      optional cameras), loaded from the calibration YAML by
//      readCalibration().
//
// Everything here is a plain global variable ('extern' declarations; the
// matching definitions live in src/parameter/parameter.cpp). Each node calls
// readGlobalparam() and readCalibration() once at startup, and afterwards
// all code simply reads these globals.
//
// Naming convention for transforms: T_parent_child maps points from the child
// frame into the parent frame (p_parent = T_parent_child * p_child). For
// example T_imu_lidar maps lidar-frame points into the IMU frame.
// ============================================================================
#pragma once

#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "super_odometry/utils/Twist.h"

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/transform_datatypes.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

/// Supported lidar models. The choice affects how per-point timestamps and
/// ring indices are interpreted by the feature extraction node.
enum class SensorType {VELODYNE, OUSTER, LIVOX};

// ---- ROS topic names (set from the parameter file) -------------------------
extern std::string IMU_TOPIC;        // raw IMU input (~200 Hz)
extern std::string LASER_TOPIC;      // raw lidar point cloud input
extern std::string ODOM_TOPIC;       // odometry output topic
extern std::string DepthUP_TOPIC;    // optional upward-facing depth camera cloud
extern std::string DepthDown_TOPIC;  // optional downward-facing depth camera cloud
extern std::string ProjectName;      // prefix added to all published topic names

// ---- TF frame names ---------------------------------------------------------
extern std::string WORLD_FRAME;      // fixed odometry origin (e.g. "sensor_init")
extern std::string WORLD_FRAME_ROT;  // world frame in the rotated (camera-style) convention
extern std::string SENSOR_FRAME;     // moving body/lidar frame (e.g. "sensor")
extern std::string SENSOR_FRAME_ROT; // sensor frame in the rotated convention
extern SensorType sensor;            // lidar model parsed from the SENSOR string

/// Nonzero if the calibration YAML directly provides T_imu_lidar. If zero,
/// T_imu_lidar is chained through the camera instead:
/// T_imu_lidar = T_imu_camera * T_camera_lidar.
extern int PROVIDE_IMU_LASER_EXTRINSIC;

// Per-camera extrinsics (rotation/translation of each camera w.r.t. the IMU);
// kept for multi-camera setups, unused by the lidar-only pipeline.
extern std::vector<Eigen::Matrix3d> RIC;

extern std::vector<Eigen::Vector3d> TIC;

/// Rotation part of T_imu_lidar, read from the calibration YAML key
/// "extrinsicRotation_imu_laser". Per the YAML convention, it maps
/// lidar-frame vectors into the IMU frame
/// (v_imu = R_imu_lidar * v_lidar). All shipped calibrations set it to
/// identity.
extern Eigen::Matrix3d R_imu_lidar;

/// Translation part of T_imu_lidar: position of the lidar origin expressed
/// in the IMU frame, in meters
/// ("extrinsicTranslation_imu_laser" in the YAML).
extern Eigen::Vector3d t_imu_lidar;

/// Rotation part of T_camera_lidar; maps lidar-frame vectors into the camera
/// frame. Used only when T_imu_lidar is not given directly.
extern Eigen::Matrix3d R_camera_lidar;

extern Eigen::Vector3d t_camera_lidar;

/// Rotation part of T_imu_camera; maps camera-frame vectors into the IMU
/// frame. Used only when T_imu_lidar is not given directly.
extern Eigen::Matrix3d R_imu_camera;

extern Eigen::Vector3d t_imu_camera;

/// Extra hand-tuned roll/pitch/yaw correction in degrees
/// ("imu_laser_rotation_offset" in the YAML), applied on top of
/// R_imu_lidar inside readCalibration().
extern Eigen::Vector3d rpy_imu_lidar_offset_deg;

/// Camera <- lidar transform built from R_camera_lidar / t_camera_lidar.
extern Transformd T_camera_lidar;

/// IMU <- camera transform built from R_imu_camera / t_imu_camera.
extern Transformd T_imu_camera;

/// IMU <- lidar transform: maps lidar-frame points into the IMU frame
/// (p_imu = T_imu_lidar * p_lidar). Built as
/// Transformd(R_imu_lidar, t_imu_lidar),
/// or chained through the camera when the direct extrinsic is unavailable.
extern Transformd T_imu_lidar;

/// Lidar <- IMU transform, the inverse of T_imu_lidar.
extern Transformd T_lidar_imu;

// ---- Optional RealSense depth camera extrinsics -----------------------------
// Pose of the upward-facing RealSense camera w.r.t. the lidar, as roll/
// pitch/yaw angles plus x/y/z translation in meters. Read from the
// calibration YAML; each value defaults to 0 when the YAML has no entry.
extern float up_realsense_roll;

extern float up_realsense_pitch;

extern float up_realsense_yaw;

extern float up_realsense_x;

extern float up_realsense_y;

extern float up_realsense_z;

// Same for the downward-facing RealSense camera.
extern float down_realsense_roll;

extern float down_realsense_pitch;

extern float down_realsense_yaw;

extern float down_realsense_x;

extern float down_realsense_y;

extern float down_realsense_z;

/// Yaw drift correction in degrees per meter traveled, applied by the lidar
/// SLAM optimization (calibration YAML entry; 0 in all shipped files).
extern float yaw_ratio;

// Sanity limits on the accelerometer reading (m/s^2) used by the IMU health
// checks; a stationary IMU exceeding these is considered faulty.
extern float IMU_ACC_X_LIMIT;

extern float IMU_ACC_Y_LIMIT;

extern float IMU_ACC_Z_LIMIT;

/// If true, take roll/pitch from the IMU driver's internal attitude filter
/// instead of the optimized estimate.
extern bool USE_IMU_ROLL_PITCH;

/// If true, laser mapping writes the accumulated map to a .ply file.
extern bool SAVE_PLY;

/// Lidar model as a lowercase string ("velodyne", "ouster" or "livox");
/// parsed into the 'sensor' enum by readGlobalparam().
extern std::string SENSOR; 

/// Fixed T_sensor_ouster transform mapping points from the Ouster internal
/// lidar frame into its "sensor" parent frame. The lidar frame is rotated
/// 180 degrees about z and offset 36.18 mm along z. Hardcoded in
/// readCalibration().
extern Transformd T_sensor_ouster;

extern Eigen::Matrix3d R_sensor_ouster;

extern Eigen::Vector3d t_sensor_ouster;

/// Loads topic names, frame names and generic settings from the ROS
/// parameter file into the globals above. Returns false on an unsupported
/// sensor type. Call once at node startup.
bool readGlobalparam(rclcpp::Node::SharedPtr);

/// Loads the sensor extrinsics from the calibration YAML (path given by the
/// "calibration_file" ROS parameter) and builds T_imu_lidar / T_lidar_imu.
/// Returns false if the file cannot be opened. Call once at node startup.
bool readCalibration(rclcpp::Node::SharedPtr);

