//
// Created by shibo zhao on 2020-09-27.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// This file defines the global configuration variables declared in
// config/parameter.h and implements the two functions that fill them in:
//
//   - readGlobalparam(): reads topic names, TF frame names and generic
//     settings from the ROS parameter file.
//   - readCalibration(): reads the sensor extrinsics from the calibration
//     YAML (an OpenCV FileStorage file whose path comes from the
//     "calibration_file" ROS parameter) and builds the IMU<->lidar
//     transforms T_imu_lidar / T_lidar_imu used throughout the pipeline.
//
// Every SuperOdom node calls both functions once at startup; afterwards the
// globals are treated as read-only.
// ============================================================================
#include "super_odometry/config/parameter.h"

// Define color escape codes for ~beautification~
#define RESET "\033[0m"
#define BLACK "\033[30m"   /* Black */
#define RED "\033[31m"     /* Red */
#define GREEN "\033[32m"   /* Green */
#define YELLOW "\033[33m"  /* Yellow */
#define BLUE "\033[34m"    /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m"    /* Cyan */
#define WHITE "\033[37m"   /* White */

#define BOLD "\033[1m"
#define UNDERLINE "\033[4m"
#define ITALIC "\033[3m"

std::string IMU_TOPIC;
std::string LASER_TOPIC;
std::string ODOM_TOPIC;
std::string DepthUP_TOPIC;
std::string DepthDown_TOPIC;
std::string ProjectName;

std::string WORLD_FRAME;
std::string WORLD_FRAME_ROT;
std::string SENSOR_FRAME;
std::string SENSOR_FRAME_ROT;
SensorType sensor;

int PROVIDE_IMU_LASER_EXTRINSIC;

Eigen::Matrix3d R_imu_lidar;

Eigen::Vector3d t_imu_lidar;

Eigen::Vector3d rpy_imu_lidar_offset_deg;

Eigen::Matrix3d R_camera_lidar;

Eigen::Vector3d t_camera_lidar;

Eigen::Matrix3d R_imu_camera;

Eigen::Vector3d t_imu_camera;

Transformd T_camera_lidar;

Transformd T_imu_camera;

Transformd T_imu_lidar;

Transformd T_lidar_imu;

Transformd T_sensor_ouster;

Eigen::Matrix3d R_sensor_ouster;

Eigen::Vector3d t_sensor_ouster;

float lidar_imu_offset_roll;

float up_realsense_roll;

float up_realsense_pitch;

float up_realsense_yaw;

float up_realsense_x;

float up_realsense_y;

float up_realsense_z;

float down_realsense_roll;

float down_realsense_pitch;

float down_realsense_yaw;

float down_realsense_x;

float down_realsense_y;

float down_realsense_z;

float yaw_ratio;

float IMU_ACC_X_LIMIT;

float IMU_ACC_Y_LIMIT;

float IMU_ACC_Z_LIMIT;

bool USE_IMU_ROLL_PITCH;

bool SAVE_PLY;

std::string SENSOR;


/// Reads a single ROS parameter that must already be declared; shuts the
/// node down if the parameter is missing (all parameters here are required).
template <typename T>
T readParam(rclcpp::Node::SharedPtr node, std::string name)
{
    T ans;
    // node->declare_parameter<T>(name);
    if (node->get_parameter(name, ans)) {
        RCLCPP_INFO(node->get_logger(),  "Loaded %s: ", name.c_str());
    }
    else {
        RCLCPP_ERROR(node->get_logger(), "Failed to load %s", name.c_str());
        rclcpp::shutdown();
    }
    return ans;
}

/// Loads sensor extrinsics from the calibration YAML into the globals and
/// builds T_imu_lidar and its inverse T_lidar_imu. Two paths exist:
///  - PROVIDE_IMU_LASER_EXTRINSIC set: read extrinsicRotation_imu_laser /
///    extrinsicTranslation_imu_laser directly as T_imu_lidar, mapping
///    lidar-frame points into the IMU frame, and apply the optional
///    roll/pitch/yaw offset from imu_laser_rotation_offset.
///  - otherwise: chain the transform through the camera,
///    T_imu_lidar = T_imu_camera * T_camera_lidar.
bool readCalibration(rclcpp::Node::SharedPtr node)
{
    RCLCPP_INFO(node->get_logger(), "[super_odometry] read parameter");
    std::string calib_file;
    // calib_file = readParam<std::string>(node, "calib_file");
    calib_file = node->declare_parameter("calibration_file", std::string(""));
    RCLCPP_INFO(node->get_logger(), "[super_odometry] calib_file: %s", calib_file.c_str());
    cv::FileStorage fsSettings(calib_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
        return false;
    }
    PROVIDE_IMU_LASER_EXTRINSIC = node->declare_parameter("provide_imu_laser_extrinsic", true);
    RCLCPP_INFO(node->get_logger(), "PROVIDE_IMU_LASER_EXTRINSIC: %d", PROVIDE_IMU_LASER_EXTRINSIC);

    // Defaults to 0 if no entry
    up_realsense_roll = fsSettings["up_realsense_roll"];
    up_realsense_pitch = fsSettings["up_realsense_pitch"];
    up_realsense_yaw = fsSettings["up_realsense_yaw"];
    up_realsense_x = fsSettings["up_realsense_x"];
    up_realsense_y = fsSettings["up_realsense_y"];
    up_realsense_z = fsSettings["up_realsense_z"];

    down_realsense_roll = fsSettings["down_realsense_roll"];
    down_realsense_pitch = fsSettings["down_realsense_pitch"];
    down_realsense_yaw = fsSettings["down_realsense_yaw"];
    down_realsense_x = fsSettings["down_realsense_x"];
    down_realsense_y = fsSettings["down_realsense_y"];
    down_realsense_z = fsSettings["down_realsense_z"];
    
    yaw_ratio=fsSettings["yaw_ratio"];

    RCLCPP_INFO(node->get_logger(), "up realsense extrinsic to velodyne (RPYXYZ): %f, %f, %f, %f, %f, %f",
                up_realsense_roll,
                up_realsense_pitch,
                up_realsense_yaw,
                up_realsense_x,
                up_realsense_y,
                up_realsense_z);

    RCLCPP_INFO(node->get_logger(), "down realsense extrinsic to velodyne (RPYXYZ): %f, %f, %f, %f, %f, %f",
                down_realsense_roll,
                down_realsense_pitch,
                down_realsense_yaw,
                down_realsense_x,
                down_realsense_y,
                down_realsense_z);

    RCLCPP_INFO(node->get_logger(), "yaw ratio: %f", yaw_ratio);
    
    if (PROVIDE_IMU_LASER_EXTRINSIC)
    {
        cv::Mat cv_R_imu_lidar, cv_t_imu_lidar;

        cv::Mat cv_rpy_imu_lidar_offset_deg;
        fsSettings["imu_laser_rotation_offset"] >> cv_rpy_imu_lidar_offset_deg;
        fsSettings["extrinsicRotation_imu_laser"] >> cv_R_imu_lidar;
        fsSettings["extrinsicTranslation_imu_laser"] >> cv_t_imu_lidar;
        cv::cv2eigen(cv_R_imu_lidar, R_imu_lidar);
        cv::cv2eigen(cv_t_imu_lidar, t_imu_lidar);
        cv::cv2eigen(cv_rpy_imu_lidar_offset_deg, rpy_imu_lidar_offset_deg);
        // RCLCPP_INFO(node->get_logger(),  "\n R_imu_lidar: \n"
        //           << R_imu_lidar;
        // RCLCPP_INFO(node->get_logger(),  "\n t_imu_lidar: \n"
        //           << t_imu_lidar.transpose();
        
        // RCLCPP_INFO(node->get_logger(),  "\n rpy_imu_lidar_offset_deg: \n" << rpy_imu_lidar_offset_deg.transpose();
        
        // RCLCPP_INFO(node->get_logger(), BLUE <<"\n Before applying offset to R_imu_lidar: \n"<<RESET
        //           << R_imu_lidar;
        // RCLCPP_INFO(node->get_logger(), "\n Before applying offset to R_imu_lidar: \n"
        //           << R_imu_lidar;

        // Build the IMU <- lidar transform from the raw YAML values:
        // p_imu = T_imu_lidar * p_lidar. T_lidar_imu is its inverse.
        T_imu_lidar = Transformd(R_imu_lidar, t_imu_lidar);
        T_lidar_imu = T_imu_lidar.inverse();
        
        // Log the roll/pitch/yaw of the raw extrinsic (degrees) so a bad
        // calibration file is easy to spot in the console.
        double roll, pitch, yaw;
        tf2::Quaternion q_imu_lidar_raw(T_imu_lidar.rot.x(), T_imu_lidar.rot.y(),
                                       T_imu_lidar.rot.z(), T_imu_lidar.rot.w());
        tf2::Matrix3x3(q_imu_lidar_raw).getRPY(roll, pitch, yaw);
        RCLCPP_INFO(node->get_logger(), BLUE"\n previous roll: %f previous pitch: %f previous yaw: %f" RESET, roll *180/M_PI, pitch *180/M_PI, yaw *180/M_PI); 

        // Apply the optional hand-tuned correction:
        // rpy_imu_lidar_offset_deg is converted to radians and composed on
        // the left of the YAML rotation.
        tf2::Quaternion q_imu_lidar_offset;
        q_imu_lidar_offset.setRPY(rpy_imu_lidar_offset_deg[0] * M_PI / 180,
                                  rpy_imu_lidar_offset_deg[1] * M_PI / 180,
                                  rpy_imu_lidar_offset_deg[2] * M_PI / 180);

        tf2::Quaternion q_imu_lidar_before_offset(
            T_imu_lidar.rot.x(), T_imu_lidar.rot.y(), T_imu_lidar.rot.z(),
            T_imu_lidar.rot.w());
        tf2::Quaternion q_imu_lidar_corrected_tf2 =
            q_imu_lidar_offset * q_imu_lidar_before_offset;
        Eigen::Quaterniond q_imu_lidar;
        q_imu_lidar = Eigen::Quaterniond(
            q_imu_lidar_corrected_tf2.w(), q_imu_lidar_corrected_tf2.x(),
            q_imu_lidar_corrected_tf2.y(), q_imu_lidar_corrected_tf2.z());
         
        // Store the corrected rotation back into the globals so that every
        // consumer (including R_imu_lidar itself) sees the offset applied.
        T_imu_lidar.rot=q_imu_lidar;
        T_lidar_imu = T_imu_lidar.inverse();
        R_imu_lidar=T_imu_lidar.rot.toRotationMatrix();
        
        RCLCPP_INFO_STREAM(node->get_logger(),  GREEN BOLD "T_imu_lidar extrinsic:\n" << T_imu_lidar.matrix());
        RCLCPP_INFO_STREAM(node->get_logger(),  GREEN BOLD "T_lidar_imu extrinsic:\n" << T_lidar_imu.matrix());

        // Log the roll/pitch/yaw after the offset (degrees) for comparison.
        double updated_roll, updated_pitch, updated_yaw;
        tf2::Quaternion q_imu_lidar_corrected(
            q_imu_lidar_corrected_tf2.x(), q_imu_lidar_corrected_tf2.y(),
            q_imu_lidar_corrected_tf2.z(), q_imu_lidar_corrected_tf2.w());
        tf2::Matrix3x3(q_imu_lidar_corrected).getRPY(updated_roll, updated_pitch, updated_yaw);
        
        RCLCPP_INFO(node->get_logger(), GREEN BOLD"\n updated roll: %f updated pitch: %f updated yaw: %f" RESET, updated_roll*180/M_PI, updated_pitch *180/M_PI, updated_yaw*180/M_PI); 

        // RCLCPP_INFO(node->get_logger(), "\n After applying offset to R_imu_lidar: \n"
        //           << R_imu_lidar;
        // RCLCPP_INFO(node->get_logger(), "\n After applying offset to T_imu_lidar: \n"
        //           << T_imu_lidar;
    }
    else
    {
        // No direct T_imu_lidar calibration: read T_camera_lidar and
        // T_imu_camera, both mapping child-frame points into their parent,
        // and compose
        // T_imu_lidar = T_imu_camera * T_camera_lidar.
        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation_camera_laser"] >> cv_R;
        fsSettings["extrinsicTranslation_camera_laser"] >> cv_T;
        cv::cv2eigen(cv_R, R_camera_lidar);
        cv::cv2eigen(cv_T, t_camera_lidar);

        T_camera_lidar = Transformd(R_camera_lidar, t_camera_lidar);

        // RCLCPP_INFO(node->get_logger(),  "\n R_camera_lidar: \n"
        //           << R_camera_lidar;
        // RCLCPP_INFO(node->get_logger(),  "\n t_camera_lidar: \n"
        //           << t_camera_lidar.transpose();
        // RCLCPP_INFO(node->get_logger(),  "\n T_camera_lidar: \n"
        //           << T_camera_lidar;

        fsSettings["extrinsicRotation_imu_camera"] >> cv_R;
        fsSettings["extrinsicTranslation_imu_camera"] >> cv_T;

        cv::cv2eigen(cv_R, R_imu_camera);
        cv::cv2eigen(cv_T, t_imu_camera);
        // Round-trip through a quaternion to clean up numerical errors and
        // guarantee the matrix is a proper rotation.
        Eigen::Quaterniond q_imu_camera(R_imu_camera);
        R_imu_camera = q_imu_camera.normalized();

        T_imu_camera = Transformd(R_imu_camera, t_imu_camera);

        T_imu_lidar = T_imu_camera * T_camera_lidar;
        T_lidar_imu = T_imu_lidar.inverse();

        // RCLCPP_INFO(node->get_logger(),  "R_imu_camera: \n"
        //           << R_imu_camera;
        // RCLCPP_INFO(node->get_logger(),  "t_imu_camera: \n"
        //           << t_imu_camera;

        RCLCPP_INFO_STREAM(node->get_logger(),  GREEN BOLD "T_imu_lidar extrinsic:\n" << T_imu_lidar.matrix());
        RCLCPP_INFO_STREAM(node->get_logger(),  GREEN BOLD "T_lidar_imu extrinsic:\n" << T_lidar_imu.matrix());
    }

    // Fixed transform between the Ouster "sensor" frame and its internal
    // lidar frame (from the Ouster datasheet): the lidar frame is rotated
    // 180 degrees about z and sits 36.18 mm above the sensor origin.
    R_sensor_ouster << -1, 0,  0,
                       0, -1, 0,
                       0,  0,  1;

    t_sensor_ouster << 0, 0, 0.036180;

    T_sensor_ouster = Transformd(R_sensor_ouster, t_sensor_ouster);

    return true;
}

/// Loads topic names, TF frame names and generic settings from the ROS
/// parameter file. Every parameter has a default, so the node still starts
/// with an incomplete configuration; only an unsupported sensor string makes
/// this function fail.
bool readGlobalparam(rclcpp::Node::SharedPtr node)
{
    node->declare_parameter<std::string>("imu_topic","imu/data");
    node->declare_parameter<std::string>("laser_topic","velodyne_points");
    node->declare_parameter<std::string>("odom_topic","integrated_to_init");
    node->declare_parameter<std::string>("depthup_topic","/rs_up/depth/cloud_filtered");
    node->declare_parameter<std::string>("depthdown_topic","/rs_down/depth/cloud_filtered");
    node->declare_parameter<std::string>("world_frame", "sensor_init");
    node->declare_parameter<std::string>("world_frame_rot", "sensor_init_rot");
    node->declare_parameter<std::string>("sensor_frame", "sensor");
    node->declare_parameter<std::string>("sensor_frame_rot", "sensor_rot");
    node->declare_parameter<std::string>("PROJECT_NAME", "");
    node->declare_parameter<std::string>("sensor", "livox");
    node->declare_parameter<double>("imu_acc_x_limit", 0.5);
    node->declare_parameter<double>("imu_acc_y_limit", 0.2);
    node->declare_parameter<double>("imu_acc_z_limit", 0.4);
    node->declare_parameter<bool>("save_ply", false);
    // node->declare_parameter<bool>("use_imu_roll_pitch", false);

    
    LASER_TOPIC = node->get_parameter("laser_topic").as_string();
    IMU_TOPIC = node->get_parameter("imu_topic").as_string();
    ODOM_TOPIC = node->get_parameter("odom_topic").as_string();
    DepthUP_TOPIC = node->get_parameter("depthup_topic").as_string();
    DepthDown_TOPIC = node->get_parameter("depthdown_topic").as_string();
    WORLD_FRAME = node->get_parameter("world_frame").as_string();
    WORLD_FRAME_ROT = node->get_parameter("world_frame_rot").as_string();
    SENSOR_FRAME = node->get_parameter("sensor_frame").as_string();
    SENSOR_FRAME_ROT = node->get_parameter("sensor_frame_rot").as_string();
    ProjectName = node->get_parameter("PROJECT_NAME").as_string();
    SENSOR = node->get_parameter("sensor").as_string();
    // USE_IMU_ROLL_PITCH = node->get_parameter("use_imu_roll_pitch").as_bool();
    SAVE_PLY = node->get_parameter("save_ply").as_bool();
    IMU_ACC_X_LIMIT = node->get_parameter("imu_acc_x_limit").as_double();
    IMU_ACC_Y_LIMIT = node->get_parameter("imu_acc_y_limit").as_double();
    IMU_ACC_Z_LIMIT = node->get_parameter("imu_acc_z_limit").as_double();
    //check whether sensor is support 
    const std::unordered_map<std::string, SensorType> sensorTypeMap = {
        {"velodyne", SensorType::VELODYNE},
        {"ouster", SensorType::OUSTER},
        {"livox", SensorType::LIVOX}
    };

    if (sensorTypeMap.find(SENSOR) == sensorTypeMap.end()) {
        RCLCPP_ERROR(node->get_logger(), "Unsupported sensor type: %s", SENSOR.c_str());
        return false;
    }
    
    RCLCPP_INFO(node->get_logger(), "LASER_TOPIC %s", LASER_TOPIC.c_str());
    RCLCPP_INFO(node->get_logger(), "IMU_TOPIC %s", IMU_TOPIC.c_str());
    RCLCPP_INFO(node->get_logger(), "ODOM_TOPIC %s", ODOM_TOPIC.c_str());
    RCLCPP_INFO(node->get_logger(), "DepthUP_TOPIC %s", DepthUP_TOPIC.c_str());
    RCLCPP_INFO(node->get_logger(), "DepthDown_TOPIC %s", DepthDown_TOPIC.c_str());
    RCLCPP_INFO(node->get_logger(), "WORLD_FRAME %s", WORLD_FRAME.c_str());
    RCLCPP_INFO(node->get_logger(), "WORLD_FRAME_ROT %s", WORLD_FRAME_ROT.c_str());
    RCLCPP_INFO(node->get_logger(), "SENSOR_FRAME %s", SENSOR_FRAME.c_str());
    RCLCPP_INFO(node->get_logger(), "SENSOR_FRAME_ROT %s", SENSOR_FRAME_ROT.c_str());
    RCLCPP_INFO(node->get_logger(), "ProjectName %s", ProjectName.c_str());
    RCLCPP_INFO(node->get_logger(), "SENSOR %s", SENSOR.c_str());
    RCLCPP_INFO(node->get_logger(), "SAVE_PLY %d", SAVE_PLY);
    RCLCPP_INFO(node->get_logger(), "SAVE_PLY %d", SAVE_PLY);

    return true;
}