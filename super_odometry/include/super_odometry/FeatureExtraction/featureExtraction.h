//
// Created by shibo zhao on 2020-09-27.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// This node is the FRONT END of the lidar-inertial odometry pipeline. It is
// the first thing that touches raw sensor data, and it prepares each lidar
// scan so that the laserMapping node can register it against the map.
//
// It consumes three streams:
//   1. Raw lidar scans (~10 Hz) from a Velodyne, Ouster or Livox sensor.
//   2. IMU messages (~200 Hz): gyro + accelerometer.
//   3. (Optional) an external "visual/VIO" odometry stream.
//
// and performs two jobs per scan:
//
//   a) UNDISTORTION (also called "deskewing" or "motion compensation").
//      A spinning lidar does not capture all points at once: a 10 Hz scan
//      takes ~100 ms, and the robot moves during that time. Each point is
//      therefore expressed in the sensor pose at ITS OWN capture instant,
//      not in a single common frame. Undistortion uses the IMU rotation (or
//      the VIO pose, when available) interpolated at each point's timestamp
//      to move every point into the sensor frame at the START of the scan,
//      as if the whole cloud had been captured instantaneously.
//
//   b) FEATURE EXTRACTION. The full cloud is too dense to register in real
//      time, so a subset of geometrically useful points is selected. This
//      implementation keeps a uniformly downsampled set of "surface" points
//      (classic LOAM would additionally pick high-curvature "edge" points;
//      here the edge cloud is published but left empty).
//
// The result is bundled into one LaserFeature message (undistorted cloud +
// feature clouds + the IMU/VIO attitude at scan start) and published for the
// laserMapping node. laserMapping estimates the lidar pose, and the
// imuPreintegration node fuses that pose with the IMU in a factor graph.
// ============================================================================

#ifndef super_odometry_FEATUREEXTRACTION_H
#define super_odometry_FEATUREEXTRACTION_H

// #include "super_odometry/logging.h"


#include <cmath>
#include <string>
#include <vector>
#include <sophus/so3.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/console/print.h>

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <super_odometry_msgs/msg/laser_feature.hpp>

#include "super_odometry/container/MapRingBuffer.h"
#include "super_odometry/sensor_data/imu/imu_data.h"
#include "super_odometry/sensor_data/pointcloud/point_os.h"
#include "super_odometry/tic_toc.h"
#include "super_odometry/utils/Twist.h"
#include "super_odometry/config/parameter.h"

#include <mutex>

#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "super_odometry/utils/superodom_utils.h"


namespace super_odometry {


    using std::atan2;
    using std::cos;
    using std::sin;
    std::vector<std::queue<sensor_msgs::msg::PointCloud2::SharedPtr>> all_cloud_buf(2);

    // Timing constants derived from the Velodyne data sheet. A Velodyne packet
    // consists of 12 firing "blocks"; each block takes 55.296 microseconds
    // (firing all lasers + recharge). LIDAR_MESSAGE_TIME approximates the
    // duration of one full scan message (12 blocks * 151 packets), in seconds.
    constexpr unsigned int BLOCK_TIME_NS = 55296;   // Time in ns for one block (measurement + recharge)
    constexpr std::size_t NUM_BLOCKS = 12;    // Number of blocks in a Velodyne packet
    constexpr double LIDAR_MESSAGE_TIME = (double)(NUM_BLOCKS * BLOCK_TIME_NS * 151) * 1e-9;
    // Allowed fractional jitter of the IMU period before a timestamp gap is
    // treated as a dropout (see calculateDeltaTime).
    constexpr double IMU_TIME_LENIENCY = 0.1;


    // A "blind" box around the robot: points inside it (usually returns from
    // the robot's own body) are ignored. Distances in meters, sensor frame.
    struct bounds_t
    {
        double blindFront;
        double blindBack;
        double blindRight;
        double blindLeft;
    };

    // Tuning knobs, loaded from the ROS parameter file in readParameters().
    struct feature_extraction_config{
        bounds_t box_size;          // self-hit rejection box (see bounds_t)
        int skipFrame;              // process every skipFrame-th scan (1 = all)
        int N_SCANS;                // number of laser rings (16/32/64/128, 4 for Livox Mid-360)
        int provide_point_time;     // 1 if the driver stamps each point; 0 = reconstruct times here
        bool use_dynamic_mask;
        bool use_imu_roll_pitch;    // seed the first orientation with IMU roll/pitch (yaw removed)
        bool debug_view_enabled;
        float min_range;            // discard points closer than this (m)
        float max_range;            // discard points farther than this (m)
        int filter_point_size;      // downsample stride for surface features (keep 1 of N)
        SensorType sensor;          // VELODYNE / OUSTER / LIVOX
        double imu_acc_x_limit;     // sanity limits on measured acceleration (m/s^2)
        double imu_acc_y_limit;
        double imu_acc_z_limit;
    };

    // A raw IMU sample unpacked from the ROS message into Eigen types.
    struct ImuMeasurement {
        double timestamp;               // seconds
        Eigen::Vector3d accel;          // specific force, IMU body frame (m/s^2)
        Eigen::Vector3d gyr;            // angular velocity, IMU body frame (rad/s)
        Eigen::Quaterniond q_world_imu_driver; // attitude from the IMU driver (if provided)
    };

    typedef feature_extraction_config feature_extraction_config;

    class featureExtraction : public rclcpp::Node {
    public:

        /* TODO: return this as a parameter */

        // Velodyne firing timing (seconds), used only when the driver does not
        // stamp each point (provide_point_time == 0) and per-point times must
        // be reconstructed from the firing pattern:
        //  - scanPeriod: duration of one full 360-degree sweep (~0.1 s).
        //  - columnTime: time between successive firing columns (one column =
        //    all N_SCANS lasers fired once).
        //  - laserTime: time between two consecutive lasers within a column.
        static constexpr double scanPeriod = 0.100859904 - 20.736e-6;
        static constexpr double columnTime = 55.296e-6;
        static constexpr double laserTime = 2.304e-6;

        featureExtraction(const rclcpp::NodeOptions & options);

        /// Creates subscribers/publishers, loads parameters and calibration.
        /// The cloud subscription depends on the sensor type: PointCloud2 for
        /// Velodyne/Ouster, the Livox CustomMsg format for Livox.
        void initInterface();

        /// Checks that the measurement buffer (IMU or VIO) fully covers the
        /// oldest lidar scan in time, i.e. measurements exist both before the
        /// first point and after the last point, so every point timestamp can
        /// be interpolated. Returns false (and possibly drops the scan) if not.
        template <typename Meas>
        bool synchronize_measurements(MapRingBuffer<Meas> &measureBuf,
                                        MapRingBuffer<pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr> &lidarBuf);

        /// Legacy entry point for IMU-based undistortion (rotation only).
        /// The current code path uses the removePointDistortion template.
        void imuRemovePointDistortion(double lidar_start_time, double lidar_end_time, MapRingBuffer<Imu::Ptr> &imuBuf,
                                    pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr &lidar_msg);

        /// Legacy entry point for VIO-based undistortion (full 6-DOF pose).
        /// The current code path uses the removePointDistortion template.
        void vioRemovePointDistortion(double lidar_start_time, double lidar_end_time, MapRingBuffer<nav_msgs::msg::Odometry::SharedPtr>&vioBuf,
                                    pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr &lidar_msg);

        /// Per-scan pipeline: check IMU/VIO synchronization, undistort the
        /// oldest buffered scan with whichever source is available (VIO
        /// preferred over IMU), then extract and publish features.
        void undistortionAndFeatureExtraction();

        /// Selects feature points from the undistorted cloud and publishes
        /// everything. Currently only surface points are filled (uniform
        /// downsampling); the edge and depth-camera clouds stay empty.
        void extractFeatures(
            double lidar_start_time,
            const pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr& lidar_msg,
            const Eigen::Quaterniond& q_world_lidar_start);

        /// Callback for raw IMU messages (~200 Hz): converts, integrates the
        /// gyro into an orientation, buffers the sample, and (once, at
        /// startup) triggers the IMU bias/gravity initialization.
        void imu_Handler(const sensor_msgs::msg::Imu::SharedPtr msg_in);

        /// Callback for the optional external odometry stream: just buffers
        /// the message with its timestamp.
        void visual_odom_Handler(const nav_msgs::msg::Odometry::SharedPtr visualOdometry);

        /// Callback for Velodyne/Ouster PointCloud2 scans: converts the cloud
        /// to the common PointcloudXYZITR format (x,y,z,intensity,time,ring),
        /// buffers it, and runs undistortion + feature extraction.
        void laserCloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg);

        /// Callback for Livox CustomMsg scans: filters points by tag/line,
        /// keeps them in the physical lidar frame, converts to the common
        /// format, and runs undistortion + feature extraction.
        void livoxHandler(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg);

        /// Simple feature selection: keeps every skip_num-th point that is not
        /// a duplicate of its predecessor and lies outside block_range. The
        /// survivors are published as "surface" (planar) features.
        void uniformFeatureExtraction(const pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr &pc_in, 
            pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_surf, int skip_num, float block_range);

        /// Fallback for drivers without per-point timestamps: computes each
        /// point's ring id from its vertical angle and its capture time from
        /// the Velodyne firing pattern (columnTime/laserTime).
        void assignTimeforPointCloud(pcl::PointCloud<PointType>::Ptr laserCloudIn_ptr_);

        /// Converts a PCL cloud to a stamped PointCloud2 message (returned,
        /// not published, despite the name; the caller embeds it into the
        /// LaserFeature message).
        template <typename Point>
        sensor_msgs::msg::PointCloud2 publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub, typename pcl::PointCloud<Point>::Ptr thisCloud, rclcpp::Time thisStamp, std::string thisFrame);

        /// Loads the node's tuning parameters into config_.
        bool readParameters();

        /// Assembles and publishes the LaserFeature message consumed by the
        /// laserMapping node: undistorted cloud, edge/surface feature clouds,
        /// and the world-frame attitude of the lidar at scan start.
        void publishTopic(double lidar_start_time, 
                                         pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr laser_no_distortion_points,
                                         pcl::PointCloud<PointType>::Ptr edgePoints,
                                         pcl::PointCloud<PointType>::Ptr plannerPoints, 
                                         pcl::PointCloud<PointType>::Ptr depthPoints,
                                         Eigen::Quaterniond q_world_lidar_start);

        /// Adds a scan to lidarBuf, dropping the oldest scans if the buffer
        /// backs up (i.e. processing cannot keep up with the sensor).
        void manageLidarBuffer(pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr pointCloud, double timestamp);

        /// Unpacks a ROS IMU message into an ImuMeasurement (Eigen types).
        ImuMeasurement parseImuMessage(const sensor_msgs::msg::Imu::SharedPtr& msg);

        /// Returns the time step since the previous IMU sample, falling back
        /// to the nominal IMU period when the timestamp gap looks wrong.
        double calculateDeltaTime(double current_timestamp);

        /// Wraps a measurement into the internal Imu struct. For Livox, also
        /// applies the leveling rotation and rescales the accelerometer from
        /// units of g to m/s^2 (Livox IMUs report ~1.0 when stationary).
        Imu::Ptr createImuData(const ImuMeasurement& measurement);

        /// Dead-reckons q_world_imu by integrating the gyro from the
        /// previous sample (midpoint rule on SO(3)). For the very first sample
        /// it optionally seeds roll/pitch from the driver attitude, yaw = 0.
        void updateImuOrientation(Imu::Ptr& imudata);

        /// One-time IMU initialization: after ~1 s of stationary data, runs
        /// Imu::imuInit() to estimate gyro/accel statistics, gravity, and the
        /// initial sensor tilt used for diagnostics.
        void imuInitialization(double timestamp);

        /// Core undistortion routine (templated over the pose source: IMU
        /// buffer or VIO odometry buffer). Interpolates a pose for every point
        /// timestamp and moves the point into the scan-start sensor frame.
        template<typename BufferType>
        void removePointDistortion(
            double lidar_start_time, 
            double lidar_end_time,
            MapRingBuffer<BufferType> &buffer,
            pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr &lidar_msg);


        /// Linearly interpolates a pose at the given timestamp between the two
        /// buffered measurements that bracket it (slerp for the rotation,
        /// lerp for the translation).
        template<typename BufferType>
        Transformd getInterpolatedPose(double timestamp, MapRingBuffer<BufferType> &buffer,
                                const std::function<Transformd(const BufferType&)>& extractPose);

        /// Maps one point from its capture-time frame into the scan-start
        /// frame; for IMU poses the relative motion is conjugated with the
        /// IMU<->lidar extrinsics (T_lidar_imu * T * T_imu_lidar).
        Eigen::Vector3d transformPoint(
            const point_os::PointcloudXYZITR& point,
            const Transformd& T_world_body_start,
            const Transformd& T_world_body_point,
            bool is_imu_data);

        /// Writes the corrected xyz back into the point (helper).
        void updatePointPosition(point_os::PointcloudXYZITR& point, const Eigen::Vector3d& new_pos);

        /// A point is usable only if all of its coordinates are finite (lidar
        /// drivers emit NaN/inf for missed returns).
        bool isPointValid(const point_os::PointcloudXYZITR& point);

        // ---- Data buffers (public: sized/allocated by the node's main()) ----
        Imu::Ptr imu_Init = std::make_shared<Imu>(); // results of IMU init: biases, gravity, leveling rotation
        MapRingBuffer<Imu::Ptr> imuBuf;              // time-indexed IMU samples with integrated orientation
        MapRingBuffer<pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr> lidarBuf;   // scans awaiting processing
        MapRingBuffer<nav_msgs::msg::Odometry::SharedPtr> visualOdomBuf;            // optional VIO poses


    private:
        // ---- ROS interface --------------------------------------------------
        // Subscribers
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloud; // Velodyne/Ouster scans
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;                // raw IMU (~200 Hz)
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom;             // optional VIO odometry
        rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr subLivoxCloud; // Livox scans

        // Publishers
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloud;     // undistorted full cloud
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubEdgePoints;     // edge features (unused, empty)
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPlannerPoints;  // surface (planar) features
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubBobPoints;      // depth-camera points (unused)
        rclcpp::Publisher<super_odometry_msgs::msg::LaserFeature>::SharedPtr pubLaserFeatureInfo; // the bundle for laserMapping
        std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> pubEachScan; // per-ring debug clouds

        rclcpp::CallbackGroup::SharedPtr cb_group_;

        // ---- State flags & bookkeeping --------------------------------------
        int delay_count_;
        std::mutex m_buf;               // guards all buffers (callbacks are reentrant)
        int frameCount = 0;             // incoming scan counter, used for frame skipping

        bool PUB_EACH_LINE = false;             // publish one debug cloud per ring
        bool LASER_IMU_SYNC_SCCUESS = false;    // IMU buffer covers the current scan
        bool LASER_CAMERA_SYNC_SUCCESS = false; // VIO buffer covers the current scan
        bool IMU_INIT=false;                    // one-time IMU init (bias/gravity) finished
        double m_imuPeriod;                     // nominal IMU sample period (s)

        // ---- Per-scan outputs ------------------------------------------------
        super_odometry_msgs::msg::LaserFeature laserFeature; // message assembled in publishTopic()
        std_msgs::msg::Header FeatureHeader;
        // The world frame here is the same with the initial IMU frame.
        Eigen::Quaterniond q_world_lidar_start; // lidar attitude in world frame at scan start
        Eigen::Vector3d t_world_lidar_start;    // lidar position in world frame at scan start (zero for IMU-only)
        pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr pointCloudwithTime=nullptr; // scratch cloud from assignTimeforPointCloud
        pcl::PointCloud<point_os::OusterPointXYZIRT>::Ptr tmpOusterCloudIn=nullptr ; // scratch cloud for Ouster conversion
        feature_extraction_config config_;
    };

} // namespace super_odometry

#endif //super_odometry_FEATUREEXTRACTION_H
