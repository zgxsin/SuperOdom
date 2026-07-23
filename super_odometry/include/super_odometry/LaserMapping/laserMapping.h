//
// Created by shibo zhao on 2020-09-27.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// This node is the "scan-to-map registration" stage of the pipeline. It sits
// between the feature extraction node and the IMU preintegration node:
//
//   featureExtraction ──> laserMapping (this node) ──> imuPreintegration
//     (edge + planar          (lidar odometry               (fuse with IMU
//      feature clouds)         pose, ~10 Hz)                 at ~200 Hz)
//
// For every lidar scan the node:
//
//   1. Receives one LaserFeature message that bundles the edge (corner)
//      features, the planar (surface) features, the full deskewed cloud, and
//      an IMU orientation prediction (laserFeatureInfoHandler).
//   2. Predicts where the robot is now ("initial guess") using, in order of
//      preference, VIO / IMU-preintegration odometry / IMU orientation /
//      constant velocity (setInitialGuess).
//   3. Downsamples the feature clouds with a voxel grid whose size adapts to
//      the environment (adjustVoxelSize).
//   4. Refines the pose by "scan-to-map registration": the LidarSLAM class
//      matches each feature point against a sliding local map and solves a
//      small nonlinear least-squares problem (Ceres) that minimizes
//      point-to-line and point-to-plane distances (performSLAMOptimization).
//      The optimized scan is then merged into the local map.
//   5. Publishes the refined pose ("laser_odometry"), the registered scan,
//      the local map, an RViz path, and optimization statistics
//      (updatePoseAndPublish / publishTopic).
//
// The heavy lifting (map storage, correspondence search, Ceres problem) lives
// in LidarProcess/LidarSlam and LidarProcess/LocalMap; this class is mostly
// the ROS wrapper: buffering, synchronization, initial-guess logic and
// publishing.
//
// It can also run in "localization mode": instead of building a map from
// scratch it loads a prior map from disk and localizes inside it.
// ============================================================================

#pragma once
#ifndef super_odometry_LASERMAPPING_H
#define super_odometry_LASERMAPPING_H

#include <cmath>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <iomanip>
#include <mutex>
#include <thread>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/transform_datatypes.h>
#include <std_msgs/msg/float32.hpp>
#include "super_odometry/LidarProcess/LidarSlam.h"
#include "super_odometry/LidarProcess/LocalMap.h"
#include "super_odometry/tic_toc.h"
#include "super_odometry/utils/Twist.h"
#include "super_odometry/container/MapRingBuffer.h"
#include <super_odometry_msgs/msg/laser_feature.hpp>
#include "super_odometry/config/parameter.h"
#include <std_msgs/msg/string.hpp>
#include "super_odometry/utils/superodom_utils.h"

namespace super_odometry {
    // Tuning knobs, loaded from the ROS parameter file in readParameters().
    struct laser_mapping_config{
        float lineRes;                      // voxel size (m) for downsampling edge features
        float planeRes;                     // voxel size (m) for downsampling planar features
        int max_iterations;                 // max ICP/Ceres iterations per scan
        bool debug_view_enabled;            // publish extra debug clouds (local map around robot)
        bool enable_ouster_data;            // accept a secondary Ouster lidar stream
        bool publish_only_feature_points;   // publish feature points instead of the full scan
        bool use_imu_roll_pitch;            // constrain roll/pitch to the IMU attitude during optimization
        int max_surface_features;           // cap on planar features used per scan (runtime bound)
        double velocity_failure_threshold;  // reject an optimized pose implying speed above this (m/s)
        bool auto_voxel_size;               // adapt lineRes/planeRes to the scene size (see adjustVoxelSize)
        bool forget_far_chunks;             // drop local-map chunks far behind the robot
        float visual_confidence_factor;     // weight of visual (VIO) predictions in the SLAM core
        float pos_degeneracy_threshold;     // eigenvalue threshold to flag position degeneracy
        float ori_degeneracy_threshold;     // eigenvalue threshold to flag orientation degeneracy
        float yaw_ratio;                    // scaling of yaw observability in the optimization
        std::string map_dir;                // path of the prior map .pcd (localization mode)
        bool localization_mode;             // true = localize in a prior map, false = build a map
        // Initial pose in the prior map (localization mode only).
        float init_x;
        float init_y;
        float init_z;
        float init_roll;
        float init_pitch;
        float init_yaw;
        float read_pose_file;               // load the initial pose from a file instead of parameters
    };

    class laserMapping : public rclcpp::Node {
    public:
        // The SLAM core: owns the sliding local map and runs the Ceres
        // scan-to-map optimization. This class feeds it data and publishes
        // its results.
        LidarSLAM slam;
    // Everything belonging to one lidar scan, popped from the input buffers
    // as a consistent bundle (see extractSensorData()).
    struct SensorData {
        pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; //not used in optimization
        pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
        pcl::PointCloud<PointType>::Ptr laserCloudFullRes;
        Eigen::Quaterniond q_world_lidar_prediction; // lidar attitude at scan time from feature extraction
        Transformd T_lidar_prev_lidar_current_vio;    // relative scan motion predicted by VIO
        Transformd T_lidar_prev_lidar_current_lio;    // relative scan motion predicted by LIO
        Transformd T_lidar_prev_lidar_current_neural; // relative scan motion predicted by the IMU network
        // Which of the predictions above are valid for this scan.
        bool vio_prediction_status;
        bool lio_prediction_status;
        bool nio_prediction_status;
        bool imu_orientation_status;
        double timestamp;
    };

    SensorData sensorMeas;
    
    // Where the initial pose guess for the current scan came from. A good
    // guess keeps the (locally convergent) scan-to-map optimization from
    // falling into a wrong local minimum.
    enum class PredictionSource {IMU_ORIENTATION, LIO_ODOM, VIO_ODOM, NEURAL_IMU_ODOM, CONSTANT_VELOCITY};
    PredictionSource prediction_source;

    public:
        laserMapping(const rclcpp::NodeOptions & options);

        /// Creates subscribers/publishers, loads parameters and calibration,
        /// configures the SLAM core, and starts the periodic process() timer.
        void initInterface();

        /// Allocates the point cloud buffers, zeroes the odometry-frame
        /// transforms, and loads the prior map when in localization mode.
        void initializationParam();

        /// Merges feature clouds from two lidars (e.g. Velodyne + Ouster)
        /// into one set before optimization.
        void preprocessDualLidarFeatures(Transformd T_world_lidar_current, SensorType sensor_type);

        /// Picks the voxel-filter leaf size from the average point distance
        /// (small indoors, large outdoors) and downsamples the feature clouds.
        void adjustVoxelSize();

        void mappingOptimization(Eigen::Vector3i &postion_in_locamap, Transformd T_world_lidar_start,tf2::Quaternion q_world_lidar_roll_pitch,
                            const int laserCloudCornerStackNum, const int laserCloudSurfStackNum);

        void publishTopic(Eigen::Vector3i postion_in_locamap);


        /// Callback for the feature message from the featureExtraction node.
        /// Buffers the corner/surface/full clouds and the IMU orientation
        /// prediction; the actual work happens later in process().
        void laserFeatureInfoHandler(const super_odometry_msgs::msg::LaserFeature::SharedPtr msgIn);

        /// Callback for the raw (unprocessed) point cloud topic; only buffers.
        void laserCloudRawDataHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudRawdata);

        /// Looks up the IMU-preintegration odometry at the scan time and uses
        /// it to predict the current lidar pose T_world_lidar.
        void extractIMUOdometry(double timeLaserFrame, Transformd &T_world_lidar);

        /// Same as above but for visual(-inertial) odometry; returns false if
        /// no usable VIO message covers the scan time.
        bool extractVisualIMUOdometryAndCheck(Transformd &T_world_lidar);

        /// Interpolates position T and orientation Q from an odometry buffer
        /// at the requested timestamp.
        void getOdometryFromTimestamp(MapRingBuffer<nav_msgs::msg::Odometry::SharedPtr> &buf, const double &timestamp,
                                 Eigen::Vector3d &t_odom_lidar, Eigen::Quaterniond &q_odom_lidar);

        /// Computes the relative motion between the previous and current scan
        /// times from an odometry buffer (used as the motion prediction).
        void extractRelativeTransform(MapRingBuffer<nav_msgs::msg::Odometry::SharedPtr> &buf,
                                      Transformd &T_lidar_prev_lidar_current,
                                      bool imu_prediction);

        /// Computes the initial pose guess for the current scan. Dispatches to
        /// initializeFirstFrame(), initializeWithIMU() or selectPosePrediction()
        /// depending on how far along the system is.
        void setInitialGuess();

        void selectposePrediction();

        /// Chooses the best available prediction source: VIO/neural-IMU when
        /// the lidar geometry is degenerate, otherwise LIO, then IMU
        /// orientation, then constant velocity as the last resort.
        laserMapping::PredictionSource determinePredictionSource();

        /// Very first scan: orients the world frame with IMU roll/pitch (yaw
        /// zeroed) so gravity is aligned with -z, or uses the configured pose
        /// in localization mode.
        void initializeFirstFrame();

        /// During the startup period: trusts the IMU orientation directly and
        /// keeps the last position (motion is assumed negligible).
        void initializeWithIMU();

        /// Applies the chosen prediction source to propagate T_world_lidar to the
        /// current scan time (the "initial guess" for optimization).
        void selectPosePrediction();

        //TODO: organize the publish topics and odometry
        void publishOdometry();

        /// Publishes everything for the current scan: prediction source, local
        /// map (throttled), registered full cloud, odometry, path and stats.
        void publishTopic();

        /// Main loop, driven by a wall timer: waits for a complete data
        /// bundle, then runs guess -> downsample -> optimize -> publish.
        void process();

        /// Loads the node's tuning parameters into config_.
        bool readParameters();

        /// True when all buffers for one scan (corner, surface, full cloud,
        /// IMU prediction) are non-empty.
        bool checkDataAvailable() const;

        /// Pops one synchronized bundle (feature clouds + IMU prediction) off
        /// the input buffers and converts it to PCL types.
        SensorData extractSensorData();

        /// Drops any leftover buffered messages so the node always works on
        /// the freshest scan (older scans are skipped, not queued up).
        void clearSensorData();

        /// Returns true (and stores the orientation) if the IMU prediction
        /// quaternion is valid (w != 0 means it was actually filled in).
        bool useIMUPrediction(const Eigen::Quaterniond& q_world_lidar_prediction);

        /// Hands the downsampled feature clouds, the initial guess and the
        /// IMU roll/pitch constraint to the LidarSLAM core, which runs the
        /// Ceres scan-to-map optimization and updates the local map.
        void performSLAMOptimization();

        /// Copies the optimized pose back, computes body-frame velocities by
        /// finite differences, publishes all topics and stores the pose for
        /// the next iteration.
        void updatePoseAndPublish();
        



        /// Converts a message header stamp to seconds as a double.
        template<typename T>
        double secs(T msg) {
            return msg->header.stamp.sec + msg->header.stamp.nanosec*1e-9;
        }

    private:
        // ---- Constants ------------------------------------------------------
        // Fixed time offset (s) between camera and lidar clocks.
        static constexpr float vision_laser_time_offset = 0.0;
        // Legacy LOAM-style map cube grid: the local map is a 21 x 21 x 11
        // array of cubes and the "Cen" values are the index of the center
        // cube (where the robot starts). Kept for reference; the actual map
        // storage now lives in LocalMap.
        static constexpr int laserCloudCenWidth = 10;
        static constexpr int laserCloudCenHeight = 10;
        static constexpr int laserCloudCenDepth = 5;
        static constexpr int laserCloudWidth = 21;
        static constexpr int laserCloudHeight = 21;
        static constexpr int laserCloudDepth = 11;
        static constexpr int laserCloudNum =laserCloudWidth * laserCloudHeight * laserCloudDepth; // 4851

        // ros::NodeHandle *pub_node_;
        // ros::NodeHandle *private_node_;
        
        // ---- ROS interface --------------------------------------------------
        // Subscribers (only subLaserFeatureInfo is wired up in initInterface;
        // the others are kept for optional inputs such as VIO or raw clouds).
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloudCornerLast;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloudSurfLast;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subIMUOdometry;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subVisualOdometry;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloudFullRes;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserRawdata;
        rclcpp::Subscription<super_odometry_msgs::msg::LaserFeature>::SharedPtr subLaserFeatureInfo;
        // rclcpp::Subscription<>::SharedPtr subTakeoffAlignment;

        // Publishers. The most important ones are pubOdomAftMapped (the lidar
        // odometry pose consumed by imuPreintegration), pubLaserCloudFullRes
        // (the scan re-expressed in the world frame) and pubLaserCloudMap
        // (the accumulated local map for visualization).
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudSurround;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudPrior;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes_rot;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullResOusterWithFeatures;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullResOuster;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudRawRes;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_rot;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMappedHighFrec;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubLaserAfterMappedPath;
        rclcpp::Publisher<super_odometry_msgs::msg::OptimizationStats>::SharedPtr pubOptimizationStats;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryIncremental;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPreviousCloud;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubPreviousPose;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pubprediction_source;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubVIOPrediction; 
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLIOPrediction;

        // Timer that runs process() (the main loop) every 100 ms.
        rclcpp::TimerBase::SharedPtr process_timer_;

        rclcpp::CallbackGroup::SharedPtr cb_group_;

        // ---- Odometry history buffers ---------------------------------------
        // Timestamp-indexed ring buffers of IMU-preintegration odometry and
        // visual odometry, used to interpolate a pose prediction at scan time.
        MapRingBuffer<nav_msgs::msg::Odometry::SharedPtr> imu_odom_buf;
        MapRingBuffer<nav_msgs::msg::Odometry::SharedPtr> visual_odom_buf;

        // ---- Counters & timestamps ------------------------------------------
        int frameCount = 0;             // scans processed so far (throttles map publishing)
        int waiting_takeoff_timeout = 300;
        int startupCount = 10;          // scans left in the "trust the IMU" startup period
        int localizationCount = 0;
        int laserCloudValidInd[125];    // legacy LOAM cube-index scratch (unused here)
        int laserCloudSurroundInd[125];

        // Timestamps (s) of the latest received clouds and the current scan.
        double timeLaserCloudCornerLast = 0;
        double timeLaserCloudSurfLast = 0;
        double timeLaserCloudFullRes = 0;
        double timeLaserOdometry = 0;       // timestamp of the scan being processed
        double timeLaserOdometryPrev = 0;   // previous scan time (for velocity estimation)


        // ---- Status flags -----------------------------------------------------
        bool got_previous_map = false;
        bool force_initial_guess = false;
        bool odomAvailable = false;          // VIO covers the current scan time
        bool lastOdomAvailable = false;
        bool laser_imu_sync = false;         // lidar and IMU timestamps agree
        bool use_imu_roll_pitch_this_step = false;
        bool initialization = false;         // first frame has been processed
        bool imuodomAvailable = false;       // IMU-preintegration odometry covers the scan
        bool imuorientationAvailable = false;
        bool lastimuodomAvaliable=false;
        bool imu_initialized = false;


        // Voxel-grid downsampling filters for edge and planar features (leaf
        // sizes lineRes / planeRes, possibly adapted per scan).
        pcl::VoxelGrid<PointType> downSizeFilterCorner;
        pcl::VoxelGrid<PointType> downSizeFilterSurf;


        // ---- Input buffers (filled by callbacks, drained by process()) -------
        // The four queues below are pushed together by laserFeatureInfoHandler,
        // so their fronts always belong to the same scan.
        std::queue<sensor_msgs::msg::PointCloud2> cornerLastBuf;   // edge features
        std::queue<sensor_msgs::msg::PointCloud2> surfLastBuf;     // planar features
        std::queue<sensor_msgs::msg::PointCloud2> realsenseBuf;    // optional depth-camera cloud
        std::queue<sensor_msgs::msg::PointCloud2> fullResBuf;      // full deskewed scan
        std::queue<sensor_msgs::msg::PointCloud2> rawWithFeaturesBuf;
        std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> rawDataBuf;
        std::queue<nav_msgs::msg::Odometry::SharedPtr> odometryBuf;
        std::queue<Eigen::Quaterniond> q_world_lidar_prediction_buf; // lidar orientation prediction per scan
        std::queue<SensorType> sensorTypeLastBuf;
        SensorType last_sensor_type_= SensorType::VELODYNE;
     
        
        // variables for stacking 2 scans (dual-lidar setups)
        std::queue<pcl::PointCloud<PointType>> cornerDualScanBuf;
        std::queue<pcl::PointCloud<PointType>> surfDualScanBuf;
        std::queue<pcl::PointCloud<pcl::PointXYZHSV>> fullResDualScanBuf;
        std::queue<Transformd> T_dual_lidar_scan_buffer;
        std::queue<SensorType> sensorTypeBuf;

        // ---- Working point clouds (current scan) ------------------------------
        pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;  // edge features, lidar frame
        pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;    // planar features, lidar frame
        pcl::PointCloud<PointType>::Ptr laserCloudRealsense;

        pcl::PointCloud<PointType>::Ptr laserCloudSurround;    // local map around robot (debug)
        pcl::PointCloud<PointType>::Ptr laserCloudFullRes;     // full scan; transformed to world for publishing
        pcl::PointCloud<PointType>::Ptr laserCloudFullRes_rot;
        pcl::PointCloud<PointType>::Ptr laserCloudRawRes;
        pcl::PointCloud<PointType>::Ptr laserCloudCornerStack; // downsampled edge features fed to the optimizer
        pcl::PointCloud<PointType>::Ptr laserCloudSurfStack;   // downsampled planar features fed to the optimizer
        pcl::PointCloud<pcl::PointXYZHSV>::Ptr laserCloudRawWithFeatures;
        pcl::PointCloud<pcl::PointXYZHSV>::Ptr velodyneLaserCloudRawWithFeatures;
        pcl::PointCloud<pcl::PointXYZHSV>::Ptr ousterLaserCloudRawWithFeatures;

        // Prior map loaded from disk in localization mode.
        pcl::PointCloud<PointType>::Ptr laserCloudPriorOrg;
        pcl::PointCloud<PointType>::Ptr laserCloudPrior;
        sensor_msgs::msg::PointCloud2 priorCloudMsg;

        // ---- Pose state --------------------------------------------------------
        Transformd T_world_lidar;          // current lidar pose in the world (map) frame
        Transformd T_world_lidar_prev;     // pose of the previous scan
        Transformd T_world_imu_prev;
        Transformd T_lidar_initial_guess_lidar_current;
        Transformd T_world_lidar_incremental; // pose published on the "incremental" topic
        Transformd T_world_lidar_forced_initial_guess;

        // Transform between the mapping world frame and the odometry frame
        // (LOAM convention: q/t_map_odom maps odometry-frame poses into the
        // map frame), plus the current/previous lidar attitude predictions
        // and legacy odometry-frame translations.
        Eigen::Quaterniond q_map_odom;
        Eigen::Vector3d t_map_odom;
        Eigen::Quaterniond q_world_lidar_prediction_current;
        Eigen::Vector3d t_odom_lidar_current;
        Eigen::Quaterniond q_world_lidar_prediction_prev;
        Eigen::Vector3d t_odom_lidar_prev;
        Eigen::Quaterniond q_world_imu_prev;
        Eigen::Vector3d t_world_imu_prev;


        // ---- Misc ---------------------------------------------------------------
        laser_mapping_config config_;
        nav_msgs::msg::Path laserAfterMappedPath;  // full trajectory for RViz
        std::mutex mBuf;                           // guards the input buffers
        rclcpp::Time timeLatestImuOdometry;        // used to estimate end-to-end latency
        rclcpp::Time timeLastMappingResult;
        PointType pointOri, pointSel;
           
    }; // class laserMapping

} // namespace super_odometry
#endif //super_odometry_LASERMAPPING_H
