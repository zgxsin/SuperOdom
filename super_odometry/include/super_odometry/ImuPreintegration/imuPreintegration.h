//
// Created by shibo zhao on 2020-09-27.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// This node fuses two data streams to produce a high-rate odometry estimate:
//
//   1. IMU messages (~200 Hz): accelerometer + gyroscope. Integrating them
//      gives smooth, high-rate motion, but the estimate drifts quickly
//      because of sensor noise and slowly-changing biases.
//   2. Lidar odometry poses (~10 Hz, from the laserMapping node): accurate
//      and drift-free over short spans, but low-rate and delayed.
//
// The fusion is done with "IMU preintegration" + factor-graph optimization
// (the GTSAM library), the same idea as in LIO-SAM:
//
//   - Between two lidar poses, all IMU samples are summed ("preintegrated")
//     into a single relative-motion constraint (gtsam::ImuFactor).
//   - Each new lidar pose is added as a unary prior factor.
//   - A sliding optimizer (gtsam::ISAM2) solves for the pose, velocity and
//     IMU bias at each lidar keyframe:  X(k) = pose, V(k) = velocity,
//     B(k) = bias, where k is the integer 'key' counter.
//
// After every optimization, the newest bias estimate is used to re-integrate
// the IMU samples that arrived after the lidar pose, so the node can publish
// odometry at full IMU rate that is anchored to the latest optimized state.
//
// Data flow:
//
//   IMU topic ──> imuHandler ──> imuConverter (same physical IMU frame)
//                 ──> two queues:
//                   - imuQueOpt: consumed by the optimizer thread of work
//                   - imuQueImu: used to propagate high-rate odometry
//                 ──> predict + publish odometry at IMU rate
//
//   laser_odometry topic ──> laserodometryHandler ──> initialize system /
//                 integrate IMU between lidar poses ──> build & solve factor
//                 graph ──> updated state + bias ──> re-propagate imuQueImu
// ============================================================================
#pragma once
#ifndef IMUPREINTEGRATION_H
#define IMUPREINTEGRATION_H

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>


#include "utility.h"
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include "super_odometry/utils/Twist.h"
#include "super_odometry/container/MapRingBuffer.h"
#include "super_odometry/config/parameter.h"
#include "super_odometry/tic_toc.h"
#include <glog/logging.h>
#include "super_odometry/sensor_data/imu/imu_data.h"


namespace super_odometry {

    // GTSAM identifies each unknown in the factor graph by a symbol made of a
    // letter + an index. At lidar keyframe k the three unknowns are:
    using gtsam::symbol_shorthand::B; // B(k): IMU Bias  (ax,ay,az,gx,gy,gz)
    using gtsam::symbol_shorthand::V; // V(k): Velocity in world frame (xdot,ydot,zdot)
    using gtsam::symbol_shorthand::X; // X(k): Pose3 (x,y,z,roll,pitch,yaw)
    using FrameId = std::uint64_t;

    // Tuning knobs, loaded from the ROS parameter file in readParameters().
    struct imuPreintegration_config{
        float imuAccNoise;             // accelerometer white-noise sigma (continuous time)
        float imuAccBiasN;             // accelerometer bias random-walk sigma
        float imuGyrNoise;             // gyroscope white-noise sigma (continuous time)
        float imuGyrBiasN;             // gyroscope bias random-walk sigma
        float imuGravity;              // local gravity magnitude (m/s^2)
        float lidar_correction_noise;  // sigma of the lidar pose prior factor:
                                       // small = trust lidar more, large = trust IMU more
        float smooth_factor;           // smoothing weight (currently unused here)
        bool  use_imu_roll_pitch;      // publish orientation from the IMU driver instead
                                       // of the optimized one (useful if the IMU has a
                                       // good internal attitude filter)
        SensorType sensor;             // LIVOX / VELODYNE / OUSTER

        // Sanity limits on measured acceleration (used for health checks).
        double imu_acc_x_limit;
        double imu_acc_y_limit;
        double imu_acc_z_limit;
    };

    class imuPreintegration : public rclcpp::Node {
    public:

        imuPreintegration(const rclcpp::NodeOptions & options);

        // Time tolerance (s) when deciding which IMU samples are "before" a
        // lidar pose. 0 = use the lidar timestamp as-is.
        static constexpr double delta_t = 0;
        // If the newest IMU sample is more than this many seconds older than
        // the lidar pose, the IMU stream is considered broken (health check).
        static constexpr double imu_laser_timedelay= 0.8;

    public:
        /// Creates subscribers/publishers, loads parameters and calibration,
        /// and configures the two GTSAM preintegrators (noise models, gravity).
        void initInterface();

        /// Loads the node's tuning parameters into config_.
        bool readParameters();

        /// Callback for lidar odometry poses (~10 Hz). This drives the factor
        /// graph: first call initializes the system; later calls integrate the
        /// buffered IMU data, add factors, optimize, and re-propagate.
        void laserodometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg);

        /// Callback for raw IMU messages (~200 Hz). Converts and buffers the
        /// measurement and (once the first
        /// optimization has run) publishes high-rate odometry by IMU prediction.
        void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imu_raw);

        /// First lidar pose: sets the initial pose/velocity/bias priors of the
        /// factor graph and marks the system initialized.
        void initial_system(double currentCorrectionTime, gtsam::Pose3 T_world_lidar_meas);

        /// Main per-lidar-pose pipeline: integrate IMU -> build & solve the
        /// graph -> failure check -> re-propagate the high-rate IMU odometry.
        void process_imu_odometry(double currentCorrectionTime, gtsam::Pose3 T_world_lidar_meas);

        /// Adds one keyframe to the factor graph (lidar pose prior + IMU factor
        /// + bias random-walk factor), runs ISAM2, and stores the new optimum.
        bool build_graph(gtsam::Pose3 T_world_lidar_meas, double curLaserodomtimestamp);

        /// After optimization: re-integrates the IMU samples newer than the
        /// lidar pose on top of the freshly optimized state and bias, so the
        /// IMU-rate odometry published in imuHandler starts from the optimum.
        void repropagate_imuodometry(double currentCorrectionTime);

        /// Sanity check on the optimized state: absurd velocity or bias means
        /// the optimization diverged and the system should be reset.
        bool failureDetection(const gtsam::Vector3 &velCur,
                         const gtsam::imuBias::ConstantBias &biasCur);

        void obtainCurrodometry(nav_msgs::msg::Odometry::SharedPtr &odomMsg, double &currentCorrectionTime,
                           gtsam::Pose3 &T_world_lidar_meas,
                           int &currentResetId);

        /// Feeds all IMU samples older than the lidar pose into the
        /// preintegrator used by the optimizer (imuIntegratorOpt_).
        void integrate_imumeasurement(double currentCorrectionTime);

        /// Periodic reset that keeps the graph small: starts a fresh graph
        /// whose priors are the current estimate with its marginal covariance.
        void reset_graph();

        /// Replaces the ISAM2 optimizer and clears factors/values.
        void resetOptimization();

        /// Resets the bookkeeping flags after a failure (forces re-init).
        void resetParams();

        /// Runs the one-time IMU initialization (bias/gravity estimation) and
        /// the Livox gravity rescaling. Returns false until init has finished.
        bool handleIMUInitialization(const sensor_msgs::msg::Imu::SharedPtr&imu_raw, 
        sensor_msgs::msg::Imu& thisImu);


        /// Appends the current pose to a short rolling path (for RViz).
        void updateAndPublishPath(nav_msgs::msg::Odometry &odometry, const sensor_msgs::msg::Imu& thisImu);

        /// Broadcasts the world -> sensor TF for the current pose.
        void publishTransform(nav_msgs::msg::Odometry &odometry, const sensor_msgs::msg::Imu& thisImu);

        /// Fills the odometry message from the predicted NavState: converts the
        /// IMU pose to the lidar pose, adds body-frame velocity and bias info.
        void prepareOdometryMessage(nav_msgs::msg::Odometry &odometry, const sensor_msgs::msg::Imu& thisImu, const gtsam::NavState &currentState);

        /// Publishes the odometry (decimated to every 4th IMU sample) + health.
        void publishOdometry(const sensor_msgs::msg::Imu& thisImu, const gtsam::NavState& currentState, nav_msgs::msg::Odometry &odometry);

        void publishTransformsAndPath(nav_msgs::msg::Odometry &odometry, const sensor_msgs::msg::Imu& thisImu);

        /// Computes dt bookkeeping and pushes the prepared IMU message into
        /// both queues (imuQueOpt for optimization, imuQueImu for propagation).
        void processTiming(const sensor_msgs::msg::Imu& thisImu);

        /// Accumulates ~1 s of raw IMU data, then runs Imu::imuInit() to
        /// estimate gravity direction, biases and the leveling rotation.
        void initializeImu(const sensor_msgs::msg::Imu::SharedPtr& imu_raw);

        /// Livox IMUs report acceleration in units of g (~1.0 when static);
        /// rescale to m/s^2 using the measured stationary norm.
        void correctLivoxGravity(sensor_msgs::msg::Imu& thisImu);


        /**
         * @brief Prepares a raw message without changing its physical frame.
         *
         * GTSAM state X(k) is T_world_imu, and its preintegrators expect
         * acceleration, angular velocity, and bias in the physical IMU frame at
         * the IMU origin. Consequently this function does not apply lidar
         * extrinsics, startup leveling, or lever-arm corrections. It only
         * normalizes a valid optional orientation quaternion.
         *
         * @param imu_in Raw IMU message expressed in the physical IMU frame.
         * @return A same-frame copy with a normalized valid orientation.
         */
        sensor_msgs::msg::Imu
        imuConverter(const sensor_msgs::msg::Imu &imu_in);

        /// Converts a message header stamp to seconds as a double.
        template<typename T>
        double secs(T msg) {
            return msg->header.stamp.sec + msg->header.stamp.nanosec*1e-9;
        }


    private:

        // ---- ROS interface ------------------------------------------------
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;            // raw IMU (~200 Hz)
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subLaserOdometry;// lidar odometry (~10 Hz)
        

        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;  // fused odometry at IMU rate
        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pubHealthStatus;     // false if IMU stream looks broken
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubImuPath;          // short rolling path for RViz


        rclcpp::CallbackGroup::SharedPtr cb_group_;
    public:
        // ---- GTSAM machinery ----------------------------------------------
        // Noise models = "how much do I trust each piece of information".
        // A small sigma means high confidence.
        gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;   // initial pose prior
        gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;    // initial velocity prior
        gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;   // initial bias prior
        gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;  // per-keyframe lidar pose prior
        gtsam::Vector noiseModelBetweenBias;                      // bias random-walk between keyframes

        // Two independent preintegrators over the SAME IMU stream:
        //  - imuIntegratorOpt_ sums samples BETWEEN two lidar poses; its result
        //    becomes the ImuFactor in the graph.
        //  - imuIntegratorImu_ sums samples AFTER the newest lidar pose; it is
        //    used to predict/publish odometry at IMU rate ahead of the last
        //    optimized state.
        std::shared_ptr<gtsam::PreintegratedImuMeasurements> imuIntegratorOpt_;
        std::shared_ptr<gtsam::PreintegratedImuMeasurements> imuIntegratorImu_;

        // Latest optimized state (updated after each graph solve).
        gtsam::Pose3 T_world_imu_prev;
        gtsam::Vector3 prevVel_;
        gtsam::NavState prevState_;              // pose + velocity together
        gtsam::imuBias::ConstantBias prevBias_;

        // Snapshot of the optimized state used as the starting point for the
        // high-rate IMU propagation (copied in repropagate_imuodometry()).
        gtsam::NavState prevStateOdom;
        gtsam::imuBias::ConstantBias prevBiasOdom;

        // Incremental smoothing and mapping optimizer + the new factors/values
        // that will be added to it at the next update.
        gtsam::ISAM2 optimizer;
        gtsam::NonlinearFactorGraph graphFactors;
        gtsam::Values graphValues;
        gtsam::Pose3 T_world_lidar_meas_prev; // previous measured lidar pose in world frame
        gtsam::Pose3 T_world_lidar_meas;      // current measured lidar pose in world frame


    public:
        // ---- Extrinsics (rigid transforms between sensors) -----------------
        // Used to convert between "pose of the lidar" and "pose of the IMU":
        // the graph estimates the IMU pose, while lidar odometry measures the
        // lidar pose. If no direct IMU<->lidar calibration is provided they
        // are chained through the camera.
        gtsam::Pose3 T_imu_camera;
        gtsam::Pose3 T_camera_lidar;
        gtsam::Pose3 T_imu_lidar;
        gtsam::Pose3 T_lidar_imu;

    public:
        // ---- Buffers --------------------------------------------------------
        MapRingBuffer<Imu::Ptr> imuBuf;             // raw IMU kept ~1 s for one-time initialization
        std::deque<sensor_msgs::msg::Imu> imuQueOpt;// physical IMU data awaiting graph integration
        std::deque<sensor_msgs::msg::Imu> imuQueImu;// physical IMU data newer than the last lidar pose
        MapRingBuffer<nav_msgs::msg::Odometry::SharedPtr> lidarOdomBuf;
        std::mutex mBuf;                            // guards both callbacks (they share the queues)
        Imu::Ptr imu_Init = std::make_shared<Imu>();// results of IMU init: biases, gravity, leveling rotation

    public:
        // ---- State flags & bookkeeping --------------------------------------
        bool systemInitialized = false; // first lidar pose received, graph has priors
        bool doneFirstOpt = false;      // at least one graph solve succeeded
        bool health_status = true;      // published on pubHealthStatus
        bool imu_init_success = false;  // one-time IMU init (bias/gravity) finished

       
        Eigen::Quaterniond q_world_imu_first;

        double first_imu_time_stamp;
        double last_processed_lidar_time = -1;
        double lastImuT_imu = -1;       // timestamp of last IMU sample seen by imuHandler
        double lastImuT_opt = -1;       // timestamp of last IMU sample fed to the optimizer
        int key = 1;                    // index of the current keyframe: X(key), V(key), B(key)
        int imuPreintegrationResetId = 0;
        int frame_count = 0;            // used to decimate publishing (every 4th message)

        // Health flag embedded into odometry.pose.covariance[0] for consumers.
        enum IMU_STATE : uint8_t {
        FAIL=0,    //lose imu information 
        SUCCESS=1, //Obtain the good imu data 
        UNKNOW=2
        };  

        IMU_STATE RESULT;
        nav_msgs::msg::Odometry::SharedPtr cur_frame = nullptr;  // newest lidar odometry msg
        nav_msgs::msg::Odometry::SharedPtr last_frame = nullptr; // previous lidar odometry msg
        imuPreintegration_config config_;
    };

}

#endif // IMUPREINTEGRATION_H
