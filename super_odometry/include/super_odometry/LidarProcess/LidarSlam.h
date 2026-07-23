//
// Created by ubuntu on 2020/9/26.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// This class is the core "scan-to-map registration" engine of the lidar
// odometry pipeline. It is called by the laserMapping node once per lidar
// scan and answers the question: "given the edge/planar feature points of
// the new scan and a rough initial guess of the pose, where exactly is the
// lidar in the world?"
//
// Inputs (per scan, produced upstream by the featureExtraction node):
//   - edge feature points   (points lying on sharp 3D edges, e.g. poles,
//     corners of walls)
//   - planar feature points (points lying on flat surfaces, e.g. ground,
//     walls)
//   - an initial pose guess T_world_lidar (from IMU preintegration, visual
//     odometry, or constant-velocity extrapolation; see PredictionSource)
//
// Output:
//   - the refined pose T_world_lidar of the scan in the world frame, plus an
//     uncertainty/degeneracy estimate of that pose.
//
// The refinement is an ICP-style ("Iterative Closest Point") loop against a
// sliding-window LocalMap (a voxel grid of previously registered feature
// points centered around the robot). Each iteration does:
//
//   1. Transform every feature point into the world frame using the current
//      pose estimate (the "initial guess" on the first iteration).
//   2. Data association: for each point, find its nearest neighbors in the
//      LocalMap (edge points search the edge map, planar points the surf map).
//   3. Model fitting via eigen-decomposition of the neighbors' 3x3
//      covariance matrix (PCA):
//        - edge point:   the neighbors should form a LINE. The line direction
//          is the eigenvector of the largest eigenvalue; the fit is accepted
//          only if that eigenvalue clearly dominates the others.
//        - planar point: the neighbors should form a PLANE. The plane normal
//          is the eigenvector of the smallest eigenvalue; the fit is accepted
//          only if that eigenvalue is clearly smaller than the others.
//   4. Residual construction: each accepted match contributes one Ceres cost
//      term, either point-to-line distance (edge) or point-to-plane distance
//      (planar), weighted by the fit quality and wrapped in a robust Tukey
//      loss so outliers cannot dominate.
//   5. Nonlinear least-squares solve (Ceres, Levenberg-Marquardt style) for
//      the 6-DoF pose, parameterized as translation + unit quaternion.
//   6. If the solver converged in a single step (the pose barely moved) or
//      the maximum number of ICP iterations is reached, stop; otherwise
//      re-associate with the improved pose and repeat.
//
// After convergence the registered feature points are inserted back into the
// LocalMap, and the pose covariance is estimated from the Ceres problem to
// detect DEGENERACY (environments like long corridors or open fields where
// some directions of motion are unobservable from geometry alone). The
// resulting pose is published by laserMapping and consumed by the
// imuPreintegration node, which fuses it with the IMU at high rate.
// ============================================================================

#pragma once
#ifndef LIDARSLAM_H
#define LIDARSLAM_H

#include <atomic>

#include <tbb/concurrent_vector.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "super_odometry/sensor_data/pointcloud/LidarPoint.h"
#include "super_odometry/LidarProcess/LocalMap.h"


#include "super_odometry/utils/EigenTypes.h"
#include "super_odometry/utils/Twist.h"
#include <super_odometry_msgs/msg/optimization_stats.hpp>

#include <sophus/se3.hpp>
#include <std_msgs/msg/float32.hpp>
#include <filesystem>

#include "super_odometry/LidarProcess/factor/SE3AbsolutatePoseFactor.h"
#include "super_odometry/LidarProcess/factor/lidarOptimization.h"
#include "super_odometry/LidarProcess/factor/pose_local_parameterization.h"
#include <ceres/ceres.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <Eigen/Dense>
#include <glog/logging.h>
#include "super_odometry/tic_toc.h"
#include "super_odometry/config/parameter.h"
#include "super_odometry/utils/superodom_utils.h"

namespace super_odometry {

    class LidarSLAM {


    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using Point = pcl::PointXYZI;
        using PointCloud = pcl::PointCloud<Point>;
        // Where the initial pose guess for the current scan came from. The
        // optimizer starts from this guess; a good guess means fewer ICP
        // iterations and less risk of converging to a wrong local minimum.
        enum class PredictionSource {
            IMU_ORIENTATION, LIO_ODOM, VIO_ODOM, NEURAL_IMU_ODOM, CONSTANT_VELOCITY
        };


    public:
        
        // The world-frame body axes: the unit x/y/z axes of the lidar rotated
        // by the current orientation estimate. Used by the observability
        // analysis to relate each feature to the 6 motion directions.
        struct RotatedAxes {
        Eigen::Vector3f x, y, z;};


        // EGO_MOTION = scan-to-scan matching, LOCALIZATION = scan-to-map
        // matching. Only LOCALIZATION is used in this implementation.
        enum class MatchingMode {
            EGO_MOTION = 0, LOCALIZATION = 1
        };

        enum UndistortionMode {
            //! No undistortion is performed:
            //! -End scan pose is optimized using rigid registration of raw scan and map.
            //! -Raw input scan is added to maps.
            NONE = 0,

            //! Minimal undistortion is performed:
            //! - begin scan pose is linearly interpolated between previous and current end scan poses.
            //! - End scan pose is optimized using rigid registration of undistorted scan and map.
            //! - Scan is linearly undistorted between begin and end scan poses.
            APPROXIMATED = 1,

            //! Ceres-optimized undistorted is performed:
            //! -both begin and end scan are optimized using registration of undistorted scan
            //! and map.
            //! -Scan is linearly undistorted between begin and scan poses.
            OPTIMIZED = 2
        };

        //! Result of keypoint matching, explaining rejection.
        //! Every feature point goes through a chain of validity checks
        //! (enough neighbors? neighbors close enough? does the PCA really
        //! look like a line/plane? is the fit residual small?). A point only
        //! contributes a residual to the optimizer if the result is SUCCESS;
        //! histograms of the rejection causes are kept for diagnostics.
        enum MatchingResult : uint8_t {
            SUCCESS = 0,               // keypoint has been successfully matched
            NOT_ENOUGH_NEIGHBORS = 1,  // not enough neighbors to match keypoint
            NEIGHBORS_TOO_FAR = 2,     // neighbors are too far to match keypoint
            BAD_PCA_STRUCTURE = 3,     // PCA eigenvalues analysis discards neighborhood fit to model
            INVAVLID_NUMERICAL = 4,    // optimization parameter computation has numerical invalidity
            MSE_TOO_LARGE = 5,         // mean squared error to model is too large to accept fitted model
            UNKNON = 6,                // unkown status (matching not performed yet)
            nRejectionCauses = 7
        };

        // Labels for "which of the 6 pose degrees of freedom does this
        // feature constrain best". Each accepted plane feature votes for the
        // rotation/translation axes it observes; the per-axis vote counts
        // (PlaneFeatureHistogramObs) are later turned into per-axis
        // uncertainty and used for degeneracy detection. For example, points
        // on the ground constrain z translation and roll/pitch, but say
        // nothing about x/y translation or yaw.
        enum Feature_observability : uint8_t {
            rx_cross = 0,               // evaluate for x rotation estimation
            neg_rx_cross = 1,           // evaluate for neg x  rotation estimation
            ry_cross = 2,               // evaluate for y rotation estimation
            neg_ry_cross = 3,           // evaluate for neg y rotation estimation
            rz_cross = 4,               // evaluate for z rotation estimation
            neg_rz_cross = 5,           // evaluate for neg z rotation estimation
            tx_dot = 6,                 // evaluate for x translation
            ty_dot = 7,                 // evaluate for y translation
            tz_dot = 8,                    // evaluate for z translation
            nFeatureObs = 9
        };

        enum FeatureType : uint8_t {
            EdgeFeature = 0,
            PlaneFeature = 1,
        };


    public:

        // Tuning knobs for the optimization, loaded from ROS parameters by
        // the laserMapping node.
        struct LaserOptSet {
            tf2::Quaternion q_world_lidar_roll_pitch; // optional lidar roll/pitch prior from IMU
            bool  debug_view_enabled;
            bool  use_imu_roll_pitch;           // trust the IMU attitude for roll/pitch
            float velocity_failure_threshold;   // m/s; larger jumps are rejected as failures
            float yaw_ratio;                    // deg of yaw correction per meter traveled (calibration fudge)
            int max_surface_features;           // cap on planar features per scan (runtime control)
        };

        //! Estimation of registration error, derived from the covariance of
        //! the final Ceres problem. The eigen-decomposition of the 3x3
        //! position (resp. orientation) covariance block gives the size and
        //! direction of the worst-case error; the inverse condition number
        //! (smallest/largest eigenvalue) is a degeneracy score: values near 0
        //! mean one direction is far less constrained than another.
        struct RegistrationError {
            // Estimation of the maximum position error
            double PositionError = 0.;
            double PositionUncertainty = 0.;
            double MaxPositionError = 0.1;
            double PosInverseConditionNum = 1.0;

            //Estimation of Lidar Uncertainty
            Eigen::VectorXf LidarUncertainty;

            // Direction of the maximum position error
            Eigen::Vector3d PositionErrorDirection = Eigen::Vector3d::Zero();

            // Estimation of the maximum orientation error (in radians)
            double OrientationError = 0.;
            double OrientationUncertainty = 0.;
            double MaxOrientationError = 10;
            double OriInverseConditionNum = 1.0;
            // Direction of the maximum orientation error
            Eigen::Vector3d OrientationErrorDirection = Eigen::Vector3d::Zero();

            // Covariance matrix encoding the estimation of the pose's errors about the 6-DoF parameters
            // (DoF order :  X, Y, Z,rX, rY, rZ)
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Covariance = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();
        };

        // Square roots of the PCA eigenvalues of a neighborhood, sorted
        // lamada1 > lamada2 > lamada3. Their relative sizes describe the
        // local shape: one dominant value = line, two dominant = plane,
        // all similar = unstructured blob.
        struct eigenValue // Eigen Value ,lamada1 > lamada2 > lamada3;
        {
            double lamada1;
            double lamada2;
            double lamada3;
        };

        struct eigenVector //the eigen vector corresponding to the eigen value
        {
            Eigen::Vector3f principalDirection; // largest-eigenvalue direction (line direction)
            Eigen::Vector3f normalDirection;    // smallest-eigenvalue direction (plane normal)
        };

        // Everything the observability analysis knows about one feature
        // point: its PCA shape descriptors plus how strongly it constrains
        // each rotation/translation axis.
        struct pcaFeature //PCA
        {
            eigenValue values;
            eigenVector vectors;
            double curvature;    // lamada3 / (lamada1+lamada2+lamada3): 0 for a perfect plane
            double linear;
            double planar;
            double spherical;
            double linear_2;     // (lamada1-lamada2)/lamada1: close to 1 for a line
            double planar_2;     // (lamada2-lamada3)/lamada1: close to 1 for a plane
            double spherical_2;  // lamada3/lamada1: close to 1 for an unstructured blob
            pcl::PointNormal pt; // the feature point itself (world frame)
            size_t ptId;
            size_t ptNum = 0;
            std::vector<int> neighbor_indices;
            std::array<int, 4> observability; // top-2 rotation + top-2 translation axes this feature observes
            // Per-axis observability scores. Rotation about axis a is
            // observable if (p x n) . a is large (p = point position,
            // n = plane normal): moving the point by a small rotation then
            // changes its point-to-plane distance. Translation along axis a
            // is observable if n . a is large.
            double rx_cross;
            double neg_rx_cross;
            double ry_cross;
            double neg_ry_cross;
            double rz_cross;
            double neg_rz_cross;
            double tx_dot;
            double ty_dot;
            double tz_dot;
        };

        // Per-axis uncertainty in [0, 1] derived from the observability
        // histogram: 0 = many features constrain this axis, 1 = almost none
        // do (that axis of the pose should not be trusted).
        struct LidarOdomUncertainty {
            double uncertainty_x;
            double uncertainty_y;
            double uncertainty_z;
            double uncertainty_roll;
            double uncertainty_pitch;
            double uncertainty_yaw;
        };

        // ---- Debug publishers for the per-axis uncertainty ------------------
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pubUncertaintyX;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pubUncertaintyY;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pubUncertaintyZ;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pubUncertaintyRoll;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pubUncertaintyPitch;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pubUncertaintyYaw;;

        // One matched feature = one residual for the optimizer. Produced by
        // ComputeLine/PlaneDistanceParameters, consumed by
        // addFeatureConstraints when building the Ceres problem.
        struct OptimizationParameter {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            MatchingResult match_result;   // SUCCESS or the rejection cause
            FeatureType feature_type;      // edge (line residual) or plane residual
            pcaFeature feature;            // observability info (planes only)
            Eigen::Matrix3d Avalue;        // point-to-line projection matrix I - d*d^T (lines only)
            Eigen::Vector3d Pvalue;        // centroid of the matched map neighborhood (world frame)
            Eigen::Vector3d Xvalue;        // the raw scan point in the LIDAR frame (the optimizer
                                           // re-transforms it with the pose being estimated)
            Eigen::Vector3d NormDir;       // unit plane normal n (planes only, world frame)
            double negative_OA_dot_norm;   // plane offset d in the plane equation n.x + d = 0
            std::pair<Eigen::Vector3d, Eigen::Vector3d> corres; // two world points spanning the matched line
            double residualCoefficient;    // fit-quality weight in [0,1]; scales the residual
            double TimeValue;              // per-point timestamp (for undistortion; currently 1.0)
        };

    public:
        // ---- Map & bookkeeping ----------------------------------------------
        // Sliding-window voxel map of past edge/planar points, kept centered
        // around the robot. All nearest-neighbor queries go here.
        LocalMap localMap;
    
        super_odometry_msgs::msg::OptimizationStats stats; // diagnostics published per scan
        RegistrationError LocalizationUncertainty;         // covariance-based error estimate
        LidarOdomUncertainty lidarOdomUncer;               // observability-based per-axis uncertainty
        LaserOptSet OptSet;                                // tuning knobs (see struct above)

        // ---- Pose state -------------------------------------------------------
        Transformd T_world_lidar;               // current lidar pose in the world frame (the output)
        Transformd T_world_lidar_prev;          // pose of the previous scan (for motion checks)
        Transformd T_world_lidar_initial_guess; // the prediction we started this scan from
        Eigen::Vector3i pos_in_localmap; // voxel index of the robot inside the local map

        int frame_count;
        int laser_imu_sync;
        int startupCount = 0;          // countdown after a detected failure

        // ---- Degeneracy / localization-mode parameters ------------------------
        float Pos_degeneracy_threshold;
        float Ori_degeneracy_threshold;
        float Visual_confidence_factor; // weight of the VIO pose prior when degenerate
        
        std::string map_dir;
        float init_x;
        float init_y;
        float init_z;
        float init_roll;
        float init_pitch;
        float init_yaw;
        float localization_mode;
        float update_map;
        double lasttimeLaserOdometry;   // timestamp of the previous scan (s)

        // ---- Per-scan feature clouds -------------------------------------------
        PointCloud::Ptr EdgesPoints;    // this scan's edge features (lidar frame)
        PointCloud::Ptr PlanarsPoints;  // this scan's planar features (lidar frame)


        PointCloud::Ptr WorldEdgesPoints;   // same clouds transformed by the final pose
        PointCloud::Ptr WorldPlanarsPoints; // (world frame), ready to insert into the map

        bool bInitialization = false;
        bool isDegenerate = false;  // set when the geometry does not constrain all 6 DoF
        bool save_ply = false;

        UndistortionMode Undistortion = UndistortionMode::NONE;
        // Histograms filled during data association (atomics because matching
        // may run in parallel): which axes the plane features observe, and
        // why line/plane matches were rejected.
        std::array<std::atomic_int, Feature_observability::nFeatureObs> PlaneFeatureHistogramObs;
        std::array<std::atomic_int, MatchingResult::nRejectionCauses> MatchRejectionHistogramLine;
        std::array<std::atomic_int, MatchingResult::nRejectionCauses> MatchRejectionHistogramPlane;
        tbb::concurrent_vector<OptimizationParameter> OptimizationData;

    
        // ---- ICP / matching parameters ------------------------------------------
        size_t LocalizationICPMaxIter = 4;  // max outer loops of (associate -> solve)

        size_t LocalizationLineDistanceNbrNeighbors = 10;   // k for edge nearest-neighbor search
        size_t LocalizationMinmumLineNeighborRejection = 4; // min neighbors for a valid line fit
        size_t LocalizationPlaneDistanceNbrNeighbors = 5;   // k for plane nearest-neighbor search

        double LocalizationLineDistancefactor = 5.0;

        double LocalizationLineMaxDistInlier = 0.2; // max distance (m) for edge neighbors
 
        rclcpp::Node::SharedPtr node_;

        // PLY file saving variables
        pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_to_save;

    public:
        LidarSLAM();

        void Reset(bool resetLog = true);

        /// Main entry point, called once per scan by the laserMapping node.
        /// On the very first scan it only seeds the map (initializeMapping);
        /// afterwards it runs the full scan-to-map ICP loop
        /// (performLocalizationAndMapping). T_world_lidar is the initial guess in
        /// and the refined pose out (stored in the member of the same name).
        void Localization(bool initialization, PredictionSource predictodom,
                          Transformd T_world_lidar_guess,
                          pcl::PointCloud<Point>::Ptr edge_point, pcl::PointCloud<Point>::Ptr planner_point, double timeLaserOdometry);

        /// Computes a point's position in the lidar frame (pInit) and in the
        /// world frame under the current pose estimate (pFinal). The extra
        /// branches would apply motion undistortion, but only NONE is used.
        void ComputePointInitAndFinalPose(MatchingMode matchingMode,const Point &p,Eigen::Vector3d &pInit,Eigen::Vector3d &pFinal);

        /// Data association for one EDGE point: finds map neighbors, fits a
        /// 3D line to them via PCA, validates the fit, and returns the
        /// point-to-line residual parameters (or the rejection cause).
        OptimizationParameter ComputeLineDistanceParameters(LocalMap &local_map, const Point &p);

        /// Data association for one PLANAR point: finds map neighbors, fits a
        /// plane to them, validates the fit, analyzes observability, and
        /// returns the point-to-plane residual parameters (or the rejection).
        OptimizationParameter ComputePlaneDistanceParameters(LocalMap &local_map, const Point &p);

        /// Estimates the 6x6 pose covariance from the solved Ceres problem
        /// and extracts worst-case position/orientation errors and their
        /// directions via eigen-decomposition (degeneracy indicator).
        RegistrationError EstimateRegistrationError(ceres::Problem &problem, const double eigen_thresh);

        /// Scores how much one plane feature contributes to observing each of
        /// the 6 pose DoF, and stores its best axes in feature.observability.
        void FeatureObservabilityAnalysis(pcaFeature &feature, const Eigen::Vector3d &pFinal, const Eigen::Vector3d &eigenvalues, 
                                          const Eigen::Vector3d &normal_direction, const Eigen::Vector3d &principal_direction);

        /// Clears the per-iteration match data and the rejection/observability
        /// histograms before re-running data association.
        inline void ResetDistanceParameters();

        /// Applies a small hand-tuned yaw drift compensation proportional to
        /// the distance traveled (OptSet.yaw_ratio, deg per meter).
        void MannualYawCorrection();

        /// Stores the node handle and creates the uncertainty debug publishers.
        void initROSInterface(rclcpp::Node::SharedPtr);

        /// Adopts the caller's pose prediction as the working pose and the
        /// initial guess for this scan.
        void initializeState(bool initialization,
                             const Transformd& T_world_lidar_guess);

        /// Copies the incoming edge/planar feature clouds into the members.
        void processInputClouds(const pcl::PointCloud<Point>::Ptr&edge_point, const pcl::PointCloud<Point>::Ptr&planner_point);

        /// First-scan handling: centers the local map at the initial pose and
        /// inserts the first feature clouds (nothing to register against yet).
        void initializeMapping(double timeLaserOdometry);

        /// The heart of the class: runs the ICP loop (associate -> build Ceres
        /// problem -> solve -> repeat), then estimates uncertainty and updates
        /// the map. See the file-level overview for the algorithm.
        void performLocalizationAndMapping(PredictionSource predictodom, double timeLaserOdometry);

        /// Transforms a feature cloud into the world frame with the current
        /// pose and inserts it into the local map (edge or surf layer).
        void transformAndAddToMap(const pcl::PointCloud<Point>::Ptr&source_cloud, 
        pcl::PointCloud<Point>::Ptr&world_cloud, bool is_edge);

        /// Records feature counts (scan and map) into the stats message.
        void updateFeatureStats(size_t edge_count, size_t planner_count);

        /// The pose is only optimized if the surrounding map holds more than
        /// 50 planar points; otherwise the problem is too weakly constrained.
        bool hasEnoughFeatures();

        /// Runs ComputeLineDistanceParameters on every edge point and keeps
        /// the successful matches.
        void processEdgeFeatures(tbb::concurrent_vector<OptimizationParameter>&features_corres, int &edge_num);

        /// Runs ComputePlaneDistanceParameters on (a subsample of) the planar
        /// points, keeps successful matches, and updates the observability
        /// histogram used for degeneracy detection.
        void processPlannerFeatures(tbb::concurrent_vector<OptimizationParameter>&features_corres, int &planner_num);

        /// Returns the fraction of planar points to keep so that at most
        /// max_surface_features are processed (-1 means "keep all").
        double calculateSamplingRate(size_t num_points);

        /// Deterministic decimation: decides whether point 'index' survives
        /// the subsampling at the given rate.
        bool shouldProcessPoint(size_t index, double sampling_rate);

        /// Data-association step of one ICP iteration: matches all edge and
        /// planar features against the map and collects the residuals.
        void extractFeaturesConstraints(tbb::concurrent_vector<OptimizationParameter>&feature_corres, int &edge_num, int &planner_num);

        /// Re-centers the local map around the robot and copies T_world_lidar
        /// into the global pose_parameters array that Ceres optimizes.
        void prepareOptimizationState();

        /// Builds the Ceres problem: SE(3) parameter block + one residual per
        /// matched feature (+ an absolute pose prior when degenerate).
        ceres::Problem setupOptimizationProblem(const tbb::concurrent_vector<OptimizationParameter>&features_corres, 
        PredictionSource predictsource, const Transformd& T_world_lidar_guess);

        /// Adds point-to-line (edge) and point-to-plane (planar) residual
        /// blocks, each with a Tukey robust loss scaled by the fit quality.
        void addFeatureConstraints(ceres::Problem&problem, const tbb::concurrent_vector<OptimizationParameter>&features_corres);

        /// A VIO pose prior is added only when visual odometry is available
        /// AND the lidar geometry is degenerate AND the prior has a weight.
        bool shouldAddAbsolutePoseConstraints(PredictionSource predictodom);    

        /// Adds a prior factor pulling the solution toward the VIO pose, with
        /// per-axis weights that are larger where the lidar is less certain.
        void addAbsolutePoseConstraints(
            ceres::Problem& problem, const Transformd& T_world_lidar_guess,
            int good_feature_num);
        
        /// Runs Ceres (dense QR, at most 4 inner iterations) on the problem.
        ceres::Solver::Summary solveOptimizationProblem(ceres::Problem&problem);

        /// Logs feature counts and the pose change of one ICP iteration.
        void recordIterationStats(super_odometry_msgs::msg::IterationStats&iter_stats,
        int surf_num, int edge_num, Transformd& T_world_lidar_prev,
        Transformd& T_world_lidar_current);

        /// After the ICP loop: yaw correction, stats, motion sanity checks,
        /// and insertion of the registered scan into the local map.
        void performPostOptimizationProcessing(double timeLaserOdometry, TicToc &t_opt, super_odometry_msgs::msg::OptimizationStats &stats);

        /// Sanity checks on the estimated motion (too fast = likely failure,
        /// too small = not worth accumulating into the map).
        bool checkMotionThresholds(double timeLaserOdometry, super_odometry_msgs::msg::OptimizationStats &stats);

        /// Computes the pose change relative to the initial guess and to the
        /// previous scan, and stores them in the stats message.
        void updateOptimizationStats(TicToc &t_opt, super_odometry_msgs::msg::OptimizationStats &stats);
        
        /// Thin wrapper around ComputePointInitAndFinalPose for the
        /// LOCALIZATION matching mode.
        bool initializeAndTransformPoint(const Point &p, Eigen::Vector3d &pInit, Eigen::Vector3d &pFinal);
        
        /// K-nearest-neighbor search in the surf map, with rejection when too
        /// few neighbors are found or the farthest one is beyond
        /// square_max_dist (the neighborhood would not be a local surface).
        bool findNearestNeighbors(LocalMap &local_map,
                                    const Eigen::Vector3d &pFinal,
                                    std::vector<Point> &nearest_pts,
                                    std::vector<float> &nearest_dist,
                                    size_t requiredNearest,
                                    size_t min_neighbors,
                                    double square_max_dist,
                                    OptimizationParameter &result);

        /// Eigen-decomposition of the neighbors' covariance matrix, plus the
        /// shape test: a valid LINE needs one dominant eigenvalue, a valid
        /// PLANE needs the smallest eigenvalue to be much smaller than the rest.
        bool computePCAForFeature(const std::vector<Point> &nearest_pts,
                                  Eigen::Vector3d &mean,
                                  Eigen::Vector3d &eigenvalues,
                                  Eigen::Matrix3d &eigenvectors,
                                  OptimizationParameter &result,
                                  FeatureType feature_type);

        bool computeFeatureObservability(pcaFeature &feature, const Eigen::Vector3d &pFinal, 
                                          const Eigen::Vector3d &eigenvalues, 
                                          const Eigen::Vector3d &normal_direction, 
                                          const Eigen::Vector3d &principal_direction);

        /// Packs the plane fit (normal, offset, quality, observability) into
        /// the OptimizationParameter consumed by the optimizer.
        void setPlaneResults(OptimizationParameter &result, const Eigen::Vector3d &mean, 
                               const Eigen::Vector3d &pInit, const Eigen::Vector3d &plane_normal, 
                               double negative_OA_dot_norm, const pcaFeature &feature, double fitQualityCoeff);
        
        /// Fits the plane n.x + d = 0 to the 5 neighbors by least squares and
        /// returns the mean point-to-plane distance; rejects the match if any
        /// neighbor is too far from the fitted plane.
        double computePlaneQualityMetrics(const std::vector<Point>& nearest_pts,
                                          Eigen::Vector3d &plane_normal,
                                          double &negative_OA_dot_norm,
                                          OptimizationParameter &result);

        /// Converts raw eigenvalues into the shape descriptors (curvature,
        /// linear_2, planar_2, spherical_2) stored in the feature.
        void computeEigenProperties(pcaFeature &feature, const Eigen::Vector3d &eigenvalues);

        /// Rotation observability: projects (p x n) onto each body axis.
        void computeCrossProducts(pcaFeature &feature, const RotatedAxes &axes);

        /// Returns the body x/y/z axes rotated into the world frame.
        RotatedAxes computeRotatedAxes();

        /// Translation observability: projects the plane normal onto each
        /// body axis, weighted by how plane-like the neighborhood is.
        void computeTranslationObservability(pcaFeature &feature, const RotatedAxes &axes);

        /// Sorts the per-axis scores and keeps the top-2 rotation and top-2
        /// translation axes as this feature's observability labels.
        void analyzeFeatureObservability(pcaFeature &feature);

        /// Turns the observability histogram into per-axis uncertainties in
        /// [0,1] and publishes them (degeneracy indicator for consumers).
        void EstimateLidarUncertainty();

        /// Publishes the six per-axis uncertainty values on their debug topics.
        void publishUncertainty(double uncer_x, double uncer_y, double uncer_z,
            double uncer_roll, double uncer_pitch, double uncer_yaw);
        
        /// Rejection checks shared by the edge search: enough neighbors, and
        /// the farthest neighbor close enough to form a local structure.
        bool validateNeighborSearch( bool found,const std::vector<Point> &nearest_pts,
        const std::vector<float> &nearest_dist, OptimizationParameter &result);

        /// Builds the point-to-line residual from a validated line fit:
        /// projection matrix, quality weight, and two points spanning the line.
        OptimizationParameter processLineResults(
                                  const Eigen::Vector3d &pInit,
                                  const Eigen::Vector3d &mean,
                                  const Eigen::Vector3d &eigenvalues,
                                  const Eigen::Matrix3d &eigenvectors,
                                  const std::vector<Point> &nearest_pts,
                                  double square_max_dist);



        

    };      // class lidarslam
}
#endif  // LIDARSLAM_H
