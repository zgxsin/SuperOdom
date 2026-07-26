//
// Created by shiboz on 2021-10-18.
//
// ============================================================================
// OVERVIEW
// ============================================================================
// Implementation of the laserMapping node (see laserMapping.h for the big
// picture). The per-scan flow, driven by the process() loop, is:
//
//   laserFeatureInfoHandler   buffer the feature message (callback thread)
//   process()                 main loop (timer thread):
//     extractSensorData()       pop one synchronized scan bundle
//     setInitialGuess()         predict the pose at scan time
//     adjustVoxelSize()         downsample features (adaptive voxel size)
//     performSLAMOptimization() scan-to-map registration via LidarSLAM/Ceres
//     updatePoseAndPublish()    accept the refined pose, publish topics
//
// "Scan-to-map registration" means: given a predicted pose, transform each
// feature point into the world frame, find its closest line/plane in the
// local map, and adjust the pose so the summed point-to-line and
// point-to-plane distances are minimal. That refined pose is the lidar
// odometry output.
// ============================================================================

#include "super_odometry/LaserMapping/laserMapping.h"

// The pose being optimized, stored in the memory layout Ceres expects:
// parameters = [tx, ty, tz, qx, qy, qz, qw]. The two Eigen::Map objects are
// views into this same array, so writing q_world_lidar / t_world_lidar updates the
// parameter block and vice versa. This is the lidar pose in the world frame.
namespace {
double parameters[7] = {0, 0, 0, 0, 0, 0, 1};
Eigen::Map<Eigen::Vector3d> t_world_lidar(parameters);
Eigen::Map<Eigen::Quaterniond> q_world_lidar(parameters+3);

// Body-frame linear and angular velocity of the latest scan, computed by
// finite-differencing consecutive optimized poses (filled in
// updatePoseAndPublish, published in the odometry twist).
Eigen::Vector3d vel_b;
Eigen::Vector3d ang_vel_b;
}  // namespace

namespace super_odometry {

    laserMapping::laserMapping(const rclcpp::NodeOptions & options)
    : Node("laser_mapping_node", options) {
    this->get_logger().set_level(rclcpp::Logger::Level::Debug);
    }

    // Wires the node up: loads calibration and parameters, creates the
    // subscriber for the feature message and all publishers, configures the
    // LidarSLAM core with the tuning values, and starts the 100 ms timer
    // that drives process().
    void laserMapping::initInterface() {
        //! Callback Groups
        // A reentrant group lets the subscription callback and the process()
        // timer run concurrently on the MultiThreadedExecutor; shared buffers
        // are protected by mBuf.
        cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = cb_group_;

        if(!readGlobalparam(shared_from_this()))
        {
            RCLCPP_ERROR(this->get_logger(), "[SuperOdometry::laserMapping] Could not read calibration. Exiting...");
            rclcpp::shutdown();
        }

        if (!readParameters())
        {
            RCLCPP_ERROR(this->get_logger(), "[SuperOdometry::laserMapping] Could not read parameters. Exiting...");
            rclcpp::shutdown();
        }

        if (!readCalibration(shared_from_this()))
        {
            RCLCPP_ERROR(this->get_logger(), "[AriseSlam::laserMapping] Could not read parameters. Exiting...");
            rclcpp::shutdown();
        }

        RCLCPP_INFO(this->get_logger(), "DEBUG VIEW: %d", config_.debug_view_enabled);
        RCLCPP_INFO(this->get_logger(), "ENABLE OUSTER DATA: %d", config_.enable_ouster_data);
        RCLCPP_INFO(this->get_logger(), "line resolution %f plane resolution %f vision_laser_time_offset %f",
                config_.lineRes, config_.planeRes, vision_laser_time_offset);

        // Voxel-grid leaf sizes: planar features can be sparser than edge
        // features, so planeRes is typically twice lineRes.
        downSizeFilterCorner.setLeafSize(config_.lineRes, config_.lineRes, config_.lineRes);
        downSizeFilterSurf.setLeafSize(config_.planeRes, config_.planeRes, config_.planeRes);


        subLaserFeatureInfo = this->create_subscription<super_odometry_msgs::msg::LaserFeature>(
            ProjectName+"/feature_info", 2,
            std::bind(&laserMapping::laserFeatureInfoHandler, this,
                        std::placeholders::_1), sub_options);
                        

        pubLaserCloudSurround = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            ProjectName+"/laser_cloud_surround", 2);

        pubLaserCloudMap = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            ProjectName+"/laser_cloud_map", 2);

        pubLaserCloudPrior = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            ProjectName+"/overall_map", 2);

        pubLaserCloudFullRes = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            ProjectName+"/registered_scan", 2);

        // Main output: the global lidar pose at the end of optimization.
        pubOdomAftMapped = this->create_publisher<nav_msgs::msg::Odometry>(
            ProjectName+"/laser_odometry", 1);

        pubLaserOdometryIncremental = this->create_publisher<nav_msgs::msg::Odometry>(
            ProjectName+"/aft_mapped_to_init_incremental", 1);


        pubVIOPrediction=  this->create_publisher<nav_msgs::msg::Odometry>(
            ProjectName+"/vio_prediction", 1);

        pubLIOPrediction= this->create_publisher<nav_msgs::msg::Odometry>(
            ProjectName+"/lio_prediction", 1);


        pubLaserAfterMappedPath = this->create_publisher<nav_msgs::msg::Path>(
            ProjectName+"/laser_odom_path", 1);

        pubOptimizationStats = this->create_publisher<super_odometry_msgs::msg::OptimizationStats>(
            ProjectName+"/super_odometry_stats", 1);

  

        pubprediction_source = this->create_publisher<std_msgs::msg::String>(
            ProjectName+"/prediction_source", 1);

        // process() contains its own while(rclcpp::ok()) loop, so this timer
        // effectively just launches it once on an executor thread.
        process_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(100.)),
            std::bind(&laserMapping::process, this));

        // Push the tuning parameters down into the SLAM core.
        slam.initROSInterface(shared_from_this());
        slam.localMap.lineRes_ = config_.lineRes;
        slam.localMap.planeRes_ = config_.planeRes;
        slam.Visual_confidence_factor=config_.visual_confidence_factor;
        slam.Pos_degeneracy_threshold=config_.pos_degeneracy_threshold;
        slam.Ori_degeneracy_threshold=config_.ori_degeneracy_threshold;
        slam.LocalizationICPMaxIter=config_.max_iterations;
        slam.OptSet.debug_view_enabled=config_.debug_view_enabled;
        slam.OptSet.velocity_failure_threshold=config_.velocity_failure_threshold;
        slam.OptSet.max_surface_features=config_.max_surface_features;
        slam.OptSet.yaw_ratio=yaw_ratio;
        slam.map_dir=config_.map_dir;
        slam.localization_mode=config_.localization_mode;
        slam.init_x=config_.init_x;
        slam.init_y=config_.init_y;
        slam.init_z=config_.init_z;
        slam.init_roll=config_.init_roll;
        slam.init_pitch=config_.init_pitch;
        slam.init_yaw=config_.init_yaw;

        prediction_source = PredictionSource::IMU_ORIENTATION;
        timeLatestImuOdometry = rclcpp::Time(0,0,RCL_ROS_TIME);

        initializationParam();

    }

    // Allocates all working point clouds, resets the odometry/map frame
    // transforms to identity, sizes the odometry history buffers, and (in
    // localization mode) loads the prior map from disk into the local map.
    void laserMapping::initializationParam() {

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());

        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurround.reset(new pcl::PointCloud<PointType>());
        laserCloudFullRes.reset(new pcl::PointCloud<PointType>());
        laserCloudFullRes_rot.reset(new pcl::PointCloud<PointType>());
        laserCloudRawRes.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerStack.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfStack.reset(new pcl::PointCloud<PointType>());
        laserCloudRealsense.reset(new pcl::PointCloud<PointType>());
        laserCloudPriorOrg.reset(new pcl::PointCloud<PointType>());
        laserCloudPrior.reset(new pcl::PointCloud<PointType>());

        Eigen::Quaterniond q_map_odom_initial(1, 0, 0, 0);
        Eigen::Vector3d t_map_odom_initial(0, 0, 0);
        Eigen::Quaterniond q_world_lidar_prediction_current_initial(1, 0, 0, 0);
        Eigen::Vector3d t_odom_lidar_current_initial(0, 0, 0);
        Eigen::Quaterniond q_world_lidar_prediction_prev_initial(1, 0, 0, 0);
        Eigen::Vector3d t_odom_lidar_prev_initial(0, 0, 0);

        q_map_odom = q_map_odom_initial;
        t_map_odom = t_map_odom_initial;
        q_world_lidar_prediction_current =
            q_world_lidar_prediction_current_initial;
        t_odom_lidar_current = t_odom_lidar_current_initial;
        q_world_lidar_prediction_prev = q_world_lidar_prediction_prev_initial;
        t_odom_lidar_prev = t_odom_lidar_prev_initial;

        imu_odom_buf.allocate(5000);
        visual_odom_buf.allocate(5000);
        
        slam.localMap.setOrigin(Eigen::Vector3d(slam.init_x, slam.init_y, slam.init_z));

        if (slam.localization_mode) {
            RCLCPP_INFO(this->get_logger(), "\033[1;32m Loading GT Map now.... Please wait for 10 sec before running rosbag.\033[0m");
            if(utils::readPointCloud(config_.map_dir, laserCloudPrior)) {
                slam.localMap.addSurfPointCloud(*laserCloudPrior);
                pcl::toROSMsg(*laserCloudPrior, priorCloudMsg);
                priorCloudMsg.header.frame_id = WORLD_FRAME;
                RCLCPP_INFO(this->get_logger(), "\033[1;32m Loading GT Map Succesfully. Localization mode is Ready.\033[0m");
            } else {
                slam.localization_mode = false;
                RCLCPP_INFO(this->get_logger(), "\033[1;32mCannot read map file, switch to mapping mode.\033[0m");
            }
        } else {
            RCLCPP_INFO(this->get_logger(), "\033[1;32mStart SLAM in mapping mode.\033[0m");
        }
    }

    // Declares and reads all ROS parameters into config_. If read_pose_file
    // is set, the initial pose for localization mode comes from a saved
    // odometry file instead of the individual init_* parameters.
    bool laserMapping::readParameters()
    {
        // Declare with default values
        this->declare_parameter("laser_mapping_node.mapping_line_resolution", 0.1);
        this->declare_parameter("laser_mapping_node.mapping_plane_resolution", 0.2);
        this->declare_parameter("laser_mapping_node.max_iterations", 4);
        this->declare_parameter("laser_mapping_node.debug_view", false);
        this->declare_parameter("laser_mapping_node.enable_ouster_data", false);
        this->declare_parameter("laser_mapping_node.publish_only_feature_points", false);
        this->declare_parameter("laser_mapping_node.use_imu_roll_pitch", false);
        this->declare_parameter("laser_mapping_node.max_surface_features", 2000);
        this->declare_parameter("laser_mapping_node.velocity_failure_threshold", 30.0);
        this->declare_parameter("laser_mapping_node.auto_voxel_size", true);
        this->declare_parameter("laser_mapping_node.forget_far_chunks", false);
        this->declare_parameter("laser_mapping_node.visual_confidence_factor", 1.0);
        this->declare_parameter("laser_mapping_node.localization_mode", false); // Add default value!
        this->declare_parameter("laser_mapping_node.read_pose_file", false);
        this->declare_parameter("laser_mapping_node.init_x", 0.0);
        this->declare_parameter("laser_mapping_node.init_y", 0.0);
        this->declare_parameter("laser_mapping_node.init_z", 0.0);
        this->declare_parameter("laser_mapping_node.init_roll", 0.0);
        this->declare_parameter("laser_mapping_node.init_pitch", 0.0);
        this->declare_parameter("laser_mapping_node.init_yaw", 0.0);
        this->declare_parameter("map_dir", "pointcloud_local.pcd");


        // Get parameters
        config_.lineRes = this->get_parameter("laser_mapping_node.mapping_line_resolution").as_double();
        config_.planeRes = this->get_parameter("laser_mapping_node.mapping_plane_resolution").as_double();
        config_.max_iterations = this->get_parameter("laser_mapping_node.max_iterations").as_int();
        config_.debug_view_enabled = this->get_parameter("laser_mapping_node.debug_view").as_bool();
        config_.enable_ouster_data = this->get_parameter("laser_mapping_node.enable_ouster_data").as_bool();
        config_.publish_only_feature_points = this->get_parameter("laser_mapping_node.publish_only_feature_points").as_bool();
        // config_.use_imu_roll_pitch = this->get_parameter("laser_mapping_node.use_imu_roll_pitch").as_bool();
        config_.max_surface_features = this->get_parameter("laser_mapping_node.max_surface_features").as_int();
        config_.velocity_failure_threshold = this->get_parameter("laser_mapping_node.velocity_failure_threshold").as_double();
        config_.auto_voxel_size = this->get_parameter("laser_mapping_node.auto_voxel_size").as_bool();
        config_.forget_far_chunks = this->get_parameter("laser_mapping_node.forget_far_chunks").as_bool();
        config_.visual_confidence_factor = this->get_parameter("laser_mapping_node.visual_confidence_factor").as_double();
        config_.map_dir = this->get_parameter("map_dir").as_string(); 
        config_.localization_mode = this->get_parameter("laser_mapping_node.localization_mode").as_bool();
        config_.read_pose_file = this->get_parameter("laser_mapping_node.read_pose_file").as_bool();
        config_.use_imu_roll_pitch = USE_IMU_ROLL_PITCH;

        if(config_.read_pose_file)
        {   
            std::vector<utils::OdometryData> odometryResults;
            utils::readLocalizationPose(config_.map_dir, odometryResults);
            config_.init_x= odometryResults[0].x;
            config_.init_y= odometryResults[0].y;
            config_.init_z= odometryResults[0].z;
            config_.init_roll= odometryResults[0].roll;
            config_.init_pitch= odometryResults[0].pitch;
            config_.init_yaw= odometryResults[0].yaw;
        }
        else
        {  
            config_.init_x = get_parameter("laser_mapping_node.init_x").as_double(); 
            config_.init_y = get_parameter("laser_mapping_node.init_y").as_double(); 
            config_.init_z = get_parameter("laser_mapping_node.init_z").as_double(); 
            config_.init_roll = get_parameter("laser_mapping_node.init_roll").as_double();
            config_.init_pitch = get_parameter("laser_mapping_node.init_pitch").as_double();
            config_.init_yaw = get_parameter("laser_mapping_node.init_yaw").as_double(); 
        }

        return true;
    }
    
   

    // Callback for the LaserFeature message from the featureExtraction node.
    // One message carries everything for a single scan (edge features, planar
    // features, deskewed full cloud, IMU orientation), so pushing each part
    // onto its own queue under one lock keeps the queue fronts synchronized:
    // process() can later pop one element from each and know they belong to
    // the same scan. No processing happens here to keep the callback fast.
    void laserMapping::laserFeatureInfoHandler(const super_odometry_msgs::msg::LaserFeature::SharedPtr msgIn) {
       
        mBuf.lock();
        cornerLastBuf.push(msgIn->cloud_corner);
        surfLastBuf.push(msgIn->cloud_surface);
        realsenseBuf.push(msgIn->cloud_realsense);
        fullResBuf.push(msgIn->cloud_nodistortion);
        Eigen::Quaterniond q_world_lidar_prediction(
            msgIn->initial_quaternion_w, msgIn->initial_quaternion_x,
            msgIn->initial_quaternion_y, msgIn->initial_quaternion_z);

        q_world_lidar_prediction_buf.push(q_world_lidar_prediction);
        mBuf.unlock();
    }


// Computes the initial pose guess for the current scan. The scan-to-map
// optimization is a local method: it only converges to the right answer if
// it starts close to it, so a good prediction matters, especially under
// fast motion. Three regimes:
void laserMapping::setInitialGuess()
{
  //Case1: First Frame Initialization 
  if(!initialization){
    initializeFirstFrame();
    return;
  }
  //Case2: Startup period -continue using IMU for stability 
  if(startupCount>0){
    initializeWithIMU();
    startupCount--;
    return; 
  }
  //Case3: Normal operation -select prediction source 
  selectPosePrediction();
}

// First scan ever: there is no map yet, so this pose DEFINES the world
// frame. Using the IMU's roll/pitch (with yaw zeroed, since a gyro cannot
// observe heading) makes the world frame gravity-aligned: z points up and
// the ground plane in the map is horizontal.
void laserMapping::initializeFirstFrame(){

    //Get initial orientation from IMU prediction 
    if(sensorMeas.q_world_lidar_prediction.w()!=0){   //Have IMU data
        //Extract roll and pitch, zero out yaw 
        tf2::Quaternion q_world_lidar_roll_pitch =
        // The world in sensorMeas.q_world_lidar_prediction is the first IMU frame of initializaiton.
            utils::extractRollPitch(sensorMeas.q_world_lidar_prediction);
        q_world_lidar =
            Eigen::Quaterniond(q_world_lidar_roll_pitch.w(),
                               q_world_lidar_roll_pitch.x(),
                               q_world_lidar_roll_pitch.y(),
                               q_world_lidar_roll_pitch.z());
        // Feature extraction already publishes the physical lidar attitude,
        // q_world_lidar = q_world_imu * q_imu_lidar. Do not apply the
        // extrinsic a second time here.
    }else{

        q_world_lidar=Eigen::Quaterniond(1,0,0,0); //If no IMU data, use identity rotation

    }

    //initialize position 
    q_world_lidar_prediction_prev=q_world_lidar;
    T_world_lidar.rot=q_world_lidar;
    T_world_lidar.pos=Eigen::Vector3d::Zero();

    //Overide with predefined pose if localization mode 
    if(slam.localization_mode){
        T_world_lidar.pos=Eigen::Vector3d(slam.init_x,slam.init_y,slam.init_z);
        tf2::Quaternion q_world_lidar_localization;
        q_world_lidar_localization.setRPY(
            slam.init_roll, slam.init_pitch, slam.init_yaw);
        T_world_lidar.rot =
            Eigen::Quaterniond(q_world_lidar_localization.w(),
                               q_world_lidar_localization.x(),
                               q_world_lidar_localization.y(),
                               q_world_lidar_localization.z());
        slam.T_world_lidar_prev=T_world_lidar;
    }

}

// Scans 2..~11 (the startup period): the map is still tiny and the
// registration is unreliable, so trust the IMU attitude outright and assume
// the robot has not moved. This keeps the first map chunks consistent.
void laserMapping::initializeWithIMU(){
    if(sensorMeas.q_world_lidar_prediction.w()!=0){  //Have IMU data
    //Use IMU Orientation directly during startup for seconds 
    tf2::Quaternion q_world_lidar_prediction(
        sensorMeas.q_world_lidar_prediction.w(),
        sensorMeas.q_world_lidar_prediction.x(),
        sensorMeas.q_world_lidar_prediction.y(),
        sensorMeas.q_world_lidar_prediction.z());
    
    //Keep position from last frame 
    t_world_lidar=T_world_lidar_prev.pos;
    T_world_lidar.pos=t_world_lidar;

    //Update rotation 
    q_world_lidar =
        Eigen::Quaterniond(q_world_lidar_prediction.w(),
                           q_world_lidar_prediction.x(),
                           q_world_lidar_prediction.y(),
                           q_world_lidar_prediction.z());
    T_world_lidar.rot=q_world_lidar;


    }else
    {
      //No IMU data, use last rotation 
      q_world_lidar=T_world_lidar_prev.rot;
      t_world_lidar=T_world_lidar_prev.pos;
      T_world_lidar=T_world_lidar_prev;

    } 
}

// Normal operation: propagate the last optimized pose T_world_lidar forward to
// the current scan time using the best available motion prediction. The
// odometry sources (LIO/VIO/NIO) provide a RELATIVE transform between the
// previous and current scan times, which is right-multiplied onto the pose
// (i.e. composed in the body frame).
void laserMapping::selectPosePrediction(){

// Step1: Decide prediction source based on system state 
prediction_source=determinePredictionSource();

//Step2: Get prediction from selected source 
switch(prediction_source){
    case PredictionSource::LIO_ODOM:{
    T_world_lidar =
        T_world_lidar * sensorMeas.T_lidar_prev_lidar_current_lio;
    break;
    } 
   
    case PredictionSource::VIO_ODOM:{
    T_world_lidar =
        T_world_lidar * sensorMeas.T_lidar_prev_lidar_current_vio;
    break; 
    } 

    case PredictionSource::NEURAL_IMU_ODOM:{
    T_world_lidar =
        T_world_lidar * sensorMeas.T_lidar_prev_lidar_current_neural;
    break; 
    } 
    case PredictionSource::IMU_ORIENTATION:{
    // Only orientation is predicted: apply the IMU's rotation change since
    // the previous scan (q_world_lidar_prediction_prev^-1 *
    // q_world_lidar_prediction_current) to the current
    // world orientation. Position is left as-is.
    Eigen::Quaterniond q_world_lidar_predicted =
        q_world_lidar * q_world_lidar_prediction_prev.inverse() *
        q_world_lidar_prediction_current;
    q_world_lidar_predicted.normalize();
    T_world_lidar.rot=q_world_lidar_predicted;
    q_world_lidar_prediction_prev=q_world_lidar_prediction_current;
    break;
    } 
   
    case PredictionSource::CONSTANT_VELOCITY:{  
    // Assume the same motion as between the last two poses ("the robot
    // keeps doing what it was doing").
    Transformd T_lidar_prev_lidar_current =
        T_world_lidar_prev.inverse()*T_world_lidar;
    T_world_lidar=T_world_lidar*T_lidar_prev_lidar_current;
    break; 
    }
}

//Step4: Update current pose (mirror T_world_lidar into the Ceres parameter block)
q_world_lidar=T_world_lidar.rot;
t_world_lidar=T_world_lidar.pos;

}

// Picks the prediction source. "Degenerate" means the lidar geometry does
// not constrain all 6 degrees of freedom (e.g. a long featureless corridor
// leaves forward translation unobservable); in that case lidar-based
// odometry cannot be trusted and camera- or learning-based sources are
// preferred. Falls back to constant velocity if nothing else is available.
laserMapping::PredictionSource laserMapping::determinePredictionSource(){
// If system is degerenate, prefer VIO or learning imu odom

if(slam.isDegenerate){
    if(sensorMeas.vio_prediction_status){
        return PredictionSource::VIO_ODOM;
    }
    if(sensorMeas.nio_prediction_status){
        return PredictionSource::NEURAL_IMU_ODOM;
    }

}else{
    // If system is not degenerate, use IMU orientation 
    if(sensorMeas.lio_prediction_status){
        return PredictionSource::LIO_ODOM;
    }
    sensorMeas.imu_orientation_status =
        useIMUPrediction(sensorMeas.q_world_lidar_prediction);
    if(sensorMeas.imu_orientation_status){
        return PredictionSource::IMU_ORIENTATION;
    }
   
}



// If no prediction source is available, use constant velocity
return PredictionSource::CONSTANT_VELOCITY;

}

    // Publishes all per-scan outputs:
    //  - the name of the prediction source used (for debugging),
    //  - the 5x5-chunk local map around the robot (every 5th scan, debug only),
    //  - the whole local map and, in localization mode, the prior map
    //    (every 20th scan; these clouds are big),
    //  - the full deskewed scan transformed into the world frame
    //    ("registered_scan"),
    //  - the optimized pose as two odometry messages (laser_odometry, which
    //    imuPreintegration consumes, and an "incremental" variant),
    //  - the accumulated trajectory path for RViz,
    //  - optimization statistics (iteration counts, latency, ...).
    void laserMapping::publishTopic(){

        TicToc t_pub;
        std_msgs::msg::String prediction_source_msg;
        switch (prediction_source) {
            case PredictionSource::IMU_ORIENTATION :
                prediction_source_msg.data = "IMU Only Orientation Prediction";
                break;
            case PredictionSource::LIO_ODOM :
                prediction_source_msg.data = "Using Laser-Inertial Odometry (LIO)";
                break;
            case PredictionSource::VIO_ODOM :
                prediction_source_msg.data = "Using Visual-Inertial Odometry (VIO)";
                break;
            case PredictionSource::NEURAL_IMU_ODOM :
                prediction_source_msg.data = "Using Neural-Inertial Odometry (Neural-IMU)";
                break;
            case PredictionSource::CONSTANT_VELOCITY :
                prediction_source_msg.data = "Using Constant Velocity Prediction";
                break;
        }
        pubprediction_source->publish(prediction_source_msg);

        if (frameCount % 5 == 0 && config_.debug_view_enabled) {
            laserCloudSurround->clear();
            *laserCloudSurround = slam.localMap.get5x5LocalMap(slam.pos_in_localmap);
            sensor_msgs::msg::PointCloud2 laserCloudSurround3;
            pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
            laserCloudSurround3.header.stamp =
                    rclcpp::Time(timeLaserOdometry*1e9);
            laserCloudSurround3.header.frame_id = WORLD_FRAME;
            pubLaserCloudSurround->publish(laserCloudSurround3);
        }

        if (frameCount % 20 == 0) {
            pcl::PointCloud<PointType> laserCloudMap;
            laserCloudMap = slam.localMap.getAllLocalMap();
            sensor_msgs::msg::PointCloud2 laserCloudMsg;
            pcl::toROSMsg(laserCloudMap, laserCloudMsg);
            laserCloudMsg.header.stamp = rclcpp::Time(timeLaserOdometry*1e9);
            laserCloudMsg.header.frame_id = WORLD_FRAME;
            pubLaserCloudMap->publish(laserCloudMsg);
            
            if (slam.localization_mode) {
                priorCloudMsg.header.stamp = rclcpp::Time(timeLaserOdometry*1e9);
                pubLaserCloudPrior->publish(priorCloudMsg);
            }
        }

        // Transform the full scan from the lidar frame into the world frame
        // using the optimized pose ("registering" the scan). Points closer
        // than 0.1 m (squared distance < 0.01) are self-returns from the
        // robot body and are skipped.
        int laserCloudFullResNum = laserCloudFullRes->points.size();
        for (int i = 0; i < laserCloudFullResNum; i++) {
            PointType const *const &pi = &laserCloudFullRes->points[i];
            if (pi->x* pi->x+ pi->y * pi->y + pi->z* pi->z < 0.01)
            {
                continue;
            }

            utils::pointAssociateToMap(&laserCloudFullRes->points[i],
                                &laserCloudFullRes->points[i],
                                q_world_lidar,
                                t_world_lidar);
        }

        // Round-trip through a ROS message, then drop the near-origin points
        // that were skipped above, so the published cloud contains only
        // valid, world-frame points.
        pcl::PointCloud<pcl::PointXYZI> laserCloudFullResCvt, laserCloudFullResClean;
        sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
        pcl::fromROSMsg(laserCloudFullRes3, laserCloudFullResCvt);
        for (int i = 0; i < laserCloudFullResNum; i++) {
          PointType const *const &pi = &laserCloudFullResCvt.points[i];
          if (pi->x* pi->x+ pi->y * pi->y + pi->z* pi->z > 0.01)
          {
             laserCloudFullResClean.push_back(*pi);
          }
        }
        pcl::toROSMsg(laserCloudFullResClean, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = rclcpp::Time(timeLaserOdometry*1e9);
        laserCloudFullRes3.header.frame_id = WORLD_FRAME;
        pubLaserCloudFullRes->publish(laserCloudFullRes3);

        laserCloudFullResCvt.clear();
        laserCloudFullResClean.clear();
        laserCloudFullRes_rot->clear();
        laserCloudFullRes_rot->resize(laserCloudFullResNum);

        // Axis-permuted copy of the registered cloud (x,y,z) -> (y,z,x), for
        // consumers that use the camera-style z-forward convention.
        for (int i = 0; i < laserCloudFullResNum; i++) {
            laserCloudFullRes_rot->points[i].x = laserCloudFullRes->points[i].y;
            laserCloudFullRes_rot->points[i].y = laserCloudFullRes->points[i].z;
            laserCloudFullRes_rot->points[i].z = laserCloudFullRes->points[i].x;
            laserCloudFullRes_rot->points[i].intensity = laserCloudFullRes->points[i].intensity;
        }

        // Main odometry output: the optimized world-frame pose plus the
        // finite-difference body-frame velocities in the twist field. This
        // is what the imuPreintegration node fuses with the IMU.
        nav_msgs::msg::Odometry odomAftMapped;
        odomAftMapped.header.frame_id = WORLD_FRAME;
        odomAftMapped.child_frame_id = SENSOR_FRAME;
        odomAftMapped.header.stamp = rclcpp::Time(timeLaserOdometry*1e9);

        odomAftMapped.pose.pose.orientation.x = q_world_lidar.x();
        odomAftMapped.pose.pose.orientation.y = q_world_lidar.y();
        odomAftMapped.pose.pose.orientation.z = q_world_lidar.z();
        odomAftMapped.pose.pose.orientation.w = q_world_lidar.w();

        odomAftMapped.pose.pose.position.x = t_world_lidar.x();
        odomAftMapped.pose.pose.position.y = t_world_lidar.y();
        odomAftMapped.pose.pose.position.z = t_world_lidar.z();

        odomAftMapped.twist.twist.linear.x = vel_b.x();
        odomAftMapped.twist.twist.linear.y = vel_b.y();
        odomAftMapped.twist.twist.linear.z = vel_b.z();

        odomAftMapped.twist.twist.angular.x = ang_vel_b.x();
        odomAftMapped.twist.twist.angular.y = ang_vel_b.y();
        odomAftMapped.twist.twist.angular.z = ang_vel_b.z();

        // Secondary "incremental" odometry: same pose but published under a
        // separate topic, mirroring LIO-SAM's odometry_incremental (a stream
        // guaranteed to be smooth/continuous for downstream consumers).
        nav_msgs::msg::Odometry laserOdomIncremental;

        if (initialization == false)
        {
            laserOdomIncremental.header.stamp = rclcpp::Time(timeLaserOdometry*1e9);
            laserOdomIncremental.header.frame_id = WORLD_FRAME;
            laserOdomIncremental.child_frame_id =  SENSOR_FRAME;
            laserOdomIncremental.pose.pose.position.x = t_world_lidar.x();
            laserOdomIncremental.pose.pose.position.y = t_world_lidar.y();
            laserOdomIncremental.pose.pose.position.z = t_world_lidar.z();
            laserOdomIncremental.pose.pose.orientation.x = q_world_lidar.x();
            laserOdomIncremental.pose.pose.orientation.y = q_world_lidar.y();
            laserOdomIncremental.pose.pose.orientation.z = q_world_lidar.z();
            laserOdomIncremental.pose.pose.orientation.w = q_world_lidar.w();
        }
        else
        {

            T_world_lidar_incremental = T_world_lidar;
            T_world_lidar_incremental.rot.normalized();

            laserOdomIncremental.header.stamp = rclcpp::Time(timeLaserOdometry*1e9);
            laserOdomIncremental.header.frame_id = WORLD_FRAME;
            laserOdomIncremental.child_frame_id =  SENSOR_FRAME;
            laserOdomIncremental.pose.pose.position.x = T_world_lidar_incremental.pos.x();
            laserOdomIncremental.pose.pose.position.y = T_world_lidar_incremental.pos.y();
            laserOdomIncremental.pose.pose.position.z = T_world_lidar_incremental.pos.z();
            laserOdomIncremental.pose.pose.orientation.x = T_world_lidar_incremental.rot.x();
            laserOdomIncremental.pose.pose.orientation.y = T_world_lidar_incremental.rot.y();
            laserOdomIncremental.pose.pose.orientation.z = T_world_lidar_incremental.rot.z();
            laserOdomIncremental.pose.pose.orientation.w = T_world_lidar_incremental.rot.w();
        }

        pubLaserOdometryIncremental->publish(laserOdomIncremental);


        // Overload covariance[0] as a degeneracy flag (1 = the scan-to-map
        // problem was ill-conditioned) so consumers can lower their trust.
        if (slam.isDegenerate) {
            odomAftMapped.pose.covariance[0] = 1;
        } else {
            odomAftMapped.pose.covariance[0] = 0;
        }

        rclcpp::Time pub_time = rclcpp::Clock{RCL_ROS_TIME}.now(); //PARV_TODO - find how to syncrynoise this with rosbag time
        pubOdomAftMapped->publish(odomAftMapped);

        geometry_msgs::msg::PoseStamped laserAfterMappedPose;
        laserAfterMappedPose.header = odomAftMapped.header;
        laserAfterMappedPose.pose = odomAftMapped.pose.pose;
        laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
        laserAfterMappedPath.header.frame_id = WORLD_FRAME;
        laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
        pubLaserAfterMappedPath->publish(laserAfterMappedPath);


        // Optimization statistics: latency between the newest IMU odometry
        // and this mapping result, plus per-iteration convergence data.
        slam.stats.header = odomAftMapped.header;
        if (timeLatestImuOdometry.seconds() < 1.0)
        {
            timeLatestImuOdometry = pub_time;
        }
        rclcpp::Duration latency = timeLatestImuOdometry - pub_time;  
        slam.stats.latency = latency.seconds() * 1000;
        slam.stats.n_iterations = slam.stats.iterations.size();
        // Avoid breaking rqt_multiplot
        while (slam.stats.iterations.size() < 4)
        {
            slam.stats.iterations.push_back(super_odometry_msgs::msg::IterationStats());
        }

        pubOptimizationStats->publish(slam.stats);
        slam.stats.iterations.clear();
    }


    // Adapts the voxel-filter resolution to the scene, then downsamples the
    // feature clouds. Reasoning: indoors, points are close together and a
    // fine grid (0.1/0.2 m) keeps enough detail; outdoors, points are spread
    // out and a coarse grid (0.4/0.8 m) keeps the point count (and thus the
    // Ceres problem size) manageable. Downsampling also ensures features are
    // roughly evenly distributed so no region dominates the cost function.
    void laserMapping::adjustVoxelSize(){

        // Calculate cloud statistics
        bool increase_blind_radius = false;
        if(config_.auto_voxel_size)
        {
            Eigen::Vector3f average(0,0,0);
            int count_far_points = 0;
            for (auto &point : *laserCloudSurfLast)
            {
                average(0) += fabs(point.x);
                average(1) += fabs(point.y);
                average(2) += fabs(point.z);
                // Count points farther than 3 m (squared distance > 9 m^2).
                if(point.x*point.x + point.y*point.y + point.z*point.z>9){
                    count_far_points++;
                }
            }
            // With plenty of far points, close-range returns (often the robot
            // itself or dust) can be ignored by enlarging the blind radius.
            if (count_far_points > 3000)
            {
                increase_blind_radius = true;
            }

            // Scene-size proxy: product of the mean absolute x, y and z
            // coordinates (roughly the volume the scan spans). < 25 means a
            // tight indoor space, > 65 a wide outdoor space.
            average /= laserCloudSurfLast->points.size();
            slam.stats.average_distance = average(0)*average(1)*average(2);
            if (slam.stats.average_distance < 25)
            {
                config_.lineRes = 0.1;
                config_.planeRes = 0.2;
            }
            else if (slam.stats.average_distance > 65)
            {
                config_.lineRes = 0.4;
                config_.planeRes = 0.8;
            }
            downSizeFilterSurf.setLeafSize(config_.planeRes , config_.planeRes , config_.planeRes );
            downSizeFilterCorner.setLeafSize(config_.lineRes , config_.lineRes , config_.lineRes );
        }

        laserCloudCornerStack->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerStack);


        laserCloudSurfStack->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfStack);
      

        slam.localMap.lineRes_ = config_.lineRes;
        slam.localMap.planeRes_ = config_.planeRes;

    }

    
    // True once every buffer holds at least one message, i.e. a complete
    // scan bundle is ready to be processed.
    bool  laserMapping::checkDataAvailable() const{
        return !cornerLastBuf.empty() && !surfLastBuf.empty()
               && !fullResBuf.empty()
               && !q_world_lidar_prediction_buf.empty();
        //Note: in pure laser odometry, IMU Prediction will be identy. 
    }

    // Pops the front element of each input buffer (they are pushed together,
    // so the fronts belong to the same scan) and converts the ROS clouds to
    // PCL. Caller must hold mBuf.
    laserMapping::SensorData laserMapping::extractSensorData(){
        
        SensorData data;
        //1. Extract timestamp
        data.timestamp=secs(&fullResBuf.front());
        timeLaserOdometry=data.timestamp;

        //2. Extract point cloud data 
        pcl::fromROSMsg(cornerLastBuf.front(), *laserCloudCornerLast);
        cornerLastBuf.pop();
        pcl::fromROSMsg(surfLastBuf.front(), *laserCloudSurfLast);
        surfLastBuf.pop();
        pcl::fromROSMsg(fullResBuf.front(), *laserCloudFullRes);
        fullResBuf.pop();

        //3. Extract IMU prediction 
        data.q_world_lidar_prediction =
            q_world_lidar_prediction_buf.front();
        data.q_world_lidar_prediction.normalize();
        q_world_lidar_prediction_buf.pop();

        //4 set status for prediction source (TODO: didn't release code other prediction source yet) 
        data.vio_prediction_status=false;
        data.lio_prediction_status=false;
        data.nio_prediction_status=false;
        data.imu_orientation_status=false;

        return data;
    }

    // Empties all input buffers. Called right after extractSensorData: if the
    // optimizer is slower than the sensor, older scans are deliberately
    // dropped so the node always works on the latest scan instead of falling
    // further and further behind real time.
    void laserMapping::clearSensorData(){
        auto clearBuffer=[](auto&buffer){
            while(!buffer.empty()){
                buffer.pop();
            }
        };
        clearBuffer(cornerLastBuf);
        clearBuffer(surfLastBuf);
        clearBuffer(fullResBuf);
        clearBuffer(q_world_lidar_prediction_buf);
    }


    // Runs the actual scan-to-map registration. Optionally passes the IMU's
    // roll/pitch to the SLAM core so those two angles can be constrained to
    // gravity during optimization (an IMU observes roll/pitch absolutely,
    // whereas the lidar only observes them relative to the map). Then calls
    // slam.Localization(), which finds point-to-line / point-to-plane
    // correspondences in the local map, builds the Ceres problem, solves for
    // T_world_lidar, and merges the scan into the map.
    void laserMapping::performSLAMOptimization(){
        tf2::Quaternion q_world_lidar_roll_pitch;
        if(config_.use_imu_roll_pitch){  // TODO: Livox mid360 not use roll pitch angle
            slam.OptSet.use_imu_roll_pitch=true;
            q_world_lidar_roll_pitch =
                utils::extractRollPitch(sensorMeas.q_world_lidar_prediction);
            slam.OptSet.q_world_lidar_roll_pitch =
                q_world_lidar_roll_pitch;
        }else{
            slam.OptSet.use_imu_roll_pitch=false;
            slam.OptSet.q_world_lidar_roll_pitch =
                tf2::Quaternion(0,0,0,1);
        }

        slam.Localization(initialization, static_cast<LidarSLAM::PredictionSource>(prediction_source), T_world_lidar,
                 laserCloudCornerStack, laserCloudSurfStack, timeLaserOdometry);
    }


    // A zero quaternion (w == 0) marks "no IMU data" upstream; a valid
    // rotation always has w != 0 here. Stores the orientation as the current
    // odometry-frame prediction when valid.
    bool laserMapping::useIMUPrediction(
        const Eigen::Quaterniond& q_world_lidar_prediction) {
        if (q_world_lidar_prediction.w()!=0)
        {
            q_world_lidar_prediction_current =
                q_world_lidar_prediction;
            q_world_lidar_prediction_current.normalize();
            return true;
        }
        else
        {
            return false;
        }
    }

    // Accepts the optimized pose from the SLAM core, derives velocities, and
    // publishes everything for this scan.
    void laserMapping::updatePoseAndPublish(){

        //1. Update pose 
        q_world_lidar=slam.T_world_lidar.rot;
        t_world_lidar=slam.T_world_lidar.pos;
        T_world_lidar.rot=slam.T_world_lidar.rot;
        T_world_lidar.pos=slam.T_world_lidar.pos;
        startupCount=slam.startupCount;
        frameCount++;
        slam.frame_count=frameCount;
        slam.laser_imu_sync=laser_imu_sync;
        initialization = true;

        // Calculate linear and angular velocity by finite-differencing the
        // last two optimized poses over the scan interval dt (~0.1 s).
        double dt = timeLaserOdometry - timeLaserOdometryPrev;

        if (dt > 1e-6) {  
            // World-frame velocity, rotated into the body frame (odometry
            // twist is conventionally expressed in the child/body frame).
            Eigen::Vector3d v_world =
                (t_world_lidar - T_world_lidar_prev.pos) / dt;
            vel_b = q_world_lidar.inverse() * v_world;

            // Let R_prev and R_curr map lidar vectors into the world frame.
            // The left/inertial increment is defined by
            //   R_curr = delta_R_world * R_prev,
            // therefore
            //   delta_R_world = R_curr * R_prev.inverse().
            // This is an active rotation whose axis is expressed in world
            // coordinates; it is not another world<-lidar frame transform.
            Eigen::Quaterniond q_rotation_increment_world =
                q_world_lidar * T_world_lidar_prev.rot.inverse();
            Eigen::AngleAxisd angle_axis(q_rotation_increment_world);
            Eigen::Vector3d ang_vel_world =
                angle_axis.axis() * angle_axis.angle() / dt;

            // Odometry twist is expressed in the current child (lidar) frame.
            ang_vel_b = q_world_lidar.inverse() * ang_vel_world;
        } else {
            vel_b = Eigen::Vector3d::Zero();
            ang_vel_b = Eigen::Vector3d::Zero();
        }

        //2. Publish results 
        publishTopic();

        //3. Store current pose and time for next iteration
        T_world_lidar_prev = slam.T_world_lidar;
        timeLaserOdometryPrev = timeLaserOdometry;
    }

    // Main loop. Started by the wall timer but never returns: it keeps one
    // executor thread busy polling for new scan bundles and running the
    // guess -> downsample -> optimize -> publish pipeline on each of them.
    void laserMapping::process() {

        while (rclcpp::ok()) {
            if(!checkDataAvailable()){
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            try{
                utils::ScopedTimer timer("Frame Processing");
                mBuf.lock(); 
                sensorMeas=extractSensorData();
                clearSensorData();
                mBuf.unlock();
                setInitialGuess();
                adjustVoxelSize();
                performSLAMOptimization();
                updatePoseAndPublish();
               
                //updateStatsAndDebugInfo();

            }catch(const std::exception&e){
                RCLCPP_ERROR(this->get_logger(), "Error in frame processing: %s", e.what());
            }
        }

    }


#pragma clang diagnostic pop

} // namespace super_odometry
