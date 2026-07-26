//
// Created by shibo zhao on 2020-09-27.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// Implementation of the feature-extraction front end (see the header for the
// big picture). The flow for every lidar scan is:
//
//   laserCloudHandler / livoxHandler  (sensor-specific ingestion)
//     -> convert points to the common XYZ + intensity + time + ring format
//     -> manageLidarBuffer            (queue the scan)
//     -> undistortionAndFeatureExtraction
//          -> synchronize_measurements (do IMU/VIO samples cover the scan?)
//          -> removePointDistortion    (deskew: move every point into the
//                                       sensor frame at scan start)
//          -> extractFeatures          (select surface points)
//          -> publishTopic             (bundle + publish LaserFeature)
//
// In parallel, imu_Handler runs at IMU rate (~200 Hz): it integrates the
// gyro into a rolling orientation estimate q_world_imu for each sample (that is
// what the deskewing interpolates between) and runs the one-time IMU
// initialization that estimates sensor statistics, gravity, and initial tilt.
// ============================================================================

#include <super_odometry/FeatureExtraction/featureExtraction.h>
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


namespace super_odometry {
    
    featureExtraction::featureExtraction(const rclcpp::NodeOptions & options)
    : Node("feature_extraction_node", options) {
    }

    // One-time setup: parameters, calibration, subscriptions and publishers.
    // Called once from main() after construction (it needs shared_from_this(),
    // which is not available inside the constructor).
    void featureExtraction::initInterface() {      
        // A Reentrant callback group lets the lidar, IMU and odometry
        // callbacks run concurrently on the multi-threaded executor; they
        // synchronize on the m_buf mutex where needed.
        cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = cb_group_;

        // BEST_EFFORT = do not retry lost messages; for high-rate sensor
        // streams a fresh sample is more useful than a re-sent old one.
        rclcpp::QoS imu_qos(10);
        imu_qos.best_effort();  // Use BEST_EFFORT reliability
        imu_qos.keep_last(10);  // Keep last 10 messages

        rclcpp::QoS laser_qos(10);
        laser_qos.best_effort();  // Use BEST_EFFORT reliability
        laser_qos.keep_last(2);   // Keep only the last 2 scans

        if(!readGlobalparam(shared_from_this()))
        {
            RCLCPP_ERROR(this->get_logger(), "[super_odometry::featureExtraction] Could not read calibration. Exiting...");
            rclcpp::shutdown();
        }
        if (!readParameters())
        {
            RCLCPP_ERROR(this->get_logger(), "[super_odometry::featureExtraction] Could not read parameters. Exiting...");
            rclcpp::shutdown();
        }
        RCLCPP_INFO(this->get_logger(), "calibration");
        if (!readCalibration(shared_from_this()))
        {
            RCLCPP_ERROR(this->get_logger(), "[super_odometry::featureExtraction] Could not read parameters. Exiting...");
            rclcpp::shutdown();
        }
         
        RCLCPP_WARN(this->get_logger(), "config_.skipFrame: %d", config_.skipFrame);
        RCLCPP_INFO(this->get_logger(), "scan line number %d \n", config_.N_SCANS);      
        RCLCPP_INFO(this->get_logger(), "use imu roll and pitch %d \n", config_.use_imu_roll_pitch);

        if (config_.N_SCANS != 16 && config_.N_SCANS != 32 && config_.N_SCANS != 64 && config_.N_SCANS != 4 && config_.N_SCANS != 128)
        {
            RCLCPP_ERROR(this->get_logger(), "only support velodyne, livox, ouster with 16, 32, 64 or 128 scan line! and livox mid 360");
            rclcpp::shutdown();
        }

        // Velodyne and Ouster publish standard PointCloud2 messages; Livox
        // uses its own CustomMsg format (per-point offset_time, line, tag),
        // so the subscription type depends on the configured sensor.
        if (config_.sensor == SensorType::VELODYNE || config_.sensor == SensorType::OUSTER) {
            subLaserCloud = this->create_subscription<sensor_msgs::msg::PointCloud2>(LASER_TOPIC, laser_qos, 
                    std::bind(&featureExtraction::laserCloudHandler, this,
                    std::placeholders::_1), sub_options);
        } else if (config_.sensor == SensorType::LIVOX) {
            subLivoxCloud = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(LASER_TOPIC, 20, 
                    std::bind(&featureExtraction::livoxHandler, this,
                    std::placeholders::_1), sub_options);
        } //TODO: add this to config

        subImu = this->create_subscription<sensor_msgs::msg::Imu>(
            IMU_TOPIC, imu_qos, 
            std::bind(&featureExtraction::imu_Handler, this,
                        std::placeholders::_1), sub_options);

        subOdom = this->create_subscription<nav_msgs::msg::Odometry>(
            ODOM_TOPIC, 10, 
            std::bind(&featureExtraction::visual_odom_Handler, this,
                        std::placeholders::_1), sub_options);

        // Output topics. The individual clouds are mainly for visualization;
        // laserMapping consumes the combined "feature_info" message, which
        // embeds copies of all of them plus the initial attitude.
        pubLaserCloud = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            ProjectName+"/velodyne_cloud_2", 2);

        pubLaserFeatureInfo = this->create_publisher<super_odometry_msgs::msg::LaserFeature>(
            ProjectName+"/feature_info", 2);

        pubBobPoints = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            ProjectName+"/bob_points", 2);

        pubPlannerPoints = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            ProjectName+"/planner_points", 2);

        pubEdgePoints = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            ProjectName+"/edge_points", 2);

        // Optional debug mode: publish each laser ring as a separate topic.
        if (PUB_EACH_LINE)
        {
            for (int i = 0; i < config_.N_SCANS; i++)
            {
                auto tmp_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                    "laser_scanid_" + std::to_string(i), 2);
                pubEachScan.push_back(tmp_publisher_);
            }
        }

        delay_count_ = 0;
        m_imuPeriod = 1.0/imu_Init->imu_frequency;
    }

    // Declares and reads the node's ROS parameters into config_. Note that a
    // few values (use_imu_roll_pitch, the acceleration limits) are then
    // overwritten by the globals loaded from the shared calibration file, so
    // the calibration wins over the per-node parameter file for those.
    bool featureExtraction::readParameters()
    {          

        this->declare_parameter<int>("feature_extraction_node.scan_line", 4);
        this->declare_parameter<int>("feature_extraction_node.mapping_skip_frame", 1);
        this->declare_parameter<double>("feature_extraction_node.blindFront", 0.1);
        this->declare_parameter<double>("feature_extraction_node.blindBack", -1.0);
        this->declare_parameter<double>("feature_extraction_node.blindLeft", 0.1);
        this->declare_parameter<double>("feature_extraction_node.blindRight", -0.1);
        this->declare_parameter<bool>("feature_extraction_node.use_dynamic_mask", false);
        this->declare_parameter<bool>("feature_extraction_node.use_imu_roll_pitch", false);
        this->declare_parameter<float>("feature_extraction_node.min_range", 0.2);
        this->declare_parameter<float>("feature_extraction_node.max_range", 130.0);
        this->declare_parameter<int>("feature_extraction_node.filter_point_size", 3);
        this->declare_parameter<int>("feature_extraction_node.provide_point_time", 1);
        this->declare_parameter<bool>("feature_extraction_node.debug_view", false);
        this->declare_parameter<double>("feature_extraction_node.imu_acc_x_limit", 1.0);
        this->declare_parameter<double>("feature_extraction_node.imu_acc_y_limit", 1.0);
        this->declare_parameter<double>("feature_extraction_node.imu_acc_z_limit", 1.0);
        this->declare_parameter<std::string>("feature_extraction_node.sensor", "livox");

                
        config_.N_SCANS = this->get_parameter("feature_extraction_node.scan_line").as_int();
        config_.skipFrame = this->get_parameter("feature_extraction_node.mapping_skip_frame").as_int();
        config_.box_size.blindFront = this->get_parameter("feature_extraction_node.blindFront").as_double();
        config_.box_size.blindBack = this->get_parameter("feature_extraction_node.blindBack").as_double();
        config_.box_size.blindLeft = this->get_parameter("feature_extraction_node.blindLeft").as_double();
        config_.box_size.blindRight = this->get_parameter("feature_extraction_node.blindRight").as_double();
        // config_.use_imu_roll_pitch = this->get_parameter("feature_extraction_node.use_imu_roll_pitch").as_bool();
        config_.min_range = this->get_parameter("feature_extraction_node.min_range").as_double();
        config_.max_range = this->get_parameter("feature_extraction_node.max_range").as_double();
        config_.filter_point_size = this->get_parameter("feature_extraction_node.filter_point_size").as_int();
        config_.provide_point_time = this->get_parameter("feature_extraction_node.provide_point_time").as_int();
        config_.use_dynamic_mask = this->get_parameter("feature_extraction_node.use_dynamic_mask").as_bool(); 
        config_.debug_view_enabled = this->get_parameter("feature_extraction_node.debug_view").as_bool();
        config_.imu_acc_x_limit = this->get_parameter("feature_extraction_node.imu_acc_x_limit").as_double();
        config_.imu_acc_y_limit = this->get_parameter("feature_extraction_node.imu_acc_y_limit").as_double();
        config_.imu_acc_z_limit = this->get_parameter("feature_extraction_node.imu_acc_z_limit").as_double();
        config_.use_imu_roll_pitch = USE_IMU_ROLL_PITCH;
        config_.imu_acc_x_limit = IMU_ACC_X_LIMIT;
        config_.imu_acc_y_limit = IMU_ACC_Y_LIMIT;
        config_.imu_acc_z_limit = IMU_ACC_Z_LIMIT;

        if (SENSOR == "livox") {
            config_.sensor = SensorType::LIVOX;
        } else if (SENSOR == "velodyne") {
            config_.sensor = SensorType::VELODYNE;
        } else if (SENSOR == "ouster") {
            config_.sensor = SensorType::OUSTER;
        } 
        return true;
    }


    // Decides whether the pose source (IMU or VIO buffer) covers the oldest
    // buffered scan in time. Undistortion interpolates a pose at every point's
    // timestamp, so we need at least one measurement BEFORE the first point
    // and one AFTER the last point; otherwise we would have to extrapolate,
    // which is unreliable. Returns true only when interpolation is safe.
    template <typename Meas>
    bool featureExtraction::synchronize_measurements(MapRingBuffer<Meas> &measureBuf,
                                                     MapRingBuffer<pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr> &lidarBuf)
    {

        if (lidarBuf.getSize() == 0 or measureBuf.getSize() == 0)
            return false;

        double lidar_start_time;
        lidarBuf.getFirstTime(lidar_start_time);

        pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr lidar_msg;
        lidarBuf.getFirstMeas(lidar_msg);

        // Point times are stored relative to scan start, so the last point's
        // 'time' field is the scan duration (~0.1 s).
        double lidar_end_time = lidar_start_time + lidar_msg->back().time;

        // obtain the current imu message
        double meas_start_time=0;
        measureBuf.getFirstTime(meas_start_time);

        double meas_end_time=0;
        measureBuf.getLastTime(meas_end_time);

        if (meas_end_time <= lidar_end_time) // make sure imu message arrives after lidar message
        {
            RCLCPP_WARN_STREAM(this->get_logger(), "meas_end_time < lidar_end_time ||"
                            " message order is not perfect! please restart velodyne and imu driver!");
            RCLCPP_WARN(this->get_logger(), "meas_end_time %f <  %f lidar_end_time", meas_end_time, lidar_end_time);
            RCLCPP_WARN(this->get_logger(), "All the lidar data is more recent than all the imu data. Will throw away lidar frame");

            return false;
        }

        // No measurement older than the scan start: cannot interpolate the
        // start pose. Drop this scan (normal while buffers fill up at launch).
        if (meas_start_time >= lidar_start_time)          
        {
            RCLCPP_WARN(this->get_logger(), "throw laser scan, only should happen at the beginning");
            lidarBuf.clean(lidar_start_time);
            RCLCPP_WARN(this->get_logger(), "removed the lidarBuf size % d, measureBuf size % d ", lidarBuf.getSize(), measureBuf.getSize());
            RCLCPP_WARN(this->get_logger(), "meas_start_time: %f > lidar_start_time: %f ", meas_start_time, lidar_start_time);
            return false;
        }
        else
        {
            return true;
        }
        
    }


    

    // Core undistortion ("deskewing") routine.
    //
    // Problem: a spinning lidar measures points one at a time over ~100 ms.
    // Each point's coordinates are expressed in the sensor frame at that
    // point's OWN capture instant. If the robot moved during the sweep, the
    // raw cloud is smeared ("distorted"). We fix this by moving every point
    // into a single common frame: the sensor frame at the scan START time.
    //
    // Ingredients: a buffer of timestamped poses (either IMU orientations,
    // rotation only, or full 6-DOF VIO poses), interpolated at each point's
    // timestamp. For a point captured at time t:
    //
    //   T_body_start_body_current =
    //       T_world_body_start^-1 * T_world_body_current(t)
    //
    // is the motion of the tracked body between scan start and time t.
    // Applying it to the point re-expresses the point in the scan-start
    // frame. (Rotation-only IMU deskewing ignores translation, which is fine
    // for short 0.1 s sweeps at moderate speeds.)
    template<typename BufferType>
    void featureExtraction::removePointDistortion(
        double lidar_start_time, 
        double lidar_end_time,
        MapRingBuffer<BufferType> &buffer,
        pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr &lidar_msg)
    {
        // Step 1: Define how to extract pose based on buffer type
        auto extractPose = [](const BufferType& data) -> Transformd {
            Transformd T_world_body;
            if constexpr (std::is_same_v<BufferType, Imu::Ptr>) {
                // For IMU data: only rotation, zero translation. q_world_imu is the
                // gyro-integrated attitude computed in updateImuOrientation().
                T_world_body.rot = data->q_world_imu;
                T_world_body.pos = Eigen::Vector3d::Zero();
            } else {
                // For VIO data: both rotation and translation
                T_world_body.rot = Eigen::Quaterniond(
                    data->pose.pose.orientation.w,
                    data->pose.pose.orientation.x,
                    data->pose.pose.orientation.y,
                    data->pose.pose.orientation.z
                );
                T_world_body.pos = Eigen::Vector3d(
                    data->pose.pose.position.x,
                    data->pose.pose.position.y,
                    data->pose.pose.position.z
                );
            }
            return T_world_body;
        };

        
        // Step 2: Pose lookup with interpolation. Finds the two buffered
        // measurements that bracket the timestamp and blends them: slerp
        // (spherical linear interpolation) for the rotation, plain linear
        // interpolation for the translation. synchronize_measurements()
        // guaranteed that such a bracket exists for every point time.
        auto getInterpolatedPoseAtTime = [&buffer, &extractPose](double timestamp) -> Transformd {
        auto after_ptr = buffer.measMap_.upper_bound(timestamp);
        if (after_ptr->first < 0.0001) {
            after_ptr = buffer.measMap_.begin();
        }

        if (after_ptr == buffer.measMap_.begin()) {
            return extractPose(after_ptr->second);
        }

        auto before_ptr = std::prev(after_ptr);
        double ratio = (timestamp - before_ptr->first) / 
                      (after_ptr->first - before_ptr->first);

        Transformd T_world_body_before = extractPose(before_ptr->second);
        Transformd T_world_body_after = extractPose(after_ptr->second);

        Transformd T_world_body_interpolated;
        T_world_body_interpolated.rot =
            T_world_body_before.rot.slerp(ratio, T_world_body_after.rot);
        T_world_body_interpolated.pos =
            (1 - ratio) * T_world_body_before.pos +
            ratio * T_world_body_after.pos;
        return T_world_body_interpolated;
    };

    // Step 3: Pose of the tracked body (IMU or VIO body) at scan start.
    Transformd T_world_body_start =
        getInterpolatedPoseAtTime(lidar_start_time);

    // Step 4: Convert that body pose into the LIDAR pose at scan start.
    // For IMU poses we append T_imu_lidar; VIO poses are
    // already reported for the lidar/body convention used downstream.
    // This lidar pose is exported (q_world_lidar_start /
    // t_world_lidar_start) and
    // shipped in the LaserFeature message as the initial orientation guess
    // for laserMapping's scan-to-map registration.
    bool is_imu_data = std::is_same_v<BufferType, Imu::Ptr>;
    Transformd T_world_lidar_start = is_imu_data ?
                                    T_world_body_start * T_imu_lidar :
                                    T_world_body_start;

    q_world_lidar_start = T_world_lidar_start.rot;
    t_world_lidar_start = T_world_lidar_start.pos;

    // Step 5: Deskew each point.
    for (auto &point : lidar_msg->points) {
        // Lidar drivers emit NaN/inf coordinates for missed returns.
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }

        // point.time is relative to scan start (seconds).
        double point_time = point.time + lidar_start_time;
        Transformd T_world_body_point =
            getInterpolatedPoseAtTime(point_time);
        
        // Relative motion of the tracked body between scan start and this
        // point's capture time.
        Transformd T_world_body_current(
            T_world_body_point.rot, T_world_body_point.pos);
        Transformd T_body_start_body_current =
            T_world_body_start.inverse() * T_world_body_current;
        // The point lives in the LIDAR frame, but for IMU data the relative
        // motion above is expressed in the IMU frame. Conjugating with the
        // extrinsics converts it into a lidar-frame motion:
        //   T_lidar_start_lidar_current =
        //       T_lidar_imu * T_body_start_body_current * T_imu_lidar
        // reads right-to-left as: lidar -> IMU, apply IMU motion, IMU -> lidar.
        // (All shipped calibrations have identity rotation, so in practice
        // this is close to a no-op conjugation.)
        Transformd T_lidar_start_lidar_current = is_imu_data ?
            T_lidar_imu * T_body_start_body_current * T_imu_lidar :
            T_body_start_body_current;

        Eigen::Vector3d pt(point.x, point.y, point.z);
        pt = T_lidar_start_lidar_current * pt;
            
        point.x = pt.x();
        point.y = pt.y();
        point.z = pt.z();
        }
    }

    

    // Standalone version of the pose interpolation used above (same math:
    // slerp on rotation, lerp on translation between the two bracketing
    // measurements). Kept as a reusable helper.
    template<typename BufferType>
    Transformd featureExtraction::getInterpolatedPose(
        double timestamp, 
        MapRingBuffer<BufferType> &buffer,
        const std::function<Transformd(const BufferType&)>& extractPose)
    {
        auto after_ptr = buffer.measMap_.upper_bound(timestamp);
        if (after_ptr->first < 0.0001) {
            after_ptr = buffer.measMap_.begin();
        }

        if (after_ptr == buffer.measMap_.begin()) {
            return extractPose(after_ptr->second);
        }

        auto before_ptr = std::prev(after_ptr);
        double ratio = (timestamp - before_ptr->first) / 
                    (after_ptr->first - before_ptr->first);

        Transformd T_world_body_before = extractPose(before_ptr->second);
        Transformd T_world_body_after = extractPose(after_ptr->second);

        Transformd T_world_body_interpolated;
        T_world_body_interpolated.rot =
            T_world_body_before.rot.slerp(ratio, T_world_body_after.rot);
        T_world_body_interpolated.pos =
            (1 - ratio) * T_world_body_before.pos +
            ratio * T_world_body_after.pos;
        return T_world_body_interpolated;
    }

    // Missed lidar returns come through as NaN/inf coordinates; reject them.
    bool featureExtraction::isPointValid(const point_os::PointcloudXYZITR& point) {
        return std::isfinite(point.x) && 
            std::isfinite(point.y) && 
            std::isfinite(point.z);
    }

    // Standalone version of the per-point deskew transform (see the comments
    // inside removePointDistortion): computes the body motion between scan
    // start and the point's capture time, conjugates it with the IMU<->lidar
    // extrinsics for IMU data, and applies it to the point.
    Eigen::Vector3d featureExtraction::transformPoint(
        const point_os::PointcloudXYZITR& point,
        const Transformd& T_world_body_start,
        const Transformd& T_world_body_point,
        bool is_imu_data)
    {
        Transformd T_world_body_current(
            T_world_body_point.rot, T_world_body_point.pos);
        Transformd T_body_start_body_current =
            T_world_body_start.inverse() * T_world_body_current;

        Transformd T_lidar_start_lidar_current = is_imu_data ?
            T_lidar_imu * T_body_start_body_current * T_imu_lidar :
            T_body_start_body_current;

        Eigen::Vector3d pt(point.x, point.y, point.z);
        return T_lidar_start_lidar_current * pt;
    }

    // Writes corrected coordinates back into a point (helper).
    void featureExtraction::updatePointPosition(point_os::PointcloudXYZITR& point, const Eigen::Vector3d& new_pos) {
        point.x = new_pos.x();
        point.y = new_pos.y();
        point.z = new_pos.z();
    }


    // Converts a PCL cloud into a stamped PointCloud2 message. Despite the
    // name it does not publish anything itself; the caller stores the
    // returned message inside the LaserFeature bundle.
    template <typename Point>
    sensor_msgs::msg::PointCloud2
    featureExtraction::publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub, typename pcl::PointCloud<Point>::Ptr thisCloud,
                                    rclcpp::Time thisStamp, std::string thisFrame)
    {
        sensor_msgs::msg::PointCloud2 tempCloud;
        pcl::toROSMsg(*thisCloud, tempCloud);
        tempCloud.header.stamp = thisStamp;
        tempCloud.header.frame_id = thisFrame;
        return tempCloud;
    }

    // Assembles the LaserFeature message and publishes it. This is the single
    // hand-off point to the laserMapping node: one message per scan carrying
    //  - cloud_nodistortion: the full undistorted cloud,
    //  - cloud_corner / cloud_surface: edge and planar feature clouds
    //    (corner is currently always empty; see extractFeatures),
    //  - cloud_realsense: depth-camera points (unused here),
    //  - initial_quaternion / initial_pose: the lidar pose in the world frame
    //    at scan start, used by laserMapping as its registration prior.
    void featureExtraction::publishTopic(double lidar_start_time, 
                                         pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr laser_no_distortion_points,
                                         pcl::PointCloud<PointType>::Ptr edgePoints,
                                         pcl::PointCloud<PointType>::Ptr plannerPoints, 
                                         pcl::PointCloud<PointType>::Ptr depthPoints,
                                         Eigen::Quaterniond q_world_lidar_start)
    {
        // Stamp the message with the scan-start time (all clouds are
        // expressed in the sensor frame at that instant).
        FeatureHeader.frame_id = WORLD_FRAME;
        FeatureHeader.stamp = rclcpp::Time(lidar_start_time*1e9);
        laserFeature.header = FeatureHeader;
        laserFeature.imu_available = false;
        laserFeature.odom_available = false;

      
        laserFeature.cloud_nodistortion = publishCloud<point_os::PointcloudXYZITR>(pubLaserCloud, laser_no_distortion_points, FeatureHeader.stamp, SENSOR_FRAME);
        laserFeature.cloud_corner = publishCloud<PointType>(pubEdgePoints, edgePoints, FeatureHeader.stamp, SENSOR_FRAME);
        laserFeature.cloud_surface = publishCloud<PointType>(pubPlannerPoints, plannerPoints, FeatureHeader.stamp, SENSOR_FRAME);
        laserFeature.cloud_realsense=publishCloud<PointType>(pubBobPoints, depthPoints, FeatureHeader.stamp, SENSOR_FRAME);
       
        laserFeature.initial_quaternion_x = q_world_lidar_start.x();
        laserFeature.initial_quaternion_y = q_world_lidar_start.y();
        laserFeature.initial_quaternion_z = q_world_lidar_start.z();
        laserFeature.initial_quaternion_w= q_world_lidar_start.w();

        laserFeature.initial_pose_x = t_world_lidar_start.x();
        laserFeature.initial_pose_y = t_world_lidar_start.y();
        laserFeature.initial_pose_z = t_world_lidar_start.z();

        laserFeature.imu_available = true;
        laserFeature.sensor = 0; 
        pubLaserFeatureInfo->publish(laserFeature);
    }

    // Feature selection stage. Classic LOAM computes a "curvature" score for
    // each point (how much it sticks out from its neighbors along the ring)
    // and labels high-curvature points as edges and low-curvature points as
    // planar surfaces. This implementation takes a simpler route: it only
    // produces uniformly downsampled surface points, and publishes empty edge
    // and depth-camera clouds so the downstream message format stays the same.
    void featureExtraction::extractFeatures(
        double lidar_start_time,
        const pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr& lidar_msg,
        const Eigen::Quaterniond& q_world_lidar_start)
    {
        pcl::PointCloud<PointType>::Ptr plannerPoints(new pcl::PointCloud<PointType>());
        plannerPoints->reserve(lidar_msg->points.size());
        pcl::PointCloud<PointType>::Ptr edgePoints(new pcl::PointCloud<PointType>());
        edgePoints->reserve(lidar_msg->points.size());
        pcl::PointCloud<PointType>::Ptr bobPoints(new pcl::PointCloud<PointType>());
        bobPoints->reserve(lidar_msg->points.size());

        uniformFeatureExtraction(lidar_msg, plannerPoints, config_.filter_point_size, config_.min_range);
        
        publishTopic(
            lidar_start_time, lidar_msg, edgePoints, plannerPoints, bobPoints,
            q_world_lidar_start);
    }


    // Per-scan driver: picks the pose source for undistortion and runs the
    // deskew + feature extraction + publish sequence on the oldest buffered
    // scan. Preference order: VIO (full 6-DOF pose) over IMU (rotation only);
    // with neither available the scan is passed through undistorted.
    void featureExtraction::undistortionAndFeatureExtraction()      
    {
        LASER_IMU_SYNC_SCCUESS = synchronize_measurements<Imu::Ptr>(imuBuf, lidarBuf);
        LASER_CAMERA_SYNC_SUCCESS = synchronize_measurements<nav_msgs::msg::Odometry::SharedPtr>(visualOdomBuf, lidarBuf);

        // Only trust the VIO stream after a warm-up of 100 scans (it needs
        // time to converge before its poses are good enough for deskewing).
        if (frameCount > 100 and LASER_CAMERA_SYNC_SUCCESS == true)
            LASER_CAMERA_SYNC_SUCCESS = true;
        else
            LASER_CAMERA_SYNC_SUCCESS = false;

        if ((LASER_IMU_SYNC_SCCUESS == true or LASER_CAMERA_SYNC_SUCCESS == true) and lidarBuf.getSize() > 0)
        {
            double lidar_start_time;
            lidarBuf.getFirstTime(lidar_start_time);
            pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr lidar_msg;
            lidarBuf.getFirstMeas(lidar_msg);

            double lidar_end_time = lidar_start_time + lidar_msg->back().time;

            // VIO wins whenever it is synchronized (it provides translation
            // as well as rotation); otherwise fall back to IMU rotation-only
            // deskewing.
            if (LASER_IMU_SYNC_SCCUESS == true and LASER_CAMERA_SYNC_SUCCESS == true)
            {
                RCLCPP_INFO(this->get_logger(), "\033[1;32m----> Both IMU ,VIO laserscan are synchronized!.\033[0m");
                removePointDistortion<nav_msgs::msg::Odometry::SharedPtr>(lidar_start_time, lidar_end_time, visualOdomBuf, lidar_msg);
            }

            if (LASER_IMU_SYNC_SCCUESS == false and LASER_CAMERA_SYNC_SUCCESS == true)
            {
                removePointDistortion<nav_msgs::msg::Odometry::SharedPtr>(lidar_start_time, lidar_end_time, visualOdomBuf, lidar_msg);
            }

            if (LASER_IMU_SYNC_SCCUESS == true and LASER_CAMERA_SYNC_SUCCESS == false)
            {
                // RCLCPP_INFO(this->get_logger(), "\033[1;32m----> IMU and laserscan is synchronized!.\033[0m");
                removePointDistortion<Imu::Ptr>(lidar_start_time, lidar_end_time, imuBuf, lidar_msg);
            }

            // Extract features and publish
            extractFeatures(lidar_start_time, lidar_msg, q_world_lidar_start);

            LASER_CAMERA_SYNC_SUCCESS = false;
            LASER_IMU_SYNC_SCCUESS = false;
        
        }
        else if (imuBuf.empty())
        {
            // No IMU at all: skip deskewing entirely and publish the raw scan
            // with an identity attitude (pure lidar odometry mode).
            double lidar_start_time;
            lidarBuf.getFirstTime(lidar_start_time);
            pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr lidar_msg;
            lidarBuf.getFirstMeas(lidar_msg);
            double lidar_end_time = lidar_start_time + lidar_msg->back().time;

            RCLCPP_INFO(this->get_logger(), "\033[1;32m----> no IMU data, running LiDAR Odometry only.\033[0m");
            Eigen::Quaterniond q_world_lidar_identity =
                Eigen::Quaterniond::Identity();
            
            // Extract features and publish with default quaternion
            extractFeatures(lidar_start_time, lidar_msg,
                            q_world_lidar_identity);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "sync unsuccessfull, skipping scan frame");
        }
        
    }

    // Uniform downsampling that stands in for LOAM-style feature extraction:
    // keep every skip_num-th point, provided it differs from its predecessor
    // (spinning lidars repeat the same coordinates for consecutive missed
    // returns) and lies outside the block_range sphere around the sensor
    // (very close points are usually the robot's own body). The survivors are
    // treated as "surface" features for scan-to-map matching. Note the point
    // capture time is smuggled through the PointXYZI intensity channel so
    // that laserMapping can still access per-point timing.
    void featureExtraction::uniformFeatureExtraction(const pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr &pc_in, 
        pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_surf, int skip_num, float block_range)
    {   
        for (uint i=1; i <(int)pc_in->points.size(); i+=skip_num)
        {   
            pcl::PointXYZI point;
            point.x=pc_in->points[i].x;
            point.y=pc_in->points[i].y;
            point.z=pc_in->points[i].z;
            point.intensity=pc_in->points[i].time;

            // Because of C++ precedence the range check only binds to the
            // z-difference clause, so in practice a point is kept if it is
            // not an exact duplicate of the previous one.
            if ((abs(pc_in->points[i].x - pc_in->points[i-1].x) > 1e-7)
                || (abs(pc_in->points[i].y - pc_in->points[i-1].y) > 1e-7)
                || (abs(pc_in->points[i].z - pc_in->points[i-1].z) > 1e-7)
                && (pc_in->points[i].x * pc_in->points[i].x + pc_in->points[i].y * pc_in->points[i].y + pc_in->points[i].z * pc_in->points[i].z > (block_range * block_range)))
            {
                pc_out_surf->push_back(point);
            }
        
        }
        
    }

    // Unpacks a ROS IMU message into Eigen types (no processing).
    ImuMeasurement featureExtraction::parseImuMessage(const sensor_msgs::msg::Imu::SharedPtr& msg) {
        ImuMeasurement measurement;
        measurement.timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        measurement.accel << msg->linear_acceleration.x, 
                            msg->linear_acceleration.y,
                            msg->linear_acceleration.z;
        measurement.gyr << msg->angular_velocity.x, 
                        msg->angular_velocity.y,
                        msg->angular_velocity.z;
        // Raw IMUs commonly publish only acceleration and angular velocity; 
        // orientation may be zero or identity.
        // In this code, q_world_imu_driver is parsed but never used. updateImuOrientation() 
        // instead integrates gyroscope readings, starting from identity. 
        measurement.q_world_imu_driver = Eigen::Quaterniond(
            msg->orientation.w, msg->orientation.x,
            msg->orientation.y, msg->orientation.z);
        return measurement;
    }

    // Time step since the previous IMU sample. If the measured gap deviates
    // from the nominal period by more than IMU_TIME_LENIENCY (10 percent),
    // the timestamp likely jumped (dropped packets, clock glitch), so we fall
    // back to the nominal period instead of integrating over a bogus dt.
    double featureExtraction::calculateDeltaTime(double current_timestamp) {
        double lastImuTime = 0.0;
        double dt = m_imuPeriod;
        
        if(imuBuf.getLastTime(lastImuTime)) {
            dt = current_timestamp - lastImuTime;
            if(abs(dt - m_imuPeriod) > m_imuPeriod * IMU_TIME_LENIENCY) {
                dt = m_imuPeriod; // IMU timestamp jumped - quietly assume normal delta t
            }
        }
        return dt;
    }

    // Wraps a parsed measurement into the internal Imu struct.
    Imu::Ptr featureExtraction::createImuData(const ImuMeasurement& measurement) {
        Imu::Ptr imudata = std::make_shared<Imu>();
        imudata->time = measurement.timestamp;
        
        // Livox reports acceleration in units of g (a stationary sensor reads
        // ~1.0), so convert its magnitude to m/s^2 after initialization.
        // Keep the vector in the physical IMU frame used by deskewing.
        if(IMU_INIT && config_.sensor == SensorType::LIVOX) {
            double gravity = imu_Init->gravity_norm;
            imudata->acc =
                measurement.accel * gravity / imu_Init->acc_mean.norm();
        } else {
            imudata->acc = measurement.accel;
        }
        
        // Gyro likewise remains in the physical IMU frame.
        imudata->gyr = measurement.gyr;
        return imudata;
    }

    // Maintains the rolling orientation estimate q_world_imu (attitude of the IMU
    // body in a fixed "world" frame) that the deskewing interpolates between.
    // This is pure gyro dead reckoning: each new sample rotates the previous
    // attitude by the angle swept since the last sample. It drifts slowly,
    // but only pose DIFFERENCES within a 0.1 s scan matter for deskewing,
    // so the drift cancels out.
    void featureExtraction::updateImuOrientation(Imu::Ptr& imudata) {
        if (!imuBuf.empty()) {
            const auto& last_imu = imuBuf.measMap_.rbegin()->second;
            const double dt = imudata->time - last_imu->time;
            
            // Midpoint rule: average the previous and current angular rates,
            // multiply by dt to get the rotation vector (axis * angle, rad),
            // and map it onto SO(3) with the exponential map.
            Eigen::Vector3d delta_angle = dt * 0.5 * (imudata->gyr + last_imu->gyr);
            Eigen::Quaterniond q_imu_prev_imu_current =
                Sophus::SO3d::exp(delta_angle).unit_quaternion();
            
            imudata->q_world_imu =
                last_imu->q_world_imu * q_imu_prev_imu_current;
            imudata->q_world_imu.normalize();
        } else if (config_.use_imu_roll_pitch) {
            // Very first sample: intended to seed the attitude with the
            // sensor's roll/pitch while removing yaw (pre-multiplying by a
            // -yaw rotation), so the world frame starts gravity-aligned with
            // zero heading. Note that createImuData() never copies the driver
            // orientation into q_world_imu, so it is still identity here and
            // this branch is effectively a no-op; integration simply starts
            // from identity.
            tf2::Quaternion q_world_imu_current(
                imudata->q_world_imu.x(), imudata->q_world_imu.y(),
                imudata->q_world_imu.z(), imudata->q_world_imu.w());
            double roll, pitch, yaw;
            tf2::Matrix3x3(q_world_imu_current).getRPY(roll, pitch, yaw);
            
            tf2::Quaternion q_world_yaw_correction;
            q_world_yaw_correction.setRPY(0, 0, -yaw);
            tf2::Quaternion q_world_imu_initial =
                q_world_yaw_correction * q_world_imu_current;
            
            imudata->q_world_imu =
                Eigen::Quaterniond(q_world_imu_initial.w(),
                                   q_world_imu_initial.x(),
                                   q_world_imu_initial.y(),
                                   q_world_imu_initial.z());
        }
    }

    // One-time IMU initialization trigger. Once lidar data has started
    // flowing and at least 1 second of IMU samples has accumulated, run
    // Imu::imuInit() on the buffered (assumed stationary) data. That routine
    // estimates the gyro/accel statistics, gravity vector, and initial tilt.
    // Sensor measurements remain in their physical frames; the buffer is
    // cleared afterwards so deskewing starts from fresh samples.
    void featureExtraction::imuInitialization(double timestamp) {
        double lidar_first_time = 0;
        if(lidarBuf.getFirstTime(lidar_first_time) && 
            timestamp > lidar_first_time + LIDAR_MESSAGE_TIME + 0.05) {
            
            double first_time = 0.0;
            imuBuf.getFirstTime(first_time);
            
            if (timestamp - first_time > 1.0 && !IMU_INIT) {
                imu_Init->imuInit(imuBuf);
                IMU_INIT = true;
                imuBuf.clean(timestamp);
                RCLCPP_INFO(this->get_logger(), "IMU Initialization Process Finish!");
            }
        }
    }

    // IMU callback (~200 Hz): parse -> wrap into the internal Imu struct
    // (with Livox unit rescaling) -> integrate gyro into q_world_imu ->
    // buffer the sample for deskewing -> possibly run the one-time init.
    void featureExtraction::imu_Handler(const sensor_msgs::msg::Imu::SharedPtr msg_in) {
        m_buf.lock();
        
        auto measurement = parseImuMessage(msg_in);
        
        calculateDeltaTime(measurement.timestamp);
        
        auto imudata = createImuData(measurement);
        
        updateImuOrientation(imudata);
        
        imuBuf.addMeas(imudata, measurement.timestamp);
        
        // only do it at the beginning
        imuInitialization(measurement.timestamp);
        
        m_buf.unlock();
    }

    // Buffers the optional external (visual/VIO) odometry stream, keyed by
    // its timestamp, for use as the 6-DOF pose source in deskewing.
    void featureExtraction::visual_odom_Handler(const nav_msgs::msg::Odometry::SharedPtr visualOdometry)
    {
        m_buf.lock();
        visualOdomBuf.addMeas(visualOdometry, visualOdometry->header.stamp.sec + visualOdometry->header.stamp.nanosec*1e-9);
        m_buf.unlock();
    }

    // Fallback for drivers that provide only x/y/z/intensity per point
    // (provide_point_time == 0): reconstructs the two fields undistortion
    // needs, the ring id and the per-point capture time.
    //
    // Ring id: a spinning lidar has N_SCANS lasers stacked at fixed vertical
    // angles, so the elevation angle of a point identifies which laser
    // ("ring") produced it. The formulas below invert the known vertical
    // angle layout of each Velodyne model (VLP-16: -15 to +15 deg in 2 deg
    // steps; HDL-32: -30.67 to +10.67 deg in 4/3 deg steps; HDL-64: two
    // blocks with different angular spacing).
    //
    // Capture time: points arrive in firing order, so the index i encodes
    // when the point was measured. Each group of N_SCANS consecutive points
    // is one firing column (columnTime apart); within a column, consecutive
    // lasers fire laserTime apart. The result is stored relative to scan
    // start, matching what native driver timestamps would contain.
    void featureExtraction::assignTimeforPointCloud(pcl::PointCloud<PointType>::Ptr laserCloudIn_ptr_)
    {
        size_t cloud_size = laserCloudIn_ptr_->size();
        RCLCPP_DEBUG(this->get_logger(), "\n\ninput cloud size: %zu \n",cloud_size);
        pointCloudwithTime.reset(new pcl::PointCloud<point_os::PointcloudXYZITR>());
        pointCloudwithTime->reserve(cloud_size);

        point_os::PointcloudXYZITR point;
        for (size_t i = 0; i < cloud_size; i++)
        {
            point.x = laserCloudIn_ptr_->points[i].x;
            point.y = laserCloudIn_ptr_->points[i].y;
            point.z = laserCloudIn_ptr_->points[i].z;
            point.intensity = laserCloudIn_ptr_->points[i].intensity;

            // Elevation angle of the point in degrees; z over the horizontal
            // range. This is what selects the ring.
            float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
            int scanID = 0;

            if (config_.N_SCANS == 16)
            {
                // VLP-16: rings at -15,-13,...,+15 deg -> ring = (angle+15)/2.
                scanID = int((angle + 15) / 2 + 0.5);
                if (scanID > (config_.N_SCANS - 1) || scanID < 0)
                {
                    cloud_size--;
                    continue;
                }
            }
            else if (config_.N_SCANS == 32)
            {
                // HDL-32: rings from -30.67 deg upward in 4/3 deg steps.
                scanID = int((angle + 92.0 / 3.0) * 3.0 / 4.0);
                if (scanID > (config_.N_SCANS - 1) || scanID < 0)
                {
                    cloud_size--;
                    continue;
                }
            }
            else if (config_.N_SCANS == 64)
            {
                // HDL-64: upper block (angle >= -8.83 deg) has 1/3 deg
                // spacing, lower block has 1/2 deg spacing.
                if (angle >= -8.83)
                    scanID = int((2 - angle) * 3.0 + 0.5);
                else
                    scanID = config_.N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

                // use [0 50]  > 50 remove outlies
                if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
                {
                    cloud_size--;
                    continue;
                }
            }
            else
            {
                printf("wrong scan number\n");
                // ROS_BREAK();
            }

            point.ring = scanID;
            // Time from firing order: column index * columnTime plus the
            // laser's slot within the column * laserTime, in seconds
            // relative to scan start.
            float rel_time = (columnTime * int(i / config_.N_SCANS) + laserTime * (i % config_.N_SCANS)) / scanPeriod;
            float pointTime = rel_time * scanPeriod;
            point.time = pointTime;
            pointCloudwithTime->push_back(point);
        }
    }

    // Scan callback for Velodyne and Ouster (PointCloud2 messages). Converts
    // the incoming cloud to the common PointcloudXYZITR layout (x, y, z,
    // intensity, per-point time relative to scan start, ring id), buffers it,
    // and kicks off processing.
    void featureExtraction::laserCloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg)
    {  
        // Check if we should process this frame based on skip count
        frameCount = frameCount + 1;
        if (frameCount % config_.skipFrame != 0)
            return;

        m_buf.lock();

        pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr pointCloud(
            new pcl::PointCloud<point_os::PointcloudXYZITR>());
        
        tmpOusterCloudIn.reset(new pcl::PointCloud<point_os::OusterPointXYZIRT>());

        if (config_.provide_point_time)
        {

            if (config_.sensor == SensorType::VELODYNE)
            {
                // Velodyne clouds already match the target layout (float
                // 'time' in seconds, 'ring'), so a direct conversion works.
                pcl::fromROSMsg(*laserCloudMsg, *pointCloud);

            }
            else if (config_.sensor == SensorType::OUSTER)
            {
                // Ouster differs in two ways: per-point time 't' is a uint32
                // in nanoseconds, and the points are expressed in the ouster
                // frame, so each point is transformed by the fixed
                // T_sensor_ouster extrinsic and the time is rescaled to
                // float seconds.
                pcl::fromROSMsg(*laserCloudMsg, *tmpOusterCloudIn);
                pointCloud->points.resize(tmpOusterCloudIn->size());
                pointCloud->is_dense = tmpOusterCloudIn->is_dense;

                for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
                {
                    auto &src = tmpOusterCloudIn->points[i];
                    auto &dst = pointCloud->points[i];
                    utils::transformOusterPoints(
                        &src, &dst, T_sensor_ouster); // Ouster frame to sensor frame
                    dst.time = src.t * 1e-9f;
                }
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(),"Unknown sensor type: %d", int(sensor));
                rclcpp::shutdown();
            }
        }
        else
        {
            // Driver gives no per-point timestamps: reconstruct ring ids and
            // times from the firing geometry (see assignTimeforPointCloud).
            pcl::PointCloud<PointType>::Ptr laserCloudIn_ptr_(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn_ptr_);
            assignTimeforPointCloud(laserCloudIn_ptr_);
            pointCloud = pointCloudwithTime;
        }

        manageLidarBuffer(pointCloud, laserCloudMsg->header.stamp.sec + laserCloudMsg->header.stamp.nanosec * 1e-9);

        // Process only after the one-time IMU init has finished (so gravity
        // scale and initial attitude are available), or immediately in IMU-less
        // mode. The processed scan is then removed from the buffer.
        if(IMU_INIT==true or imuBuf.empty())
        {   
            undistortionAndFeatureExtraction();
            double lidar_first_time;
            lidarBuf.getFirstTime(lidar_first_time);
            lidarBuf.clean(lidar_first_time);
        }

        m_buf.unlock();
    }


    // Scan callback for Livox sensors, which publish a CustomMsg instead of
    // PointCloud2. Each Livox point carries: offset_time (ns since scan
    // start), line (the ring/scan-line id), and a 'tag' byte with the
    // driver's own quality classification.
    void featureExtraction::livoxHandler(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg)
    {   
        frameCount = frameCount + 1;
        if (frameCount % config_.skipFrame != 0)
            return; 

        m_buf.lock();
        
        pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr pointCloud(
            new pcl::PointCloud<point_os::PointcloudXYZITR>());
        
        pointCloud->points.resize(msg->point_num);

        // Keep Livox points in the physical lidar frame, matching the
        // Velodyne/Ouster paths and the T_imu_lidar deskew extrinsic.
        if(config_.provide_point_time) {     
            for (uint i=0; i < msg->point_num; i++) {
                // Keep only points on configured scan lines whose tag bits
                // 4-5 mark the return as normal (0x00) or slightly noisy
                // (0x10); stronger noise classes are dropped.
                if ((msg->points[i].line < config_.N_SCANS) &&
                    ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {   
                    pointCloud->points[i].x = msg->points[i].x;
                    pointCloud->points[i].y = msg->points[i].y;
                    pointCloud->points[i].z = msg->points[i].z;
                    pointCloud->points[i].intensity = msg->points[i].reflectivity;
                    // offset_time is nanoseconds since scan start; convert to
                    // the float seconds expected by the deskewing code.
                    pointCloud->points[i].time = msg->points[i].offset_time / float(1000000000);
                    pointCloud->points[i].ring = msg->points[i].line;
                }
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Please check yaml or livox driver to provide the timestamp for each point");
            rclcpp::shutdown();
        }

        manageLidarBuffer(pointCloud, msg->header.stamp.sec + msg->header.stamp.nanosec*1e-9);

        // Same gating as laserCloudHandler: wait for IMU init unless there
        // is no IMU at all.
        if(IMU_INIT==true or imuBuf.empty())
        {   
            undistortionAndFeatureExtraction();
            double lidar_first_time;
            lidarBuf.getFirstTime(lidar_first_time);
            lidarBuf.clean(lidar_first_time);
        }

        m_buf.unlock();
    }

    // Queues a converted scan for processing. If the buffer has backed up to
    // 50 scans (processing slower than the sensor), the oldest scans are
    // dropped: for real-time odometry a fresh scan is worth more than a
    // stale one.
    void featureExtraction::manageLidarBuffer(
        pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr pointCloud, 
        double timestamp)
    {
        // Check buffer size and drop oldest frames if necessary
        std::size_t curLidarBufferSize = lidarBuf.getSize();
        
        while (curLidarBufferSize >= 50) {
            double lidar_first_time;
            lidarBuf.getFirstTime(lidar_first_time);
            lidarBuf.clean(lidar_first_time);
            RCLCPP_WARN(this->get_logger(), "Lidar buffer too large, dropping frame");
            curLidarBufferSize = lidarBuf.getSize();
        }
        
        // Add measurement to buffer
        lidarBuf.addMeas(pointCloud, timestamp);
    }

} // namespace super_odometry
