//
// Created by shibo zhao on 2020-09-27.
//
// ============================================================================
// OVERVIEW
// ============================================================================
// Executable entry point for the feature-extraction node, the front end of
// the SuperOdom pipeline. It instantiates the featureExtraction class (see
// FeatureExtraction/featureExtraction.h for what the node actually does:
// point cloud undistortion and edge/planar feature selection), sizes its
// data buffers, and spins a multi-threaded executor so the lidar, IMU and
// odometry callbacks can run concurrently.
// ============================================================================
#include "rclcpp/rclcpp.hpp"
#include "super_odometry/FeatureExtraction/featureExtraction.h"

int main(int argc, char **argv)
{
    // Silence PCL's chatty conversion warnings; only real errors get through.
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    rclcpp::init(argc,argv);
    rclcpp::NodeOptions options;
    options.arguments({"feature_extraction_node"});
    
    std::shared_ptr<super_odometry::featureExtraction> featureExtraction =
        std::make_shared<super_odometry::featureExtraction>(options);

    // Pre-size the ring buffers: 2000 IMU/odometry samples covers ~10 s at
    // 200 Hz, and 50 scans covers ~5 s at 10 Hz, plenty for the
    // synchronization windows used during undistortion.
    featureExtraction->imuBuf.allocate(2000);
    featureExtraction->visualOdomBuf.allocate(2000);
    featureExtraction->lidarBuf.allocate(50);
    // Subscriptions/publishers are created here rather than in the
    // constructor because initInterface() needs shared_from_this().
    featureExtraction->initInterface();

    // A multi-threaded executor + the node's Reentrant callback group allow
    // the sensor callbacks to be processed in parallel.
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(featureExtraction);
    executor.spin();
    // rclcpp::spin(featureExtraction->get_node_base_interface());
    rclcpp::shutdown();

    return 0;
}
