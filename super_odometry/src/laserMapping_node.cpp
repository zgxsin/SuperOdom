//
// Created by shibo zhao on 2020-09-27.
//
// Entry point for the laser mapping node. It only instantiates the
// laserMapping class (which does all the work: scan-to-map registration of
// the feature clouds produced by the featureExtraction node, yielding the
// lidar odometry pose consumed by the imuPreintegration node) and spins it.
#include "rclcpp/rclcpp.hpp"
#include "super_odometry/LaserMapping/laserMapping.h"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.arguments({"laser_mapping_node"});

    std::shared_ptr<super_odometry::laserMapping> laserMapping =
      std::make_shared<super_odometry::laserMapping>(options);

    // Two-phase init: initInterface needs shared_from_this(), which is only
    // valid after the shared_ptr above exists, so it cannot run in the
    // constructor.
    laserMapping->initInterface();
    
    // A multi-threaded executor is required: the process() loop occupies one
    // thread permanently, while subscription callbacks run on others.
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(laserMapping);
    executor.spin();
    // rclcpp::spin(laserMapping->get_node_base_interface());
    rclcpp::shutdown();

    return 0;
}

