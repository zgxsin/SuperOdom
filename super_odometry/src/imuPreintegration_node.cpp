//
// Created by shibo zhao on 2020-09-27.
//
// Entry point (main) for the IMU preintegration node. The actual logic lives
// in the imuPreintegration class (ImuPreintegration/imuPreintegration.cpp);
// this file only constructs the node, sizes its buffers, and spins it.
#include "rclcpp/rclcpp.hpp"
#include "super_odometry/ImuPreintegration/imuPreintegration.h"

int main(int argc, char **argv)
{
    rclcpp::init(argc,argv);
    rclcpp::NodeOptions options;
    options.arguments({"imu_preintegration_node"});

    std::shared_ptr<super_odometry::imuPreintegration> imuPreintegration =
        std::make_shared<super_odometry::imuPreintegration>(options);
    
    // Ring-buffer capacities: ~1000 IMU samples (about 5 s at 200 Hz) for the
    // one-time IMU initialization, and 100 lidar odometry poses.
    imuPreintegration->imuBuf.allocate(1000);
    imuPreintegration->lidarOdomBuf.allocate(100);
    // Sets up parameters, calibration, subscribers/publishers and the GTSAM
    // preintegrators (must run after construction; uses shared_from_this()).
    imuPreintegration->initInterface();

    // A multi-threaded executor lets the IMU callback (~200 Hz) and the lidar
    // odometry callback (~10 Hz, expensive optimization) run concurrently;
    // the node's Reentrant callback group opts into this.
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(imuPreintegration);
    executor.spin();
    // rclcpp::spin(imuPreintegration->get_node_base_interface());
    rclcpp::shutdown();
    
    return 0;
}
