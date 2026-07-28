## Super Odometry

Super Odometry is a high-performance odometry library designed for robotics applications. It integrates advanced algorithms for real-time localization and mapping, supporting a wide range of sensors.

## How to Run the localization mode? 
Please change the yaml file 
```
        localization_mode: true             # if true, localization mode is on; Otherwise, SLAM mode is on  
        read_pose_file: false        # read the txt pose as the initial pose for localization
        init_x: 13.983960            # initial pose from yaml file for localization
        init_y: 1.305790
        init_z: 0.002673
        init_roll: 0.0
        init_pitch: 0.0
        init_yaw: -1.150664 
```
## Test Sample 

```
ros2 bag play cic_office_stopping.db3 --start-offset 50  
ros2 launch super_odometry arize_slam.launch.py 

```
## Load the start_pose.txt (below are format samples) 
```  
timespan x        y        z      roll     pitch     yaw
50s 13.983960 1.305790 0.002673   0.00     0.0 -1.150664
```
## Put the start_pose.txt same directory of groundtruth map (pointcloud_local.pcd)

## Test Bag file
https://drive.google.com/drive/u/0/folders/1R8Tx8nLDC184gjUaMPZjTiklIhsH7RLV


## Reference Frames

### Feature extraction

`featureExtraction` uses the stationary accelerometer estimate to initialize a
gravity-aligned, yaw-free IMU reference frame. It then integrates angular
velocity to estimate `q_world_imu`, an SO(3) orientation only; it does not
estimate IMU position. This orientation is used to rotationally deskew each
lidar scan and to provide the scan-start orientation
`q_world_lidar = q_world_imu * q_imu_lidar` to `laserMapping`. The fixed
IMU-lidar translation accounts for the rotational lever arm during deskewing,
but there is no estimated translational motion.

### Laser mapping

`laserMapping` establishes the persistent SLAM world frame. In mapping mode,
the first lidar position is set to zero, its IMU-derived roll and pitch are
used, and yaw is set to zero. Without IMU data the first lidar frame defines
the world directly. In localization mode, the configured initial SE(3) pose
defines the lidar pose in the existing map. Scan-to-map registration then
estimates the full SE(3) transform `T_world_lidar` (rotation and translation)
for every scan. The feature-extraction orientation is only an initial guess;
the optimized map pose defines the final estimate. The world is gravity-aligned
because the feature-extraction quaternion is initialized from the measured
gravity direction; its heading remains arbitrary.

### IMU preintegration

`imuPreintegration` does not create another world frame. It inherits the SLAM
world from the `laser_odometry` pose and initializes the IMU pose using
`T_world_imu = T_world_lidar * T_lidar_imu`. It estimates a full navigation
state: the IMU SE(3) pose, velocity, and accelerometer/gyroscope biases. Lidar
poses provide periodic corrections, while preintegrated IMU measurements
propagate the state between lidar scans. GTSAM assumes the inherited world has
z pointing up and gravity `(0, 0, -g)`, so physically correct preintegration
requires the world initialized by `laserMapping` to be gravity-aligned.
