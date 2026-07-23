//
// Created by ubuntu on 2020/6/29.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// This header defines Imu, which plays two roles:
//
//   1. A container for a single IMU sample (timestamp, accelerometer and
//      gyroscope readings, driver orientation).
//   2. The one-time IMU initialization routine imuInit(), run by the
//      imuPreintegration node on roughly 1 second of data collected while
//      the robot is STATIONARY.
//
// imuInit() estimates, from that stationary window:
//   - the mean and covariance of the accelerometer and gyroscope readings,
//   - the gravity vector expressed in the initial IMU frame,
//   - the gyroscope and accelerometer biases (the stationary means),
//   - the initial roll/pitch tilt of the IMU w.r.t. gravity (yaw is not
//     observable from an accelerometer, so it is fixed to 0), and
//   - R_gravity_lidar_initial, the rotation later used by
//     imuPreintegration::imuConverter() to re-express every IMU sample in a
//     gravity-leveled lidar frame.
//
// Frame conventions used below:
//   - "world" is a gravity-aligned frame with z pointing up; only its
//     roll/pitch relation to the sensor is estimated here (yaw-free).
//   - R_a_b rotates vectors expressed in frame b into frame a.
// ============================================================================

#ifndef IMU_DATA_H
#define IMU_DATA_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include "super_odometry/container/MapRingBuffer.h"
#include <sensor_msgs/msg/imu.hpp>
#include "super_odometry/config/parameter.h"
#include "super_odometry/utils/Twist.h"

// Assumed magnitude of gravity in m/s^2, used to scale the estimated
// gravity direction to a full gravity vector.
#define Gravity_Norm (9.81)

/// One IMU sample plus the one-time stationary initialization (bias,
/// gravity and initial tilt estimation). See the file overview above.
struct Imu {

public:
  typedef std::shared_ptr<Imu> Ptr;
  typedef std::shared_ptr<const Imu> ConstPtr;

public:
  Imu() : Imu(0.0, Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.0, 0.0, 0.0)) {}
  /// Wraps one sample: timestamp (s), accelerometer reading (m/s^2) and
  /// gyroscope reading (rad/s), both in the IMU body frame.
  Imu(double time, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
      : time(time), acc(acc), gyr(gyr) 
      {
        initialize();
      }
  ~Imu() {}
  
  /// Sets the statistics to neutral defaults before imuInit() overwrites
  /// them. (Note: the second gyr_bias line is a duplicate, so acc_bias is
  /// left uninitialized here; imuInit() fills it in.)
  void initialize()
  {
    acc_mean<<0.0,0.0,-1.0;
    gyr_mean<<0.0,0.0,0.0;
    acc_cov<<0.1,0.1,0.1;
    gyr_cov<<0.1,0.1,0.1;
    gravity<<0.0,0.0,0.0;
    gyr_bias<<0.0,0.0,0.0;
    gyr_bias<<0.0,0.0,0.0;
    q_world_imu.setIdentity();
  }

  /// Estimates the initial roll/pitch tilt from the mean accelerometer
  /// reading (ax, ay, az) taken while the IMU is STATIONARY, and returns the
  /// yaw-free rotation R_imu_world.
  ///
  /// How it works: when stationary the accelerometer measures only the
  /// gravity reaction, so its direction encodes the tilt (assumption: the
  /// accelerometer reads +g on z when level, i.e. (0, 0, +9.81)). Yaw does
  /// not change the measured gravity direction, so it is unobservable and
  /// fixed to 0.
  ///
  /// The function builds R = R_x(phi) * R_y(theta). With the true yaw-free
  /// (ZYX convention) attitude being R_world_imu = R_y(pitch) * R_x(roll),
  /// the returned matrix is exactly its inverse, i.e. R_imu_world: it maps
  /// vectors from the gravity-aligned world frame into the initial IMU
  /// frame. Consequently the caller stores it as R_imu_gravity_initial,
  /// and R_imu_gravity_initial.inverse() is the yaw-free initial
  /// attitude R_world_imu.
  ///
  /// Side effect: stores theta/phi (radians) in pitch_offset_gravity and
  /// roll_offset_gravity.
  Eigen::Matrix3d calculatePitchRollMatrix(double ax, double ay, double az) {
      // Pitch angle theta: rotation about y that tilts the x axis out of
      // the horizontal plane.
      double theta = std::atan2(ax, std::sqrt(ay * ay + az * az));

      // Roll angle phi: rotation about x that tilts the y axis.
      double phi = std::atan2(-ay, az);

      // Construct the pitch rotation matrix R_y(theta).
      Eigen::Matrix3d R_y;
      R_y << std::cos(theta), 0, std::sin(theta),
            0, 1, 0,
            -std::sin(theta), 0, std::cos(theta);

      // Construct the roll rotation matrix R_x(phi).
      Eigen::Matrix3d R_x;
      R_x << 1, 0, 0,
            0, std::cos(phi), -std::sin(phi),
            0, std::sin(phi), std::cos(phi);
      
      // R = R_x(phi) * R_y(theta) = (R_y(pitch) * R_x(roll))^-1 = R_imu_world.
      Eigen::Matrix3d R = R_x * R_y;
      pitch_offset_gravity=theta;
      roll_offset_gravity=phi;
      return R;
  }

  /// One-time initialization from ~1 second of IMU samples collected while
  /// the robot is STATIONARY (imuBuf is filled by the imuPreintegration
  /// node). Runs only once: first_imu guards against repeated calls.
  ///
  /// Computes running means/covariances of the accelerometer and gyroscope,
  /// then derives gravity, the biases and the leveling rotations described
  /// in the file overview. If the robot moves during this window, all of
  /// these estimates are corrupted.
  void imuInit(MapRingBuffer<Imu::Ptr> imuBuf) {
    
    int Num = 0;
    if (first_imu==false){
        return;
    }
    
    // Seed the running means with the first sample in the buffer.
    if (!imuBuf.empty()) {
        first_imu = false;
        const double &time_first = imuBuf.measMap_.begin()->second->time;
        const Eigen::Quaterniond q_world_imu_first =
            imuBuf.measMap_.begin()->second->q_world_imu;
        const Eigen::Vector3d gyr_first = imuBuf.measMap_.begin()->second->gyr;
        const Eigen::Vector3d acc_first = imuBuf.measMap_.begin()->second->acc;
        acc_mean = acc_first;
        gyr_mean = gyr_first;
        time = time_first;
      
        Num = 1;
    }
  
    // Variables for frequency calculation
    double time_prev = 0;
    double total_time_diff = 0;
    int time_diff_count = 0;
    
    // One pass over the buffer, updating the means and covariances
    // incrementally (Welford-style running statistics).
    for (std::map<double, Imu::Ptr>::iterator itMeas_ = imuBuf.measMap_.begin(); itMeas_ != imuBuf.measMap_.end(); ++itMeas_) {

        const double &time_cur = itMeas_->second->time;
        const Eigen::Quaterniond q_world_imu_cur =
            itMeas_->second->q_world_imu;
        const Eigen::Vector3d gyr_cur = itMeas_->second->gyr;
        const Eigen::Vector3d acc_cur = itMeas_->second->acc;
        
        // Calculate time difference for frequency estimation
        if (Num > 1) {
            double time_diff = time_cur - time_prev;
            if (time_diff > 0) {  // Ensure valid time difference
                total_time_diff += time_diff;
                time_diff_count++;
            }
        }
        time_prev = time_cur;
      
        // Update means
        acc_mean += (acc_cur - acc_mean) / Num;
        gyr_mean += (gyr_cur - gyr_mean) / Num;

        // Update covariances
        acc_cov = acc_cov * (Num - 1.0) / Num + (acc_cur - acc_mean).cwiseProduct(acc_cur - acc_mean) / (Num - 1.0);
        gyr_cov = gyr_cov * (Num - 1.0) / Num + (gyr_cur - gyr_mean).cwiseProduct(gyr_cur - gyr_mean) / (Num - 1.0);
        Num++;
    }
  
    // if (Num > 1 && time_diff_count > 0) {
    //     // Get total time span
    //     double total_time_span = imuBuf.measMap_.rbegin()->second->time - imuBuf.measMap_.begin()->second->time;
    //     // Calculate actual number of intervals (Num-1)
    //     imu_frequency = Num / total_time_span;
    // }
    
    // A stationary accelerometer measures the reaction to gravity (+g
    // "upward" in its own frame), so gravity itself points the opposite
    // way: gravity is the gravity vector expressed in the initial IMU
    // frame, scaled to Gravity_Norm (m/s^2).
    gravity= - acc_mean / acc_mean.norm() *Gravity_Norm;
    // While stationary the true angular rate is zero and the true specific
    // force is pure gravity, so the means serve as bias estimates. (Note
    // that acc_bias still contains gravity; downstream code accounts for
    // it as the full stationary mean, not as a residual bias.)
    gyr_bias = gyr_mean;
    acc_bias = acc_mean;
    first_imu = false;

    // Estimate the initial tilt w.r.t. gravity in case the robot starts on
    // a slope. R_imu_gravity_initial maps the gravity-aligned frame into the
    // initial IMU frame, so its inverse maps initial IMU into gravity.
    R_imu_gravity_initial=calculatePitchRollMatrix(acc_mean.x(),
    acc_mean.y(), acc_mean.z());


    std::cout<<"R_imu_lidar: "<<R_imu_lidar<<std::endl;
    // R_imu_lidar maps lidar-frame vectors into the IMU frame. Composing it
    // with the leveling rotation gives R_gravity_lidar_initial =
    // R_imu_gravity_initial.inverse() * R_imu_lidar, which maps lidar-frame
    // vectors into the initial gravity-aligned frame. The
    // imuPreintegration node uses it to re-express IMU measurements in a
    // gravity-leveled lidar frame. (All shipped calibrations set
    // R_imu_lidar to identity, which masks a direction inconsistency in how
    // imuConverter() applies this rotation; see imuConverter's docs.)
    R_gravity_lidar_initial=R_imu_gravity_initial.inverse()*R_imu_lidar;
    // The initial gravity-aligned frame is at the same origin as the IMU frame, but with the gravity direction aligned.
    // That's why the translation part is the same as the IMU frame.
    Transformd T_gravity_lidar_initial_value(R_gravity_lidar_initial,
                                             t_imu_lidar);
    T_gravity_lidar_initial=T_gravity_lidar_initial_value;
    std::cout<<"T_gravity_lidar_initial: "<<T_gravity_lidar_initial<<std::endl;
      
  
    std::cout<<"IMU Data Summary"<<std::endl;
    std::cout<<"Gravity: "<<gravity.transpose()<<std::endl;
    std::cout<<"Gyroscope Bias: "<<gyr_bias.transpose()<<std::endl;
    std::cout<<"Accelerometer Bias: "<<acc_bias.transpose()<<std::endl;
    std::cout<<"Accelerometer Mean: "<<acc_mean.transpose()<<std::endl;
    std::cout<<"IMU Frequency: "<<imu_frequency<<" Hz"<<std::endl;
    std::cout<<"pitch offset gravity: "<<pitch_offset_gravity*180/M_PI<<std::endl;
    std::cout<<"roll offset gravity: "<<roll_offset_gravity*180/M_PI<<std::endl;
    std::cout<<"R_imu_gravity_initial: "<<R_imu_gravity_initial<<std::endl;
      
  }

  /// Converts a rotation matrix to (roll, pitch, yaw) in radians, ZYX
  /// convention. Handles the gimbal-lock singularity (pitch near +/-90
  /// degrees, where sy ~ 0) by fixing yaw to 0.
  Eigen::Vector3d rotationMatrixToRPY(const Eigen::Matrix3d& R) {
    Eigen::Vector3d rpy;

    // Extract the roll, pitch, and yaw from the rotation matrix
    double sy = sqrt(R(0,0) * R(0,0) + R(1,0) * R(1,0));

    bool singular = sy < 1e-6; // If

    if (!singular) {
        rpy(0) = atan2(R(2,1), R(2,2)); // Roll
        rpy(1) = atan2(-R(2,0), sy);     // Pitch
        rpy(2) = atan2(R(1,0), R(0,0));  // Yaw
    } else {
        rpy(0) = atan2(-R(1,2), R(1,1)); // Roll
        rpy(1) = atan2(-R(2,0), sy);     // Pitch
        rpy(2) = 0;                      // Yaw
    }

    return rpy;
}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double gravity_norm = 9.8105; // measured stationary |acc| (m/s^2); used to rescale Livox g-units
  bool first_imu = true;        // true until imuInit() has run once
  double time;                  // sample timestamp (s)
  double imu_frequency = 200.0; // IMU sample rate (Hz), default 200 Hz
  // Results of imuInit(), all expressed in the initial IMU frame:
  Eigen::Vector3d gravity;  // gravity vector (m/s^2), points "down" in the IMU frame
  Eigen::Vector3d gyr_bias; // gyroscope bias = stationary gyro mean (rad/s)
  Eigen::Vector3d acc_bias; // stationary accelerometer mean (m/s^2); includes gravity
  Eigen::Vector3d acc_mean; // mean of accelerometer measurements (m/s^2)
  Eigen::Vector3d gyr_mean; // mean of gyroscope measurements (rad/s)
  // The single sample carried by this struct:
  Eigen::Vector3d acc; // accelerometer measurement (m/s^2), IMU body frame
  Eigen::Vector3d gyr; // gyroscope measurement (rad/s), IMU body frame
  Eigen::Vector3d acc_cov; // per-axis variance of the accelerometer measurements
  Eigen::Vector3d gyr_cov; // per-axis variance of the gyroscope measurements
  Eigen::Quaterniond q_world_imu; // orientation from the IMU driver: R_world_imu
  Eigen::Matrix3d R_imu_gravity_initial; // maps gravity-aligned vectors into the initial IMU frame
  Eigen::Matrix3d R_gravity_lidar_initial; // maps lidar-frame vectors into the initial gravity frame
  Transformd T_gravity_lidar_initial; // maps lidar-frame points into the initial gravity frame
  double pitch_offset_gravity; // initial pitch w.r.t. gravity (rad), from calculatePitchRollMatrix()
  double roll_offset_gravity;  // initial roll w.r.t. gravity (rad)
};

#endif // IMU_DATA_H
