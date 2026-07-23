//
// Created by shibo zhao on 2020-09-27.
//
#include "super_odometry/ImuPreintegration/imuPreintegration.h"


namespace super_odometry {

    imuPreintegration::imuPreintegration(const rclcpp::NodeOptions & options)
    : Node("imu_preintegration_node", options) {
    }
    
    // One-time setup: parameters, calibration, topics, and the GTSAM
    // preintegration settings. Called once after construction.
    void imuPreintegration::initInterface() {
        // A Reentrant callback group lets the IMU callback and the lidar
        // odometry callback run concurrently on a multi-threaded executor
        // (they synchronize on the mBuf mutex where needed).
        cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = cb_group_;

        // BEST_EFFORT = do not retry lost messages; for a 200 Hz sensor stream
        // a fresh sample is more useful than a re-sent old one.
        rclcpp::QoS imu_qos(10);
        imu_qos.best_effort();  // Use BEST_EFFORT reliability
        imu_qos.keep_last(10);  // Keep last 10 messages

        if (!readGlobalparam(shared_from_this())) {
            RCLCPP_ERROR(this->get_logger(), "[SuperOdometry::imuPreintegration] Could not read global parameters. Exiting...");
            rclcpp::shutdown();
        }

        if (!readParameters()) {
            RCLCPP_ERROR(this->get_logger(), "[SuperOdometry::imuPreintegration] Could not read local parameters. Exiting...");
            rclcpp::shutdown();
        }

        if (!readCalibration(shared_from_this()))
        {
            RCLCPP_ERROR(this->get_logger(), "[SuperOdometry::imuPreintegration] Could not read calibration parameters. Exiting...");
            rclcpp::shutdown();
        }

        RCLCPP_INFO(this->get_logger(), "[SuperOdometry::imuPreintegration] use_imu_rol_pitch:  %d", config_.use_imu_roll_pitch);

        //subscribe and publish relevant topics
        subImu = this->create_subscription<sensor_msgs::msg::Imu>(
            IMU_TOPIC, imu_qos,
            std::bind(&imuPreintegration::imuHandler, this,
                        std::placeholders::_1), sub_options);
        subLaserOdometry = this->create_subscription<nav_msgs::msg::Odometry>(
            ProjectName+"/laser_odometry", 5,
            std::bind(&imuPreintegration::laserodometryHandler, this,
                        std::placeholders::_1), sub_options);

        pubImuOdometry = this->create_publisher<nav_msgs::msg::Odometry>(
            ProjectName+"/state_estimation", 10);
        pubHealthStatus = this->create_publisher<std_msgs::msg::Bool>(
            ProjectName+"/state_estimation_health", 1);
        pubImuPath = this->create_publisher<nav_msgs::msg::Path>(
            ProjectName+"/imuodom_path", 1);
        
        // Configure how GTSAM integrates IMU measurements.
        // MakeSharedU("Up") means the world frame has z pointing up and
        // gravity = (0, 0, -imuGravity). The original converter below attempts
        // to align incoming measurements with that convention.
        std::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(config_.imuGravity);
        
        p->accelerometerCovariance =
                gtsam::Matrix33::Identity(3, 3) * pow(config_.imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance =
                gtsam::Matrix33::Identity(3, 3) * pow(config_.imuGyrNoise, 2); // gyro white noise in continuous
        p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) *
                                   pow(1e-4, 2); // error committed in integrating position from velocities

        gtsam::imuBias::ConstantBias prior_imu_bias(
                (gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

        priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m

        priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-2);                      // m/s
        priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6,
                                                             1e-1);                    // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Isotropic::Sigma(6, config_.lidar_correction_noise); // meter


        noiseModelBetweenBias = (gtsam::Vector(6)
                << config_.imuAccBiasN,
                config_.imuAccBiasN, config_.imuAccBiasN, config_.imuGyrBiasN, config_.imuGyrBiasN, config_.imuGyrBiasN)
                .finished();
        imuIntegratorImu_ = std::make_shared<gtsam::PreintegratedImuMeasurements>(p, prior_imu_bias); // setting up the IMU integration for IMU message
        imuIntegratorOpt_ = std::make_shared<gtsam::PreintegratedImuMeasurements>(p, prior_imu_bias); // setting up the IMU integration for optimization

        // The calibration stores T_imu_lidar, which maps lidar-frame points
        // into the IMU frame. Keep both transform directions available:
        // T_lidar_imu = inverse(T_imu_lidar).
        if (PROVIDE_IMU_LASER_EXTRINSIC) {
            T_imu_lidar = gtsam::Pose3(
                gtsam::Rot3(R_imu_lidar), gtsam::Point3(t_imu_lidar));
            T_lidar_imu = T_imu_lidar.inverse();
        } else {
            T_imu_camera = gtsam::Pose3(gtsam::Rot3(R_imu_camera), gtsam::Point3(t_imu_camera));
            T_camera_lidar = gtsam::Pose3(gtsam::Rot3(R_camera_lidar), gtsam::Point3(t_camera_lidar));
            T_imu_lidar = T_imu_camera.compose(T_camera_lidar);
            T_lidar_imu = T_imu_lidar.inverse();
        }

    }
    

    bool imuPreintegration::readParameters()
    {
        this->declare_parameter<float>("imu_preintegration_node.acc_n", 1e-3);
        this->declare_parameter<float>("imu_preintegration_node.acc_w", 1e-3);
        this->declare_parameter<float>("imu_preintegration_node.gyr_n", 1e-6);
        this->declare_parameter<float>("imu_preintegration_node.gyr_w", 1e-6);
        this->declare_parameter<float>("imu_preintegration_node.g_norm", 9.80511);
        this->declare_parameter<float>("imu_preintegration_node.lidar_correction_noise",0.01);
        this->declare_parameter<float>("imu_preintegration_node.smooth_factor",0.9);
        this->declare_parameter<bool>("imu_preintegration_node.use_imu_roll_pitch", false);
        this->declare_parameter<double>("imu_preintegration_node.imu_acc_x_limit", 1.0);
        this->declare_parameter<double>("imu_preintegration_node.imu_acc_y_limit", 1.0);
        this->declare_parameter<double>("imu_preintegration_node.imu_acc_z_limit", 1.0);

        config_.imuAccNoise = this->get_parameter("imu_preintegration_node.acc_n").as_double();
        config_.imuAccBiasN = this->get_parameter("imu_preintegration_node.acc_w").as_double();
        config_.imuGyrNoise = this->get_parameter("imu_preintegration_node.gyr_n").as_double();
        config_.imuGyrBiasN = this->get_parameter("imu_preintegration_node.gyr_w").as_double();
        config_.imuGravity = this->get_parameter("imu_preintegration_node.g_norm").as_double();
        config_.lidar_correction_noise = this->get_parameter("imu_preintegration_node.lidar_correction_noise").as_double();
        config_.smooth_factor = this->get_parameter("imu_preintegration_node.smooth_factor").as_double();
        // config_.use_imu_roll_pitch = this->get_parameter("imu_preintegration_node.use_imu_roll_pitch").as_bool();
        config_.imu_acc_x_limit = this->get_parameter("imu_preintegration_node.imu_acc_x_limit").as_double();
        config_.imu_acc_y_limit = this->get_parameter("imu_preintegration_node.imu_acc_y_limit").as_double();
        config_.imu_acc_z_limit = this->get_parameter("imu_preintegration_node.imu_acc_z_limit").as_double();
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

    // Throws away the optimizer and starts an empty factor graph.
    void imuPreintegration::resetOptimization() {
        gtsam::ISAM2Params optParameters;
        // Relinearize a variable when its update exceeds 0.1, and check at
        // every update. Keeps the (nonlinear) solution accurate as it evolves.
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    // Called after a failure: forces the next lidar pose to re-initialize the
    // whole system (see laserodometryHandler).
    void imuPreintegration::resetParams() {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }


    // Periodic "restart" that keeps the graph from growing forever (called
    // every 100 keyframes). The trick: ask the old optimizer how uncertain the
    // latest state is (marginal covariance), then start a brand-new graph
    // whose priors are that state with that uncertainty. Nothing is lost, but
    // the graph is small again.
    void imuPreintegration::reset_graph() {

        // get updated noise before reset
        gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise, updatedVelNoise, updatedBiasNoise;

        try
        {
            updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
            updatedVelNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
            updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));
        }
        catch (...)
        {
            RCLCPP_WARN(this->get_logger(), "Failed to reset graph. No marginal covariance key");
            updatedPoseNoise = priorPoseNoise;
            updatedVelNoise = priorVelNoise;
            updatedBiasNoise = priorBiasNoise;
        }

        // reset graph
        resetOptimization();
        // Re-seed the new graph at index 0 with the carried-over state.
        // add pose
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), T_world_imu_prev,
                                                   updatedPoseNoise);
        graphFactors.add(priorPose);
        // add velocity
        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_,
                                                    updatedVelNoise);
        graphFactors.add(priorVel);
        // add bias
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(
                B(0), prevBias_, updatedBiasNoise);
        graphFactors.add(priorBias);
        // add values (initial guesses for the unknowns = the carried-over state)
        graphValues.insert(X(0), T_world_imu_prev);
        graphValues.insert(V(0), prevVel_);
        graphValues.insert(B(0), prevBias_);
        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        key = 1;
    }

    // Called on the FIRST lidar pose (or after a reset). Anchors the factor
    // graph: pose comes from lidar odometry, velocity is assumed zero, and
    // bias is assumed zero (the priors' sigmas say how much those assumptions
    // may be bent by later measurements).
    void imuPreintegration::initial_system(
        double currentCorrectionTime, gtsam::Pose3 T_world_lidar_meas) {
        resetOptimization();

        // Drop IMU samples older than the first lidar pose; they predate the
        // state we are anchoring and can never be used.
        while (!imuQueOpt.empty()) {
            if (secs(&imuQueOpt.front()) < currentCorrectionTime - delta_t) {
                lastImuT_opt = secs(&imuQueOpt.front());
                imuQueOpt.pop_front();
            }
            else
                break;
        }

        // The graph estimates the IMU pose, but lidar odometry measures the
        // lidar pose -> convert using the extrinsic.
        // T_world_imu = T_world_lidar * T_lidar_imu.
        T_world_imu_prev = T_world_lidar_meas.compose(T_lidar_imu);

        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), T_world_imu_prev,
                                                   priorPoseNoise);
        graphFactors.add(priorPose);

        // Assume the robot starts (approximately) at rest.
        prevVel_ = gtsam::Vector3(0, 0, 0);
        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_,
                                                    priorVelNoise);
        graphFactors.add(priorVel);

        // Assume zero initial IMU bias; the optimizer will estimate the real
        // value over time via the bias random-walk factors.
        prevBias_ = gtsam::imuBias::ConstantBias();
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(
                B(0), prevBias_, priorBiasNoise);
        graphFactors.add(priorBias);

        graphValues.insert(X(0), T_world_imu_prev);
        graphValues.insert(V(0), prevVel_);
        graphValues.insert(B(0), prevBias_);

        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        // Both preintegrators start fresh with the (zero) initial bias.
        imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

        key = 1;
        systemInitialized = true;
    }

    // Feeds every buffered IMU sample older than the new lidar pose into
    // imuIntegratorOpt_. GTSAM accumulates them into a single relative
    // motion (delta position/velocity/rotation) that will become one
    // ImuFactor connecting the previous keyframe to the new one.
    void imuPreintegration::integrate_imumeasurement(double currentCorrectionTime) {
        // 1. integrate imu data and optimize

        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::msg::Imu *thisImu = &imuQueOpt.front();
            double imuTime = secs(thisImu);
            if (imuTime < currentCorrectionTime - delta_t)
            {
                // dt = time since the previous integrated sample. For the very
                // first sample there is no previous one, so assume 200 Hz.
                double dt = (lastImuT_opt < 0) ? (1.0 / 200.0) : (imuTime - lastImuT_opt);
                lastImuT_opt = imuTime;

                // Guard against garbage timestamps (duplicates or gaps).
                if(dt < 0.001 || dt > 0.5) 
                    dt = 0.005;
               
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);

                imuQueOpt.pop_front();
            }
            else
                break;  // queue is time-ordered; the rest is newer than the lidar pose
        }

    }


    // Adds one keyframe to the factor graph and solves it. Three factors are
    // added per keyframe:
    //   1. a prior on X(key) from the lidar pose ("the lidar says you are here"),
    //   2. an ImuFactor connecting keyframe key-1 to key ("the IMU says you
    //      moved this much in between"),
    //   3. a bias between-factor allowing the bias to drift slowly.
    // The optimizer balances (1) and (2); their disagreement is what makes
    // the bias observable.
    bool imuPreintegration::build_graph(
        gtsam::Pose3 T_world_lidar_meas, double curLaserodomtimestamp) {


        // Convert the measured lidar pose to the IMU pose that the graph estimates.
        gtsam::Pose3 T_world_imu_meas = T_world_lidar_meas.compose(T_lidar_imu);

        // Predict where the IMU thinks we are now, by applying the
        // preintegrated motion to the previous optimized state. Used as the
        // initial guess for the new unknowns.
        gtsam::NavState propState_ =
                imuIntegratorOpt_->predict(prevState_, prevBias_);
        auto diff = T_world_imu_meas.translation() - propState_.pose().translation();

        // (1) lidar pose prior factor
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), T_world_imu_meas,
                                                     correctionNoise);
        graphFactors.add(pose_factor);

        // (2) IMU preintegration factor: constrains pose+velocity at key-1 and
        // key, given the bias at key-1.
        const gtsam::PreintegratedImuMeasurements &preint_imu =
                dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(
                        *imuIntegratorOpt_);
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key),
                                    B(key - 1), preint_imu);
        graphFactors.add(imu_factor);

        // (3) bias random-walk factor: bias(key) should equal bias(key-1) up
        // to noise that grows with the elapsed time deltaTij.
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
                B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                gtsam::noiseModel::Diagonal::Sigmas(
                        sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));

        // Initial guesses for the new unknowns (IMU prediction + old bias).
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        
  
        // Run the incremental solver. The second update() call performs an
        // extra relinearization pass to refine the solution.
        bool systemSolvedSuccessfully = false;
        try {
            optimizer.update(graphFactors, graphValues);
            optimizer.update();
            systemSolvedSuccessfully = true;
        }
        catch (const gtsam::IndeterminantLinearSystemException &) {
            systemSolvedSuccessfully = false;
            RCLCPP_WARN(this->get_logger(), "Update failed due to underconstrained call to isam2 in imuPreintegration");
        }

        // ISAM2 keeps the factors internally; clear our staging containers.
        graphFactors.resize(0);
        graphValues.clear();

        if (systemSolvedSuccessfully) {
            // Store the new optimum, and reset the preintegrator with the
            // refined bias so the next inter-keyframe integration is cleaner.
            gtsam::Values result = optimizer.calculateEstimate();
            T_world_imu_prev = result.at<gtsam::Pose3>(X(key));
            prevVel_ = result.at<gtsam::Vector3>(V(key));
            prevState_ = gtsam::NavState(T_world_imu_prev, prevVel_);
            prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        }        
        return systemSolvedSuccessfully;
    }

    // The IMU-rate odometry published in imuHandler is always "last optimized
    // state + everything the IMU measured since". After each optimization the
    // starting point changed, so we must redo that sum: snapshot the new
    // state/bias, drop IMU samples older than the lidar pose, and re-integrate
    // the remaining (newer) samples with the refined bias.
    void imuPreintegration::repropagate_imuodometry(double currentCorrectionTime) {
        prevStateOdom = prevState_;
        prevBiasOdom = prevBias_;

        // Discard samples already covered by the optimization.
        double lastImuQT = -1;
        while (!imuQueImu.empty() && secs(&imuQueImu.front()) < currentCorrectionTime - delta_t) {
            lastImuQT = secs(&imuQueImu.front());
            imuQueImu.pop_front();
        }

        if (!imuQueImu.empty()) {
            // Restart the high-rate integrator from zero with the new bias and
            // replay the remaining samples.
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            for (int i = 0; i < (int)imuQueImu.size(); ++i) {
                sensor_msgs::msg::Imu *thisImu = &imuQueImu[i];
                double imuTime = secs(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 200.0) :(imuTime - lastImuQT);
                lastImuQT = imuTime;

                if(dt < 0.001 || dt > 0.5) 
                    dt = 0.005;

                imuIntegratorImu_->integrateMeasurement(
                    gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                    gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), 
                    dt);
                lastImuQT = imuTime;
            }
        }
    }

    // The per-lidar-pose pipeline, in order. Called from laserodometryHandler
    // for every lidar pose after the system is initialized.
    void imuPreintegration::process_imu_odometry(
        double currentCorrectionTime, gtsam::Pose3 T_world_lidar_meas) {

        // Keep the factor graph small: every 100 keyframes, carry the current
        // estimate over into a fresh graph (see reset_graph()).
        if (key > 100) {
            reset_graph();
        }

        // 1. Sum the IMU samples between the previous and this lidar pose.
        integrate_imumeasurement(currentCorrectionTime);

        this->T_world_lidar_meas = T_world_lidar_meas;

        // 2. Add this keyframe's factors and run the optimizer.
        bool successOptimization =
            build_graph(this->T_world_lidar_meas, currentCorrectionTime);
        
        // 3. If the optimizer failed or produced a physically absurd result,
        //    force a full re-initialization on the next lidar pose.
        if (failureDetection(prevVel_, prevBias_) || !successOptimization) {
            RCLCPP_WARN(this->get_logger(), "failureDetected");
            resetParams();
            return;
        }

        // 4. Rebase the high-rate IMU odometry onto the new optimum.
        repropagate_imuodometry(currentCorrectionTime);
        ++key;

        // From now on imuHandler may publish IMU-rate odometry.
        doneFirstOpt = true;
    }

    // Plausibility check on the optimized result. Thresholds are generous:
    // 30 m/s velocity, 2 m/s^2 accel bias, 1 rad/s gyro bias — real values
    // beyond these almost certainly mean the estimate diverged.
    bool imuPreintegration::failureDetection(const gtsam::Vector3 &velCur,
                                             const gtsam::imuBias::ConstantBias &biasCur) {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30) {
            RCLCPP_WARN(this->get_logger(), "Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(),
                           biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(),
                           biasCur.gyroscope().z());

        if (ba.norm() > 2.0 || bg.norm() > 1.0) {
            RCLCPP_WARN(this->get_logger(), "Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    // Callback for each lidar odometry pose (~10 Hz). This is where the
    // actual sensor fusion happens: initialize on the first pose, then for
    // every later pose run the integrate -> optimize -> re-propagate pipeline
    // and finally check the health of the IMU stream.
    void imuPreintegration::laserodometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg) {
        std::lock_guard<std::mutex> lock(mBuf);

        cur_frame = odomMsg;
        double lidarOdomTime = secs(odomMsg);

        // Without IMU data there is nothing to fuse yet.
        if (imuQueOpt.empty())
            return;

        // Unpack the lidar pose into a gtsam::Pose3.
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        gtsam::Pose3 T_world_lidar_meas(
            gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z),
            gtsam::Point3(p_x, p_y, p_z));

        // 0. initialize system
        if (systemInitialized == false) {
            initial_system(lidarOdomTime, T_world_lidar_meas);
            return;
        }

        TicToc Optimization_time;
        //1. process imu odometry
        process_imu_odometry(lidarOdomTime, T_world_lidar_meas);

        // 2. Health check: if the newest IMU sample is much older than this
        // lidar pose, the IMU stream has stalled (driver/hardware problem).
        double latest_imu_time = secs(&imuQueImu.back());

        if (lidarOdomTime - latest_imu_time < imu_laser_timedelay) {
            RESULT = IMU_STATE::SUCCESS;
            health_status = true;
          
            // The lidar odometry node signals its own failure through
            // covariance[0]; propagate it.
            if((int)odomMsg->pose.covariance[0] == 1) {
                RESULT = IMU_STATE::FAIL;
            }
         
        } else {
            health_status = false;
            if (cur_frame != nullptr && last_frame != nullptr) {
                Eigen::Vector3d velocity_curr;
                velocity_curr.x() =
                    (cur_frame->pose.pose.position.x - last_frame->pose.pose.position.x) /
                    (secs(cur_frame) - secs(last_frame));
                velocity_curr.y() = (cur_frame->pose.pose.position.y - last_frame->pose.pose.position.y) /
                                    (secs(cur_frame) - secs(last_frame));
                velocity_curr.z() = (cur_frame->pose.pose.position.z - last_frame->pose.pose.position.z) /
                                    (secs(cur_frame) - secs(last_frame));

                RCLCPP_INFO(this->get_logger(), "LOOSE CONNECTION WITH IMU DRIVER, PLEASE CHECK HARDWARE!!");
                RESULT = IMU_STATE::FAIL;
                health_status = false;
                nav_msgs::msg::Odometry odometry;

                std_msgs::msg::Bool health_status_msg;
                health_status_msg.data = health_status;
                pubHealthStatus->publish(health_status_msg);
            }
        }

        last_frame = cur_frame;
        last_processed_lidar_time = lidarOdomTime;
    }
    // Keep measurements in the physical IMU frame expected by GTSAM. Startup
    // leveling belongs in the world pose; lidar extrinsics are applied only at
    // lidar measurement and publishing boundaries.
    sensor_msgs::msg::Imu imuPreintegration::imuConverter(const sensor_msgs::msg::Imu &imu_in) {
        sensor_msgs::msg::Imu imu_out = imu_in;

        Eigen::Quaterniond q_world_imu(
            imu_in.orientation.w, imu_in.orientation.x,
            imu_in.orientation.y, imu_in.orientation.z);
        if (q_world_imu.squaredNorm() > 1e-12) {
            q_world_imu.normalize();
            imu_out.orientation.x = q_world_imu.x();
            imu_out.orientation.y = q_world_imu.y();
            imu_out.orientation.z = q_world_imu.z();
            imu_out.orientation.w = q_world_imu.w();
        }

        return imu_out;
    }


   // Callback for each raw IMU message (~200 Hz). Note that the heavy
   // math (optimization) happens in laserodometryHandler; this callback only
   // converts, buffers, and predicts.
   void imuPreintegration::imuHandler(const sensor_msgs::msg::Imu::SharedPtr imu_raw) {
    std::lock_guard<std::mutex> lock(mBuf);
    
    // 1. Preserve the physical IMU frame (orientation normalization only).
    sensor_msgs::msg::Imu thisImu = imuConverter(*imu_raw);

    // 2. During the first ~1 s, only collect data for the one-time IMU
    //    initialization (bias/gravity/leveling); nothing else can run yet.
    if (!handleIMUInitialization(imu_raw, thisImu)) {
        return;
    }

    // 3. Push the physical-frame sample into both queues (for optimization and
    //    for high-rate propagation).
    processTiming(thisImu);

    // 4. Until the first graph optimization there is no state to propagate from.
    if (!doneFirstOpt) {
        return;
    }

    // 5. Predict the current state = last optimized state + preintegrated
    //    IMU motion since then, and publish it. This is the "high-rate
    //    odometry" output of this node.
    gtsam::NavState currentState =imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);
    nav_msgs::msg::Odometry odometry;
    publishOdometry(thisImu, currentState, odometry);
    publishTransformsAndPath(odometry,  thisImu);   
   
   }

   // Gate for the one-time IMU initialization. Returns false (blocking the
   // rest of imuHandler) until Imu::imuInit() has run. Also applies the Livox
   // unit fix on every message, since Livox reports acceleration in g.
   bool imuPreintegration::handleIMUInitialization(const sensor_msgs::msg::Imu::SharedPtr&imu_raw, 
   sensor_msgs::msg::Imu& thisImu) {   

    if (!imu_init_success) {
        initializeImu(imu_raw);
    }

    if (!imu_init_success) {
        return false;
    }

    if (config_.sensor == SensorType::LIVOX) {
        correctLivoxGravity(thisImu);
    }
    
    return true;

   }

   // Collects raw IMU samples into imuBuf; once 1 s of data has accumulated,
   // Imu::imuInit() estimates gyro/accel statistics, gravity direction, and
   // initial tilt diagnostics without changing the estimator measurement frame.
   // The robot should be stationary during this window.
   void imuPreintegration::initializeImu(const sensor_msgs::msg::Imu::SharedPtr& imu_raw) {
    Imu::Ptr imudata = std::make_shared<Imu>();
    imudata->time = imu_raw->header.stamp.sec + imu_raw->header.stamp.nanosec * 1e-9;
    imudata->acc = Eigen::Vector3d(imu_raw->linear_acceleration.x,
                                  imu_raw->linear_acceleration.y,
                                  imu_raw->linear_acceleration.z);
    imudata->gyr = Eigen::Vector3d(imu_raw->angular_velocity.x,
                                  imu_raw->angular_velocity.y,
                                  imu_raw->angular_velocity.z);
    imudata->q_world_imu = Eigen::Quaterniond(imu_raw->orientation.w,
                                             imu_raw->orientation.x,
                                             imu_raw->orientation.y,
                                             imu_raw->orientation.z);

    imuBuf.addMeas(imudata, imudata->time);

    double first_imu_time = 0.0;
    imuBuf.getFirstTime(first_imu_time);

    if (imudata->time - first_imu_time > 1.0) {
        imu_Init->imuInit(imuBuf);
        imu_init_success = true;
        imuBuf.clean(imudata->time);
      std::cout<<"IMU Initialization Process Finish! "<<std::endl;
    }
}


// Livox IMUs report acceleration in units of g (a static sensor reads norm
// ~1.0 instead of ~9.81 m/s^2). Rescale using the stationary mean measured
// during initialization so a static sensor reads exactly 'gravity'.
void imuPreintegration::correctLivoxGravity(sensor_msgs::msg::Imu& thisImu) {
    const double gravity = 9.8105;
    Eigen::Vector3d acc(thisImu.linear_acceleration.x,
                       thisImu.linear_acceleration.y,
                       thisImu.linear_acceleration.z);
    acc = acc * gravity / imu_Init->acc_mean.norm();
    thisImu.linear_acceleration.x = acc.x();
    thisImu.linear_acceleration.y = acc.y();
    thisImu.linear_acceleration.z = acc.z();
}


// Updates the last-IMU-timestamp bookkeeping and appends the physical-frame sample
// to both queues: imuQueOpt (consumed by the optimizer between lidar poses)
// and imuQueImu (used for high-rate propagation past the last lidar pose).
void imuPreintegration::processTiming(const sensor_msgs::msg::Imu& thisImu) {
    double imuTime = secs(&thisImu);
    double dt = (lastImuT_imu < 0) ? (1.0 / 200.0) : (imuTime - lastImuT_imu);
    lastImuT_imu = imuTime;
    
    if (dt < 0.001 || dt > 0.5) {
        dt = 0.005;
    }

    imuQueOpt.push_back(thisImu);
    imuQueImu.push_back(thisImu);
}



void imuPreintegration::publishOdometry(
    const sensor_msgs::msg::Imu& thisImu,
    const gtsam::NavState& currentState, nav_msgs::msg::Odometry &odometry) {
    
    prepareOdometryMessage(odometry, thisImu, currentState);
    
    // Decimate: publish every 4th IMU sample (e.g. 200 Hz -> 50 Hz).
    if (frame_count++ % 4 == 0) {
        pubImuOdometry->publish(odometry);
    }

    // Publish health status
    std_msgs::msg::Bool health_status_msg;
    health_status_msg.data = health_status;
    pubHealthStatus->publish(health_status_msg);
}



void imuPreintegration::publishTransformsAndPath(nav_msgs::msg::Odometry &odometry, const sensor_msgs::msg::Imu& thisImu) {
    publishTransform(odometry, thisImu);
    updateAndPublishPath(odometry,thisImu);
}

void imuPreintegration::publishTransform(nav_msgs::msg::Odometry &odometry, const sensor_msgs::msg::Imu& thisImu){
    
    tf2_ros::TransformBroadcaster br(this);
    geometry_msgs::msg::TransformStamped transform_stamped_;
    tf2::Transform T_world_sensor_tf;
    transform_stamped_.header.stamp  = thisImu.header.stamp;
    transform_stamped_.header.frame_id = WORLD_FRAME;
    transform_stamped_.child_frame_id = SENSOR_FRAME;
    
    tf2::Quaternion q_world_sensor;
    T_world_sensor_tf.setOrigin(tf2::Vector3(
        odometry.pose.pose.position.x,
        odometry.pose.pose.position.y,
        odometry.pose.pose.position.z));

    q_world_sensor.setW(odometry.pose.pose.orientation.w);
    q_world_sensor.setX(odometry.pose.pose.orientation.x);
    q_world_sensor.setY(odometry.pose.pose.orientation.y);
    q_world_sensor.setZ(odometry.pose.pose.orientation.z);
    T_world_sensor_tf.setRotation(q_world_sensor);
    transform_stamped_.transform = tf2::toMsg(T_world_sensor_tf);
    if(frame_count%4==0)
        br.sendTransform(transform_stamped_);
}

// Maintains a short visualization path: one pose every 0.1 s, keeping only
// the last 3 s, published when someone (e.g. RViz) is subscribed.
void imuPreintegration::updateAndPublishPath(nav_msgs::msg::Odometry &odometry, const sensor_msgs::msg::Imu& thisImu){
    static nav_msgs::msg::Path imuPath;
    static double last_path_time = -1;
    double curimuTime = secs(&thisImu);
    if (curimuTime - last_path_time > 0.1)
    {
        last_path_time = curimuTime;
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = thisImu.header.stamp;
        pose_stamped.header.frame_id = WORLD_FRAME;
        pose_stamped.pose = odometry.pose.pose;
        imuPath.poses.push_back(pose_stamped);
        while (!imuPath.poses.empty() &&
                abs(secs(&imuPath.poses.front()) -
                    secs(&imuPath.poses.back())) > 3.0)
            imuPath.poses.erase(imuPath.poses.begin());
        if (pubImuPath->get_subscription_count() != 0)
        {
            imuPath.header.stamp = thisImu.header.stamp;
            imuPath.header.frame_id = WORLD_FRAME;
            pubImuPath->publish(imuPath);
        }
    }
}

// Converts the predicted NavState (which is the IMU pose in the world frame)
// into the odometry message consumers expect: lidar pose in the world frame,
// body-frame velocity, bias-corrected angular rate, plus health/bias values
// smuggled into the covariance array.
void imuPreintegration::prepareOdometryMessage( nav_msgs::msg::Odometry &odometry, 
const sensor_msgs::msg::Imu &thisImu, const gtsam::NavState &currentState){
    
    // Orientation source: either the IMU driver's own attitude filter
    // (use_imu_roll_pitch) or the optimized/predicted state.
    Eigen::Quaterniond q_world_imu;
    if (config_.use_imu_roll_pitch) {
        q_world_imu = Eigen::Quaterniond(
            thisImu.orientation.w, thisImu.orientation.x,
            thisImu.orientation.y, thisImu.orientation.z);
    } else {
        q_world_imu = Eigen::Quaterniond(
            currentState.quaternion().w(), currentState.quaternion().x(),
            currentState.quaternion().y(), currentState.quaternion().z());
    }

    // The state refers to the IMU; convert to the lidar pose via the extrinsic.
    gtsam::Rot3 R_world_imu(q_world_imu);
    gtsam::Pose3 T_world_imu(R_world_imu, currentState.position());
    gtsam::Pose3 T_world_lidar = T_world_imu.compose(T_imu_lidar);

    // Velocity is estimated in the world frame; rotate it into the body frame
    // for the twist field (ROS convention: twist is in child_frame_id).
    Eigen::Vector3d velocity_world_current(
        currentState.velocity().x(), currentState.velocity().y(),
        currentState.velocity().z());
    Eigen::Vector3d velocity_imu_current =
        currentState.quaternion().inverse() * velocity_world_current;
    
    
    odometry.header.stamp = thisImu.header.stamp;
    odometry.header.frame_id = WORLD_FRAME;
    odometry.child_frame_id = SENSOR_FRAME;
    
    Eigen::Quaterniond q_world_lidar(
        T_world_lidar.rotation().toQuaternion().w(),
        T_world_lidar.rotation().toQuaternion().x(),
        T_world_lidar.rotation().toQuaternion().y(),
        T_world_lidar.rotation().toQuaternion().z());
    q_world_lidar.normalize();

    odometry.pose.pose.position.x = T_world_lidar.translation().x();
    odometry.pose.pose.position.y = T_world_lidar.translation().y();
    odometry.pose.pose.position.z = T_world_lidar.translation().z();
    odometry.pose.pose.orientation.x = q_world_lidar.x();
    odometry.pose.pose.orientation.y = q_world_lidar.y();
    odometry.pose.pose.orientation.z = q_world_lidar.z();
    odometry.pose.pose.orientation.w = q_world_lidar.w();

    odometry.twist.twist.linear.x = velocity_imu_current.x();
    odometry.twist.twist.linear.y = velocity_imu_current.y();;
    odometry.twist.twist.linear.z = velocity_imu_current.z();;
    // GTSAM's bias convention is measurement = truth + bias, so recover the
    // physical IMU angular rate by subtracting the estimated gyro bias.
    odometry.twist.twist.angular.x =
            thisImu.angular_velocity.x - prevBiasOdom.gyroscope().x();
    odometry.twist.twist.angular.y =
            thisImu.angular_velocity.y - prevBiasOdom.gyroscope().y();
    odometry.twist.twist.angular.z =
            thisImu.angular_velocity.z - prevBiasOdom.gyroscope().z();

    // The covariance array is repurposed as a side channel:
    // [0] = IMU health state (IMU_STATE enum), [1..3] = accel bias,
    // [4..6] = gyro bias, [7] = gravity magnitude. Downstream nodes read
    // these instead of a real covariance.
    odometry.pose.covariance[0] = double(RESULT);

    odometry.pose.covariance[1] = prevBiasOdom.accelerometer().x();
    odometry.pose.covariance[2] = prevBiasOdom.accelerometer().y();
    odometry.pose.covariance[3] = prevBiasOdom.accelerometer().z();
    odometry.pose.covariance[4] = prevBiasOdom.gyroscope().x();
    odometry.pose.covariance[5] = prevBiasOdom.gyroscope().y();
    odometry.pose.covariance[6] = prevBiasOdom.gyroscope().z();
    odometry.pose.covariance[7] = config_.imuGravity;
}


} // end namespace super_odometry