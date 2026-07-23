
// LOCAL
#include "super_odometry/LidarProcess/LidarSlam.h"

// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// Implementation of the scan-to-map registration engine (see LidarSlam.h for
// the big picture). The call graph for one scan is:
//
//   Localization()
//     -> initializeState()                 adopt the pose prediction
//     -> processInputClouds()              copy edge/planar features
//     -> first scan:  initializeMapping()  seed the local map
//     -> later scans: EstimateLidarUncertainty()
//                     performLocalizationAndMapping()
//         -> prepareOptimizationState()    shift map, load pose into Ceres
//         -> loop up to LocalizationICPMaxIter times:
//              extractFeaturesConstraints()   ICP data association:
//                -> ComputeLineDistanceParameters()   per edge point
//                -> ComputePlaneDistanceParameters()  per planar point
//              setupOptimizationProblem()     build Ceres residuals
//              solveOptimizationProblem()     nonlinear least squares
//              (stop early when converged)
//         -> EstimateRegistrationError()   pose covariance / degeneracy
//         -> performPostOptimizationProcessing()  map update, stats
// ============================================================================

//TODO: add to header file
// The 6-DoF pose being optimized, in the memory layout Ceres works on
// directly: [x, y, z, qx, qy, qz, qw] (translation + unit quaternion, world
// frame). The two Eigen::Map objects are zero-copy views into this array, so
// writing t_world_lidar / q_world_lidar updates the solver state and vice versa.
namespace {
double pose_parameters[7] = {0, 0, 0, 0, 0, 0, 1};
Eigen::Map<Eigen::Vector3d> t_world_lidar(pose_parameters);
Eigen::Map<Eigen::Quaterniond> q_world_lidar(pose_parameters + 3);
}  // namespace

namespace super_odometry {


    // Allocates the point cloud buffers reused for every scan.
    LidarSLAM::LidarSLAM() {
        EdgesPoints.reset(new PointCloud());
        PlanarsPoints.reset(new PointCloud());
        WorldEdgesPoints.reset(new PointCloud());
        WorldPlanarsPoints.reset(new PointCloud());
        pcl_to_save.reset(new pcl::PointCloud<pcl::PointXYZI>());
    }
    // Stores the ROS node handle (this class is not a node itself; it lives
    // inside the laserMapping node) and creates the six debug publishers for
    // the per-axis pose uncertainty.
    void LidarSLAM::initROSInterface(rclcpp::Node::SharedPtr node) {
        node_ = node;
        pubUncertaintyX=node_->create_publisher<std_msgs::msg::Float32>(ProjectName+"uncertainty_X", 1);
        pubUncertaintyY=node_->create_publisher<std_msgs::msg::Float32>(ProjectName+"uncertainty_Y", 1);
        pubUncertaintyZ=node_->create_publisher<std_msgs::msg::Float32>(ProjectName+"uncertainty_Z", 1);
        pubUncertaintyRoll=node_->create_publisher<std_msgs::msg::Float32>(ProjectName+"uncertainty_roll", 1);
        pubUncertaintyPitch=node_->create_publisher<std_msgs::msg::Float32>(ProjectName+"uncertainty_pitch", 1);
        pubUncertaintyYaw=node_->create_publisher<std_msgs::msg::Float32>(ProjectName+"uncertainty_yaw", 1);
    }

    // Entry point called once per scan by the laserMapping node.
    // 'T_world_lidar_guess' is the pose prediction (from IMU preintegration, VIO, or
    // constant velocity); 'initialization' is false only for the very first
    // scan, which just seeds the map because there is nothing to register
    // against yet.
    void LidarSLAM::Localization(
        bool initialization,
        PredictionSource predictodom,
        Transformd T_world_lidar_guess,
        pcl::PointCloud<Point>::Ptr edge_point,
        pcl::PointCloud<Point>::Ptr planner_point,
        double timeLaserOdometry){  
       
       // Initialize state with the current world-from-lidar pose guess.
       initializeState(initialization, T_world_lidar_guess);

       //ProcessInputClouds (we remove the edge points in optimization step)
       processInputClouds(edge_point, planner_point);

       //Intialize and perform localization and mapping
       if(!initialization){
        initializeMapping(timeLaserOdometry);
       } else{
        // Uncertainty is computed from the observability histogram filled
        // during the PREVIOUS scan's data association; it also feeds the
        // per-axis weights of the optional VIO pose prior below.
        EstimateLidarUncertainty();
        performLocalizationAndMapping(predictodom, timeLaserOdometry); 
       }
    }
     
    // Adopts the caller's prediction as both the working pose (which the
    // optimizer will refine) and the remembered initial guess (used later to
    // report how far the optimization moved the pose).
    void LidarSLAM::initializeState(
        bool initialization, const Transformd& T_world_lidar_guess) {
        T_world_lidar=T_world_lidar_guess;
        T_world_lidar_initial_guess=T_world_lidar_guess;
        T_world_lidar_prev=T_world_lidar;
    }
    

    // Transforms one feature cloud from the lidar frame into the world frame
    // using the (freshly optimized) pose T_world_lidar and inserts it into the
    // matching layer of the local map. This is the "mapping" half of SLAM:
    // the registered scan becomes part of the reference map for future scans.
    void LidarSLAM::transformAndAddToMap(const pcl::PointCloud<Point>::Ptr&source_cloud, 
        pcl::PointCloud<Point>::Ptr&world_cloud, bool is_edge){

         //prepare point cloud 
         world_cloud->clear();
         world_cloud->points.reserve(source_cloud->size());
         world_cloud->header=source_cloud->header;

         //Transform points to world frame 
         for (const Point&p: *source_cloud){
            world_cloud->push_back(utils::TransformPointd(p,T_world_lidar));
         }

         //Add to local map 
         if(is_edge){
            localMap.addEdgePointCloud(*world_cloud);
         }else{
            localMap.addSurfPointCloud(*world_cloud);
         }

         // Save world cloud to .ply file
         if (SAVE_PLY) {
            *pcl_to_save += *world_cloud;
            utils::savePly(pcl_to_save, node_);
         }

    }
    

    // First-scan handling: there is no map yet, so the scan cannot be
    // registered. Center the local map at the initial pose and insert the
    // first feature clouds as the seed map.
    void LidarSLAM::initializeMapping(double timeLaserOdometry){
        //clear map and reset statistics 
        
        //set origin for local map 
        localMap.setOrigin(T_world_lidar.pos);

        //Transform and add feature points to map 
        transformAndAddToMap(EdgesPoints, WorldEdgesPoints, true);
        transformAndAddToMap(PlanarsPoints, WorldPlanarsPoints, false);

        lasttimeLaserOdometry=timeLaserOdometry;
    }
    

    // Copies the incoming feature clouds (still in the lidar frame) into the
    // member buffers used throughout the rest of the pipeline.
    void LidarSLAM::processInputClouds(const pcl::PointCloud<Point>::Ptr&edge_point, const pcl::PointCloud<Point>::Ptr&planner_point){
        //clear and reserve space for efficiency 
        EdgesPoints->clear();
        PlanarsPoints->clear();
        EdgesPoints->reserve(edge_point->size());
        PlanarsPoints->reserve(planner_point->size());
        *EdgesPoints=*edge_point;
        *PlanarsPoints=*planner_point;
    }    
    
    // The pose estimation loop. Classic ICP structure: because we do not know
    // the true point correspondences, we alternate between (a) matching each
    // feature to the map using the CURRENT pose estimate and (b) solving for
    // the pose that best fits those matches. As the pose improves, the
    // matches improve, so a few outer iterations usually suffice.
     void LidarSLAM::performLocalizationAndMapping(PredictionSource predictodom, double timeLaserOdometry)
    {  
        //intialize optimization state 
        prepareOptimizationState();

        //Check if we have enough features for optimization 
        if(!hasEnoughFeatures()){
            RCLCPP_WARN(node_->get_logger(), "Not enough features for optimization");
            return;
        }
        //Perform ICP iteration 
        TicToc t_opt;
        for (size_t icp_iter=0; icp_iter<LocalizationICPMaxIter; ++icp_iter){
        //Extract features 
        int edge_num=0; int planner_num=0;
        ResetDistanceParameters();
        super_odometry_msgs::msg::IterationStats iter_stats;

        // Data association: match every feature point against the local map
        // and collect one residual description per successful match.
        tbb::concurrent_vector<OptimizationParameter> feature_corres;
        extractFeaturesConstraints(feature_corres, edge_num, planner_num);
        
        //Setup and solve the optimization problem 
        Transformd T_world_lidar_before_iteration(T_world_lidar);
      
        auto problem=setupOptimizationProblem(
            feature_corres, predictodom, T_world_lidar_initial_guess);
        auto summary=solveOptimizationProblem(problem);
        
        // Copy the solution back from the raw Ceres parameter array
        // (t_world_lidar / q_world_lidar are views into pose_parameters).
        T_world_lidar.pos=t_world_lidar;
        T_world_lidar.rot=q_world_lidar;
        //Record iteration statistics 
        recordIterationStats(iter_stats, planner_num, edge_num,
                             T_world_lidar_before_iteration, T_world_lidar);

        // Convergence test: if Ceres accepted only ONE step, the pose was
        // already so close that re-associating would change nothing, so we
        // treat the estimate as converged. Either way (converged or out of
        // iterations) we estimate the pose covariance from the final problem
        // for the degeneracy / uncertainty report.
        if ((summary.num_successful_steps == 1) ||(icp_iter == this->LocalizationICPMaxIter - 1)) {
            this->LocalizationUncertainty =
                    EstimateRegistrationError(problem, 100);
            break;
        
        }

      }
      
      //post-optimization processing 
      performPostOptimizationProcessing(timeLaserOdometry, t_opt, stats);
    }

    
    // Wrap-up after the ICP loop: apply the yaw drift correction, compute
    // motion statistics, and (if the motion passes the sanity checks) insert
    // the registered scan into the local map so future scans can match it.
    void LidarSLAM::performPostOptimizationProcessing(double timeLaserOdometry, TicToc &t_opt, super_odometry_msgs::msg::OptimizationStats &stats) {
        // Apply manual yaw correction
        MannualYawCorrection();
        
        // Update statistics
        updateOptimizationStats(t_opt, stats);
        
        // Check motion thresholds and update map
        if (checkMotionThresholds(timeLaserOdometry, stats)) {
            // Transform and add new features to map
            transformAndAddToMap(EdgesPoints, WorldEdgesPoints, true);
            transformAndAddToMap(PlanarsPoints, WorldPlanarsPoints, false);
        }
        
        // Update timing
        lasttimeLaserOdometry = timeLaserOdometry;
    }

    // Sanity checks on the estimated motion since the previous scan.
    // Intended behavior: reject the result if the implied velocity is
    // impossibly large (registration probably diverged; fall back to the
    // previous pose) or if the motion is negligible (< 2 cm and < 0.005 rad;
    // adding a near-duplicate scan would only bloat the map). NOTE: the
    // unconditional "acceptResult = true" before the return overrides both
    // checks, so the pose reverts above but the map is currently always
    // updated; only the warnings remain effective.
    bool LidarSLAM::checkMotionThresholds(double timeLaserOdometry, super_odometry_msgs::msg::OptimizationStats &stats) {
    
        bool acceptResult = true;
        double delta_t = timeLaserOdometry - lasttimeLaserOdometry;
        
        // Check velocity threshold
        if (stats.translation_from_last/delta_t > OptSet.velocity_failure_threshold) {
            T_world_lidar = T_world_lidar_prev;
            startupCount = 5;
            acceptResult = false;
            RCLCPP_WARN(node_->get_logger(), "large motion detected, ignoring predictor for a while");
        }
        
        // Check small motion threshold
        if (stats.translation_from_last < 0.02 && stats.rotation_from_last < 0.005) {
            acceptResult = false;
            T_world_lidar = T_world_lidar_prev;
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                "very small motion, not accumulating. %f", stats.translation_from_last);
        }
    acceptResult = true;
    return acceptResult;
}


    // Fills the stats message with how far the optimizer moved the pose.
    // "total_*" compares against the initial guess (how much the scan
    // matching corrected the prediction); "*_from_last" compares against the
    // previous scan's pose (the actual motion). The rotation angle of a
    // quaternion q is recovered as 2*atan2(|q.vec|, q.w).
    void LidarSLAM::updateOptimizationStats(TicToc &t_opt, super_odometry_msgs::msg::OptimizationStats &stats){
        double time_duration = t_opt.toc();
        stats.time_elapsed = time_duration;
        Transformd T_lidar_initial_guess_lidar_optimized;
        T_lidar_initial_guess_lidar_optimized =
            T_world_lidar_initial_guess.inverse() * T_world_lidar;
        stats.total_translation =
            T_lidar_initial_guess_lidar_optimized.pos.norm();
        stats.total_rotation =
            2 * atan2(T_lidar_initial_guess_lidar_optimized.rot.vec().norm(),
                      T_lidar_initial_guess_lidar_optimized.rot.w());
        Transformd T_lidar_prev_lidar_current =
            T_world_lidar_prev.inverse() * T_world_lidar;

        stats.translation_from_last = T_lidar_prev_lidar_current.pos.norm();
        stats.rotation_from_last =
            2 * atan2(T_lidar_prev_lidar_current.rot.vec().norm(),
                      T_lidar_prev_lidar_current.rot.w());
        T_world_lidar_prev=T_world_lidar;
    }


    // Builds the nonlinear least-squares problem for one ICP iteration.
    // The single parameter block is the 7-value pose array; the
    // PoseLocalParameterization tells Ceres the pose really has only 6
    // degrees of freedom (updates keep the quaternion on the unit sphere,
    // i.e. optimization happens on the SE(3) manifold).
    ceres::Problem LidarSLAM::setupOptimizationProblem(
        const tbb::concurrent_vector<OptimizationParameter>& features_corres,
        PredictionSource predictsource,
        const Transformd& T_world_lidar_guess) {
        ceres::Problem::Options problem_options; 
        ceres::Problem problem(problem_options);
        problem.AddParameterBlock(pose_parameters, 7, new PoseLocalParameterization());

        //Add feature constraints
        addFeatureConstraints(problem, features_corres);

       
        //Add absolute pose constraints if needed 
        if(shouldAddAbsolutePoseConstraints(predictsource)){
            addAbsolutePoseConstraints(
                problem, T_world_lidar_guess, features_corres.size());
        }
        return problem;
    }

    // Runs the Ceres solver (Levenberg-Marquardt by default). The inner
    // iteration count is kept small (4) on purpose: it is cheaper to take a
    // few steps, re-associate the points with the improved pose in the outer
    // ICP loop, and solve again, than to over-optimize stale matches.
    ceres::Solver::Summary LidarSLAM::solveOptimizationProblem(ceres::Problem&problem){
        ceres::Solver::Options options;
        options.max_num_iterations=4;
        options.linear_solver_type=ceres::DENSE_QR;
        options.minimizer_progress_to_stdout=false;
        options.check_gradients=false;
        options.gradient_check_relative_precision=1e-4;
        ceres::Solver::Summary summary; 
        ceres::Solve(options, &problem, &summary);
        return summary;
    }

    // Logs, for one ICP iteration, how many features matched and how much
    // the pose moved (used to judge convergence behavior offline).
    void LidarSLAM::recordIterationStats(
        super_odometry_msgs::msg::IterationStats& iter_stats, int surf_num,
        int edge_num, Transformd& T_world_lidar_prev,
        Transformd& T_world_lidar_current) {
        //Record iteration statistics 
        iter_stats.num_surf_from_scan=surf_num;
        iter_stats.num_corner_from_scan=edge_num;
        Transformd T_lidar_prev_lidar_current =
            T_world_lidar_prev.inverse() * T_world_lidar_current;
        iter_stats.translation_norm =
            T_lidar_prev_lidar_current.pos.norm();
        iter_stats.rotation_norm =
            2 * atan2(T_lidar_prev_lidar_current.rot.vec().norm(),
                      T_lidar_prev_lidar_current.rot.w());
        stats.iterations.push_back(iter_stats);
    }
    

    // Converts each successful match into a Ceres residual block.
    //  - Edge match: EdgeAnalyticCostFunction measures the distance from the
    //    transformed scan point to the 3D line through corres.first and
    //    corres.second (point-to-line residual).
    //  - Plane match: SurfNormAnalyticCostFunction measures the signed
    //    distance n . (T * p) + d to the fitted plane (point-to-plane
    //    residual).
    // Each residual is wrapped in a Tukey robust loss, which smoothly caps
    // the influence of large residuals so a few wrong matches (outliers)
    // cannot drag the pose away, then scaled by the match's fit quality
    // (residualCoefficient in [0,1]).
    void LidarSLAM::addFeatureConstraints(ceres::Problem&problem, const tbb::concurrent_vector<OptimizationParameter>&features_corres){
        //Add edge constraints 
       int edge_num=0;
       int planner_num=0;
        for(const auto&constraint: features_corres){
            if(constraint.feature_type==FeatureType::EdgeFeature){
            ceres::CostFunction*cost_function=new EdgeAnalyticCostFunction
            (constraint.Xvalue, constraint.corres.first, constraint.corres.second);
            // Use a robustifier to limit the outlier contribution 
            auto *loss_function=new ceres::TukeyLoss(std::sqrt(3*localMap.lineRes_));
            // Weight the contribution of the given match by its reliability 
            auto *weight_function=new ceres::ScaledLoss(loss_function, constraint.residualCoefficient, ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(cost_function, weight_function, pose_parameters);
            edge_num++;
            }else if(constraint.feature_type==FeatureType::PlaneFeature){
                ceres::CostFunction*cost_function=new SurfNormAnalyticCostFunction(constraint.Xvalue, constraint.NormDir, constraint.negative_OA_dot_norm);
                // Use a robustifier to limit the outlier contribution 
                auto *loss_function=new ceres::TukeyLoss(std::sqrt(3*localMap.planeRes_));
                // Weight the contribution of the given match by its reliability 
                auto *weight_function=new ceres::ScaledLoss(loss_function, constraint.residualCoefficient, ceres::TAKE_OWNERSHIP);
                problem.AddResidualBlock(cost_function, weight_function, pose_parameters);
                planner_num++;
            }
        }
        stats.prediction_source=0;
    }
    
    // The external pose prior is a rescue mechanism, used only when all three
    // hold: the prediction comes from visual-inertial odometry, the lidar
    // geometry is degenerate (some pose directions unconstrained by the scan
    // matching), and the operator gave the prior a non-zero weight.
    bool LidarSLAM::shouldAddAbsolutePoseConstraints(PredictionSource predictodom){
        return predictodom==PredictionSource::VIO_ODOM and isDegenerate==true and Visual_confidence_factor!=0;
    }

    // Adds a unary factor pulling the solution toward the VIO pose. The 6x6
    // information matrix (inverse covariance; DoF order x,y,z,rx,ry,rz) sets
    // per-axis strength: translation axes are weighted by how UNcertain the
    // lidar is on that axis (1 - uncertainty is the lidar confidence, so the
    // prior is strongest exactly where the lidar is weakest), scaled with the
    // number of good feature matches to stay comparable to the summed feature
    // residuals. The yaw weight is multiplied by 0, i.e. the VIO yaw is
    // deliberately ignored (lidar yaw is usually more reliable than VIO yaw).
    void LidarSLAM::addAbsolutePoseConstraints(
        ceres::Problem& problem, const Transformd& T_world_lidar_guess,
        int good_feature_num) {
        //Add absolute pose constraint 
       Eigen::Matrix<double, 6, 6, Eigen::RowMajor> information;
       information.setIdentity();
       information(0, 0) =(1 - lidarOdomUncer.uncertainty_x) * std::max(50, int(good_feature_num*0.1))* Visual_confidence_factor;
       information(1, 1) =(1 - lidarOdomUncer.uncertainty_y) * std::max(50, int(good_feature_num*0.1))* Visual_confidence_factor;
       information(2, 2) =(1 - lidarOdomUncer.uncertainty_z) * std::max(50, int(good_feature_num*0.1))* Visual_confidence_factor;
       information(3, 3) = std::max(10, int(good_feature_num*0.01)) * Visual_confidence_factor;
       information(4, 4) = std::max(10, int(good_feature_num*0.01)) * Visual_confidence_factor;
       information(5, 5) = std::max(5, int(good_feature_num*0.001)) * 0;     
       SE3AbsolutatePoseFactor *absolutatePoseFactor =
           new SE3AbsolutatePoseFactor(T_world_lidar_guess, information);
       problem.AddResidualBlock(absolutatePoseFactor, nullptr, pose_parameters);
       stats.prediction_source=1;
    }
    // One round of ICP data association: matches every feature point of the
    // scan against the local map and collects the successful residuals.
    void LidarSLAM::extractFeaturesConstraints(
        tbb::concurrent_vector<LidarSLAM::OptimizationParameter>&feature_corres,
        int &edge_num, int &planner_num){

        //Process edge features 
        processEdgeFeatures(feature_corres, edge_num);

        //Process planner features 
        processPlannerFeatures(feature_corres,planner_num);
    }

    // Tries to match every edge point to a line in the map; keeps successful
    // matches and tallies the rejection causes for diagnostics.
    void LidarSLAM::processEdgeFeatures(tbb::concurrent_vector<OptimizationParameter>&features_corres, int &edge_num){
        if(EdgesPoints->empty()) return; 
        edge_num=0;
        for(const auto&p: *EdgesPoints){
            auto constraint=ComputeLineDistanceParameters(localMap, p);
            if(constraint.match_result==MatchingResult::SUCCESS){
                features_corres.push_back(constraint);
                edge_num++;
            }
        MatchRejectionHistogramLine[constraint.match_result]++;
        }
    }

    // Same as processEdgeFeatures but for planar points, with two additions:
    // the cloud is subsampled to at most max_surface_features points (planar
    // points are plentiful and processing all of them is unnecessary), and
    // each successful match votes in the observability histogram for the
    // pose axes it constrains (input to the degeneracy analysis).
    void LidarSLAM::processPlannerFeatures(tbb::concurrent_vector<OptimizationParameter>&features_corres, int &planner_num){
        if(PlanarsPoints->empty()) return;
        
        double sampling_rate=calculateSamplingRate(PlanarsPoints->size());
        planner_num=0;
        for(size_t i=0; i<PlanarsPoints->size(); ++i){
            if(!shouldProcessPoint(i,sampling_rate)) continue;
            const Point&p=PlanarsPoints->points[i];
            auto constraint=ComputePlaneDistanceParameters(localMap, p);
            if(constraint.match_result==MatchingResult::SUCCESS){
                features_corres.push_back(constraint);
                planner_num++;
                // Each match votes for its top-2 rotation axes and top
                // translation axis (see analyzeFeatureObservability).
                const auto&obs=constraint.feature.observability;
                PlaneFeatureHistogramObs[obs[0]]++;
                PlaneFeatureHistogramObs[obs[1]]++;
                PlaneFeatureHistogramObs[obs[2]]++;
            }
            MatchRejectionHistogramPlane[constraint.match_result]++;
        }
       
    } 

    // Fraction of planar points to keep so that roughly max_surface_features
    // survive; -1 is the sentinel for "cloud already small enough, keep all".
    double LidarSLAM::calculateSamplingRate(size_t num_points){
        if(num_points>OptSet.max_surface_features){   
            return 1.0*OptSet.max_surface_features/num_points;
        }
        return -1.0;
    }

    // Deterministic decimation that spreads the kept points evenly over the
    // cloud (better spatial coverage than taking the first N points):
    // index*rate advances by 'rate' per point, and a point is kept exactly
    // when that value crosses an integer boundary. The 0.001 guards against
    // floating-point rounding at the boundary.
    bool LidarSLAM::shouldProcessPoint(size_t index, double sampling_rate){
        if(sampling_rate<0.0) return true;
        double remainder = fmod(index*sampling_rate, 1.0);
        if (remainder + 0.001 > sampling_rate)
            return false;
        return true;
    }

    // Prepares one scan's optimization: re-centers the sliding local map on
    // the predicted position (dropping voxels that fall out of the window),
    // seeds the Ceres parameter array with the predicted pose (the "initial
    // guess" of the solve), and records how many map features surround us.
    void LidarSLAM::prepareOptimizationState(){
        
        pos_in_localmap=localMap.shiftMap(T_world_lidar.pos);
        t_world_lidar=T_world_lidar.pos;
        q_world_lidar=T_world_lidar.rot;

        auto [edge_count, planner_count]=localMap.get5x5LocalMapFeatureSize(pos_in_localmap);
        updateFeatureStats(edge_count, planner_count);
    } 

    // Copies the per-scan feature counts into the stats message and clears
    // the per-iteration log for the new scan.
    void LidarSLAM::updateFeatureStats(size_t edge_count, size_t planner_count){
        stats.laser_cloud_corner_from_map_num = edge_count;
        stats.laser_cloud_surf_from_map_num = planner_count;
        stats.laser_cloud_corner_stack_num = EdgesPoints->size();
        stats.laser_cloud_surf_stack_num = PlanarsPoints->size();
        stats.iterations.clear();
    }
    
    // Registration needs a minimum amount of map geometry around the robot;
    // with fewer than ~50 planar map points the problem would be too weakly
    // constrained to trust.
    bool LidarSLAM::hasEnoughFeatures(){
         return stats.laser_cloud_surf_from_map_num>50;
    }

    // Produces the two representations of a scan point that matching needs:
    // pInit is the point in the LIDAR frame (kept raw so the optimizer can
    // re-transform it as the pose changes), and pFinal is the point in the
    // WORLD frame under the current pose estimate (used only to query the
    // map for neighbors). The first two branches are placeholders for the
    // OPTIMIZED / APPROXIMATED undistortion modes, which are not implemented;
    // with UndistortionMode::NONE the scan is treated as rigid.
    void LidarSLAM::ComputePointInitAndFinalPose(
            LidarSLAM::MatchingMode matchingMode, const LidarSLAM::Point &p,
            Eigen::Vector3d &pInit, Eigen::Vector3d &pFinal) {
      
        const bool is_local_lization_step =
                matchingMode == MatchingMode::LOCALIZATION;
        const Eigen::Vector3d pos = p.getVector3fMap().cast<double>();

        if (this->Undistortion == UndistortionMode::OPTIMIZED and
            is_local_lization_step) {

        } else if (this->Undistortion == UndistortionMode::APPROXIMATED and
                   is_local_lization_step) {

        } else {
            pInit = pos;
            pFinal = this->T_world_lidar * pos;
        }
    }

// Full matching pipeline for one EDGE point. The idea: an edge point in the
// scan (e.g. on a pole or wall corner) should land near map points from the
// same physical edge, and those map points should form a straight LINE. If
// they do, the perpendicular distance from the scan point to that line is an
// error the optimizer can shrink by adjusting the pose.
LidarSLAM::OptimizationParameter LidarSLAM::ComputeLineDistanceParameters(
            LocalMap &local_map, const LidarSLAM::Point &p) {
    // 1. Initialize point
    Eigen::Vector3d pInit, pFinal;
    OptimizationParameter result;
    Eigen::Vector3d mean;
    Eigen::Vector3d eigenvalues;
    Eigen::Matrix3d eigenvectors;
    if (!initializeAndTransformPoint(p, pInit, pFinal)) {
        result.match_result = MatchingResult::INVAVLID_NUMERICAL;
        return result;
    }

    // 2. Nearest-neighbor data association: query the edge layer of the map
    // around the world-frame position predicted by the current pose.
    std::vector<Point> nearest_pts;
    std::vector<float> nearest_dist;
    Point query{pFinal.x(), pFinal.y(), pFinal.z()};
    bool found = local_map.nearestKSearchSpecificEdgePoint(
                query, nearest_pts, nearest_dist, LocalizationLineDistanceNbrNeighbors,
                static_cast<float>(this->LocalizationLineMaxDistInlier));
 
    if (!validateNeighborSearch(found, nearest_pts, nearest_dist, result)) {
        return result;
    }

    // 3. Fit a line to the neighbors by eigen-decomposition of their
    // covariance and reject if they do not really form a line.
    if (!computePCAForFeature(nearest_pts, mean, eigenvalues, eigenvectors, result, FeatureType::EdgeFeature)) {
        return result;
    }

    // 4. Build the point-to-line residual from the validated fit.
    result=processLineResults(pInit, mean, eigenvalues, eigenvectors, nearest_pts, 3*local_map.lineRes_);
    return result;
}


// Turns a validated line fit into a point-to-line residual description.
// Inputs come from the PCA of the neighbors: 'mean' is a point on the line
// (the neighborhood centroid) and the largest-eigenvalue eigenvector is the
// line direction.
LidarSLAM::OptimizationParameter LidarSLAM::processLineResults(
                                  const Eigen::Vector3d &pInit,
                                  const Eigen::Vector3d &mean,
                                  const Eigen::Vector3d &eigenvalues,
                                  const Eigen::Matrix3d &eigenvectors,
                                  const std::vector<Point> &nearest_pts,
                                  double square_max_dist) {
    OptimizationParameter result;
  // 1. Line direction = eigenvector of the LARGEST eigenvalue (Eigen's
  // SelfAdjointEigenSolver sorts eigenvalues ascending, so column 2).
    Eigen::Vector3d line_direction = eigenvectors.col(2);
    line_direction.normalize();

    // 2. P = I - d*d^T removes the component along the line direction d, so
    // P*(x - mean) is the perpendicular offset of x from the line and
    // (x-mean)^T * P * (x-mean) its squared point-to-line distance.
    Eigen::Matrix3d projection_matrix = Eigen::Matrix3d::Identity() - 
    line_direction * line_direction.transpose();
    
    // 3. Validate projection matrix
    if (!projection_matrix.allFinite()) {
        result.match_result = MatchingResult::INVAVLID_NUMERICAL;
        return result;
    }

    // 4. Quality check: every neighbor must lie close to the fitted line,
    // otherwise the "edge" is not clean and the match is rejected. The
    // threshold scales with the edge-map voxel resolution lineRes_ (m).
    double meanSquareDist = 0.0;
    for (const auto &pt : nearest_pts) {
        Eigen::Vector3d point_vec(pt.x, pt.y, pt.z);
        double squareDist = (point_vec - mean).transpose() * 
                           projection_matrix * (point_vec - mean);

        if (squareDist > 3*localMap.lineRes_) {
            result.match_result = MatchingResult::MSE_TOO_LARGE;
            return result;
        }
        meanSquareDist += squareDist;
    }
    meanSquareDist /= static_cast<double>(nearest_pts.size());

    // 5. Map the mean fit error to a weight in (0,1]: tight fits get weight
    // near 1, sloppy fits near the rejection threshold get weight near 0.
    double fitQualityCoeff = 1.0 - std::sqrt(meanSquareDist / (3*localMap.lineRes_));

    // 6. The Ceres edge cost function expects the line as two points, so
    // sample one point 10 cm along the direction on each side of the centroid.
    const double line_segment_length = 0.1; // 10cm line segment
    Eigen::Vector3d point_a = line_segment_length * line_direction + mean;
    Eigen::Vector3d point_b = -line_segment_length * line_direction + mean;

    // 7. Set result parameters
    result.feature_type = FeatureType::EdgeFeature;
    result.match_result = MatchingResult::SUCCESS;
    result.Avalue = projection_matrix;
    result.Pvalue = mean;
    result.Xvalue = pInit;
    result.corres = std::make_pair(point_a, point_b);
    result.TimeValue = 1.0;  // TODO: should be point cloud time
    result.residualCoefficient = fitQualityCoeff;
    return result;
}   

// Gate-keeping after the edge nearest-neighbor search: a line fit needs at
// least a handful of neighbors, and the farthest neighbor (nearest_dist is
// sorted ascending, so .back() is the largest) must still be close to the
// query, otherwise the neighborhood spans unrelated geometry.
bool LidarSLAM::validateNeighborSearch(
        bool found,
        const std::vector<Point> &nearest_pts,
        const std::vector<float> &nearest_dist,
        OptimizationParameter &result) {
    
    if (!found || nearest_pts.size() < LocalizationMinmumLineNeighborRejection) {
        result.match_result = MatchingResult::NOT_ENOUGH_NEIGHBORS;
        return false;
    }

    if (nearest_dist.back() > 3*localMap.lineRes_) {
        result.match_result = MatchingResult::NEIGHBORS_TOO_FAR;
        return false;
    }

    return true;
}

// Full matching pipeline for one PLANAR point: the neighbors of the point in
// the surf map should form a PLANE (e.g. ground or a wall), and the distance
// from the scan point to that plane becomes the residual. Compared to the
// edge pipeline this one adds an explicit least-squares plane fit and the
// per-feature observability analysis used for degeneracy detection.
LidarSLAM::OptimizationParameter LidarSLAM::ComputePlaneDistanceParameters(
            LocalMap &local_map, const Point &p) {
        OptimizationParameter result;
    // 1. Initialize and transform point
  
    Eigen::Vector3d pInit, pFinal;
    if (!initializeAndTransformPoint(p, pInit, pFinal)) {
        result.match_result = MatchingResult::INVAVLID_NUMERICAL;
        return result;
    }
    // 2. Search parameters: 5 neighbors, rejection distance scaled by the
    // surf-map voxel resolution planeRes_ (m).
    const size_t requiredNearest = LocalizationPlaneDistanceNbrNeighbors;
    const double square_max_dist = 3 * local_map.planeRes_;

    // 3. Nearest-neighbor data association in the surf layer of the map.
    std::vector<Point> nearest_pts;
    std::vector<float> nearest_dist;
    if (!findNearestNeighbors(local_map, pFinal, nearest_pts, nearest_dist, 
                             requiredNearest, 5, square_max_dist, result)) {
        return result;
    }

    // 4. PCA shape test: eigen-decomposition of the neighbors' covariance
    // must show a plane-like spread (one small eigenvalue) to continue.
    Eigen::Vector3d mean;
    Eigen::Vector3d eigenvalues;
    Eigen::Matrix3d eigenvectors;
    Eigen::Vector3d plane_normal;
    double negative_OA_dot_norm;
    if (!computePCAForFeature(nearest_pts, mean, eigenvalues, eigenvectors, result, FeatureType::PlaneFeature)) {
        return result;
    }

    // 5. Least-squares plane fit (normal + offset) with an inlier check on
    // every neighbor; rejects the match if the plane is not clean.
    double meanSquareDist = computePlaneQualityMetrics(nearest_pts, plane_normal, 
                                                      negative_OA_dot_norm, result);
    if (result.match_result != MatchingResult::SUCCESS) {
        return result;
    }
    
    // 6. Resolve the sign ambiguity of the PCA normal (an eigenvector is
    // only defined up to +/-): flip it so it points away from the sensor,
    // i.e. along the viewing ray. Note pFinal is in the world frame, so the
    // ray is measured from the world origin rather than the current lidar
    // position; this only matters far from the start of the trajectory.
    Eigen::Vector3d correct_normal;
    Eigen::Vector3d curr_point(pFinal.x(), pFinal.y(), pFinal.z());
    Eigen::Vector3d viewpoint_direction = curr_point;
    Eigen::Vector3d normal=eigenvectors.col(0);
    double dot_product = viewpoint_direction.dot(normal);
    correct_normal=normal;
    if (dot_product < 0)
        correct_normal = -correct_normal;

    // 7. Score which pose axes this feature constrains (degeneracy input).
    pcaFeature feature;
    FeatureObservabilityAnalysis(
                feature, pFinal, eigenvalues, correct_normal, eigenvectors.col(2));

    // Fit quality in (0,1]: tight plane fits weigh more in the optimizer.
    double fitQualityCoeff = 1.0 - sqrt(meanSquareDist / square_max_dist);
    // 8. Set result parameters
    setPlaneResults(result, mean, pInit, plane_normal, negative_OA_dot_norm, feature, fitQualityCoeff);
    return result;
}

// Scores one plane feature's contribution to observing each of the 6 pose
// degrees of freedom. Intuition: a point-to-plane constraint only "feels"
// motion that changes the point's distance to the plane. Translation along
// the plane normal is felt (n . a large); sliding parallel to the plane is
// invisible. Rotation about axis a moves the point by roughly (a x p), which
// is felt when it has a component along n, i.e. when (p x n) . a is large.
// The best axes per feature are recorded and later aggregated over the scan.
void LidarSLAM::FeatureObservabilityAnalysis(pcaFeature &feature, const Eigen::Vector3d &pFinal, 
                                          const Eigen::Vector3d &eigenvalues, 
                                          const Eigen::Vector3d &normal_direction, 
                                          const Eigen::Vector3d &principal_direction) {


    // 1. Initialize feature point and directions. (Note: normalized() is a
    // no-op here because its return value is discarded; the inputs are
    // expected to arrive as unit vectors already.)
    feature.pt.x = pFinal.x();
    feature.pt.y = pFinal.y();
    feature.pt.z = pFinal.z();
    normal_direction.normalized();
    principal_direction.normalized();
    feature.vectors.principalDirection = principal_direction.cast<float>();
    feature.vectors.normalDirection = normal_direction.cast<float>();
    
    // 2. Compute eigenvalues and geometric properties
    computeEigenProperties(feature, eigenvalues);
    
    // 3. Compute rotation axes and cross products
    auto rotated_axes = computeRotatedAxes();
    computeCrossProducts(feature, rotated_axes);
    
    // 4. Compute translation observability
    computeTranslationObservability(feature, rotated_axes);
    
    // 5. Analyze feature quality and observability
    analyzeFeatureObservability(feature);
}

// Converts the raw PCA eigenvalues into the standard local-shape descriptors
// (Weinmann-style dimensionality features). Square roots turn variances into
// standard deviations; note the reordering: Eigen sorts ascending but
// lamada1/2/3 are stored descending (lamada1 = largest).
void LidarSLAM::computeEigenProperties(pcaFeature &feature, const Eigen::Vector3d &eigenvalues) {
    // Compute square roots of eigenvalues
    feature.values.lamada1 = std::sqrt(eigenvalues(2));
    feature.values.lamada2 = std::sqrt(eigenvalues(1));
    feature.values.lamada3 = std::sqrt(eigenvalues(0));
    
    double sum_lamada = feature.values.lamada1 + feature.values.lamada2 + feature.values.lamada3;
    
    // Compute geometric properties
    if (sum_lamada == 0) {
        feature.curvature = 0;
    } else {
        feature.curvature = feature.values.lamada3 / sum_lamada;
    }
    
    // Exactly one of these is close to 1: linear_2 for a line (one dominant
    // spread direction), planar_2 for a plane (two), spherical_2 for a blob.
    feature.linear_2 = (feature.values.lamada1 - feature.values.lamada2) / feature.values.lamada1;
    feature.planar_2 = (feature.values.lamada2 - feature.values.lamada3) / feature.values.lamada1;
    feature.spherical_2 = feature.values.lamada3 / feature.values.lamada1;
}


// Expresses the lidar body x/y/z axes in the world frame using the current
// orientation estimate. The observability scores are computed against these
// axes so they refer to the ROBOT's forward/left/up directions, not the
// world axes.
LidarSLAM::RotatedAxes LidarSLAM::computeRotatedAxes() {
    const Eigen::Vector3f x_axis(1, 0, 0);
    const Eigen::Vector3f y_axis(0, 1, 0);
    const Eigen::Vector3f z_axis(0, 0, 1);
    
    Eigen::Quaternionf q_world_lidar_float(
        T_world_lidar.rot.w(), T_world_lidar.rot.x(),
        T_world_lidar.rot.y(), T_world_lidar.rot.z());
    q_world_lidar_float.normalized();
    
    return RotatedAxes{
        q_world_lidar_float * x_axis,
        q_world_lidar_float * y_axis,
        q_world_lidar_float * z_axis
    };
}

// Translation observability: |n . axis| is 1 when the plane normal is
// aligned with the axis (constrains translation along it fully) and 0 when
// perpendicular. The planar_2^2 prefactor discounts neighborhoods that are
// not confidently plane-shaped.
void LidarSLAM::computeTranslationObservability(
        pcaFeature &feature, 
        const RotatedAxes &axes) {
    
    float planar_squared = feature.planar_2 * feature.planar_2;
    
    feature.tx_dot = planar_squared * 
        std::abs(feature.vectors.normalDirection.dot(axes.x));
    feature.ty_dot = planar_squared * 
        std::abs(feature.vectors.normalDirection.dot(axes.y));
    feature.tz_dot = planar_squared * 
        std::abs(feature.vectors.normalDirection.dot(axes.z));
}

// Picks the axes this feature observes best: the per-axis scores are sorted
// (descending, see compare_pair_first) and the top-2 rotation axes plus
// top-2 translation axes are stored as the feature's observability labels.
// These labels are what gets counted in the scan-wide histogram.
void LidarSLAM::analyzeFeatureObservability(pcaFeature &feature) {
    using QualityPair = std::pair<float, Feature_observability>;
    std::vector<QualityPair> rotation_quality = {
        {feature.rx_cross, Feature_observability::rx_cross},
        {feature.neg_rx_cross, Feature_observability::neg_rx_cross},
        {feature.ry_cross, Feature_observability::ry_cross},
        {feature.neg_ry_cross, Feature_observability::neg_ry_cross},
        {feature.rz_cross, Feature_observability::rz_cross},
        {feature.neg_rz_cross, Feature_observability::neg_rz_cross}
    };
    
    std::vector<QualityPair> trans_quality = {
        {feature.tx_dot, Feature_observability::tx_dot},
        {feature.ty_dot, Feature_observability::ty_dot},
        {feature.tz_dot, Feature_observability::tz_dot}
    };
    // Sort quality measures
    std::sort(rotation_quality.begin(), rotation_quality.end(), utils::compare_pair_first);
    std::sort(trans_quality.begin(), trans_quality.end(), utils::compare_pair_first);
    
    // Assign top observability measures
    feature.observability.at(0) = rotation_quality.at(0).second;
    feature.observability.at(1) = rotation_quality.at(1).second;
    feature.observability.at(2) = trans_quality.at(0).second;
    feature.observability.at(3) = trans_quality.at(1).second;
}


// Rotation observability: (p x n) . axis measures how strongly a rotation
// about 'axis' changes this point's distance to its plane (the lever-arm
// effect). The negated copies exist because the histogram tracks positive
// and negative rotation directions separately.
void LidarSLAM::computeCrossProducts(pcaFeature &feature, const RotatedAxes &axes) {
    Eigen::Vector3f point(feature.pt.x, feature.pt.y, feature.pt.z);
    Eigen::Vector3f cross = point.cross(feature.vectors.normalDirection);
    
    // Compute cross products with rotated axes
    feature.rx_cross = cross.dot(axes.x);
    feature.neg_rx_cross = -feature.rx_cross;
    feature.ry_cross = cross.dot(axes.y);
    feature.neg_ry_cross = -feature.ry_cross;
    feature.rz_cross = cross.dot(axes.z);
    feature.neg_rz_cross = -feature.rz_cross;
}

// Packs a successful plane match into the OptimizationParameter record. The
// Ceres plane cost will evaluate n . (T * pInit) + d, where n = plane_normal
// and d = negative_OA_dot_norm define the plane in the world frame.
void LidarSLAM::setPlaneResults(OptimizationParameter &result, const Eigen::Vector3d &mean, 
                               const Eigen::Vector3d &pInit, const Eigen::Vector3d &plane_normal, 
                               double negative_OA_dot_norm, const pcaFeature &feature, double fitQualityCoeff) {

   result.feature_type = FeatureType::PlaneFeature;
   result.feature = feature;
   result.match_result = MatchingResult::SUCCESS;
   result.Pvalue = mean;
   result.Xvalue = pInit;
   result.NormDir = plane_normal;
   result.negative_OA_dot_norm = negative_OA_dot_norm;
   result.TimeValue =static_cast<double>(1.0);  // TODO:should be the point cloud time
   result.residualCoefficient = fitQualityCoeff;
}   




// Convenience wrapper: gives the point in the lidar frame (pInit) and in the
// world frame under the current pose (pFinal) for the LOCALIZATION mode.
bool LidarSLAM::initializeAndTransformPoint(const Point &p, 
                                          Eigen::Vector3d &pInit,
                                          Eigen::Vector3d &pFinal) {
    ComputePointInitAndFinalPose(MatchingMode::LOCALIZATION, p, pInit, pFinal);
    return true;
}

// K-nearest-neighbor lookup in the surf map with the two standard rejection
// checks: enough neighbors found, and the farthest neighbor (largest entry
// of the sorted distance list) within square_max_dist of the query. Failing
// either means the query point has no plane-like support in the map.
bool LidarSLAM::findNearestNeighbors(LocalMap &local_map,
                                    const Eigen::Vector3d &pFinal,
                                    std::vector<Point> &nearest_pts,
                                    std::vector<float> &nearest_dist,
                                    size_t requiredNearest,
                                    size_t min_neighbors,
                                    double square_max_dist,
                                    OptimizationParameter &result) {
    Point pFinal_query;
    pFinal_query.x = pFinal.x();
    pFinal_query.y = pFinal.y();
    pFinal_query.z = pFinal.z();

    bool found = local_map.nearestKSearchSurf(pFinal_query, nearest_pts,
                                            nearest_dist, requiredNearest);

    if (!found || nearest_pts.size() < min_neighbors) {
        result.match_result = MatchingResult::NOT_ENOUGH_NEIGHBORS;
        return false;
    }

    if (nearest_dist.back() > square_max_dist) {
        result.match_result = MatchingResult::NEIGHBORS_TOO_FAR;
        return false;
    }

    return true;
}

// PCA of a neighborhood + the geometric shape test. ComputePCA stacks the
// neighbors into an Nx3 matrix, subtracts the centroid, and eigen-decomposes
// the resulting 3x3 covariance. The eigenvalues (ascending: 0 = smallest,
// 2 = largest) measure the spread of the points along each eigenvector, so
// their ratios reveal the local shape.
bool LidarSLAM::computePCAForFeature(const std::vector<Point> &nearest_pts,
                                  Eigen::Vector3d &mean,
                                  Eigen::Vector3d &eigenvalues,
                                  Eigen::Matrix3d &eigenvectors,
                                  OptimizationParameter &result,
                                  FeatureType feature_type) {

    Eigen::MatrixXd data(nearest_pts.size(), 3);
    for (size_t k = 0; k < nearest_pts.size(); k++) {
        const Point &pt = nearest_pts[k];
        data.row(k) << pt.x, pt.y, pt.z;
    }
    // 2. Compute PCA
    try {
        auto eig = utils::ComputePCA(data, mean);
        eigenvalues = eig.eigenvalues();
        eigenvectors = eig.eigenvectors();
    } catch (const std::exception& e) {
        result.match_result = MatchingResult::INVAVLID_NUMERICAL;
        return false;
    }
    
    if(feature_type == FeatureType::PlaneFeature){
        // A plane must have genuine 2D extent: reject degenerate/collinear
        // neighborhoods (smallest eigenvalue ~0 together with a middle
        // eigenvalue much smaller than the largest means the points lie on
        // a line, not a plane).
        if (eigenvalues(0) < 1e-6 || eigenvalues(1) / eigenvalues(2) < 0.1) {
            result.match_result = MatchingResult::BAD_PCA_STRUCTURE;
            return false;
        }
    }else if(feature_type == FeatureType::EdgeFeature){
        if(!eigenvalues.allFinite())
        {
            result.match_result = MatchingResult::INVAVLID_NUMERICAL;
            return false;
        }
        
        // A line must have one DOMINANT direction: the largest eigenvalue
        // has to exceed the second by a factor (4x here), otherwise the
        // neighborhood is a plane or blob and a line fit would be arbitrary.
        if(eigenvalues(2) < LocalizationMinmumLineNeighborRejection * eigenvalues(1)){
            result.match_result = MatchingResult::BAD_PCA_STRUCTURE;
            return false;
        }

    }
    return true;
}

// Least-squares plane fit through the 5 neighbors (the same trick used in
// LOAM/A-LOAM). A plane can be written m . x = -1 for some vector m (valid
// for any plane not passing through the origin), so stacking the 5 points
// into A and solving A*m = -1 by QR gives the plane directly. Then
// n = m/|m| is the unit normal and d = 1/|m| the offset, giving the
// normalized plane equation n . x + d = 0, whose left side evaluated at any
// point is its signed distance to the plane.
double LidarSLAM::computePlaneQualityMetrics(const std::vector<Point>& nearest_pts,
                                          Eigen::Vector3d &plane_normal,
                                          double &negative_OA_dot_norm,
                                          OptimizationParameter &result) {
    
     // 1. Set up the system of equations
    Eigen::Matrix<double, 5, 3> matA0;
    Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
    // 2. Fill matrix with point coordinates
    for (int i = 0; i < 5; i++) {
        matA0.row(i) << nearest_pts[i].x, nearest_pts[i].y, nearest_pts[i].z;
    }
    
    // 3. Solve for plane normal
    plane_normal = matA0.colPivHouseholderQr().solve(matB0);
    
    // 4. Check if solution is valid
    if (!plane_normal.allFinite()) {
        result.match_result = MatchingResult::INVAVLID_NUMERICAL;
        return 0.0;
    }
    
    // 5. Normalize: after this, plane_normal is the unit normal n and
    // negative_OA_dot_norm is the offset d in n . x + d = 0.
    negative_OA_dot_norm = 1.0 / plane_normal.norm();
    plane_normal.normalize();
    

    double meanSquareDist = 0.0;
    // Inlier threshold: half a voxel of the surf map resolution (m).
    const double max_point_distance = localMap.planeRes_ / 2.0;
    
    // 1. Every neighbor must lie on the fitted plane; a single stray point
    // means the neighborhood is not a clean plane (e.g. it straddles a
    // corner) and the match is rejected.
    for (const auto& pt : nearest_pts) {
        double point_to_plane_dist = std::abs(
            plane_normal.x() * pt.x + 
            plane_normal.y() * pt.y +
            plane_normal.z() * pt.z + 
            negative_OA_dot_norm
        );
        
        // 2. Check if point is too far from plane
        if (point_to_plane_dist > max_point_distance) {
            result.match_result = MatchingResult::MSE_TOO_LARGE;
            return 0.0;
        }
        
        meanSquareDist += point_to_plane_dist;
    }
    
    // 3. Mean absolute distance (despite the variable name) of the
    // neighbors to the plane, used as the fit-quality measure.
    meanSquareDist /= nearest_pts.size();
    result.match_result = MatchingResult::SUCCESS;
    return meanSquareDist;
}


    // Clears the match buffers and histograms before a new round of data
    // association (called at the top of every ICP iteration).
    void LidarSLAM::ResetDistanceParameters() {
        this->OptimizationData.clear();
        for (auto &ele : MatchRejectionHistogramLine) ele = 0;
        for (auto &ele : MatchRejectionHistogramPlane) ele = 0;
        for (auto &ele : PlaneFeatureHistogramObs) ele = 0;
    }

    // Uncertainty of the registered pose, computed from the solved Ceres
    // problem. Ceres approximates the covariance as the inverse of the
    // Gauss-Newton Hessian J^T * J: directions in which the residuals react
    // strongly to pose changes get small covariance, directions the features
    // do not constrain get large covariance. This is the second, complementary
    // degeneracy signal next to the observability histogram.
    LidarSLAM::RegistrationError LidarSLAM::EstimateRegistrationError(
            ceres::Problem &problem, const double eigen_thresh) {
        RegistrationError err;

        // Covariance computation options. DENSE_SVD with null_space_rank=-1
        // tolerates a rank-deficient (degenerate) Hessian instead of failing;
        // the covariance is evaluated in the 6-DoF tangent space of the pose
        // (3 translation + 3 rotation), not the raw 7 parameters.
        ceres::Covariance::Options covOptions;
        covOptions.apply_loss_function = true;
        covOptions.algorithm_type = ceres::CovarianceAlgorithmType::DENSE_SVD;
        covOptions.null_space_rank = -1;
        covOptions.num_threads = 2;

        ceres::Covariance covarianceSolver(covOptions);
        std::vector<std::pair<const double *, const double *>> covarianceBlocks;
        const double *paramBlock = pose_parameters;
        covarianceBlocks.emplace_back(paramBlock, paramBlock);
        covarianceSolver.Compute(covarianceBlocks, &problem);
        covarianceSolver.GetCovarianceBlockInTangentSpace(paramBlock, paramBlock,
                                                          err.Covariance.data());

        // Eigen-decompose the 3x3 position block: the largest eigenvalue is
        // the variance along the least-constrained direction, so its square
        // root is the worst-case standard deviation (m) and its eigenvector
        // the direction of that weakness. The inverse condition number
        // sqrt(min/max eigenvalue) is 1 for equally-constrained directions
        // and goes to 0 as the problem becomes degenerate.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigPosition(err.Covariance.topLeftCorner<3, 3>());

        err.PositionError = std::sqrt(eigPosition.eigenvalues()(2));
        err.PositionErrorDirection = eigPosition.eigenvectors().col(2);
        err.PosInverseConditionNum = std::sqrt(eigPosition.eigenvalues()(0)) / std::sqrt(eigPosition.eigenvalues()(2));

        // Same analysis for the 3x3 rotation block (converted to degrees).
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigOrientation(err.Covariance.bottomRightCorner<3, 3>());
        err.OrientationError = utils::Rad2Deg(std::sqrt(eigOrientation.eigenvalues()(2)));
        err.OrientationErrorDirection = eigOrientation.eigenvectors().col(2);
        err.OriInverseConditionNum =
                std::sqrt(eigOrientation.eigenvalues()(0)) / std::sqrt(eigOrientation.eigenvalues()(2));

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigPosition2(err.Covariance.inverse());

        return err;
    }
    
   // Hand-tuned yaw drift compensation: adds OptSet.yaw_ratio degrees of yaw
   // per meter traveled since the last scan. This is a calibration fudge for
   // a known systematic yaw bias (e.g. a slightly misaligned sensor), not
   // part of the estimation theory.
   void LidarSLAM::MannualYawCorrection()
   {
     
    Transformd T_lidar_prev_lidar_current =
        T_world_lidar_prev.inverse() * T_world_lidar;
    float translation_norm = T_lidar_prev_lidar_current.pos.norm();

    double roll, pitch, yaw;
    tf2::Quaternion q_world_lidar(
        T_world_lidar.rot.x(), T_world_lidar.rot.y(), T_world_lidar.rot.z(),
        T_world_lidar.rot.w());
    tf2::Matrix3x3(q_world_lidar).getRPY(roll, pitch, yaw);
    
    tf2::Quaternion q_world_lidar_corrected_tf2;

   
    double correct_yaw=yaw+translation_norm*OptSet.yaw_ratio*M_PI/180;
    q_world_lidar_corrected_tf2.setRPY(roll, pitch, correct_yaw);
    
    Eigen::Quaterniond q_world_lidar_corrected =
        Eigen::Quaterniond(q_world_lidar_corrected_tf2.w(),
                           q_world_lidar_corrected_tf2.x(),
                           q_world_lidar_corrected_tf2.y(),
                           q_world_lidar_corrected_tf2.z());
    
    T_world_lidar.rot = q_world_lidar_corrected.normalized();
   }

   // Converts the observability histogram (votes collected during data
   // association) into a per-axis uncertainty in [0, 1]. The formula
   // (votes/total)*3 compares each axis's share of the votes against a
   // uniform split (1/3 per translation axis); capped at 1. NOTE the
   // counter-intuitive convention: MORE votes for an axis yields a HIGHER
   // "uncertainty" value here, and consumers such as
   // addAbsolutePoseConstraints use (1 - uncertainty) as the axis weight,
   // so the value effectively acts as a lidar confidence score.
   void LidarSLAM::EstimateLidarUncertainty() {
        // Total votes cast for the translation axes (histogram slots 6-8 =
        // tx, ty, tz).
        double TotalTransFeature = PlaneFeatureHistogramObs.at(6) +
                                   PlaneFeatureHistogramObs.at(7) +
                                   PlaneFeatureHistogramObs.at(8);
        
        //uncertainty x
        double uncertaintyX = (PlaneFeatureHistogramObs.at(6) / TotalTransFeature) * 3;
        lidarOdomUncer.uncertainty_x = std::min(uncertaintyX, 1.0);

        //uncertainty y
        double uncertaintyY = (PlaneFeatureHistogramObs.at(7) / TotalTransFeature) * 3;
        lidarOdomUncer.uncertainty_y = std::min(uncertaintyY, 1.0);

        //uncertainty Z
        double uncertaintyZ = (PlaneFeatureHistogramObs.at(8) / TotalTransFeature) * 3;
        lidarOdomUncer.uncertainty_z = std::min(uncertaintyZ, 1.0);

        // Total votes for the rotation axes (slots 0-5 = +/-rx, +/-ry,
        // +/-rz); each rotation axis sums its positive and negative slots.
        double TotalRotationFeature = PlaneFeatureHistogramObs.at(0) +
                                      PlaneFeatureHistogramObs.at(1) +
                                      PlaneFeatureHistogramObs.at(2) +
                                      PlaneFeatureHistogramObs.at(3) +
                                      PlaneFeatureHistogramObs.at(4) +
                                      PlaneFeatureHistogramObs.at(5);

        //uncertainty roll
        double uncertaintyRoll =
                (PlaneFeatureHistogramObs.at(0) + PlaneFeatureHistogramObs.at(1)) / TotalRotationFeature * 3;
        lidarOdomUncer.uncertainty_roll = std::min(uncertaintyRoll, 1.0);

        //uncertainty pitch
        double uncertaintyPitch =
                (PlaneFeatureHistogramObs.at(2) + PlaneFeatureHistogramObs.at(3)) / TotalRotationFeature * 3;
        lidarOdomUncer.uncertainty_pitch = std::min(uncertaintyPitch, 1.0);

        //uncertainty yaw
        double uncertaintyYaw =
                (PlaneFeatureHistogramObs.at(4) + PlaneFeatureHistogramObs.at(5)) / TotalRotationFeature * 3;
        lidarOdomUncer.uncertainty_yaw = std::min(uncertaintyYaw, 1.0);

              
        // No votes at all (e.g. first scans, or no plane features matched):
        // report zeros rather than dividing by zero above.
        if(TotalTransFeature==0 || TotalRotationFeature==0)
        {
          lidarOdomUncer.uncertainty_x=0;
          lidarOdomUncer.uncertainty_y=0;
          lidarOdomUncer.uncertainty_z=0;
          lidarOdomUncer.uncertainty_roll=0;
          lidarOdomUncer.uncertainty_pitch=0;
          lidarOdomUncer.uncertainty_yaw=0;   
        }

        publishUncertainty(lidarOdomUncer.uncertainty_x, lidarOdomUncer.uncertainty_y, lidarOdomUncer.uncertainty_z,
                                     lidarOdomUncer.uncertainty_roll, lidarOdomUncer.uncertainty_pitch, lidarOdomUncer.uncertainty_yaw);

        stats.uncertainty_x=lidarOdomUncer.uncertainty_x;
        stats.uncertainty_y=lidarOdomUncer.uncertainty_y;
        stats.uncertainty_z=lidarOdomUncer.uncertainty_z;
        stats.uncertainty_roll=lidarOdomUncer.uncertainty_roll;
        stats.uncertainty_pitch=lidarOdomUncer.uncertainty_pitch;
        stats.uncertainty_yaw=lidarOdomUncer.uncertainty_yaw;

        // if (lidarOdomUncer.uncertainty_x<0.2 or lidarOdomUncer.uncertainty_y<0.1 or lidarOdomUncer.uncertainty_z<0.2) {
        //     isDegenerate = true;

        // }else if (PlaneFeatureHistogramObs.at(6)<20 or PlaneFeatureHistogramObs.at(7)<10 or PlaneFeatureHistogramObs.at(8)<10)
        // {
        //     isDegenerate = true;   
        // }
        // else {
        //     isDegenerate = false;
        // }
    }

    // Publishes the six per-axis uncertainty values on their debug topics
    // (one Float32 topic per axis, mainly for plotting/monitoring).
    void LidarSLAM::publishUncertainty(double uncer_x, double uncer_y, double uncer_z,
        double uncer_roll, double uncer_pitch, double uncer_yaw)
    {

        std_msgs::msg::Float32 uncertainty_x;
        uncertainty_x.data = uncer_x;
        pubUncertaintyX->publish(uncertainty_x);

        std_msgs::msg::Float32 uncertainty_y;
        uncertainty_y.data = uncer_y;
        pubUncertaintyY->publish(uncertainty_y);

        std_msgs::msg::Float32 uncertainty_z;
        uncertainty_z.data = uncer_z;
        pubUncertaintyZ->publish(uncertainty_z);

        std_msgs::msg::Float32 uncertainty_roll;
        uncertainty_roll.data = uncer_roll;
        pubUncertaintyRoll->publish(uncertainty_roll);

        std_msgs::msg::Float32 uncertainty_pitch;
        uncertainty_pitch.data = uncer_pitch;
        pubUncertaintyPitch->publish(uncertainty_pitch);

        std_msgs::msg::Float32 uncertainty_yaw;
        uncertainty_yaw.data = uncer_yaw;
        pubUncertaintyYaw->publish(uncertainty_yaw);

    };

} /* super_odometry */
