//
// Created by ubuntu on 2020/6/27.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// This file implements the rolling LOCAL MAP used by the scan-registration
// (laserMapping) stage of the pipeline:
//
//   featureExtraction (deskewed edge/planar features)
//     -> laserMapping registers each new scan against THIS LocalMap
//     -> the resulting pose is fused with the IMU in imuPreintegration.
//
// Why a "local" map? Registering a scan needs many nearest-neighbor lookups
// ("give me the 5 map points closest to this scan point"). Searching one
// giant, ever-growing global cloud would get slower every second. Instead we
// keep only a fixed-size neighborhood of the robot, organized for fast
// queries, and let old geometry fall off the back as the robot moves.
//
// Data structure (three levels):
//
//   1. A fixed 3D grid of 21 x 21 x 11 = 4851 blocks (MapBlock). Each block
//      is a 50 m cube, so the map spans roughly 1050 x 1050 x 550 m of the
//      world around the robot. The grid is stored as a flat std::array and a
//      cell (i,j,k) lives at index
//          cubeInd = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k
//      (the usual row-major flattening of a 3D array).
//
//   2. Each MapBlock stores two separate point clouds: EDGE points (sharp,
//      line-like features) and SURF points (flat, plane-like features),
//      because scan registration matches edges to lines and surfs to planes.
//
//   3. Each cloud inside a block has its own octree (see flann/octree.h)
//      rebuilt after every insertion, so nearest-neighbor queries only ever
//      search the single 50 m block containing the query point.
//
// Rolling behavior: the grid never grows. origin_ maps world coordinates to
// grid indices. When the robot approaches the border of the grid, shiftMap()
// slides all blocks over by whole cells (dropping the row that falls off the
// far side) and updates origin_, exactly like the cube-shifting logic in
// LOAM / A-LOAM's laserMapping.
//
// Coordinate convention used everywhere below: a world point p falls in cell
//   i = floor((p.x + 25) / 50) + origin_.x()   (same for j/k with y/z)
// The +25 (half a block) centers cell boundaries so that cell (0,0,0) in
// world-voxel coordinates covers [-25, 25) on each axis. std::floor() gives
// the same half-open intervals for both positive and negative coordinates.
// ============================================================================

#ifndef LOCALMAPOCTREE_H
#define LOCALMAPOCTREE_H

#include <cmath>
#include <memory>
#include <thread>
#include <vector>

#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

#include <Eigen/Dense>

#include "../utils/EigenTypes.h"
#include "super_odometry/sensor_data/pointcloud/LidarPoint.h"
#include "super_odometry/flann/nanoflann.h"
#include "super_odometry/flann/octree.h"

// If defined, fall back to PCL's kd-tree instead of the custom octree in
// super_odometry/flann/octree.h. Left disabled: the custom octree is faster
// to (re)build, which matters because every block rebuilds its tree each
// time new points are inserted.
//#define DONT_USE_SELF_OCTREE

/// One 50 m cube of the local map. It owns two feature clouds (edge and
/// surf) plus one nearest-neighbor search tree per cloud. Blocks start empty
/// and are lazily allocated on first insertion.
struct MapBlock {

    // Usefull types
    //using Point = PointXYZTIId;
    using Point =pcl::PointXYZI;
    using PointCloud = pcl::PointCloud<Point>;

    MapBlock() = default;

    // ---- point data -----------------------------------------------------
    // Edge (line-like) and surf (plane-like) features are kept in separate
    // clouds because scan registration fits different geometric models to
    // each (point-to-line vs point-to-plane residuals).
    pcl::PointCloud<Point>::Ptr pedge_pc_ = nullptr;
    pcl::PointCloud<Point>::Ptr psurf_pc_ = nullptr;

    // ---- search structures ------------------------------------------------
    // One tree per cloud, rebuilt by LocalMap::add*PointCloud() after every
    // insertion. Named "kdtree" for historical reasons; the default build
    // actually uses the custom octree.
#ifdef DONT_USE_SELF_OCTREE
    pcl::KdTreeFLANN<Point>::Ptr pkdtree_edge_from_block_ = nullptr;
    pcl::KdTreeFLANN<Point>::Ptr pkdtree_surf_from_block_ = nullptr;
#else
    std::shared_ptr<nanoflann::Octree<Point, Eigen::aligned_vector<Point>>> pkdtree_edge_from_block_ = nullptr;
    std::shared_ptr<nanoflann::Octree<Point, Eigen::aligned_vector<Point>>> pkdtree_surf_from_block_ = nullptr;
#endif

    // ---- status flags -----------------------------------------------------
    bool bnull_ = true;               // true while the block holds no points at all
    bool bline_null_ = true;          // true while the block holds no edge points
    bool bsurf_null_ = true;          // true while the block holds no surf points
    bool bnewline_points_add_ = false;// set externally when fresh edge points arrived
    bool bnewsurf_points_add_ = false;// set externally when fresh surf points arrived

    /// Drops all points and trees, returning the block to its empty state.
    /// Called when the rolling map shifts and this cell is recycled.
    inline void clear() {
        pedge_pc_ = nullptr;
        psurf_pc_ = nullptr;

        pkdtree_edge_from_block_ = nullptr;
        pkdtree_surf_from_block_ = nullptr;

        bnull_ = true;
        bline_null_ = true;
        bsurf_null_ = true;
        bnewline_points_add_ = false;
        bnewsurf_points_add_ = false;
    }

    /// Appends one edge point, allocating the cloud on first use. Note this
    /// only stores the point; the search tree is rebuilt later in batch by
    /// LocalMap::addEdgePointCloud().
    inline void insertEdgePoint(const Point &point) {
        if(pedge_pc_ == nullptr) {
            pedge_pc_.reset(new pcl::PointCloud<Point>());
        }
        if(bnull_ or bline_null_) {
            bnull_ = false;
            bline_null_ = false;
        }
        pedge_pc_->push_back(point);
    }

    /// Appends one surf point, allocating the cloud on first use.
    /// (Note: the flag update below clears bline_null_ instead of
    /// bsurf_null_, which looks like a copy-paste bug in the original code;
    /// surfPointsIsEmpty() may stay true even after surf points were added.)
    inline void insertSurfPoint(const Point &point) {
        if(psurf_pc_ == nullptr) {
            psurf_pc_.reset(new pcl::PointCloud<Point>());
        }
        if(bnull_ or bsurf_null_) {
            bnull_ = false;
            bline_null_ = false;
        }
        psurf_pc_->push_back(point);
    }

    inline int edgePointCloudSize() const {
        if(pedge_pc_ == nullptr)
            return 0;
        return pedge_pc_->size();
    }

    inline int surfPointCloudSize() const {
        if(psurf_pc_ == nullptr)
            return 0;
        return psurf_pc_->size();
    }

    inline bool empty() const { return bnull_; }

    inline bool edgePointsIsEmpty() const { return bline_null_; }

    inline bool surfPointsIsEmpty() const { return bsurf_null_; }

    inline bool haveNewEdgePoints() const { return bnewline_points_add_; }

    inline bool haveNewsurfPoints() const { return bnewsurf_points_add_; }
};

/// Rolling grid of MapBlock cells centered (approximately) on the robot.
/// laserMapping uses it in a simple cycle for every scan:
///   1. shiftMap(robot position)          - keep the robot near the center
///   2. nearestKSearch*(...)              - data association for registration
///   3. addEdgePointCloud/addSurfPointCloud - insert the newly aligned scan
class LocalMap {
public:
    // Usefull types
    //using Point = PointXYZTIId;
    using Point =pcl::PointXYZI;
    using PointCloud = pcl::PointCloud<Point>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // ---- grid dimensions -------------------------------------------------
    // Number of blocks along x, y, z. The grid is intentionally flatter in z
    // (11 vs 21) because ground robots move mostly horizontally.
    static constexpr const int laserCloudWidth = 21;
    static constexpr const int laserCloudHeight = 21;
    static constexpr int laserCloudDepth = 11;

    static constexpr int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;  // 4851

    // Edge length of one block in meters ("voxel" here means one 50 m map
    // block, not the small voxels of the downsampling filter below).
    static constexpr double voxelResulation = 50;
    static constexpr double halfVoxelResulation = voxelResulation * 0.5;

    /// Converts one world coordinate to its 50 m block coordinate.
    /// std::floor handles negative coordinates and exact negative boundaries
    /// correctly, unlike integer truncation followed by an unconditional
    /// decrement.
    static int worldToBlockCoordinate(double coordinate) {
        return static_cast<int>(std::floor(
            (coordinate + halfVoxelResulation) / voxelResulation));
    }

public:
    LocalMap() {
        // Start with the world origin mapped to the center cell of the grid,
        // so the map initially extends equally in all directions.
        // World interval     World block
        // [-75, -25)            -1
        // [-25,  25)             0
        // Initially:
        // origin_ = (10, 10, 5);
        // Therefore:
        // World block 0  → local index 10
        // World block 1  → local index 11

        // Block index
        // The block index identifies a slot in the fixed 21 × 21 × 11 array:
        
        // i: 0–20
        // j: 0–20
        // k: 0–10
        // The relationship is:
        
        // world coordinate
        //       ↓ divide into 50 m regions
        // world block coordinate
        //       ↓ add origin_
        // local block index
        origin_ = Eigen::Vector3i(laserCloudWidth * 0.5, laserCloudHeight * 0.5, laserCloudDepth * 0.5);
    }

    /// Re-anchors the grid so that the given world position t_world_current falls in
    /// cell (0,0,0). Only used for (re)initialization; during normal
    /// operation shiftMap() moves the grid incrementally instead.
    Eigen::Vector3i setOrigin(const Eigen::Vector3d &t_world_current) {
        // Which world block does the robot occupy?
        int centerCubeI = worldToBlockCoordinate(t_world_current.x());
        int centerCubeJ = worldToBlockCoordinate(t_world_current.y());
        int centerCubeK = worldToBlockCoordinate(t_world_current.z());

        origin_.x() = -centerCubeI;
        origin_.y() = -centerCubeJ;
        origin_.z() = -centerCubeK;

        return origin_;
    }  // function setOrigin end

    /// \brief Slides the rolling grid so the robot stays away from its edges.
    ///
    /// If the robot's cell gets within 3 cells of any face of the grid, all
    /// blocks are shifted by one cell along that axis (repeatedly, if
    /// needed): every block is copied to its neighbor slot, the row that
    /// falls off the far side is cleared (its geometry is forgotten), and
    /// origin_ is updated so world-to-cell conversion stays consistent.
    /// The margin of 3 guarantees the 5x5x3 query neighborhood used by the
    /// get5x5* functions always lies inside the grid.
    ///
    /// \param t_world_current current robot position in the world frame (meters)
    /// \return the robot's cell index (i,j,k) after shifting
    Eigen::Vector3i shiftMap(const Eigen::Vector3d &t_world_current) {

        // Convert the world position to rolling-grid indices.
        int centerCubeI =
            worldToBlockCoordinate(t_world_current.x()) + origin_.x();
        int centerCubeJ =
            worldToBlockCoordinate(t_world_current.y()) + origin_.y();
        int centerCubeK =
            worldToBlockCoordinate(t_world_current.z()) + origin_.z();


        // Robot too close to the low-x face: shift every block one step in
        // +x. The slab at i = laserCloudWidth-1 is overwritten (dropped) and
        // the freed slab at i = 0 receives the recycled, cleared blocks.
        // The remaining five loops below are the same pattern for the other
        // face/axis combinations.
        while(centerCubeI < 3) {
            for(int j = 0; j < laserCloudHeight; j++) {
                for(int k = 0; k < laserCloudDepth; k++) {
                    int i = laserCloudWidth - 1;
                    for(; i >= 1; i--) {
                        map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                map_[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    }

                    map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k].clear();
                }
            }

            centerCubeI++;
            origin_.x() = origin_.x() + 1;
        }

        while(centerCubeI >= laserCloudWidth - 3) {
            for(int j = 0; j < laserCloudHeight; j++) {
                for(int k = 0; k < laserCloudDepth; k++) {
                    int i = 0;
                    for(; i < laserCloudWidth - 1; i++) {
                        map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                map_[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    }
                    map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k].clear();
                }
            }

            centerCubeI--;
            origin_.x() = origin_.x() - 1;
        }

        while(centerCubeJ < 3) {
            for(int i = 0; i < laserCloudWidth; i++) {
                for(int k = 0; k < laserCloudDepth; k++) {

                    int j = laserCloudHeight - 1;

                    for(; j >= 1; j--) {
                        map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                map_[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
                    }

                    map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k].clear();
                }
            }

            centerCubeJ++;
            origin_.y() = origin_.y() + 1;
        }

        while(centerCubeJ >= laserCloudHeight - 3) {
            for(int i = 0; i < laserCloudWidth; i++) {
                for(int k = 0; k < laserCloudDepth; k++) {
                    int j = 0;

                    for(; j < laserCloudHeight - 1; j++) {
                        map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                map_[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
                    }
                    map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k].clear();
                }
            }

            centerCubeJ--;
            origin_.y() = origin_.y() - 1;
        }

        while(centerCubeK < 3) {
            for(int i = 0; i < laserCloudWidth; i++) {
                for(int j = 0; j < laserCloudHeight; j++) {
                    int k = laserCloudDepth - 1;

                    for(; k >= 1; k--) {
                        map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
                    }
                    map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k].clear();
                }
            }

            centerCubeK++;
            origin_.z() = origin_.z() + 1;
        }

        while(centerCubeK >= laserCloudDepth - 3) {
            for(int i = 0; i < laserCloudWidth; i++) {
                for(int j = 0; j < laserCloudHeight; j++) {
                    int k = 0;

                    for(; k < laserCloudDepth - 1; k++) {
                        map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
                    }
                    map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k].clear();
                }
            }

            centerCubeK--;
            origin_.z() = origin_.z() - 1;
        }

        return Eigen::Vector3i{centerCubeI, centerCubeJ, centerCubeK};
    }  // function shiftMap

    /// \brief Counts the edge and surf points in the 5x5x3 block neighborhood
    /// around the robot (2 blocks in each horizontal direction, 1 vertically,
    /// i.e. roughly 250 x 250 x 150 m). laserMapping uses these counts to
    /// decide whether the map is populated enough to attempt registration.
    /// \param position robot cell index, as returned by shiftMap()
    /// \return std::tuple<int, int> = (edge point count, surf point count)
    std::tuple<int, int> get5x5LocalMapFeatureSize(const Eigen::Vector3i &position) {

        int centerCubeI, centerCubeJ, centerCubeK;
        centerCubeI = position.x();
        centerCubeJ = position.y();
        centerCubeK = position.z();

        int laserCloudLineFromMapNum = 0;
        int laserCloudSurfFromMapNum = 0;

        for(int i = centerCubeI - 2; i <= centerCubeI + 2; i++) {
            for(int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) {
                for(int k = centerCubeK - 1; k <= centerCubeK + 1; k++) {

                    if(i >= 0 && i < laserCloudWidth && j >= 0 && j < laserCloudHeight && k >= 0 &&
                       k < laserCloudDepth) {

                        int cubeInd = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                        laserCloudLineFromMapNum += map_[cubeInd].edgePointCloudSize();
                        laserCloudSurfFromMapNum += map_[cubeInd].surfPointCloudSize();
                    }
                }
            }
        }

        return std::make_tuple(laserCloudLineFromMapNum, laserCloudSurfFromMapNum);
    }  // function get_localmap_featuresize

    /// \brief Finds the 5 edge points nearest to pt_query.
    ///
    /// Only the single block containing the query is searched. This is the
    /// key speed trick of this class: the tree in one 50 m block is tiny
    /// compared to the whole map. The trade-off is that neighbors lying just
    /// across a block boundary are missed, which is acceptable because
    /// matched features are expected to be within ~1 m of the query.
    ///
    /// \param pt_query        query point in world coordinates
    /// \param k_pts           the neighbor points themselves (output)
    /// \param k_sqr_distances squared distances to each neighbor, in m^2 (output)
    /// \return false if the query is outside the grid or the block has no tree yet
    bool nearestKSearchEdgePoint(const Point &pt_query,
                                 std::vector<Point> &k_pts,
                                 std::vector<float> &k_sqr_distances) const {

        k_pts.clear();

        // Locate the block containing the query (same floor-division
        // pattern as in shiftMap).
        int cubeI = worldToBlockCoordinate(pt_query.x) + origin_.x();
        int cubeJ = worldToBlockCoordinate(pt_query.y) + origin_.y();
        int cubeK = worldToBlockCoordinate(pt_query.z) + origin_.z();

        if(!(cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 &&
             cubeK < laserCloudDepth)) {
            return false;
        }

        int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
        if(map_[cubeInd].pkdtree_edge_from_block_ == nullptr)
            return false;

#ifdef DONT_USE_SELF_OCTREE
        std::vector<int> k_indices;
        map_[cubeInd].pkdtree_edge_from_block_->nearestKSearch(pt_query, 5, k_indices, k_sqr_distances);
#else
        const size_t num_results = 5;
        std::vector<size_t> k_indices(num_results);
        k_sqr_distances.resize(num_results);

        map_[cubeInd].pkdtree_edge_from_block_->template knnNeighbors<nanoflann::L2Distance<Point>>(
                pt_query, num_results, k_indices.data(), k_sqr_distances.data());
#endif
        for(auto id : k_indices)
            k_pts.push_back(map_[cubeInd].pedge_pc_->points[id]);

        return true;
    }  // function nearestKSearchLinePoint

    /**
     * \brief Like nearestKSearchEdgePoint, but additionally filters the
     * neighbors with a small RANSAC-style line fit, so the caller gets only
     * points that actually lie on one line.
     *
     * Edge features should come from linear structures (poles, wall
     * corners). A plain kNN result can mix points from two different
     * structures, which would corrupt the point-to-line residual. This
     * method tries every line through the closest neighbor P1 and one other
     * neighbor P2, counts how many of the remaining neighbors lie within
     * max_dist_inliner of that line, and returns P1 plus the inliers of the
     * best line.
     *
     * @param [in]  pt_query           query point in world coordinates
     * @param [out] k_pts              closest point + inliers of the best line
     * @param [out] k_sqr_distances    squared distances of those points to the query
     * @param [in]  num_nearest_search how many raw kNN candidates to fetch
     * @param [in]  max_dist_inliner   max point-to-line distance (m) to count as inlier
     * @return false if the query falls outside the grid or the block is empty
     */
    bool nearestKSearchSpecificEdgePoint(const Point &pt_query,
                                         std::vector<Point> &k_pts,
                                         std::vector<float> &k_sqr_distances,
                                         int num_nearest_search,
                                         float max_dist_inliner) const {

        int cubeI = worldToBlockCoordinate(pt_query.x) + origin_.x();
        int cubeJ = worldToBlockCoordinate(pt_query.y) + origin_.y();
        int cubeK = worldToBlockCoordinate(pt_query.z) + origin_.z();

        if(!(cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 &&
             cubeK < laserCloudDepth)) {
            return false;
        }

        int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
        if(map_[cubeInd].pkdtree_edge_from_block_ == nullptr)
            return false;

        // Get nearest neighbors of the query point
        std::vector<size_t> nearestIndex(num_nearest_search, -1);
        std::vector<float> nearestDist(num_nearest_search, -1.0);
#ifdef DONT_USE_SELF_OCTREE

#else
        map_[cubeInd].pkdtree_edge_from_block_->template knnNeighbors<nanoflann::L2Distance<Point>>(
                pt_query, num_nearest_search, nearestIndex.data(), nearestDist.data());
#endif
        // Shortcut to keypoints cloud
        const PointCloud &previousEdgePoints = *(map_[cubeInd].pedge_pc_);

        // to avoid square root when performing comparision
        const float square_max_dist_inliner = max_dist_inliner * max_dist_inliner;

        // take the closest point
        const Point &closest = previousEdgePoints[nearestIndex[0]];
        const auto P1 = closest.getVector3fMap();

        // Loop over neighbors of the neighborhood. For each of them, compute
        // the line between closest point and current point and compute the
        // number of inliers that fit this line.

        std::vector<std::vector<size_t>> inliers_list;

        for(int pt_index = 1; pt_index < num_nearest_search; ++pt_index) {
            // Fit line that links P1 and P2
            const auto P2 = previousEdgePoints[nearestIndex[pt_index]].getVector3fMap();
            Eigen::Vector3f dir = (P2 - P1).normalized();

            // Compute number of inliers of this model
            std::vector<size_t> inlier_index;
            for(int candidate_index = 1; candidate_index < num_nearest_search; ++candidate_index) {
                if(candidate_index == pt_index)
                    inlier_index.push_back(candidate_index);
                else {
                    const auto Pcdt = previousEdgePoints[nearestIndex[candidate_index]].getVector3fMap();
                    // Distance from Pcdt to the line (P1, dir): since dir is
                    // a unit vector, |(Pcdt - P1) x dir| is exactly that
                    // perpendicular distance.
                    if(((Pcdt - P1).cross(dir)).squaredNorm() < square_max_dist_inliner) {
                        inlier_index.push_back(candidate_index);
                    }
                }
            }
            inliers_list.push_back(inlier_index);
        }

        // Keep the line and its inliers with the most inliers
        size_t max_inliers = 0;
        int index_max_inlers = -1;
        for(size_t k = 0; k < inliers_list.size(); ++k) {
            if(inliers_list[k].size() > max_inliers) {
                max_inliers = inliers_list[k].size();
                index_max_inlers = k;
            }
        }

        //        std::vector<Point> &k_pts
        //        std::vector<float> &k_sqr_distances

        // fill
        k_pts.clear();
        k_sqr_distances.clear();

        k_pts.push_back(previousEdgePoints[nearestIndex[0]]);
        k_sqr_distances.push_back(nearestDist[0]);

        for(auto inlier : inliers_list[index_max_inlers]) {
            k_pts.push_back(previousEdgePoints[nearestIndex[inlier]]);
            k_sqr_distances.push_back(nearestDist[inlier]);
        }

        return true;
    }  // function nearestKSearchSpecificLinePoint

    /// \brief Finds the num_nearest_search surf points nearest to pt_query.
    /// Same single-block strategy as nearestKSearchEdgePoint; the caller
    /// typically fits a plane to the result for a point-to-plane residual.
    /// \param pt_query           query point in world coordinates
    /// \param k_pts              the neighbor points themselves (output)
    /// \param k_sqr_distances    squared distances in m^2 (output)
    /// \param num_nearest_search number of neighbors to return
    /// \return false if the query is outside the grid or the block has no tree yet
    bool nearestKSearchSurf(const Point &pt_query,
                            std::vector<Point> &k_pts,
                            std::vector<float> &k_sqr_distances,
                            int num_nearest_search) const {

        k_pts.clear();

        int cubeI = worldToBlockCoordinate(pt_query.x) + origin_.x();
        int cubeJ = worldToBlockCoordinate(pt_query.y) + origin_.y();
        int cubeK = worldToBlockCoordinate(pt_query.z) + origin_.z();

        if(!(cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 &&
             cubeK < laserCloudDepth)) {
            return false;
        }

        int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;

        if(map_[cubeInd].pkdtree_surf_from_block_ == nullptr)
            return false;

#ifdef DONT_USE_SELF_OCTREE
        std::vector<int> k_indices;

        map_[cubeInd].pkdtree_surf_from_block_->nearestKSearch(
            pt_query, num_nearest_search, k_indices, k_sqr_distances);
#else
        const size_t num_results = num_nearest_search;
        std::vector<size_t> k_indices(num_results);
        k_sqr_distances.resize(num_results);

        map_[cubeInd].pkdtree_surf_from_block_->template knnNeighbors<nanoflann::L2Distance<Point>>(
                pt_query, num_results, k_indices.data(), k_sqr_distances.data());
#endif
        for(auto id : k_indices)
            k_pts.push_back(map_[cubeInd].psurf_pc_->points[id]);
        return true;
    }  // function nearestKSearch_surf

    /// \brief Inserts a registered scan's edge points into the map, then
    /// re-downsamples and rebuilds the search tree of every touched block.
    ///
    /// Downsampling with a voxel grid (leaf size lineRes_) after each
    /// insertion keeps the map density bounded no matter how often the robot
    /// revisits an area; without it the map would grow and queries would
    /// slow down over time. Points outside the rolling grid are silently
    /// discarded.
    /// \param laserCloudEdgeStack edge features already transformed into the world frame
    void addEdgePointCloud(pcl::PointCloud<Point> &laserCloudEdgeStack) {

        // step1: route each point to its block and remember which blocks
        // were touched, so only those get re-filtered and re-indexed.
        std::set<int> blockInd;
        for(const auto &point : laserCloudEdgeStack) {
            int cubeI = worldToBlockCoordinate(point.x) + origin_.x();
            int cubeJ = worldToBlockCoordinate(point.y) + origin_.y();
            int cubeK = worldToBlockCoordinate(point.z) + origin_.z();

            if(cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 &&
               cubeK < laserCloudDepth) {
                int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                blockInd.insert(cubeInd);
                map_[cubeInd].insertEdgePoint(point);
            }
        }

        // step2: in parallel (TBB) over the touched blocks: voxel-filter the
        // block's cloud back down to lineRes_ density and rebuild its octree
        // from scratch. Parallelism is safe because each block is touched by
        // exactly one thread.
        std::vector<int> vblockInd(blockInd.begin(), blockInd.end());

        auto compute_func = [&](const tbb::blocked_range<std::vector<int>::iterator> &range) {
            for(auto &iter : range) {

                pcl::PointCloud<Point>::Ptr tmpLine(new pcl::PointCloud<Point>());
                pcl::VoxelGrid<Point> downSizeFilterLine;

                downSizeFilterLine.setLeafSize(lineRes_, lineRes_, lineRes_);
                downSizeFilterLine.setInputCloud(map_[iter].pedge_pc_);
                downSizeFilterLine.filter(*tmpLine);

                map_[iter].pedge_pc_ = tmpLine;

#ifdef DONT_USE_SELF_OCTREE
                if(map_[iter].pkdtree_edge_from_block_ == nullptr)
                    map_[iter].pkdtree_edge_from_block_.reset(new pcl::KdTreeFLANN<Point>());

                map_[iter].pkdtree_edge_from_block_->setInputCloud(map_[iter].pedge_pc_);
#else
                if(map_[iter].pkdtree_edge_from_block_ == nullptr)
                    map_[iter].pkdtree_edge_from_block_ =
                            std::make_shared<nanoflann::Octree<Point, Eigen::aligned_vector<Point>>>();

                map_[iter].pkdtree_edge_from_block_->initialize(map_[iter].pedge_pc_->points);
#endif
            }
        };

        tbb::blocked_range<std::vector<int>::iterator> range(vblockInd.begin(), vblockInd.end());
        tbb::parallel_for(range, compute_func);
    }  // function addLinePointCloud

    /// \brief Surf-point counterpart of addEdgePointCloud: routes points to
    /// blocks, voxel-filters each touched block at planeRes_ (coarser than
    /// edges, since planar areas have many redundant points), and rebuilds
    /// the block's octree.
    /// \param laserCloudSurfStack surf features already transformed into the world frame
    void addSurfPointCloud(pcl::PointCloud<Point> &laserCloudSurfStack) {

        std::set<int> blockInd;
        for(const auto &point : laserCloudSurfStack) {

            int cubeI = worldToBlockCoordinate(point.x) + origin_.x();
            int cubeJ = worldToBlockCoordinate(point.y) + origin_.y();
            int cubeK = worldToBlockCoordinate(point.z) + origin_.z();

            if(cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 &&
               cubeK < laserCloudDepth) {
                int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;

                blockInd.insert(cubeInd);

                map_[cubeInd].insertSurfPoint(point);
            }
        }

        std::vector<int> vblockInd(blockInd.begin(), blockInd.end());

        auto compute_func = [&](const tbb::blocked_range<std::vector<int>::iterator> &range) {
            for(auto &iter : range) {
                pcl::PointCloud<Point>::Ptr tmpSurf(new pcl::PointCloud<Point>());
                pcl::VoxelGrid<Point> downSizeFilterSurf;

                downSizeFilterSurf.setLeafSize(planeRes_, planeRes_, planeRes_);
                downSizeFilterSurf.setInputCloud(map_[iter].psurf_pc_);
                downSizeFilterSurf.filter(*tmpSurf);
                map_[iter].psurf_pc_ = tmpSurf;
#ifdef DONT_USE_SELF_OCTREE
                if(map_[iter].pkdtree_surf_from_block_ == nullptr)
                    map_[iter].pkdtree_surf_from_block_.reset(new pcl::KdTreeFLANN<Point>());

                map_[iter].pkdtree_surf_from_block_->setInputCloud(map_[iter].psurf_pc_);
#else
                if(map_[iter].pkdtree_surf_from_block_ == nullptr)
                    map_[iter].pkdtree_surf_from_block_ =
                            std::make_shared<nanoflann::Octree<Point, Eigen::aligned_vector<Point>>>();

                map_[iter].pkdtree_surf_from_block_->initialize(map_[iter].psurf_pc_->points);
#endif
            }
        };

        tbb::blocked_range<std::vector<int>::iterator> range(vblockInd.begin(), vblockInd.end());
        tbb::parallel_for(range, compute_func);
    }  // function addSurfPointCloud

    /// Concatenates every stored point (edge + surf, all blocks) into one
    /// cloud. Used for visualization/publishing, not for registration.
    pcl::PointCloud<Point> getAllLocalMap() const {
        pcl::PointCloud<Point> laserCloudMap;

        for(const auto &cube : map_) {
            if(cube.pedge_pc_)
                laserCloudMap += *(cube.pedge_pc_);
            if(cube.psurf_pc_)
                laserCloudMap += *(cube.psurf_pc_);
        }

        return laserCloudMap;
    }  // function get_all_localmap

    /// Returns edge + surf points from the 5x5x3 neighborhood around the
    /// given robot cell (the same neighborhood counted by
    /// get5x5LocalMapFeatureSize).
    pcl::PointCloud<Point> get5x5LocalMap(const Eigen::Vector3i &position) const {
        pcl::PointCloud<Point> laserCloudMap;

        int centerCubeI, centerCubeJ, centerCubeK;
        centerCubeI = position.x();
        centerCubeJ = position.y();
        centerCubeK = position.z();

        for(int i = centerCubeI - 2; i <= centerCubeI + 2; i++) {
            for(int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) {
                for(int k = centerCubeK - 1; k <= centerCubeK + 1; k++) {

                    if(i >= 0 && i < laserCloudWidth && j >= 0 && j < laserCloudHeight && k >= 0 &&
                       k < laserCloudDepth) {

                        int cubeInd = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                        if(map_[cubeInd].pedge_pc_)
                            laserCloudMap += *(map_[cubeInd].pedge_pc_);

                        if(map_[cubeInd].psurf_pc_)
                            laserCloudMap += *(map_[cubeInd].psurf_pc_);
                    }
                }
            }
        }

        return laserCloudMap;
    }  // function get_5x5_localmap

    /// Same neighborhood as get5x5LocalMap but returns only edge (corner)
    /// points.
    pcl::PointCloud<Point>
    get_5x5_localmap_corner(const Eigen::Vector3i &position) const
    {
        pcl::PointCloud<Point> laserCloudMap;

        int centerCubeI, centerCubeJ, centerCubeK;
        centerCubeI = position.x();
        centerCubeJ = position.y();
        centerCubeK = position.z();

        for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
        {
            for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
            {
                for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
                {

                    if (i >= 0 && i < laserCloudWidth && j >= 0 && j < laserCloudHeight &&
                        k >= 0 && k < laserCloudDepth)
                    {

                        int cubeInd = i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k;
                        if (map_[cubeInd].pedge_pc_)
                            laserCloudMap += *(map_[cubeInd].pedge_pc_);
                    }
                }
            }
        }

        return laserCloudMap;
    } // function get_5x5_localmap_corner

    /// Same neighborhood as get5x5LocalMap but returns only surf (planar)
    /// points.
    pcl::PointCloud<Point>
    get_5x5_localmap_surf(const Eigen::Vector3i &position) const
    {
        pcl::PointCloud<Point> laserCloudMap;

        int centerCubeI, centerCubeJ, centerCubeK;
        centerCubeI = position.x();
        centerCubeJ = position.y();
        centerCubeK = position.z();

        for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
        {
            for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
            {
                for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
                {

                    if (i >= 0 && i < laserCloudWidth && j >= 0 && j < laserCloudHeight &&
                        k >= 0 && k < laserCloudDepth)
                    {

                        int cubeInd = i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k;

                        if (map_[cubeInd].psurf_pc_)
                            laserCloudMap += *(map_[cubeInd].psurf_pc_);
                    }
                }
            }
        }

        return laserCloudMap;
    } // function get_5x5_localmap_corner

public:
    // ---- storage ----------------------------------------------------------
    // All 4851 blocks in a flat array, indexed by
    // i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k.
    std::array<MapBlock, laserCloudNum> map_;

    // Voxel-grid leaf sizes (m) used when downsampling block clouds. Edges
    // are kept denser (0.2 m) than surfaces (0.4 m) because there are far
    // fewer edge points and they carry more localization information.
    float lineRes_ = 0.2;
    float planeRes_ = 0.4;

    // Offset added to world-voxel indices to get grid indices; updated by
    // setOrigin()/shiftMap() as the map rolls with the robot.
    Eigen::Vector3i origin_;
};

#endif  // LOCALMAPOCTREE_H
