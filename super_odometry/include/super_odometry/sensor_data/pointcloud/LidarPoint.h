//
// Created by ubuntu on 2020/9/14.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// This header defines custom PCL point types used by the feature extraction
// and mapping nodes. Lidar drivers attach extra data to every 3D point
// (per-point timestamp, intensity, laser ring), and standard PCL types like
// pcl::PointXYZI cannot carry it, so we define our own structs and register
// them with PCL.
//
// Why the extra fields matter:
//   - time:  the lidar spins while the robot moves, so each point in a scan
//     is captured at a slightly different instant. The per-point timestamp
//     lets the pipeline undistort the scan (motion compensation).
//   - ring / laserId: multi-beam lidars have a stack of lasers at fixed
//     vertical angles; knowing which beam produced a point lets feature
//     extraction sort points into horizontal scan lines and compute
//     curvature along each line.
//
// PCL requirements explained:
//   - PCL_ADD_POINT4D adds float x, y, z plus a fourth padding float
//     (data[3]) so a point can be treated as a 4-vector in SSE code;
//     data[3] must be 1.0 for homogeneous-coordinate transforms.
//   - EIGEN_ALIGN16 / EIGEN_MAKE_ALIGNED_OPERATOR_NEW keep the struct
//     16-byte aligned for those vectorized operations.
//   - POINT_CLOUD_REGISTER_POINT_STRUCT tells PCL the field names/offsets so
//     pcl::fromROSMsg and friends can (de)serialize the type.
// ============================================================================

#ifndef LIDARPOINT_H
#define LIDARPOINT_H
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/// Raw data layout of PointXYZTIId (PCL convention: a plain "_Name" struct
/// holds the fields, and the public "Name" struct adds constructors).
struct EIGEN_ALIGN16 _PointXYZTIId {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PCL_ADD_POINT4D     // float x, y, z + padding float data[3]
  float time;         // per-point capture time, seconds relative to the scan start
  uint8_t intensity;  // return-strength / reflectivity reported by the lidar (0-255)
  uint8_t laserId;    // index of the laser beam (ring) that produced this point
} EIGEN_ALIGN16;

PCL_EXPORTS std::ostream &operator<<(std::ostream &os, const _PointXYZTIId &p);

/** \brief A point structure representing Euclidean xyz coordinates, time,
 * intensity, and laserId. \ingroup common
 */
struct PointXYZTIId : public _PointXYZTIId {
  /// Copy-constructs from the raw layout; data[3] is set to 1.0 so the
  /// point works as a homogeneous coordinate in matrix transforms.
  inline PointXYZTIId(const _PointXYZTIId &p) {
    x = p.x;
    y = p.y;
    z = p.z;
    data[3] = 1.0f;
    time = p.time;
    intensity = p.intensity;
    laserId = p.laserId;
  }

  inline PointXYZTIId() {
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
    data[3] = 1.0f;
    time = 0.0;
    intensity = 0;
    laserId = 0;
  }

  friend std::ostream &operator<<(std::ostream &os, const PointXYZTIId &p);
};

/// Point with intensity, per-point time and ring index. This is the layout
/// published by the Velodyne driver and the common working format the
/// feature extraction node converts other lidar formats into.
struct EIGEN_ALIGN16 PointXYZITR
{
  PCL_ADD_POINT4D;   // float x, y, z + padding float data[3]
  float intensity;   // return-strength / reflectivity reported by the lidar
  float time;        // per-point capture time, seconds relative to the scan start
  uint16_t ring;     // laser-beam index (0 = lowest beam); groups points into scan lines

  /// Convenience factory; the 0.f fills the data[3] padding slot.
  static inline PointXYZITR make(float x, float y, float z, float intensity, float t, uint16_t ring)
  {
    return {x, y, z, 0.f, intensity, t, ring};
  }
};
// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZTIId,
                                 (float, x, x)
                                 (float, y, y)
                                 (float, z, z)
                                 (float, time, time)
                                 (uint8_t, intensity, intensity)
                                 (uint8_t, laserId, laserId))

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZITR,
                                  (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (float, intensity, intensity)
                                    (float, time, time)
                                    (uint16_t, ring, ring))
// clang-format on

#endif // LIDARPOINT_H
