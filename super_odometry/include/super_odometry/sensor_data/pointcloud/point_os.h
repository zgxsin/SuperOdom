//
// Created by shibo zhao on 2020/6/29.
//
// ============================================================================
// OVERVIEW (read this first)
// ============================================================================
// Custom PCL point types matching the wire formats of specific lidar
// drivers, primarily the Ouster ("os" = Ouster Sensor). The feature
// extraction node deserializes incoming PointCloud2 messages into these
// structs and then converts them into the pipeline's common working format.
//
// Multi-beam lidars attach extra data to every 3D point that standard PCL
// types cannot hold: a per-point timestamp (needed to undistort the scan,
// because the lidar spins while the robot moves) and a ring index (which
// laser beam produced the point, needed to group points into horizontal
// scan lines for feature extraction).
//
// PCL_NO_PRECOMPILE is required before including PCL headers so that PCL
// templates are instantiated for these custom (non-precompiled) types, and
// POINT_CLOUD_REGISTER_POINT_STRUCT at the bottom tells PCL the field
// names/offsets for (de)serialization.
// ============================================================================

#ifndef POINT_OS_H
#define POINT_OS_H

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>

namespace point_os {
/// Minimal point with intensity and a per-point timestamp but no ring
/// index; used where the beam index is not needed.
struct EIGEN_ALIGN16 PointOS {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PCL_ADD_POINT4D;   // float x, y, z + padding float data[3]
  float intensity;   // return-strength / reflectivity reported by the lidar
  float t;           // per-point capture time, seconds relative to the scan start

  /// Convenience factory; the 0.0f fills the data[3] padding slot.
  static inline PointOS make(float x, float y, float z, float intensity,
                             float t) {
    return {x, y, z, 0.0f, intensity, t};
  }
};

/// Point with intensity, per-point time and ring index; same layout as
/// PointXYZITR in LidarPoint.h and the common working format of the
/// pipeline.
struct EIGEN_ALIGN16 PointcloudXYZITR {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PCL_ADD_POINT4D;   // float x, y, z + padding float data[3]
  float intensity;   // return-strength / reflectivity reported by the lidar
  float time;        // per-point capture time, seconds relative to the scan start
  uint16_t ring;     // laser-beam index (0 = lowest beam); groups points into scan lines
  static inline PointcloudXYZITR make(float x, float y, float z,
                                      float intensity, float t, uint16_t ring) {
    return {x, y, z, 0.0f, intensity, t, ring};
  }
};


/// Wire format of the Ouster driver. Note the units differ from the other
/// types: t is in nanoseconds since the scan start, and range is the
/// measured distance in millimeters. The ring and noise fields present in
/// some driver versions are commented out to match the deployed driver.
struct OusterPointXYZIRT {
        PCL_ADD_POINT4D;       // float x, y, z + padding float data[3]
        float intensity;       // return-strength reported by the lidar
        uint32_t t;            // per-point capture time, nanoseconds since scan start
        uint16_t reflectivity; // calibrated reflectivity of the surface
        // uint8_t ring;
        // uint16_t noise;
        uint32_t range;        // measured distance in millimeters
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    } EIGEN_ALIGN16;

} // namespace point_os

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT( point_os::PointOS,
    (float, x,x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, t, t)
    )


POINT_CLOUD_REGISTER_POINT_STRUCT( point_os::PointcloudXYZITR,
   (float, x,x)
   (float, y, y)
   (float, z, z)
   (float, intensity, intensity)
   (float, time, time)
   (uint16_t, ring, ring)
)

// POINT_CLOUD_REGISTER_POINT_STRUCT(point_os::OusterPointXYZIRT,
//                                   (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
//                                           (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
//                                           (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
// )

POINT_CLOUD_REGISTER_POINT_STRUCT(point_os::OusterPointXYZIRT,
                                  (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
                                          (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
                                          (uint32_t, range, range)
)


// clang-format on
#endif // POINT_OS_H
