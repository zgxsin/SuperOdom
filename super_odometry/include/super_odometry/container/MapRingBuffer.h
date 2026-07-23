//
// Created by ubuntu on 2020/6/29.
//
// ============================================================================
// OVERVIEW
// ============================================================================
// MapRingBuffer<Meas> is a small time-indexed buffer for sensor measurements.
// Each measurement is stored in a std::map keyed by its timestamp (seconds),
// so entries are always sorted by time and can be looked up efficiently.
// When the buffer exceeds its configured capacity the oldest entry is
// dropped, which gives it ring-buffer ("keep only the most recent N")
// behavior.
//
// It is used across the package to hold streams of timestamped data, e.g.
// raw IMU messages and lidar odometry poses in the imuPreintegration node
// (imuBuf, lidarOdomBuf) and point clouds / visual odometry in laserMapping.
// Typical usage: addMeas() in a callback, getFirst*/getLast* to inspect the
// time span, and clean(t) to discard everything already processed.
// ============================================================================

#ifndef MAPRINGBUFFER_H
#define MAPRINGBUFFER_H

#include <iostream>
#include <map>

/// Fixed-capacity buffer of timestamped measurements, sorted by time.
template <typename Meas>
class MapRingBuffer {
public:
  std::map<double, Meas> measMap_;  // timestamp (s) -> measurement, sorted ascending
  typename std::map<double, Meas>::iterator itMeas_;  // scratch iterator reused by queries

  int size;             // maximum number of stored measurements (set by allocate())
  double maxWaitTime_;  // max time to wait for a delayed measurement (used by waitTime())
  double minWaitTime_;

  MapRingBuffer() {
    maxWaitTime_ = 0.1;
    minWaitTime_ = 0.0;
  }

  virtual ~MapRingBuffer() {}

  /// Sets the buffer capacity. Returns false for a non-positive size.
  bool allocate(const int sizeBuffer) {
    if (sizeBuffer <= 0) {
      return false;
    } else {
      size = sizeBuffer;
      return true;
    }
  }

  /// Returns the number of measurements currently stored.
  int getSize() { return measMap_.size(); }

  /// Inserts a measurement with timestamp t; if the buffer is over capacity,
  /// the oldest entry is removed.
  void addMeas(const Meas& meas, const double& t) {
    measMap_.insert(std::make_pair(t, meas));

    // ensure the size of the map, and remove the oldest element
    if ((int) measMap_.size() > size) {
      measMap_.erase(measMap_.begin());
    }
  }

  /// Removes all measurements.
  void clear() { measMap_.clear(); }

  /// Removes every measurement with timestamp <= t (already-processed data).
  void clean(double t) {
    while (measMap_.size() >= 1 && measMap_.begin()->first <= t) {
      measMap_.erase(measMap_.begin());
    }
  }

  /// Finds the timestamp of the first measurement strictly after actualTime.
  /// Returns false if there is none.
  bool getNextTime(double actualTime, double& nextTime) {
    itMeas_ = measMap_.upper_bound(actualTime);
    if (itMeas_ != measMap_.end()) {
      nextTime = itMeas_->first;
      return true;
    } else {
      return false;
    }
  }
  /// Caps 'time' so downstream processing does not run ahead of the data in
  /// this buffer: it never returns a time later than the newest measurement
  /// (plus minWaitTime_) or actualTime - maxWaitTime_.
  void waitTime(double actualTime, double& time) {
    double measurementTime = actualTime - maxWaitTime_;
    if (!measMap_.empty() &&
        measMap_.rbegin()->first + minWaitTime_ > measurementTime) {
      measurementTime = measMap_.rbegin()->first + minWaitTime_;
    }
    if (time > measurementTime) {
      time = measurementTime;
    }
  }
  /// Timestamp of the newest measurement. Returns false if the buffer is empty.
  bool getLastTime(double& lastTime) {
    if (!measMap_.empty()) {
      lastTime = measMap_.rbegin()->first;
      return true;
    } else {
      return false;
    }
  }

  /// Timestamp of the oldest measurement. Returns false if the buffer is empty.
  bool getFirstTime(double& firstTime) {
    if (!measMap_.empty()) {
      firstTime = measMap_.begin()->first;
      return true;
    } else {
      return false;
    }
  }

  /// The newest measurement. Returns false if the buffer is empty.
  bool getLastMeas(Meas& lastMeas) {
    if (!measMap_.empty()) {
      lastMeas = measMap_.rbegin()->second;
      return true;
    } else {
      return false;
    }
  }

  /// The second-newest measurement. Returns false if fewer than two are stored.
  bool getLastLastMeas(Meas& lastlastMeas) {
    if (measMap_.size() >= 2) {
      auto itr = measMap_.rbegin();
      itr++;
      lastlastMeas = itr->second;
      return true;
    } else {
      return false;
    }
  }

  /// The oldest measurement. Returns false if the buffer is empty.
  bool getFirstMeas(Meas& firstMeas) {
    if (!measMap_.empty()) {
      firstMeas = measMap_.begin()->second;
      return true;
    } else {
      return false;
    }
  }

  /// True if a measurement exists with exactly this timestamp.
  bool hasMeasurementAt(double t) { return measMap_.count(t) > 0; }

  bool empty() { return measMap_.empty(); }

  /// Prints all stored measurements to stdout (debugging aid).
  void printContainer() {
    itMeas_ = measMap_.begin();
    while (measMap_.size() >= 1 && itMeas_ != measMap_.end()) {
      std::cout << itMeas_->second << " ";
      itMeas_++;
    }
  }
};
#endif // MAPRINGBUFFER_H
