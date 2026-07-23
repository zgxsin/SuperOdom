// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk
//
// ============================================================================
// OVERVIEW
// ============================================================================
// TicToc is a minimal stopwatch used across the package to measure how long
// processing steps take (e.g. feature extraction, one mapping iteration).
// Usage: create a TicToc (which starts timing), then call toc() to get the
// elapsed time in milliseconds; call tic() again to restart.
// ============================================================================

#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>

/// Simple wall-clock stopwatch; times are reported in milliseconds.
class TicToc
{
  public:
    /// Starts timing immediately on construction.
    TicToc()
    {
        tic();
    }

    /// Restarts the stopwatch.
    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    /// Returns the milliseconds elapsed since the last tic().
    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};
