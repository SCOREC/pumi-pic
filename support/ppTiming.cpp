#include "ppTiming.hpp"
#include <unordered_map>
#include <vector>
#include <mpi.h>
#include <algorithm>

namespace {
  int verbosity = 0;
  int enable_timing = 0;
  std::unordered_map<std::string, int> timing_index;

  const double PREBARRIER_TOL = .000001;
  struct TimeInfo {
    TimeInfo(std::string s) : str(s), time(0), hasPrebarrier(false), prebarrier(0), count(0) {}
    std::string str;
    double time;
    int count;
    bool hasPrebarrier;
    double prebarrier;
  };
  std::vector<TimeInfo> time_per_op;

  bool isTiming() {
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    return enable_timing > 0 || (enable_timing == 0 && comm_rank == 0);
  }
}

namespace pumipic {

  void SetTimingVerbosity(int v) {
    if (time_per_op.size() > 0) {
      fprintf(stderr, "[ERROR] Cannot change timing verbosity after first call to RecordTime\n");
      return;
    }
    verbosity = v;
  }

  void EnableTiming() {
    if (time_per_op.size() > 0) {
      fprintf(stderr, "[ERROR] Cannot enable timing after first call to RecordTime\n");
      return;
    }
    enable_timing = 1;
  }
  void DisableTiming() {
    if (time_per_op.size() > 0) {
      fprintf(stderr, "[ERROR] Cannot disable timing after first call to RecordTime\n");
      return;
    }
    enable_timing = -1;
  }

  void RecordTime(std::string str, double seconds, double prebarrierTime) {
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    if (isTiming()) {
      if (verbosity >= 0) {
        auto itr = timing_index.find(str);
        if (itr == timing_index.end()) {
          itr = (timing_index.insert(std::make_pair(str, time_per_op.size()))).first;
          time_per_op.push_back(TimeInfo(str));
        }
        int index = itr->second;
        time_per_op[index].time += seconds;
        ++(time_per_op[index].count);
        if (prebarrierTime >= PREBARRIER_TOL) {
          time_per_op[index].hasPrebarrier = true;
          time_per_op[index].prebarrier += prebarrierTime;
        }
        if (verbosity >= 1) {
          char buffer[1024];
          char* ptr = buffer + sprintf(buffer, "%d %s (seconds) %f", comm_rank, str.c_str(),
                                       seconds);
          if (prebarrierTime >= PREBARRIER_TOL) {
            ptr += sprintf(ptr, " pre-brarrier (seconds) %f", prebarrierTime);
          }
          fprintf(stderr, "%s\n", buffer);
        }
      }
    }
  }

  void PrintAdditionalTimeInfo(char* str, int v) {
    if (isTiming() && verbosity >= v) {
      fprintf(stderr, "%s\n", str);
    }
  }

  bool sortByAlpha(const TimeInfo& first, const TimeInfo& second) {
    return first.str < second.str;
  }
  bool sortByLongest(const TimeInfo& first, const TimeInfo& second) {
    return first.time > second.time;
  }
  bool sortByShortest(const TimeInfo& first, const TimeInfo& second) {
    return first.time < second.time;
  }
  void sortTimeInfo(TimingSortOption sort) {
    if (sort == SORT_ALPHA) {
      std::sort(time_per_op.begin(), time_per_op.end(), sortByAlpha);
    }
    else if (sort == SORT_ORDER) {

    }
    else if (sort == SORT_LONGEST) {
      std::sort(time_per_op.begin(), time_per_op.end(), sortByLongest);
    }
    else if (sort == SORT_SHORTEST) {
      std::sort(time_per_op.begin(), time_per_op.end(), sortByShortest);
    }
  }

  void SummarizeTime(TimingSortOption sort) {
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    if (isTiming()) {
      if (verbosity >= 0) {
        int name_length = 9;
        int tt_length = 10;
        int cc_length = 10;
        int at_length = 12;
        sortTimeInfo(sort);
        for (std::size_t index = 0; index < time_per_op.size(); ++index) {
          if (time_per_op[index].str.size() > static_cast<size_t>(name_length))
            name_length = time_per_op[index].str.size();
          int len = log10(time_per_op[index].time) + 8;
          if (len > tt_length)
            tt_length = len;
          len = log10(time_per_op[index].count) + 1;
          if (len > cc_length)
            cc_length = len;
          len = log10(time_per_op[index].time / time_per_op[index].count) + 8;
          if (len > at_length)
            len = at_length;
        }
        char buffer[8192];
        char* ptr = buffer + sprintf(buffer, "Timing Summary %d\n", comm_rank);
        ptr += sprintf(ptr, "Operation   %*sTotal Time   %*sCall Count   %*sAverage Time\n",
                       name_length - 9 , "", tt_length - 10, "",
                       cc_length - 10, "");
        for (size_t index = 0; index < time_per_op.size(); ++index) {
          ptr += sprintf(ptr, "%s   %*s%*.6f   %*d   %*.6f",
                         time_per_op[index].str.c_str(),
                         (int)(name_length - time_per_op[index].str.size()), "",
                         tt_length, time_per_op[index].time,
                         cc_length, time_per_op[index].count,
                         at_length, time_per_op[index].time / time_per_op[index].count);
          if (time_per_op[index].hasPrebarrier) {
            ptr += sprintf(ptr, "  Total Prebarrier=%f", time_per_op[index].prebarrier);
          }
          ptr += sprintf(ptr, "\n");
        }
        fprintf(stderr, "%s\n", buffer);
      }
    }
  }

  /*
    Print a summary of all recorded timing reduced by all ranks that enable recording

    Note: This is a collective call and must be called by every process
  */
  void SummarizeTimeAcrossProcesses() {
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    int total_timing = 0;
    int is_timing = isTiming();
    MPI_Reduce(&is_timing, &total_timing, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (verbosity >= 0) {
      char buffer[8192];
      char* ptr = buffer + sprintf(buffer, "Reduced Timing Summary\nNot Implemented Yet");
      //TODO
      // for (int index = 0; index < time_per_op.size(); ++index) {
      //   float input_vals[4];
      //   float output_vals[4];
      //   MPI_Reduce(&(time_per_op[index].time), &total_time
      //   if (comm_rank == 0) {
      //     ptr += sprintf(ptr, "%s: Average Total Time=%f  Average Call Count=%d",
      //     if (time_per_op[index].hasPrebarrier) {
      //       float totalPrebarrier = 0;
      //       ptr += sprintf(ptr, "  Average Prebarrier=%f", time_per_op[index].prebarrier);
      //     }
      //     ptr += sprintf(ptr, "\n");
      //   }
      // }
      if (comm_rank == 0)
        fprintf(stderr, "%s\n", buffer);
    }
  }
}
