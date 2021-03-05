#include "ppTiming.hpp"
#include <unordered_map>
#include <vector>
#include <mpi.h>
#include <algorithm>
#include <sstream>

namespace {
  int verbosity = 0;
  int enable_timing = 0;
  std::unordered_map<std::string, int> timing_index;

  const double PREBARRIER_TOL = .000001;
  struct TimeInfo {
    TimeInfo(std::string s, int index) : str(s), time(0), hasPrebarrier(false),
                                         prebarrier(0), count(0), orig_index(index) {}
    std::string str;
    double time;
    int count;
    bool hasPrebarrier;
    double prebarrier;
    int orig_index;
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
          time_per_op.push_back(TimeInfo(str,time_per_op.size()));
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
  bool sortByOrder(const TimeInfo& first, const TimeInfo& second) {
    return first.orig_index < second.orig_index;
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
      std::sort(time_per_op.begin(), time_per_op.end(), sortByOrder);
    }
    else if (sort == SORT_LONGEST) {
      std::sort(time_per_op.begin(), time_per_op.end(), sortByLongest);
    }
    else if (sort == SORT_SHORTEST) {
      std::sort(time_per_op.begin(), time_per_op.end(), sortByShortest);
    }

    //Reset indices of map
    for (std::size_t i = 0; i < time_per_op.size(); ++i) {
      timing_index[time_per_op[i].str] = i;
    }
  }

  int length(int x) {
    if (x== 0)
      return 1;
    else
      return trunc(log10(x)) + 1;
  }
  void determineLengths(int& name_length, int& tt_length, int& cc_length,
                        int& at_length) {
    for (std::size_t index = 0; index < time_per_op.size(); ++index) {
      if (time_per_op[index].str.size() > name_length)
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
        determineLengths(name_length, tt_length, cc_length, at_length);
        sortTimeInfo(sort);
        char buffer[8192];
        char* ptr = buffer + sprintf(buffer, "Timing Summary %d\n", comm_rank);
        ptr += sprintf(ptr, "Operation   %*sTotal Time   %*sCall Count   "
                       "%*sAverage Time\n", name_length - 9 , "",
                       tt_length - 10, "", cc_length - 10, "");
        for (int index = 0; index < time_per_op.size(); ++index) {
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
  void SummarizeTimeAcrossProcesses(TimingSortOption sort) {
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    int total_timing = 0;
    int is_timing = isTiming();
    MPI_Reduce(&is_timing, &total_timing, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (verbosity >= 0) {
      int name_length = 9;
      for (std::size_t index = 0; index < time_per_op.size(); ++index) {
        if (time_per_op[index].str.size() > name_length)
          name_length = time_per_op[index].str.size();
      }
      int tt_length = 15;
      int proc_length = 8;
      int one = 1;
      if (!comm_rank) {
        sortTimeInfo(sort);
        char buffer[8192];
        char* ptr = buffer + sprintf(buffer, "Reduced Timing Summary with %d ranks\n",
                                     total_timing);
        ptr += sprintf(ptr, "Operation   %*sMax Time (max proc)   %*sMin Time (min proc)   "
                       "%*sAverage Time\n", name_length - 9, "",
                       tt_length - 8, "", tt_length - 8, "");
        for (std::size_t index = 0; index < time_per_op.size(); ++index) {
          auto& op = time_per_op[index];
          int size = op.str.size() + 1;
          MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
          MPI_Bcast(&(op.str[0]), size, MPI_CHAR, 0, MPI_COMM_WORLD);
          //Code adapted from here: https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node79.html
          struct {
            double val;
            int rank;
          } time_rank, max_time, min_time;
          double avg_time;
          time_rank.val = op.time;
          time_rank.rank = comm_rank;
          int num_procs;
          MPI_Reduce(&one, &num_procs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&time_rank, &max_time, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
          MPI_Reduce(&time_rank, &min_time, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
          MPI_Reduce(&(op.time), &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          avg_time /= num_procs;
          ptr += sprintf(ptr, "%s   %*s%*.2f (%d)%*s   %*.2f (%d)%*s   %*.2f\n",
                         time_per_op[index].str.c_str(),
                         (int)(name_length - time_per_op[index].str.size()), "",
                         tt_length, max_time.val, max_time.rank,
                         proc_length - length(max_time.rank), "",
                         tt_length, min_time.val, min_time.rank,
                         proc_length - length(min_time.rank), "",
                         tt_length, avg_time);

        }
        char end[1]; end[0] = '~';
        int size = 1;
        MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(end, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        fprintf(stderr, "%s\n", buffer);

      }
      else if (comm_rank) {
        int size;
        struct {
          double val;
          int rank;
        } time_rank;
        time_rank.rank = comm_rank;
        int zero = 0;
        char op_name[128];
        MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(op_name, size, MPI_CHAR, 0, MPI_COMM_WORLD);
        while (op_name[0] != '~') {
          auto itr = timing_index.find(std::string(op_name));
          if (itr != timing_index.end()) {
            int index = timing_index[std::string(op_name)];
            const auto& op = time_per_op[index];
            time_rank.val = op.time;
            MPI_Reduce(&one, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&time_rank, NULL, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
            MPI_Reduce(&time_rank, NULL, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
            MPI_Reduce(&(op.time), NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          }
          else {
            MPI_Reduce(&zero, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            time_rank.val = 0;
            MPI_Reduce(&time_rank, NULL, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
            time_rank.val = pow(10,10);
            MPI_Reduce(&time_rank, NULL, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
            time_rank.val = 0;
            MPI_Reduce(&(time_rank.val), NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          }
          MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
          MPI_Bcast(op_name, size, MPI_CHAR, 0, MPI_COMM_WORLD);
        }
      }
    }
  }
}
