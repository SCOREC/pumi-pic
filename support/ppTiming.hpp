#pragma once

/*
  Provides a timing utility to record and output timing of operations.

  The output can be controlled using this function:
    SetTimingVerbosity(verbosity)
  Details of each level of verbosity can be found below

  By default the timing functions record on process 0 of MPI_COMM_WORLD.
  Recording can be enabled and disabled on each process using these functions:
    EnableTiming() - enable calling process to record timing
    DisableTiming() - disable calling process to record timing

  To record timing of an operation use:
    RecordTime(string, seconds, prebarrierTime (optional))
  This will accumulate all calls with the same `string` and if verbosity is set high enough print a message with the provided timing

  To print the accumulated timing information you can call either:
    SummarizeTime() - prints timing info for enabled processes
    SummarizeTimeAcrossProcesses() - prints averaged timing info over all enabled processes
*/

namespace pumipic {

  /*
    Sets the output verbosity for recording time
    -1 - No output
    0  - Summarized statistics at the end of simulation (default)
    1  - Output on each call to RecordTime
   */
  void SetTimingVerbosity(int verbosity);

  //Turns on time recording on the calling process
  void EnableTiming();
  //Turns off time recording on the calling process
  void DisableTiming();

  /*
    Adds `seconds` time for the string provided in `str` with optional prebarrier time in: `prebarrierTime`

    If verbosity has been set to 1 then a message of the following form is printed:
      <comm_rank> str (seconds) %f
    Or if prebarrierTime is provided:
      <comm_rank> str (seconds) %f pre-barrier (seconds) %f
  */
  void RecordTime(std::string str, double seconds, double prebarrierTime = 0.0);

  /*
    Allows printing additional info using the timing verbosity. `str` will only be printed if
    verbosity was set greater than or equal to the passed in `verbosity`
  */
  void PrintAdditionalTimeInfo(char* str, int verbosity);

  //Sorting options for timing summary
  enum TimingSortOption {
    SORT_ALPHA,  //sort alphabetically
    SORT_ORDER,  //sort in order of operation occurrence
    SORT_LONGEST, //sort by longest operations first
    SORT_SHORTEST //sort by shortest operations first
  };

  /*
    Print a summary of all recorded timing on each process that enabled recording
  */
  void SummarizeTime(TimingSortOption sort = SORT_ALPHA);

  /*
    Print a summary of all recorded timing reduced by all ranks that enable recording

    Note: This is a collective call and must be called by every process
  */
  void SummarizeTimeAcrossProcesses();
}
