#include "xgcm_input.hpp"

namespace xgcm {

  Input::Input(o::Library& lib, char* mesh_filename, char* partition_filename,
               int nplanes, int nppg,
               pumipic::Input::Method bm, pumipic::Input::Method sm) : library(lib) {
    int comm_size = lib.world()->size();
    int comm_rank = lib.world()->rank();

    mesh_file = mesh_filename;
    partition_file = partition_filename;
    num_planes = nplanes;
    num_processes_per_group = nppg;
    num_core_regions = comm_size / nplanes /nppg;
    buffer_method = bm;
    safe_method = sm;

    if (comm_size != num_core_regions * num_planes * num_processes_per_group) {
      if (!comm_rank)
        fprintf(stderr, "[ERROR] Total ranks must equal num_core_regions * "
                "num_planes * num_processes_per_group (%d != %d)\n", comm_size,
                num_core_regions * num_planes * num_processes_per_group);
      throw std::runtime_error("Incorrect number of ranks");
    }

    //Default gyro parameters
    gyro_rmax = 0.038;
    gyro_num_rings = 3;
    gyro_points_per_ring = 8;
    gyro_theta = 0;

  }
}
