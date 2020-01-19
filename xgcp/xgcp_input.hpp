#pragma once
#include "xgcp_types.hpp"
#include <pumipic_input.hpp>
namespace xgcp {

  class Mesh;
  /*
    This class will store all necessary input to construct an XGCp Mesh
   */
  class Input {
  public:
    Input(o::Library& library, char* mesh_filename, char* partition_filename,
          int num_planes, int num_processes_per_group,
          pumipic::Input::Method buffer_method, pumipic::Input::Method safe_method);
    //TODO create input constructor that reads inputs from a file?


    //Friends that can access private contents
    friend class Mesh;
    friend void setGyroConfig(Input& input);
  private:
    o::Library& library;
    char* mesh_file;
    char* partition_file;

    //Communicator parameters
    int num_planes;
    int num_core_regions;
    int num_processes_per_group;

    //Picpart input rules
    p::Input::Ownership ownership_rile;
    p::Input::Method buffer_method;
    p::Input::Method safe_method;

    //Gyro paramaters
    o::Real gyro_rmax;
    o::LO gyro_num_rings;
    o::LO gyro_points_per_ring;
    o::Real gyro_theta;

  };
}
