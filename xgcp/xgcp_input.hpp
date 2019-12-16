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
    friend class Mesh;
  private:
    o::Library& library;
    char* mesh_file;
    char* partition_file;
    int num_planes;
    int num_core_regions;
    int num_processes_per_group;
    p::Input::Ownership ownership_rile;
    p::Input::Method buffer_method;
    p::Input::Method safe_method;
  };
}
