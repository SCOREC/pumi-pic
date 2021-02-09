#include "pumipic_mesh.hpp"
#include <Omega_h_file.hpp>
#include <fstream>

namespace pumipic {

  //Copied from https://github.com/SNLComputation/omega_h/blob/main/src/Omega_h_file.cpp
  bool is_little_endian_cpu() {
    static std::uint16_t const endian_canary = 0x1;
    std::uint8_t const* p = reinterpret_cast<std::uint8_t const*>(&endian_canary);
    return *p == 0x1;
  }

  void write(Mesh& picparts, const char* prefix) {
    Omega_h::CommPtr comm = picparts.comm();
    char dir[120];
    sprintf(dir, "%s_%d", prefix, comm->size());

    if (!Omega_h::filesystem::exists(dir)) {
      if (!Omega_h::filesystem::create_directory(dir)) {
        fprintf(stderr, "[ERROR] Failed to create directory %s\n", dir);
        return;
      }
    }
    char mesh_file[240];
    sprintf(mesh_file, "%s/%s_%d.osh", dir, prefix, comm->rank());
    char ppm_file[240];
    sprintf(ppm_file, "%s/%s_%d.ppm", dir, prefix, comm->rank());

    //Write the omega_h mesh for the picpart
    Omega_h::binary::write(mesh_file, picparts.mesh());

    //Write file for the pumipic mesh data
    std::ofstream out_str(ppm_file);
    if (!out_str) {
      fprintf(stderr, "[ERROR] Failed to open file %s\n", ppm_file);
    }
    //TODO check if Omega_h is using ZLIB to set compress to true
    bool compress = false;
    bool swap = !is_little_endian_cpu();
    //Write Version
    Omega_h::binary::write(out_str, "1.0", false);
    //Write is_full_mesh
    Omega_h::binary::write_value(out_str, (Omega_h::I8)picparts.is_full_mesh, compress);
    for (int i = 0; i < 4; ++i) {
      //Write num_cores
      Omega_h::binary::write_value(out_str, picparts.num_cores[i], compress);
      //Write buffered_parts
      Omega_h::binary::write_array(out_str,
        Omega_h::LOs(Omega_h::Write<Omega_h::LO>(picparts.buffered_parts[i])),
                                   compress, swap);
      //Write offset_ents_per_rank_per_dim
      Omega_h::binary::write_array(out_str, picparts.offset_ents_per_rank_per_dim[i],
                                   compress, swap);
      //Write ent_to_comm_arr_index_per_dim
      Omega_h::binary::write_array(out_str, picparts.ent_to_comm_arr_index_per_dim[i],
                                   compress, swap);
      //Write is complete part
      Omega_h::binary::write_array(out_str,
        Omega_h::LOs(Omega_h::Write<Omega_h::LO>(picparts.is_complete_part[i])),
                                   compress, swap);
      //write num_bounds
      Omega_h::binary::write_value(out_str, picparts.num_bounds[i], compress);
      //write num_boundaries
      Omega_h::binary::write_value(out_str, picparts.num_boundaries[i], compress);
      //write boundary_parts
      Omega_h::binary::write_array(out_str,
        Omega_h::LOs(Omega_h::Write<Omega_h::LO>(picparts.boundary_parts[i])),
                                   compress, swap);
      //write offset_bounded_per_dim
      Omega_h::binary::write_array(out_str,
        Omega_h::LOs(Omega_h::Write<Omega_h::LO>(picparts.offset_bounded_per_dim[i])),
                                   compress, swap);
      //write bounded_ent_ids
      Omega_h::binary::write_array(out_str, picparts.bounded_ent_ids[i],
                                   compress, swap);
    }
  }

  Mesh read(Omega_h::CommPtr comm, const char* prefix) {

  }
}
