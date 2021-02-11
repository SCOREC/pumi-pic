#include "pumipic_mesh.hpp"
#include <Omega_h_file.hpp>
#include <fstream>
#include "pumipic_lb.hpp"

//Helper functions for host writes to be read/write
namespace Omega_h {
  namespace binary {
    template <typename T>
    void write_array(std::ostream& stream, HostWrite<T> array, bool compress,
                     bool needs_swapping) {
      write_array(stream, Read<T>(Write<T>(array)), compress, needs_swapping);
    }
    template <typename T>
    void read_array(std::istream& stream, HostWrite<T>& array, bool compress,
                    bool needs_swapping) {
      Read<T> temp;
      read_array(stream, temp, compress, needs_swapping);
      HostRead<T> temp_host(temp);
      array = HostWrite<T>(temp_host.size());
      for (int i = 0; i < temp_host.size(); ++i) {
        array[i] = temp_host[i];
      }
    }

  }
}

namespace {
  const char* splitPath(const char* full_path) {
    const char* lastSlash = strrchr(full_path, '/');
    if (!lastSlash)
      return full_path;
    return lastSlash + 1;
  }
  //Copied from https://github.com/SNLComputation/omega_h/blob/main/src/Omega_h_file.cpp
  bool is_little_endian_cpu() {
    static std::uint16_t const endian_canary = 0x1;
    std::uint8_t const* p = reinterpret_cast<std::uint8_t const*>(&endian_canary);
    return *p == 0x1;
  }


}
namespace pumipic {
  void write(Mesh& picparts, const char* path) {
    Omega_h::CommPtr comm = picparts.comm();
    const char* prefix = splitPath(path);
    char dir[4096];
    sprintf(dir, "%s_%d.ppm", path, comm->size());
    if (comm->rank() == 0) {
      if (!Omega_h::filesystem::exists(dir)) {
        if (!Omega_h::filesystem::create_directory(dir)) {
          fprintf(stderr, "[ERROR] Failed to create directory %s\n", dir);
          return;
        }
      }
    }
    //Wait for directory to be created
    comm->barrier();

    char mesh_file[4096];
    sprintf(mesh_file, "%s/%s_%d.osh", dir, prefix, comm->rank());
    char ppm_file[4096];
    sprintf(ppm_file, "%s/%s_%d.ppm", dir, prefix, comm->rank());

    //Write the omega_h mesh for the picpart
    Omega_h::binary::write(mesh_file, picparts.mesh());

    //Write file for the pumipic mesh data
    std::ofstream out_str(ppm_file);
    if (!out_str) {
      fprintf(stderr, "[ERROR] Failed to open file %s\n", ppm_file);
      return;
    }
#ifdef OMEGA_H_USE_ZLIB
    bool compress = true;
#else
    bool compress = false;
#endif
    bool swap = !is_little_endian_cpu();
    //Write Version
    Omega_h::I8 version = 1;
    Omega_h::binary::write_value(out_str, version, swap);
    //Write is_full_mesh
    Omega_h::binary::write_value(out_str, (Omega_h::I8)picparts.is_full_mesh, swap);
    for (int i = 0; i < 4; ++i) {
      //Write num_cores
      Omega_h::binary::write_value(out_str, picparts.num_cores[i], swap);
      //Write buffered_parts
      Omega_h::binary::write_array(out_str,picparts.buffered_parts[i], compress, swap);
      //Write offset_ents_per_rank_per_dim
      Omega_h::binary::write_array(out_str, picparts.offset_ents_per_rank_per_dim[i],
                                   compress, swap);
      //Write ent_to_comm_arr_index_per_dim
      Omega_h::binary::write_array(out_str, picparts.ent_to_comm_arr_index_per_dim[i],
                                   compress, swap);
      //Write is complete part
      Omega_h::binary::write_array(out_str,picparts.is_complete_part[i],
                                   compress, swap);
      //write num_bounds
      Omega_h::binary::write_value(out_str, picparts.num_bounds[i], swap);
      //write num_boundaries
      Omega_h::binary::write_value(out_str, picparts.num_boundaries[i], swap);
      //write boundary_parts
      Omega_h::binary::write_array(out_str,picparts.boundary_parts[i], compress, swap);
      //write offset_bounded_per_dim
      Omega_h::binary::write_array(out_str,picparts.offset_bounded_per_dim[i],
                                   compress, swap);
      //write bounded_ent_ids
      Omega_h::binary::write_array(out_str, picparts.bounded_ent_ids[i],
                                   compress, swap);
    }
  }

  void read(Omega_h::Library* library, Omega_h::CommPtr comm, const char* path,
            Mesh* mesh) {
    const char* prefix = splitPath(path);
    char dir[4096];
    sprintf(dir, "%s_%d.ppm", path, comm->size());

    if (!Omega_h::filesystem::exists(dir)) {
      fprintf(stderr, "[ERROR] Directory %s does not exist\n", dir);
      return;
    }

    char mesh_file[4096];
    sprintf(mesh_file, "%s/%s_%d.osh", dir, prefix, comm->rank());
    char ppm_file[4096];
    sprintf(ppm_file, "%s/%s_%d.ppm", dir, prefix, comm->rank());

    mesh->picpart =
      new Omega_h::Mesh(Omega_h::binary::read(mesh_file, library->self()));

    std::ifstream in_str(ppm_file);
    if (!in_str) {
      fprintf(stderr, "[ERROR] Cannot open file %s\n", ppm_file);
      return;
    }


#ifdef OMEGA_H_USE_ZLIB
    bool compress = true;
#else
    bool compress = false;
#endif
    bool swap = !is_little_endian_cpu();
    Omega_h::I8 version;
    Omega_h::binary::read_value(in_str, version, swap);
    Omega_h::I8 is_full_mesh_int;
    Omega_h::binary::read_value(in_str, is_full_mesh_int, swap);
    mesh->is_full_mesh = is_full_mesh_int;

    for (int i = 0; i < 4; ++i) {
      //Read num_cores
      Omega_h::binary::read_value(in_str, mesh->num_cores[i], swap);
      //Read buffered_parts
      Omega_h::binary::read_array(in_str, mesh->buffered_parts[i], compress, swap);
      //Read offset_ents_per_rank_per_dim
      Omega_h::binary::read_array(in_str, mesh->offset_ents_per_rank_per_dim[i],
                                   compress, swap);
      //Read ent_to_comm_arr_index_per_dim
      Omega_h::binary::read_array(in_str, mesh->ent_to_comm_arr_index_per_dim[i],
                                   compress, swap);
      //Read is complete part
      Omega_h::binary::read_array(in_str, mesh->is_complete_part[i], compress, swap);
      //read num_bounds
      Omega_h::binary::read_value(in_str, mesh->num_bounds[i], swap);
      //read num_boundaries
      Omega_h::binary::read_value(in_str, mesh->num_boundaries[i], swap);
      //read boundary_parts
      Omega_h::binary::read_array(in_str, mesh->boundary_parts[i], compress, swap);
      //read offset_bounded_per_dim
      Omega_h::binary::read_array(in_str, mesh->offset_bounded_per_dim[i],
                                  compress, swap);
      //read bounded_ent_ids
      Omega_h::binary::read_array(in_str, mesh->bounded_ent_ids[i], compress, swap);

    }

    mesh->commptr = comm;
    //Create load balancer after reading in the mesh
    mesh->ptcl_balancer = new ParticleBalancer(*mesh);

  }
}
