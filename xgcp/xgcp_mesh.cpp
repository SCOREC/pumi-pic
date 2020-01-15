#include "xgcp_mesh.hpp"
#include "xgcp_gyro_scatter.hpp"
#include <Omega_h_file.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>

namespace {
  namespace o=Omega_h;
  /*
    Splits the library world comm into three comms stored in the *_comm parameters

    - mesh_comm will be a communicator over the mesh of a single plane
    - torodial_comm will be a communicator over the copies of a given core region
        around the torodial direction
    - group_comm will be a communicator over the copies of a given core region for a given plane
   */
  void partition_communicator(o::Library& lib, int num_cores, int num_planes, int group_size,
                              o::CommPtr& mesh_comm, o::CommPtr& torodial_comm,
                              o::CommPtr& group_comm);
}
namespace xgcp {
  Mesh::Mesh(xgcp::Input& input) {
    //Read the full mesh on every process
    full_mesh = new o::Mesh(o::read_mesh_file(input.mesh_file, input.library.self()));

    //Read other information (background fields, mesh parameters?)

    //Create fields on full mesh that will transfer to picparts?
    //   Note this feature is not supported in pumipic::Mesh yet

    //Breakup group/torodial partitioning
    int num_cores = input.num_core_regions;
    int num_planes = input.num_planes;
    int group_size = input.num_processes_per_group;
    partition_communicator(input.library, num_cores, num_planes, group_size,
                           mesh_comm, torodial_comm, group_comm);

    printf("Rank %d: is Mesh (%d %d) Torodial (%d %d) Group (%d %d)\n",
           input.library.world()->rank(), mesh_comm->rank(), mesh_comm->size(),
           torodial_comm->rank(), torodial_comm->size(), group_comm->rank(),
           group_comm->size());

    //Create picparts
    p::Input pp_input(*full_mesh, input.partition_file, input.buffer_method,
                      input.safe_method, mesh_comm);


    picparts = new p::Mesh(pp_input);
    //If the mesh is not fully buffered then delete the full mesh copy
    if (!picparts->isFullMesh()) {
      delete full_mesh;
      full_mesh = NULL;
    }
    o::Mesh* mesh = picparts->mesh();

    //Build gyro mappings
    setGyroConfig(input);
    if (!worldRank())
      printGyroConfig();
    createIonGyroRingMappings(mesh, major_ion_gyro_map, minor_ion_gyro_map);

    //TODO build electron mapping

    //Build plane information
    fp_t delta_phi = 2* M_PI / num_planes;
    major_phi = delta_phi * torodial_comm->rank();
    minor_phi = major_phi - delta_phi;
    if (torodial_comm->rank() == 0)
      minor_phi += 2*M_PI;
    major_plane = GyroField(mesh->nverts(), 0.0, "major_plane_field");
    minor_plane = GyroField(mesh->nverts(), 0.0, "major_plane_field");
    mesh->add_tag(0, "major_plane", 1, GyroFieldR(major_plane));
    mesh->add_tag(0, "minor_plane", 1, GyroFieldR(minor_plane));
  }

  Mesh::~Mesh() {
    delete picparts;
    if (full_mesh)
      delete full_mesh;
  }

  void Mesh::getGyroFields(GyroField& ma_plane, GyroField& mi_plane) {
    ma_plane = major_plane;
    mi_plane = minor_plane;
  }
  void Mesh::applyGyroFieldsToTags() {
    o::Mesh* mesh = omegaMesh();
    mesh->set_tag(0, "major_plane", GyroFieldR(major_plane));
    mesh->set_tag(0, "minor_plane", GyroFieldR(minor_plane));
  }
  void Mesh::getIonGyroMappings(Omega_h::LOs& major_map, Omega_h::LOs& minor_map) {
    major_map = major_ion_gyro_map;
    minor_map = minor_ion_gyro_map;
  }

  void Mesh::render(const char* prefix) {
    auto mesh = omegaMesh();
    //Get coordinates before changing them
    auto old_coords = mesh->coords();
    Omega_h::Write<Omega_h::Real> new_coords(old_coords.size());
    printf("HERE: %d has %d verts %d coords\n", worldRank(), mesh->nverts(), old_coords.size());
    //Make a directory for the files
    char directory[128];
    sprintf(directory, "%s_g%d", prefix, groupRank());
    if (torodialRank() == 0) {
      DIR* dir = opendir(directory);
      if (dir) {
        closedir(dir);
      } else if (ENOENT == errno) {
        if (mkdir(directory, S_IRUSR | S_IWUSR | S_IXUSR) != 0) {
          fprintf(stderr, "[ERROR] Failed to create directory for rendering on rank %d\n",
                  worldRank());
          return;
        }
      }
    }
    MPI_Barrier(torodialComm());
    char filename[512];
    sprintf(filename, "%s/%s_t%d.vtu", directory, prefix, torodialRank());
    Omega_h::vtk::write_vtu(filename, omegaMesh(), dim());
  }
}


namespace {
  void partition_communicator(o::Library& lib, int num_cores, int num_planes, int group_size,
                              o::CommPtr& mesh_comm, o::CommPtr& torodial_comm,
                              o::CommPtr& group_comm) {
    o::CommPtr world = lib.world();
    int rank = world->rank();

    //Adjacent ranks form a group
    int group_id = rank % group_size;
    int group_color = rank / group_size;

    //Adjacent groups form torodial comms for each group_id
    int torodial_id = group_color % num_planes;
    int torodial_color = group_size * (group_color / num_planes) + group_id;

    //Mesh rank
    int mesh_id = torodial_color / group_size;
    int mesh_color = rank % (group_size * num_planes);

    //Create Each mpi communicator
    MPI_Comm m_comm, t_comm, g_comm;
    MPI_Comm_split(world->get_impl(), mesh_color, mesh_id, &m_comm);
    MPI_Comm_split(world->get_impl(), torodial_color, torodial_id, &t_comm);
    MPI_Comm_split(world->get_impl(), group_color, group_id, &g_comm);

    //Create each omega_h commptr

    mesh_comm = o::CommPtr(new o::Comm(&lib, m_comm));
    torodial_comm = o::CommPtr(new o::Comm(&lib, t_comm));
    group_comm = o::CommPtr(new o::Comm(&lib, g_comm));
  }
}
