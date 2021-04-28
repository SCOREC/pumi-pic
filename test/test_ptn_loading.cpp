#include <fstream>

#include <Omega_h_file.hpp>  //gmsh
#include <pumipic_mesh.hpp>
#include <Omega_h_for.hpp>

int testGlobalTag(Omega_h::Mesh& mesh, pumipic::Mesh& picparts);
int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (argc != 5) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition filename> "
              "<# of safe layers> <# of buffer layers>\n", argv[0]);
    return EXIT_FAILURE;
  }
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int safe_layers = atoi(argv[3]);
  int ghost_layers = atoi(argv[4]);

  //**********Load the mesh in serial everywhere*************//
  Omega_h::Mesh mesh = Omega_h::read_mesh_file(argv[1], lib.self());
  int dim = mesh.dim();
  int ne = mesh.nents(dim);
  if (rank == 0)
    printf("Mesh loaded with <v e f r> %d %d %d %d\n", mesh.nverts(), mesh.nedges(),
           mesh.nfaces(), mesh.nelems());

  //********* Load the partition vector ***********//
  Omega_h::HostWrite<Omega_h::LO> host_owners(ne);
  std::ifstream in_str(argv[2]);
  if (!in_str) {
    if (!rank)
      fprintf(stderr,"Cannot open file %s\n", argv[2]);
    return EXIT_FAILURE;
  }
  int own;
  int index = 0;
  while(in_str >> own)
    host_owners[index++] = own;
  //Owner of each element
  Omega_h::Write<Omega_h::LO> owner(host_owners);

  pumipic::Mesh picparts(mesh,owner,ghost_layers, safe_layers);

  int fails = testGlobalTag(mesh, picparts);

  char vtk_name[100];
  sprintf(vtk_name, "picpart%d", rank);
  Omega_h::vtk::write_parallel(vtk_name, picparts.mesh(), dim);

  return fails;
}

int testGlobalTag(Omega_h::Mesh& mesh, pumipic::Mesh& picparts) {
  const int dim = mesh.dim();

  //Test the global tag to be converted to global_serial
  Omega_h::TagBase const* new_tag = picparts->get_tagbase(dim, "global_serial");

  Omega_h::GOs global_old = mesh.get_array<Omega_h::GO>(dim, "global");
  //Create a mapping on the full mesh of global to local id
  Omega_h::Write<Omega_h::LO> global_to_local(mesh.nelems(),-1);
  auto setMapping = OMEGA_H_LAMBDA(const Omega_h::LO ent) {
    global_to_local[global_old[ent]] = ent;
  };
  Omega_h::parallel_for(mesh.nelems(), setMapping, "setMapping");

  //Compare coordinates of the elements with same global ids
  Omega_h::GOs global_new = picparts->get_array<Omega_h::GO>(dim, "global_serial");
  Omega_h::Reals coords_old = mesh.coords();
  Omega_h::Reals coords_new = picparts->coords();
  auto faces2verts_old = mesh.ask_elem_verts();
  auto faces2verts_new = picparts->ask_elem_verts();
  const double TOL = .0000001;
  Omega_h::Write<Omega_h::LO> fails(1,0);
  if (dim == 2) {
    auto checkGlobals = OMEGA_H_LAMBDA(const Omega_h::LO ent) {
      const Omega_h::GO global = global_new[ent];
      const Omega_h::LO old_ent = global_to_local[global];

      //Calculate centroid of old element
      auto verts_old = Omega_h::gather_verts<3>(faces2verts_old, old_ent);
      const auto old_coords = Omega_h::gather_vectors<3,2>(coords_old, verts_old);
      auto center_old = average(old_coords);

      //Calculate centroid of new element
      auto verts_new = Omega_h::gather_verts<3>(faces2verts_new, ent);
      const auto new_coords = Omega_h::gather_vectors<3,2>(coords_new, verts_new);
      auto center_new = average(new_coords);

      if (fabs(center_old[0] - center_new[0]) > TOL ||
          fabs(center_old[1] - center_new[1]) > TOL) {
        printf("Centroids of same global ids do not match [%ld]\n", global);
        fails[0] = 1;
      }
    };
    Omega_h::parallel_for(picparts->nelems(), checkGlobals, "checkGlobals");
  }
  else {
    auto checkGlobals = OMEGA_H_LAMBDA(const Omega_h::LO ent) {
      const Omega_h::GO global = global_new[ent];
      const Omega_h::LO old_ent = global_to_local[global];
      //Calculate centroid of old element
      auto verts_old = Omega_h::gather_verts<4>(faces2verts_old, old_ent);
      const auto old_coords = Omega_h::gather_vectors<4,3>(coords_old, verts_old);
      auto center_old = average(old_coords);

      //Calculate centroid of new element
      auto verts_new = Omega_h::gather_verts<4>(faces2verts_new, ent);
      const auto new_coords = Omega_h::gather_vectors<4,3>(coords_new, verts_new);
      auto center_new = average(new_coords);

      if (fabs(center_old[0] - center_new[0]) > TOL ||
          fabs(center_old[1] - center_new[1]) > TOL ||
          fabs(center_old[2] - center_new[2]) > TOL) {
        printf("Centroids of same global ids do not match [%ld]", global);
        fails[0] = 1;

      }
    };
    Omega_h::parallel_for(picparts->nelems(), checkGlobals, "checkGlobals");
  }
  return Omega_h::HostWrite<Omega_h::LO>(fails)[0];
}
