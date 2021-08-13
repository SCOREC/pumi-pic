#include <fstream>

#include <particle_structs.hpp>
#include <Omega_h_file.hpp>  //gmsh
#include <pumipic_mesh.hpp>
#include <Omega_h_for.hpp>

namespace o = Omega_h;
namespace p = pumipic;

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  const auto comm_rank = lib.world()->rank();
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if(argc < 3) {
    if(comm_rank == 0)
      std::cout << "Usage: " << argv[0]
        << " <mesh> <owners_file>\n";
    exit(1);
  }
  char* meshFile = argv[1];
  char* owners = argv[2];

  int nLayers = 5;

  o::CommPtr world = lib.world();
  o::Mesh full_mesh = Omega_h::read_mesh_file(meshFile, lib.self());

  {
  auto start_rev = std::chrono::system_clock::now();
  auto faces=full_mesh.ask_revClass(2);
  auto end_rev = std::chrono::system_clock::now();
  std::chrono::duration<double> dur_rev = end_rev - start_rev;
  std::cerr <<"Time in rev class (omegah mesh) " << dur_rev.count() << " seconds\n";
  }

  o::Write<o::GO> origGids(full_mesh.nelems(), 0, 1, "origGids");
  full_mesh.add_tag(o::REGION, "origGids", 1, o::GOs(origGids));

  //Create Picparts with the full mesh
  p::Input::Method bm = p::Input::Method::BFS;
  p::Input::Method safem;
  bool fullSafe = true; //set this
  if(fullSafe)
    safem = p::Input::Method::FULL;
  else
    safem = p::Input::Method::BFS;

  //add orig global id tag for dist2bdry storage
  p::Input pp_input(full_mesh, owners, bm, safem, world);
  pp_input.bridge_dim = full_mesh.dim()-1;

  int nMinLayers = 3;
  int nSafeLayers = nLayers;
  int nBuffLayers = nSafeLayers + nMinLayers;
  if(fullSafe) {
    nBuffLayers = nLayers + nMinLayers;
    //edge layers that are non-safe
    nSafeLayers = nMinLayers;
  }

  pp_input.safeBFSLayers = nSafeLayers;
  pp_input.bufferBFSLayers = nBuffLayers; // minimum buffer region size
  p::Mesh picparts(pp_input);
  o::Mesh* mesh = picparts.mesh();

  {
  std::cerr <<"done picparts\n";
  auto start_rev = std::chrono::system_clock::now();
  auto faces=mesh->ask_revClass(2);
  auto end_rev = std::chrono::system_clock::now();
  std::chrono::duration<double> dur_rev = end_rev - start_rev;
  std::cerr <<"Time in rev class (pumipic->omegah mesh) " << dur_rev.count() << " seconds\n";
  }
  return 0;
}
