#include <fstream>

#include <particle_structs.hpp>
#include <Omega_h_file.hpp>  //gmsh
#include <pumipic_mesh.hpp>
#include <Omega_h_for.hpp>

namespace o = Omega_h;
namespace p = pumipic;

void writeArray(o::LOs deg, std::string name) {
  o::HostRead<o::LO> deg_hr(deg);
  o::LO const nl = deg_hr.size();
  std::cout << name << " size " << nl << "\n";
  for(int l=0; l<nl; l++) {
    const auto d = deg_hr[l];
    if(d) std::cout << l << " " << d << "\n";
  }
}

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

  const auto nt = full_mesh.ntags(2);
  for(auto i = 0; i<nt; i++) {
    auto t = full_mesh.get_tag(2,i);
    std::cout << "tag " << t->name() << " " << t->ncomps() << "\n";
  }

  auto class_dim = mesh.get_array<o::I8>(2, "class_dim");
  writeArray(class_dim, "class_dim");
  auto const n_dim = get_max(class_dim)+1;
  o::Write<o::LO> dimCnt(n_dim, 0, "dim_count");
  auto count_dim = OMEGA_H_LAMBDA (o::LO i) {
    assert(class_dim[i]>=0);
    auto const d = class_dim[i];
    atomic_increment(&dimCnt[d]);
  };
  parallel_for(mesh.nents(2), count_dim);
  writeArray(dimCnt, "dim");

  //{
  //auto start_rev = std::chrono::system_clock::now();
  //auto faces=full_mesh.ask_revClass(2);
  //auto end_rev = std::chrono::system_clock::now();
  //std::chrono::duration<double> dur_rev = end_rev - start_rev;
  //std::cerr <<"Time in rev class (omegah mesh) " << dur_rev.count() << " seconds\n";
  //}

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
  o::binary::write("pp.osh", mesh);

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
