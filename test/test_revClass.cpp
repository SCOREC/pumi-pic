#include <fstream>

#include <particle_structs.hpp>
#include <Omega_h_file.hpp>  //gmsh
#include <pumipic_mesh.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_array_ops.hpp> //get_max
#include <Omega_h_atomics.hpp> //atomic_increment

namespace o = Omega_h;
namespace p = pumipic;


template <class T>
void writeArray(o::Read<T> arr, std::string name) {
  o::HostRead<T> arr_hr(arr);
  o::LO const nl = arr_hr.size();
  std::cout << name << " size " << nl << "\n";
  for(int l=0; l<nl; l++) {
    const auto d = arr_hr[l];
    std::cout << l << " " << (int)d << "\n";
  }
}

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  const auto comm_rank = lib.world()->rank();
  const auto comm_size = lib.world()->size();
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

  auto class_id = full_mesh.get_array<o::ClassId>(2, "class_id");
  auto class_dim = full_mesh.get_array<o::I8>(2, "class_dim");

  {
  auto const n_dim = o::get_max(class_dim)+1;
  o::Write<o::LO> dimCnt(n_dim, 0, "dim_count");
  auto count_dim = OMEGA_H_LAMBDA (o::LO i) {
    assert(class_dim[i]>=0);
    auto const d = class_dim[i];
    o::atomic_increment(&dimCnt[d]);
  };
  o::parallel_for(full_mesh.nents(2), count_dim);
  writeArray(o::Read<o::LO>(dimCnt), "dim");
  }

  {
  auto start_rev = std::chrono::system_clock::now();
  auto faces=full_mesh.ask_revClass(2);
  auto end_rev = std::chrono::system_clock::now();
  std::chrono::duration<double> dur_rev = end_rev - start_rev;
  std::cerr <<"Time in rev class (omegah mesh) " << dur_rev.count() << " seconds\n";
  }

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

  auto class_id_pp = mesh->get_array<o::ClassId>(2, "class_id");
  auto class_dim_pp = mesh->get_array<o::I8>(2, "class_dim");

  {
  auto const n_dim = o::get_max(class_dim)+1;
  o::Write<o::LO> dimCnt(n_dim, 0, "dim_count");
  auto count_dim = OMEGA_H_LAMBDA (o::LO i) {
    assert(class_dim_pp[i]>=0);
    auto const d = class_dim_pp[i];
    o::atomic_increment(&dimCnt[d]);
  };
  o::parallel_for(mesh->nents(2), count_dim);
  writeArray(o::Read<o::LO>(dimCnt), "dim");
  }

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
