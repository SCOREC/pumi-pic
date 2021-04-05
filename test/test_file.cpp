#include <fstream>

#include <Omega_h_file.hpp>
#include <pumipic_mesh.hpp>
#include <Omega_h_for.hpp>

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (argc != 6) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition filename>"
              "<buffer method=[bfs|full]> <safe method=[bfs|full]>"
              "<output prefix>\n", argv[0]);
    return EXIT_FAILURE;
  }
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  //**********Load the mesh in serial everywhere*************//
  Omega_h::Mesh mesh = Omega_h::read_mesh_file(argv[1], lib.self());
  int dim = mesh.dim();
  int ne = mesh.nents(dim);
  if (rank == 0)
    printf("Mesh loaded with <v e f r> %d %d %d %d\n", mesh.nverts(), mesh.nedges(),
           mesh.nfaces(), mesh.nelems());

  const auto bufferMethod = pumipic::Input::getMethod(argv[3]);
  const auto safeMethod = pumipic::Input::getMethod(argv[4]);
  assert(bufferMethod>=0);
  assert(safeMethod>=0);

  pumipic::Input input(mesh, argv[2], bufferMethod, safeMethod);
  pumipic::Mesh picparts(input);

  //Write picparts to a file
  pumipic::write(picparts, argv[5]);


  //Reread the picparts from file to a new mesh
  pumipic::Mesh read_picparts;
  pumipic::read(&lib, picparts.comm(), argv[5], &read_picparts);

  /************Compare picparts vs read_picparts***********/
  //Check basic values
  assert(picparts.isFullMesh() == read_picparts.isFullMesh());

  assert(picparts.dim() == read_picparts.dim());
  for (int i = 0; i < picparts.dim(); ++i) {
    assert(picparts->nents(i) == read_picparts->nents(i));
  }

  //Check communicator
  assert(picparts.comm()->size() == read_picparts.comm()->size());
  assert(picparts.comm()->rank() == read_picparts.comm()->rank());

  //Check entity dimension fields
  for (int i = 0; i < picparts.dim(); ++i) {
    assert(picparts.numBuffers(i) == read_picparts.numBuffers(i));
    for (int j = 0; j < picparts.numBuffers(i) - 1; ++j) {
      assert(picparts.bufferedRanks(i)[j] == read_picparts.bufferedRanks(i)[j]);
    }
    Omega_h::Write<Omega_h::LO> failed(1,0);
    auto gids = picparts.globalIds(i);
    auto gids_r = read_picparts.globalIds(i);
    auto owners = picparts.entOwners(i);
    auto owners_r = read_picparts.entOwners(i);
    auto rli = picparts.rankLocalIndex(i);
    auto rli_r = read_picparts.rankLocalIndex(i);
    auto cai = picparts.commArrayIndex(i);
    auto cai_r = read_picparts.commArrayIndex(i);

    auto checkValues = OMEGA_H_LAMBDA(const Omega_h::LO ent) {
      if (gids[ent] != gids_r[ent])
        failed[0] = 1;
      if (owners[ent] != owners_r[ent])
        failed[0] = 1;
      if (rli[ent] != rli_r[ent])
        failed[0] = 1;
      if (cai[ent] != cai_r[ent])
        failed[0] = 1;
    };
    Omega_h::parallel_for(picparts.nents(i), checkValues);

    Omega_h::HostWrite<Omega_h::LO> failed_h(failed);
    assert(!failed_h[0]);

    auto nentsOffsets = picparts.nentsOffsets(i);
    auto nentsOffsets_r = read_picparts.nentsOffsets(i);
    auto checkOffsets = OMEGA_H_LAMBDA(const Omega_h::LO r) {
      if (nentsOffsets[r] != nentsOffsets_r[r])
        failed[0] = 1;
    };
    Omega_h::parallel_for(picparts.comm()->size(), checkOffsets);

    failed_h = Omega_h::HostWrite<Omega_h::LO>(failed);
    assert(!failed_h[0]);
  }

  //Check safe tag
  auto safe = picparts.safeTag();
  auto safe_r = read_picparts.safeTag();
  Omega_h::Write<Omega_h::LO> failed(1,0);
  auto checkSafe = OMEGA_H_LAMBDA(const Omega_h::LO ent) {
    if (safe[ent] != safe_r[ent])
      failed[0] = 1;
  };
  Omega_h::parallel_for(picparts->nelems(), checkSafe);

  Omega_h::HostWrite<Omega_h::LO> failed_h(failed);
  assert(!failed_h[0]);

  if (!rank) {
    printf("All Tests Passed\n");
  }
  return 0;
}
