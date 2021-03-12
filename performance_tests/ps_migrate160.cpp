#include <particle_structs.hpp>
#include <ppTiming.hpp>
#include <Kokkos_Random.hpp>
#include "perfTypes.hpp"
#include "../particle_structs/test/Distribute.h"
#include <string>

PS160* createSCS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int C, int sigma, int V, std::string name);
PS160* createCSR(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids);
PS160* createCabM(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids);

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);
  // Default values if not specified on command line
  int test_num = 2;
  double percentMoved = 10;
  int team_size = 32;
  int vert_slice = 1024;

  /* Check commandline arguments */
  //Required arguments
  int num_elems = atoi(argv[1]);
  int num_ptcls = atoi(argv[2]);
  int strat = atoi(argv[3]);
  bool optimal = false;

  //Optional arguments specified with flags
  for (int i = 4; i < argc; i+=2) {
    // -p = percent_moved
    if (std::string(argv[i]) == "-p") {
      percentMoved = atoi(argv[i+1]);
    }
    // -n = test_num
    else if (std::string(argv[i]) == "-n") {
      test_num = atoi(argv[i+1]);
    }
    // -s = team_size (/chunk width)
    else if (std::string(argv[i]) == "-s") {
      team_size = atoi(argv[i+1]);
    }
    // -v = vertical slicing
    else if (std::string(argv[i]) == "-v") {
      vert_slice = atoi(argv[i+1]);
    }
    else if (std::string(argv[i]) == "--optimal") {
      optimal = true;
      i--;
    }
    else{
      fprintf(stderr, "Illegal argument: %s", argv[i]);
      // insert usage statement
    }
  }

  fprintf(stderr, "Test Command:\n");
  for(int i = 0; i < argc; i++){
    fprintf(stderr, " %s", argv[i]);
  }
  fprintf(stderr, "\n");

  /* Enable timing on every process */
  pumipic::SetTimingVerbosity(0);
  /// @todo Add prebarrier to all classes (Only on SCS at the moment)
  pumipic::enable_prebarrier();

  { //Begin Kokkos region
    
    /* Create initial distribution of particles */
    kkLidView ppe("ptcls_per_elem", num_elems);
    kkLidView ptcl_elems("ptcl_elems", num_ptcls);
    kkGidView element_gids("element_gids", num_elems);
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const int i) { // set gids, sharing between all processes
        element_gids(i) = i;
    });
    printf("Generating particle distribution with strategy: %s\n", distribute_name(strat));
    distribute_particles(num_elems, num_ptcls, strat, ppe, ptcl_elems);

    int comm_size; // get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    /* Create particle structure */
    ParticleStructures160 structures;
    if (test_num == 0) {
      structures.push_back(std::make_pair("Sell-"+std::to_string(team_size)+"-ne",
                                      createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                team_size, num_elems, vert_slice, "Sell-"+std::to_string(team_size)+"-ne")));
    }
    else if (test_num == 1) {
      structures.push_back(std::make_pair("CSR",
                                      createCSR(num_elems, num_ptcls, ppe, element_gids)));
    }
    else if (test_num == 2) {
      structures.push_back(std::make_pair("CabM",
                                      createCabM(num_elems, num_ptcls, ppe, element_gids)));
    }

    const int ITERS = 100;
    printf("Performing %d iterations of rebuild on each structure\n", ITERS);
    /* Perform push & rebuild on the particle structures */
    for (int i = 0; i < structures.size(); ++i) {
      std::string name = structures[i].first;
      PS160* ptcls = structures[i].second;

      int seed = 0; // set seed for uniformly random processes
      Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> pool(seed);

      printf("Beginning migrate on structure %s\n", name.c_str());
      for (int i = 0; i < ITERS; ++i) {
        kkLidView new_elms("new elems", ptcls->capacity());
        Kokkos::Timer t;
        redistribute_particles(ptcls, strat, percentMoved, new_elms);
        pumipic::RecordTime("redistribute", t.seconds());

        kkLidView new_process("new_process", ptcls->capacity());
        Kokkos::fill_random(new_process, pool, comm_size);

        ptcls->migrate(new_elms, new_process);
      }

    } //end loop over structures

    for (size_t i = 0; i < structures.size(); ++i)
      delete structures[i].second;
    structures.clear();

  } //end Kokkos region

  pumipic::SummarizeTime();
  Kokkos::finalize();
  return 0;
}

PS160* createSCS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int C, int sigma, int V, std::string name) {
  Kokkos::TeamPolicy<ExeSpace> policy(4, C);
  pumipic::SCS_Input<PerfTypes160> input(policy, sigma, V, num_elems, num_ptcls, ppe, elm_gids);
  input.name = name;
  return new pumipic::SellCSigma<PerfTypes160, MemSpace>(input);
}
PS160* createCSR(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids) {
  Kokkos::TeamPolicy<ExeSpace> po(32,Kokkos::AUTO);
  return new pumipic::CSR<PerfTypes160, MemSpace>(po, num_elems, num_ptcls, ppe, elm_gids);
}
PS160* createCabM(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids) {
  Kokkos::TeamPolicy<ExeSpace> po(32,Kokkos::AUTO);
  return new pumipic::CabM<PerfTypes160, MemSpace>(po, num_elems, num_ptcls, ppe, elm_gids);
}