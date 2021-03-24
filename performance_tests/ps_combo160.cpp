#include <particle_structs.hpp>
#include <ppTiming.hpp>
#include <Kokkos_Random.hpp>
#include "perfTypes.hpp"
#include "../particle_structs/test/Distribute.h"
#include <string>

PS160* createSCS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int C, int sigma, int V, std::string name);
PS160* createCSR(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int team_size);
PS160* createCabM(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int team_size);

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);
  // Default values if not specified on command line
  int test_num = 2;
  double percentMoved = 10;
  int team_size = 32;
  int vert_slice = 1024;

  /* Check commandline arguments */
  // Required arguments
  int num_elems = atoi(argv[1]);
  int num_ptcls = atoi(argv[2]);
  int strat = atoi(argv[3]);
  bool optimal = false;

  // Optional arguments specified with flags
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
    else {
      fprintf(stderr, "Illegal argument: %s", argv[i]);
      // insert usage statement
    }
  }

  int comm_rank; // get process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  int comm_size; // get number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if (!comm_rank) {
    fprintf(stderr, "Test Command:\n");
    for (int i = 0; i < argc; i++) {
      fprintf(stderr, " %s", argv[i]);
    }
    fprintf(stderr, "\n");
  }

  /* Enable timing on every process */
  pumipic::SetTimingVerbosity(0);
  /// @todo Add prebarrier to all classes (Only on SCS at the moment)
  pumipic::enable_prebarrier();

  { // Begin Kokkos region

    /* Create initial distribution of particles */
    kkLidView ppe("ptcls_per_elem", num_elems);
    kkLidView ptcl_elems("ptcl_elems", num_ptcls);
    kkGidView element_gids("element_gids", num_elems);
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const int i) { // set gids, sharing between all processes
        element_gids(i) = i;
    });
    if (!comm_rank)
      printf("Generating particle distribution with strategy: %s\n", distribute_name(strat));
    distribute_particles(num_elems, num_ptcls, strat, ppe, ptcl_elems);

    /* Create particle structure */
    ParticleStructures160 structures;
    if (test_num == 0) {
      if (optimal) {
        if (strat == 1) {
          team_size = 512;
          vert_slice = 8;
        }
        else if (strat == 2) {
          team_size = 512;
          vert_slice = 4;
        }
        else if (strat == 3) {
          team_size = 128;
          vert_slice = 8;
        }
      }
      structures.push_back(std::make_pair("Sell-"+std::to_string(team_size)+"-ne",
                                      createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                team_size, num_elems, vert_slice, "Sell-"+std::to_string(team_size)+"-ne")));
    }
    if (test_num == 1) {
      structures.push_back(std::make_pair("CSR",
                                      createCSR(num_elems, num_ptcls, ppe, element_gids, team_size)));
    }
    if (test_num == 2) {
      structures.push_back(std::make_pair("CabM",
                                      createCabM(num_elems, num_ptcls, ppe, element_gids, team_size)));
    }

    const int ITERS = 100;
    if (!comm_rank)
      printf("Performing %d iterations of rebuild on each structure\n", ITERS);
    /* Perform push & rebuild on the particle structures */
    for (int i = 0; i < structures.size(); ++i) {
      std::string name = structures[i].first;
      PS160* ptcls = structures[i].second;

      int seed = 0; // set seed for uniformly random processes
      Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> pool(seed);

      if (!comm_rank)
        printf("Beginning push on structure %s\n", name.c_str());

      // Per element data to access in pseudoPush
      Kokkos::View<double*> parentElmData("parentElmData", ptcls->nElems());
      Kokkos::parallel_for("parent_elem_data", parentElmData.size(),
          KOKKOS_LAMBDA(const int& e){
        parentElmData(e) = sqrt(e) * e;
      });

      for (int i = 0; i < ITERS; ++i) {
        /* Begin Push Setup */
        auto dbls = ptcls->get<0>();
        auto nums = ptcls->get<1>();
        auto lint = ptcls->get<2>();

        auto pseudoPush = PS_LAMBDA(const int& e, const int& p, const bool& mask) {
          if (mask) {
            for (int i = 0; i < 17; i++) {
              dbls(p,i) = 10.3;
              dbls(p,i) = dbls(p,i) * dbls(p,i) * dbls(p,i) / sqrt(p) / sqrt(e) + parentElmData(e);
            }
            for (int i = 0; i < 4; i++) {
              nums(p,i) = 4*p + i;
            }
            lint(p) = p;
          }
          else {
            for (int i = 0; i < 17; i++) {
              dbls(p,i) = 0;
            }
            for (int i = 0; i < 4; i++) {
              nums(p,i) = -1;
            }
            lint(p) = 0;
          }
        };

        Kokkos::fence();
        Kokkos::Timer pseudo_push_timer;
        /* Begin push operations */
        ps::parallel_for(ptcls,pseudoPush,"pseudo push");
        Kokkos::fence();
        /* End push */
        float pseudo_push_time = pseudo_push_timer.seconds();
        pumipic::RecordTime(name+" pseudo-push", pseudo_push_time);
      }

      if (!comm_rank)
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

    } // end loop over structures


    for (size_t i = 0; i < structures.size(); ++i)
      delete structures[i].second;
    structures.clear();

  } // end Kokkos region

  pumipic::SummarizeTime();
  Kokkos::finalize();
  return 0;
}

PS160* createSCS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int C, int sigma, int V, std::string name) {
  Kokkos::TeamPolicy<ExeSpace> policy(32, C);
  pumipic::SCS_Input<PerfTypes160> input(policy, sigma, V, num_elems, num_ptcls, ppe, elm_gids);
  input.name = name;
  return new pumipic::SellCSigma<PerfTypes160, MemSpace>(input);
}
PS160* createCSR(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int team_size) {
  Kokkos::TeamPolicy<ExeSpace> policy(32, team_size);
  return new pumipic::CSR<PerfTypes160, MemSpace>(policy, num_elems, num_ptcls, ppe, elm_gids);
}
PS160* createCabM(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int team_size) {
  Kokkos::TeamPolicy<ExeSpace> policy(32, team_size);
  return new pumipic::CabM<PerfTypes160, MemSpace>(policy, num_elems, num_ptcls, ppe, elm_gids);
}