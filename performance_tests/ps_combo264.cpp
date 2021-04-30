#include <particle_structs.hpp>
#include <ppTiming.hpp>
#include <Kokkos_Random.hpp>
#include "perfTypes.hpp"
#include "../particle_structs/test/Distribute.h"
#include <string>

PS264* createSCS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int C, int sigma, int V, std::string name);
PS264* createCSR(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int team_size);
#ifdef PP_ENABLE_CAB
PS264* createCabM(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int team_size, std::string name);
PS264* createDPS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int team_size, std::string name);
#endif

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);
  // Default values if not specified on command line
  double percentMoved = 0.5;
  double percentMovedProcess = 0.1;
  int team_size = 32;
  int vert_slice = 1024;

  /* Check commandline arguments */
  // Required arguments
  assert(argc >= 5);
  int num_elems = atoi(argv[1]);
  int num_ptcls = atoi(argv[2]);
  int strat = atoi(argv[3]);
  int structure = atoi(argv[4]);
  bool optimal = false;

  // Optional arguments specified with flags
  for (int i = 5; i < argc; i+=2) {
    // -p = percent_moved
    if (std::string(argv[i]) == "-p") {
      percentMoved = atof(argv[i+1]);
    }
    // -pp = percent_moved_to_new_process
    else if (std::string(argv[i]) == "-pp") {
      percentMovedProcess = atof(argv[i+1]);
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
      fprintf(stderr, "Usage argument: ./ps_combo264 num_elms num_ptcls distribution structure_type\n[-p percentMovedRebuild] [-pp percentMovedMigrate] [-s team_size] [-v vertical_slicing] [--optimal]");
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

    std::string name;
    PS264* ptcls;

    /* Create particle structure */
    if (structure == 0) {
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
      name = ("Sell-"+std::to_string(team_size)+"-ne");
      ptcls = createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                team_size, num_elems, vert_slice, name);
    }
    else if (structure == 1) {
      name = "CSR";
      ptcls = createCSR(num_elems, num_ptcls, ppe, element_gids, team_size);
    }
    else if (structure == 2) {
      name = "CabM";
#ifdef PP_ENABLE_CAB
      ptcls = createCabM(num_elems, num_ptcls, ppe, element_gids, team_size, name);
#else
      fprintf(stderr, "CabM requested, but PUMI-PIC was not built with Cabana enabled\n");
#endif
    }
    else if (structure == 3) {
      name = "DPS";
#ifdef PP_ENABLE_CAB
      ptcls = createDPS(num_elems, num_ptcls, ppe, element_gids, team_size, name);
#else
      fprintf(stderr, "DPS requested, but PUMI-PIC was not built with Cabana enabled\n");
#endif
    }

    const int PS_ITERS = 100;
    const int ITERS = 100;
    
    if (!comm_rank)
      printf("Performing %d iterations of push on each structure\n", PS_ITERS);

    /* Perform push & rebuild on the particle structures */
    if (!comm_rank)
      printf("Beginning push on structure %s\n", name.c_str());

    // Per element data to access in pseudoPush
    Kokkos::View<double*> parentElmData("parentElmData", ptcls->nElems());
    Kokkos::parallel_for("parent_elem_data", parentElmData.size(),
        KOKKOS_LAMBDA(const int& e){
      parentElmData(e) = std::sqrt((double)e) * e;
    });

    for (int i = 0; i < PS_ITERS; ++i) {
      /* Begin Push Setup */
      auto dbls = ptcls->get<0>();
      auto nums = ptcls->get<1>();
      auto lint = ptcls->get<2>();

      auto pseudoPush = PS_LAMBDA(const int& e, const int& p, const bool& mask) {
        if (mask) {
          for (int i = 0; i < 17; i++) {
            dbls(p,i) = 10.3;
            dbls(p,i) = dbls(p,i) * dbls(p,i) * dbls(p,i) / std::sqrt((double)p) / std::sqrt((double)e) + parentElmData(e);
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

    int seed = 0; // set seed for uniformly random processes
    Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> pool(seed);

    int* other_ranks = new int[comm_size-1];
    int counter = 0;
    for (int i = 0; i < comm_size; i++) {
      if (i != comm_rank)
        other_ranks[counter++] = i;
    }
    kkLidView other_ranks_d("other_ranks_d", comm_size-1);
    pumipic::hostToDevice(other_ranks_d, other_ranks);

    if (!comm_rank)
      printf("Performing %d iterations of migrate/rebuild on each structure\n", ITERS);

    if (!comm_rank)
      printf("Beginning migrate on structure %s\n", name.c_str());
    for (int i = 0; i < ITERS; ++i) {
      kkLidView new_elms("new elems", ptcls->capacity());
      Kokkos::Timer t;
      redistribute_particles(ptcls, strat, percentMoved, new_elms);
      pumipic::RecordTime("redistribute", t.seconds());

      Kokkos::Timer tp;
      kkLidView new_process("new_process", ptcls->capacity());
      if (comm_size > 1) {
        auto to_new_processes = PS_LAMBDA(const int& e, const int& p, const bool& mask) {
          if (mask) {
            auto generator = pool.get_state();
            double prob = generator.drand(1.0);
            if (prob < percentMovedProcess)
              new_process(p) = other_ranks_d( generator.urand(comm_size-1) ); // send particle to a different process
            else
              new_process(p) = comm_rank;
            pool.free_state(generator);
          }
        };
        pumipic::parallel_for(ptcls, to_new_processes, "to_new_processes");
      }
      pumipic::RecordTime("redistribute processes", tp.seconds());

      ptcls->migrate(new_elms, new_process);
    }

    delete ptcls;

  } // end Kokkos region

  pumipic::SummarizeTime();
  Kokkos::finalize();
  return 0;
}

PS264* createSCS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int C, int sigma, int V, std::string name) {
  Kokkos::TeamPolicy<ExeSpace> policy(32, C);
  pumipic::SCS_Input<PerfTypes264> input(policy, sigma, V, num_elems, num_ptcls, ppe, elm_gids);
  input.name = name;
  return new pumipic::SellCSigma<PerfTypes264, MemSpace>(input);
}
PS264* createCSR(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int team_size) {
  Kokkos::TeamPolicy<ExeSpace> policy(32, team_size);
  return new pumipic::CSR<PerfTypes264, MemSpace>(policy, num_elems, num_ptcls, ppe, elm_gids);
}
#ifdef PP_ENABLE_CAB
PS264* createCabM(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int team_size, std::string name) {
  Kokkos::TeamPolicy<ExeSpace> policy(32, team_size);
  pumipic::CabM_Input<PerfTypes264> input(policy, num_elems, num_ptcls, ppe, elm_gids);
  input.name = name;
  return new pumipic::CabM<PerfTypes264, MemSpace>(input);
}
PS264* createDPS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int team_size, std::string name) {
  Kokkos::TeamPolicy<ExeSpace> policy(32, team_size);
  pumipic::DPS_Input<PerfTypes264> input(policy, num_elems, num_ptcls, ppe, elm_gids);
  input.name = name;
  return new pumipic::DPS<PerfTypes264, MemSpace>(input);
}
#endif
