#include <particle_structs.hpp>
#include <ppTiming.hpp>
#include "perfTypes.hpp"
#include "../particle_structs/test/Distribute.h"

PS* createSCS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int C, int sigma, int V);
PS* createCSR(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids);
PS* createCabM(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids);

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

  /* Check commandline arguments */
  int test_num;
  if(argc == 5){
    test_num = atoi(argv[4]);
  }
  else if (argc != 4) {
    fprintf(stderr, "Usage: %s <num elems> <num ptcls> <distribution> <optional: test_num>\n",
            argv[0]);
  }

  fprintf(stderr, "Test Command:\n %s %s %s %s ", argv[0], argv[1], argv[2], argv[3]);
  if(argc == 5)
    fprintf(stderr, "%s\n", argv[4]);
  else
    fprintf(stderr, "\n");

  /* Enable timing on every process */
  pumipic::SetTimingVerbosity(0);

  {
    /* Create initial distribution of particles */
    int num_elems = atoi(argv[1]);
    int num_ptcls = atoi(argv[2]);
    int strat = atoi(argv[3]);
    kkLidView ppe("ptcls_per_elem", num_elems);
    kkGidView element_gids("",0);
    int* ppe_host = new int[num_elems];
    std::vector<int>* ids = new std::vector<int>[num_elems];
    distribute_particles(num_elems, num_ptcls, strat, ppe_host, ids);
    pumipic::hostToDevice(ppe, ppe_host);
    delete [] ppe_host;
    delete [] ids;

    /* Create particle structure */
    ParticleStructures structures;
    if(argc == 5){
      switch(test_num){
        case 0:
          structures.push_back(std::make_pair("Sell-32-ne",
                                          createSCS(num_elems, num_ptcls, ppe, element_gids,
                                          32, num_elems, 1024)));
          break;
        case 1:
          structures.push_back(std::make_pair("Sell-16-ne",
                                          createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                    16, num_elems, 1024)));
          break;
        case 2:
          structures.push_back(std::make_pair("Sell-32-1024",
                                          createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                    32, 1024, 1024)));
          break;
        case 3:
          structures.push_back(std::make_pair("Sell-16-1024",
                                          createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                    16, 1024, 1024)));
          break;
        case 4:
          structures.push_back(std::make_pair("Sell-32-1",
                                          createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                    32, 1, 1024)));
          break;
        case 5:
          structures.push_back(std::make_pair("Sell-16-1",
                                          createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                    16, 1, 1024)));
          break;
        case 6:
          structures.push_back(std::make_pair("CSR",
                                          createCSR(num_elems, num_ptcls, ppe, element_gids)));
          structures.push_back(std::make_pair("CabM",
                                          createCabM(num_elems, num_ptcls, ppe, element_gids)));
          break;
      }
    }
    else{
      structures.push_back(std::make_pair("Sell-32-ne",
                                          createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                    32, num_elems, 1024)));
      structures.push_back(std::make_pair("Sell-16-ne",
                                          createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                    16, num_elems, 1024)));
      structures.push_back(std::make_pair("Sell-32-1024",
                                          createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                    32, 1024, 1024)));
      structures.push_back(std::make_pair("Sell-16-1024",
                                          createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                    16, 1024, 1024)));
      structures.push_back(std::make_pair("Sell-32-1",
                                          createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                    32, 1, 1024)));
      structures.push_back(std::make_pair("Sell-16-1",
                                          createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                    16, 1, 1024)));
      structures.push_back(std::make_pair("CSR",
                                          createCSR(num_elems, num_ptcls, ppe, element_gids)));
      structures.push_back(std::make_pair("CabM",
                                          createCabM(num_elems, num_ptcls, ppe, element_gids)));
    }

    const int ITERS = 100;
    printf("Performing %d iterations of pseudo-push on each structure\n", ITERS);
    /* Perform pseudo-push on particle structures */
    for (int i = 0; i < structures.size(); ++i) {
      std::string name = structures[i].first;
      PS* ptcls = structures[i].second;
      printf("Beginning pseudo-push on structure %s\n", name.c_str());

      /* Begin Push Setup */
      //Per element data to access in pseudoPush
      Kokkos::View<double*> parentElmData("parentElmData", ptcls->nElems());
      Kokkos::parallel_for("parent_elem_data", parentElmData.size(), 
          KOKKOS_LAMBDA(const int& e){
        parentElmData(e) = sqrt(e) * e;
      });

      auto nums = ptcls->get<0>();
      auto dbls = ptcls->get<1>();
      auto dbl = ptcls->get<2>();

      auto pseudoPush = PS_LAMBDA(const int& e, const int& p, const bool& mask) {
        if(mask){
          dbls(p,0) = 10.3;
          dbls(p,1) = 10.3;
          dbls(p,2) = 10.3;
          dbls(p,0) = dbls(p,0) * dbls(p,0) * dbls(p,0) / sqrt(p) / sqrt(e) + parentElmData(e);
          dbls(p,1) = dbls(p,1) * dbls(p,1) * dbls(p,1) / sqrt(p) / sqrt(e) + parentElmData(e);
          dbls(p,2) = dbls(p,2) * dbls(p,2) * dbls(p,2) / sqrt(p) / sqrt(e) + parentElmData(e);
          nums(p) = p;
          dbl(p)  = parentElmData(e);
        }
        else{
          dbls(p,0) = 0;
          dbls(p,1) = 0;
          dbls(p,2) = 0;
          nums(p) = -1;
          dbl(p)  = 0;
        }
      };
      
      for (int i = 0; i < ITERS; ++i) {
        Kokkos::fence();
        Kokkos::Timer pseudo_push_timer;
        /* Begin push operations */
        ps::parallel_for(ptcls,pseudoPush,"pseudo push"); 
        /* End push */
        Kokkos::fence();
        float pseudo_push_time = pseudo_push_timer.seconds();
        pumipic::RecordTime(name+" psuedo-push", pseudo_push_time);
      }
    }

    for (size_t i = 0; i < structures.size(); ++i)
      delete structures[i].second;
    structures.clear();
  }

  pumipic::SummarizeTime();
  Kokkos::finalize();
  return 0;
}

PS* createSCS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int C, int sigma, int V) {
  Kokkos::TeamPolicy<ExeSpace> policy(4, C);
  pumipic::SCS_Input<PerfTypes> input(policy, sigma, V, num_elems, num_ptcls, ppe, elm_gids);
  return new pumipic::SellCSigma<PerfTypes, MemSpace>(input);
}
PS* createCSR(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids) {
  Kokkos::TeamPolicy<ExeSpace> po(32,Kokkos::AUTO);
  return new pumipic::CSR<PerfTypes, MemSpace>(po, num_elems, num_ptcls, ppe, elm_gids);
}
PS* createCabM(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids) {
  Kokkos::TeamPolicy<ExeSpace> po(32,Kokkos::AUTO);
  return new pumipic::CabM<PerfTypes, MemSpace>(po, num_elems, num_ptcls, ppe, elm_gids);
}