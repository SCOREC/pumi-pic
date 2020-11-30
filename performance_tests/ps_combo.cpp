#include <particle_structs.hpp>
#include <ppTiming.hpp>
#include "perfTypes.hpp"
#include "../particle_structs/test/Distribute.h"
#include <string>

PS* createSCS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int C, int sigma, int V, std::string name);
PS* createCSR(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids);

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
  double param = .01;

  //Optional arguments specified with flags
  for(int i = 4; i < argc; i+=2){
    // -p = percent_moved
    if(std::string(argv[i]) == "-p"){
      percentMoved = atoi(argv[i+1]);
    }
    // -n = test_num
    else if(std::string(argv[i]) == "-n"){
      test_num = atoi(argv[i+1]);
    }
    // -s = team_size (/chunk width)
    else if(std::string(argv[i]) == "-s"){
      team_size = atoi(argv[i+1]);
    }
    // -v = vertical slicing
    else if(std::string(argv[i]) == "-v"){
      vert_slice = atoi(argv[i+1]);
    }
    else if(std::string(argv[i]) == "--optimal"){
      optimal = true;
      i--;
    }
    else if(std::string(argv[i]) == "-r"){
      param = atof(argv[i+1]);
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

  { //Begin Kokkos region

    /* Create initial distribution of particles */
    kkLidView ppe("ptcls_per_elem", num_elems);
    kkLidView ptcl_elems("ptcl_elems", num_ptcls);
    kkGidView element_gids("",0);
    printf("Generating particle distribution with strategy: %s\n", distribute_name(strat));
    distribute_particles(num_elems, num_ptcls, strat, ppe, ptcl_elems,param);
    printView(ppe);

    /* Create particle structure */
    ParticleStructures structures;
    if(test_num % 2 == 0){ //0 or 2
      //note distribution detection is on so team_size has no impace and labels the output possibly wrong
      if(strat == 3)
        team_size = 16;
      structures.push_back(std::make_pair("Sell-"+std::to_string(team_size)+"-ne",
                                      createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                team_size, num_elems, vert_slice, "Sell-"+std::to_string(team_size)+"-ne")));
    }
    if(test_num > 0){ //1 or 2
      structures.push_back(std::make_pair("CSR",
                                      createCSR(num_elems, num_ptcls, ppe, element_gids)));
      structures.back().second->setTeamSize(team_size);
      if(optimal) //CSR optimal push may be 256 need to look a little closer still
        structures.back().second->setTeamSize(512);
    }

    const int ITERS = 100;
    fprintf(stderr,"Performing %d iterations of combo on each structure\n", ITERS);
    /* Perform push & rebuild on the particle structures */
    for (int i = 0; i < structures.size(); ++i) {
      std::string name = structures[i].first;
      PS* ptcls = structures[i].second;
      fprintf(stderr,"Beginning push on structure %s\n", name.c_str());

      //Per element data to access in pseudoPush
      Kokkos::View<double*> parentElmData("parentElmData", ptcls->nElems());
      Kokkos::parallel_for("parent_elem_data", parentElmData.size(), 
          KOKKOS_LAMBDA(const int& e){
        parentElmData(e) = sqrt(e) * e;
      });

        /* Begin Push Setup */

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

        Kokkos::Timer pseudo_push_timer;
        /* Begin push operations */
        ps::parallel_for(ptcls,pseudoPush,"pseudo push"); 
        /* End push */
        float pseudo_push_time = pseudo_push_timer.seconds();
        pumipic::RecordTime(name+" pseudo-push", pseudo_push_time);
        //fprintf(stderr, "Iter: %d\n", i);
      }
      
//      printf("Beginning rebuild on structure %s\n", name.c_str());
//      for (int i = 0; i < ITERS; ++i) {
//        kkLidView new_elms("new elems", ptcls->capacity());
//        Kokkos::Timer t;
//        redistribute_particles(ptcls, strat, percentMoved, new_elms);
//        pumipic::RecordTime("redistribute", t.seconds());
//        ptcls->rebuild(new_elms);
//      }

    } //end loop over structures


    for (size_t i = 0; i < structures.size(); ++i)
      delete structures[i].second;
    structures.clear();

  } //end Kokkos region

  cleanup_distribution_memory();
  pumipic::SummarizeTime();
  Kokkos::finalize();
  return 0;
}

PS* createSCS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int C, int sigma, int V, std::string name) {
  Kokkos::TeamPolicy<ExeSpace> policy(4, C);
  pumipic::SCS_Input<PerfTypes> input(policy, sigma, V, num_elems, num_ptcls, ppe, elm_gids);
  input.name = name;
  return new pumipic::SellCSigma<PerfTypes, MemSpace>(input);
}
PS* createCSR(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids) {
  Kokkos::TeamPolicy<ExeSpace> po(32,Kokkos::AUTO);
  return new pumipic::CSR<PerfTypes, MemSpace>(po, num_elems, num_ptcls, ppe, elm_gids);
}
