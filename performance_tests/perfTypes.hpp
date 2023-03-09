#pragma once
#include <particle_structs.hpp>
#include <vector>

typedef double Vector3d[3];
typedef pumipic::MemberTypes<int, Vector3d, double> PerfTypes;
typedef double Vector17d[17];
typedef int Vector4i[4];
typedef pumipic::MemberTypes<Vector17d, Vector4i, long int> PerfTypes160;
typedef double Vector30d[30];
typedef pumipic::MemberTypes<Vector30d, Vector4i, long int> PerfTypes264;
typedef double Vector60d[60];
typedef pumipic::MemberTypes<Vector60d, Vector4i, long int> PerfTypes504;

typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef typename ExeSpace::memory_space MemSpace;
typedef typename ExeSpace::device_type Device;
typedef pumipic::ParticleStructure<PerfTypes, MemSpace> PS;
typedef pumipic::ParticleStructure<PerfTypes160, MemSpace> PS160;
typedef pumipic::ParticleStructure<PerfTypes264, MemSpace> PS264;
typedef pumipic::ParticleStructure<PerfTypes504, MemSpace> PS504;
typedef PS::kkLidView kkLidView;
typedef PS::kkGidView kkGidView;


typedef std::vector<std::pair<std::string, PS*> > ParticleStructures;
typedef std::vector<std::pair<std::string, PS160*> > ParticleStructures160;
typedef std::vector<std::pair<std::string, PS264*> > ParticleStructures264;
typedef std::vector<std::pair<std::string, PS504*> > ParticleStructures504;


template<typename DataTypes>
int getTypeSize() {
  if(std::is_same<DataTypes, PerfTypes160>::value) return 160;
  else if(std::is_same<DataTypes, PerfTypes264>::value) return 264;
  else if(std::is_same<DataTypes, PerfTypes504>::value) return 504;
  else {
    fprintf(stderr,"Error: unknown per particle data type\n");
    exit(EXIT_FAILURE);
  }
}

template<typename DataTypes>
auto createSCS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int C, int sigma, int V, std::string name) {
  Kokkos::TeamPolicy<ExeSpace> policy(32, C);
  pumipic::SCS_Input<DataTypes> input(policy, sigma, V, num_elems, num_ptcls, ppe, elm_gids);
  input.name = name;
  return new pumipic::SellCSigma<DataTypes, MemSpace>(input);
}

template<typename DataTypes>
auto createCSR(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int team_size) {
  Kokkos::TeamPolicy<ExeSpace> policy(32, team_size);
  return new pumipic::CSR<DataTypes, MemSpace>(policy, num_elems, num_ptcls, ppe, elm_gids);
}

#ifdef PP_ENABLE_CAB
template<typename DataTypes>
auto createCabM(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int team_size, std::string name) {
  Kokkos::TeamPolicy<ExeSpace> policy(32, team_size);
  pumipic::CabM_Input<DataTypes> input(policy, num_elems, num_ptcls, ppe, elm_gids);
  input.name = name;
  return new pumipic::CabM<DataTypes, MemSpace>(input);
}

template<typename DataTypes>
auto createDPS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int team_size, std::string name) {
  Kokkos::TeamPolicy<ExeSpace> policy(32, team_size);
  pumipic::DPS_Input<DataTypes> input(policy, num_elems, num_ptcls, ppe, elm_gids);
  input.name = name;
  return new pumipic::DPS<DataTypes, MemSpace>(input);
}
#endif


