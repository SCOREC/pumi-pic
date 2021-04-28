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

typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef typename ExeSpace::memory_space MemSpace;
typedef typename ExeSpace::device_type Device;
typedef pumipic::ParticleStructure<PerfTypes, MemSpace> PS;
typedef pumipic::ParticleStructure<PerfTypes160, MemSpace> PS160;
typedef pumipic::ParticleStructure<PerfTypes264, MemSpace> PS264;
typedef PS::kkLidView kkLidView;
typedef PS::kkGidView kkGidView;


typedef std::vector<std::pair<std::string, PS*> > ParticleStructures;
typedef std::vector<std::pair<std::string, PS160*> > ParticleStructures160;
typedef std::vector<std::pair<std::string, PS264*> > ParticleStructures264;