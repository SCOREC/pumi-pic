#pragma once
#include <particle_structs.hpp>
#include <vector>

typedef double Vector3d[3];
typedef pumipic::MemberTypes<int, Vector3d, double> PerfTypes;
typedef double Vector4i[4];
typedef double Vector17d[17];
typedef pumipic::MemberTypes<Vector4i, Vector17d, double> PerfTypes160;

typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef typename ExeSpace::memory_space MemSpace;
typedef typename ExeSpace::device_type Device;
typedef pumipic::ParticleStructure<PerfTypes, MemSpace> PS;
typedef pumipic::ParticleStructure<PerfTypes160, MemSpace> PS160;
typedef PS::kkLidView kkLidView;
typedef PS::kkGidView kkGidView;


typedef std::vector<std::pair<std::string, PS*> > ParticleStructures;
typedef std::vector<std::pair<std::string, PS160*> > ParticleStructures160;