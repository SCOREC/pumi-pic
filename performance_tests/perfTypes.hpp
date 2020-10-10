#pragma once
#include <particle_structs.hpp>
#include <vector>

typedef double Vector3d[3];
typedef pumipic::MemberTypes<int, Vector3d, double> PerfTypes;

typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef typename ExeSpace::memory_space MemSpace;
typedef typename ExeSpace::device_type Device;
typedef pumipic::ParticleStructure<PerfTypes, MemSpace> PS;
typedef PS::kkLidView kkLidView;
typedef PS::kkGidView kkGidView;


typedef std::vector<std::pair<std::string, PS*> > ParticleStructures;
