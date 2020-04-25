#pragma once
#include <particle_structs.hpp>

namespace ps = particle_structs;
typedef double Vector3[3];
/* Type:
     int - particle ID
     double[3] - array example values
     short - bool value for checking
     int - int value for checking
 */
typedef ps::MemberTypes<int, Vector3, short, int> Types;
typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef typename ExeSpace::memory_space MemSpace;
typedef typename ExeSpace::device_type Device;
typedef ps::ParticleStructure<Types, MemSpace> PS;
typedef PS::kkLidView kkLidView;
typedef PS::kkGidView kkGidView;
typedef PS::kkLidHostMirror kkLidHost;
typedef PS::kkGidHostMirror kkGidHost;

template <class T>
using KView=Kokkos::View<T*, MemSpace::device_type>;
template <class T>
using KViewHost=typename KView<T>::HostMirror;
using ps::lid_t;
