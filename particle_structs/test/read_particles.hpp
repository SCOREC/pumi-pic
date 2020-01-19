#pragma once

#include <fstream>
#include <particle_structs.hpp>
#include "test_types.hpp"
namespace ps=particle_structs;


/* File format:
   <num_elems> <num_ptcls>
   //For each element
   <elem gid> <nppe>
   ...

   //For each particle
   <particle_elem> <particle_info...>
   ...
*/
void readParticles(char* particle_file, int& num_elems, int& num_ptcls,
                   kkLidView& ppe, kkGidView& eGids, kkLidView& pElems,
                   ps::MemberTypeViews<Types, Device>& pInfo) {
  std::ifstream in_str(particle_file);
  if (!in_str) {
    fprintf(stderr, "[ERROR] Cannot open file %s\n", particle_file);
    throw 1;
  }
  //Read counts
  in_str >> num_elems >> num_ptcls;
  ppe = kkLidView("particles_per_element", num_elems);
  eGids = kkGidView("elemnt_gids", num_elems);
  pElems = kkLidView("element_of_particle", num_ptcls);
  pInfo = ps::createMemberViews<Types>(num_ptcls);

  //Read Elems
  lid_t* ppe_h = new lid_t[num_elems];
  ps::gid_t* eGids_h = new ps::gid_t[num_elems];
  for (int i = 0; i < num_elems; ++i) {
    in_str >> eGids_h[i] >> ppe_h[i];
  }

  //Read Particles
  lid_t* pElems_h = new lid_t[num_ptcls];
  int* pIds = new int[num_ptcls];
  Vector3* val1s = new Vector3[num_ptcls];
  bool* val2s = new bool[num_ptcls];
  int* val3s = new int[num_ptcls];
  for (int i = 0; i < num_ptcls; ++i) {
    in_str >> pElems_h[i] >> pIds[i];
    for (int j = 0; j < 3; ++j) {
      in_str >>val1s[i][j];
    }
    in_str >> val2s[i] >> val3s[i];
  }

  //Transfer data to the device
  ps::hostToDevice<lid_t>(ppe, ppe_h);
  ps::hostToDevice<ps::gid_t>(eGids, eGids_h);
  ps::hostToDevice<lid_t>(pElems, pElems_h);
  ps::hostToDevice<int>(ps::getMemberView<Types, 0>(pInfo), pIds);
  ps::hostToDevice<Vector3>(ps::getMemberView<Types, 1>(pInfo), val1s);
  ps::hostToDevice<bool>(ps::getMemberView<Types, 2>(pInfo), val2s);
  ps::hostToDevice<int>(ps::getMemberView<Types, 3>(pInfo), val3s);

  //Cleanup
  delete [] ppe_h;
  delete [] eGids_h;
  delete [] pElems_h;
  delete [] pIds;
  delete [] val1s;
  delete [] val2s;
  delete [] val3s;
}

void writeParticles(char* particle_file, int num_elems, int num_ptcls,
                    kkLidView ppe, kkGidView eGids, kkLidView pElems,
                    ps::MemberTypeViews<Types, Device> pInfo) {
  std::ofstream out_str(particle_file);
  if (!out_str) {
    fprintf(stderr, "[ERROR] Cannot open file %s\n", particle_file);
  }

  //Transfer device to host
  kkLidHost ppe_h = ps::deviceToHost(ppe);
  kkGidHost eGids_h = ps::deviceToHost(eGids);
  kkLidHost pElems_h = ps::deviceToHost(pElems);
  KViewHost<int> pIds = ps::deviceToHost(ps::getMemberView<Types, 0>(pInfo));
  KViewHost<Vector3> vals1 = ps::deviceToHost(ps::getMemberView<Types, 1>(pInfo));
  KViewHost<bool> vals2 = ps::deviceToHost(ps::getMemberView<Types, 2>(pInfo));
  KViewHost<int> vals3 = ps::deviceToHost(ps::getMemberView<Types, 3>(pInfo));

  //Print counts
  out_str << num_elems << ' ' << num_ptcls << '\n';

  //Print elements
  for (int i = 0; i < num_elems; ++i) {
    out_str << eGids_h(i) << ' ' << ppe_h(i) << '\n';
  }
  out_str<<'\n';

  //Print particles
  for (int i = 0; i < num_ptcls; ++i) {
    out_str << pElems_h(i) << ' ' << pIds(i) << ' ';
    for (int j = 0; j < 3; ++j) {
      out_str << vals1(i,j) << ' ';
    }
    out_str << vals2(i) << ' ' << vals3(i) << '\n';
  }
}
