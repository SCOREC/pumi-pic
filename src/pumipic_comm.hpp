#pragma once

#include <Omega_h_array.hpp>
#include <Omega_h_mesh.hpp>
namespace pumipic {
  void PIC_Comm_Initialize();
  void PIC_Comm_Finalize();
  
  int PIC_Comm_Self();
  int PIC_Comm_Size();
  int PIC_Comm_Neighbors();
  void setupPICComm(Omega_h::Mesh* picpart, int dim, Omega_h::Write<Omega_h::LO>& ents_per_rank,
                    Omega_h::Write<Omega_h::LO>& picpart_ents_per_rank,
                    Omega_h::Write<Omega_h::LO>& ent_global_numbering,
                    Omega_h::Write<Omega_h::LO>& ent_owners);

  template <class T>
  typename Omega_h::Write<T> createCommArray(int dim, int num_entries_per_entity,
                                             T default_value = 0);

  enum Op {
    SUM_OP,
    MAX_OP,
    MIN_OP
  };
  template <class T>
  void reduce(int dim, Op op, Omega_h::Write<T>& array);
}
