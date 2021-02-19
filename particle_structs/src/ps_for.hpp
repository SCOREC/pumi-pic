#pragma once

#include <particle_structs.hpp>
namespace pumipic {
  template <typename ParticleStructure, typename FunctionType>
  void parallel_for(ParticleStructure* ps, FunctionType& fn,
                    std::string s) {
    ps->parallel_for(fn, s);
  }

  // template <typename MSpace, >
  // ParticleStructure<DataTypes, MSpace>* copy(ParticleStructure<DataTypes, MemSpace>* old) {
  //   SellCSigma<DataTypes, MemSpace>* scs = dynamic_cast<SellCSigma<DataTypes, MemSpace>*>(old);
  //   if (scs) {
  //     return scs->template copy<MSpace>();
  //   }
  //   CSR<DataTypes, MemSpace>* csr = dynamic_cast<CSR<DataTypes, MemSpace>*>(old);
  //   if (csr) {
  //     //return csr->template copy<MSpace>();
  //   }
  //   fprintf(stderr, "[ERROR] Structure does not support copy\n");
  //   throw 1;
  //   return NULL;
  // }
}
