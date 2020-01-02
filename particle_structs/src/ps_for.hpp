#pragma once

#include "particle_structure.hpp"
#include <SellCSigma.h>
#include <csr/CSR.hpp>
namespace particle_structs {
  template <typename FunctionType, typename DataTypes, typename MemSpace>
  void parallel_for(ParticleStructure<DataTypes, MemSpace>* ps, FunctionType& fn,
                    std::string s="") {
    SellCSigma<DataTypes, MemSpace>* scs = dynamic_cast<SellCSigma<DataTypes, MemSpace>*>(ps);
    if (scs) {
      scs->parallel_for(fn, s);
      return;
    }
    CSR<DataTypes, MemSpace>* csr = dynamic_cast<CSR<DataTypes, MemSpace>*>(ps);
    if (csr) {
      csr->parallel_for(fn, s);
      return;
    }
    fprintf(stderr, "[ERROR] Structure does not support parallel for used on kernel %s\n",
            s.c_str());
    throw 1;
  }
}
