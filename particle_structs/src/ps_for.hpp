#pragma once

#include "particle_structure.hpp"
#include <SellCSigma.h>
#include <csr/CSR.hpp>
namespace pumipic {
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

  template <typename MSpace, typename DataTypes, typename MemSpace>
  ParticleStructure<DataTypes, MSpace>* copy(ParticleStructure<DataTypes, MemSpace>* old) {
    SellCSigma<DataTypes, MemSpace>* scs = dynamic_cast<SellCSigma<DataTypes, MemSpace>*>(old);
    if (scs) {
      return scs->template copy<MSpace>();
    }
    CSR<DataTypes, MemSpace>* csr = dynamic_cast<CSR<DataTypes, MemSpace>*>(old);
    if (csr) {
      //return csr->template copy<MSpace>();
    }
    fprintf(stderr, "[ERROR] Structure does not support copy\n");
    throw 1;
    return NULL;
  }
}
