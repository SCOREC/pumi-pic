#pragma once

#include <particle_structs.hpp>
namespace pumipic {
  template <typename FunctionType, typename DataTypes, typename MemSpace>
  void parallel_for(ParticleStructure<DataTypes, MemSpace>* ps, FunctionType& fn,
                    std::string s) {
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
    CabM<DataTypes, MemSpace>* cabm = dynamic_cast<CabM<DataTypes, MemSpace>*>(ps);
    if (cabm) {
      cabm->parallel_for(fn, s);
      return;
    }
    DPS<DataTypes, MemSpace>* dps = dynamic_cast<DPS<DataTypes, MemSpace>*>(ps);
    if (dps) {
      dps->parallel_for(fn, s);
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
      return csr->template copy<MSpace>();
    }
    CabM<DataTypes, MemSpace>* cabm = dynamic_cast<CabM<DataTypes, MemSpace>*>(old);
    if (cabm) {
      return cabm->template copy<MSpace>();
    }
    DPS<DataTypes, MemSpace>* dps = dynamic_cast<DPS<DataTypes, MemSpace>*>(old);
    if (dps) {
      return dps->template copy<MSpace>();
    }

    fprintf(stderr, "[ERROR] Structure does not support copy\n");
    throw 1;
    return NULL;
  }

  //This function initializes and populates the pids and offsets arrays
  template <typename DataTypes, typename MemSpace>
  template <typename ViewT>
  void ParticleStructure<DataTypes, MemSpace>::getPIDs(ViewT& pids, ViewT& offsets) {
    pids = ViewT("pids", capacity_);
    offsets = ViewT("offsets", num_elems+1);
    ViewT ppe("ppe", num_elems+1);
    auto setPIDs = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
      if (mask) {
        pids(p) = p;
        Kokkos::atomic_increment(&ppe(e));
      }
    };
    parallel_for(this, setPIDs, "setPIDs");
    exclusive_scan(ppe, offsets, execution_space());
  }
}
