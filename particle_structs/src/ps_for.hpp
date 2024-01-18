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

  //This function returns a csr data type where the first view contains pids
  typedef MemberTypes<int> IntType;
  template <typename DataTypes, typename MemSpace>
  void ParticleStructure<DataTypes, MemSpace>::getCSRpid() {
    auto ptcl_info = ps::createMemberViews<IntType>(num_ptcls);
    auto pids = ps::getMemberView<IntType, 0>(ptcl_info);
    kkLidView ptcl_elems("ptcl_elems", num_ptcls);
    kkLidView ppe("ppe", num_elems);
    kkLidView eGid("gid", num_elems);
    auto setPIDs = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
      if (mask) {
        pids(p) = p;
        eGid(e) = p;
        ptcl_elems(p) = e;
        Kokkos::atomic_add(&ppe(e), 1);
      }
    };
    parallel_for(this, setPIDs, "setPIDs");
    auto policy = Kokkos::TeamPolicy<typename MemSpace::execution_space>(5, 5);
    auto result = new CSR<IntType, MemSpace>(policy, num_elems, num_ptcls, ppe, eGid, ptcl_elems, ptcl_info);
    delete result;
  }
}
