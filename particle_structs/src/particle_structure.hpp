#pragma once

#include "scs/SellCSigma.h"

namespace particle_structs {

  template <class DataTypes, typename MemSpace>
  class ParticleStructure {
  public:
    typedef typename MemSpace::device_type Device;
    typedef Kokkos::View<lid_t*, Device> kkLidView;
    typedef Kokkos::View<gid_t*, Device> kkGidView;
    typedef typename kkLidView::HostMirror kkLidHostMirror;
    typedef typename kkGidView::HostMirror kkGidHostMirror;
    typedef Kokkos::UnorderedMap<gid_t, lid_t, Device> GID_Mapping;
    template <std::size_t N> using DataType = typename MemberTypeAtIndex<N, DataTypes>::type;

    ~ParticleStructure();
    lid_t nElems() const;
    lid_t nPtcls() const;
    lid_t capacity() const;
    lid_t nRows() const;

    /* Provides access to the particle info for Nth time of each particle

       The segment is indexed by particle index first followed by indices for each
         dimension of the type.
     */
    template <std::size_t N>
    Segment<DataType<N>, MemSpace> get();

    void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                 MemberTypeViews<DataTypes> new_particle_info = NULL);
    void migrate(kkLidView new_element, kkLidView new_process,
                 kkLidView new_particle_elements = kkLidView(),
                 MemberTypeViews<DataTypes> new_particle_info = NULL);

    template <typename FunctionType>
    void parallel_for(FunctionType& fn, std::string name="");

    void printMetrics() const;
  private:
    ParticleStructure(SellCSigma<DataTypes, MemSpace::execution_space>* scs);
    SellCSigma<DataTypes, MemSpace::execution_space>* scs;
  };

  template <class DataTypes, typename MemSpace>
  ParticleStructure<DataTypes, MemSpace>::ParticleStructure(SellCSigma<DataTypes, MemSpace::execution_space>* scs_) {
    scs = scs_;
  }

  #include "ps_call_fns.hpp"
}
