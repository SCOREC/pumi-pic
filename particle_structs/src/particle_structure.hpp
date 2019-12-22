#pragma once

namespace particle_structs {

  template <class DataTypes, typename MemSpace = Kokkos::DefaultMemorySpace>
  class ParticleStructure {
  public:
    typedef MemSpace::device_type Device;
    typedef Kokkos::View<lid_t*, Device> kkLidView;
    typedef Kokkos::View<gid_t*, Device> kkGidView;
    typedef typename kkLidView::HostMirror kkLidHostMirror;
    typedef typename kkGidView::HostMirror kkGidHostMirror;
    typedef Kokkos::UnorderedMap<gid_t, lid_t, Device> GID_Mapping;
    template <std::size_t N>
    using DataType = typename MemberTypeAtIndex<DataTypes, N>::type;

    ParticleStructure() : num_elems(0), num_ptcls(0), capacity_(0);
    lid_t nElems() const {return num_elems;}
    lid_t nPtcls() const {return num_ptcls;}
    lid_t capacity() const {return capacity_;}

    /* Provides access to the particle info for Nth time of each particle

       The segment is indexed by particle index first followed by indices for each
         dimension of the type.
     */
    template <std::size_t N>
    Segment<DataType<N>, MemSpace> get() {
      if (num_ptcls == 0)
        return Segment<DataType<N>, MemSpace>();
      MemberTypeView<Type>* view = static_cast<MemberTypeView<DataType<N>*>(scs_data[N]);
      return Segment<Type, MemSpace>(*view);
    }

    virtual void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                         MemberTypeViews<DataTypes> new_particle_info = NULL) = 0;
    virtual void migrate(kkLidView new_element, kkLidView new_process,
                         kkLidView new_particle_elements = kkLidView(),
                         MemberTypeViews<DataTypes> new_particle_info = NULL) = 0;

    template <typename FunctionType>
    virtual void parallel_for(FunctionType& fn, std::string name="") = 0;

    void printMetrics() const = 0;
  protected:
    lid_t num_elems;
    lid_t num_ptcls;
    lid_t capacity_;

    MemberTypeViews<DataTypes> scs_data;
  };

}
