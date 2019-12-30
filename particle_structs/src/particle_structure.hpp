#pragma once

#include <PS_Types.h>

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

    ParticleStructure();
    virtual ~ParticleStructure() {}
    lid_t nElems() const {return num_elems;}
    lid_t nPtcls() const {return num_ptcls;}
    lid_t capacity() const {return capacity_;}
    lid_t numRows() const {return num_rows;}

    /* Provides access to the particle info for Nth time of each particle

       The segment is indexed by particle index first followed by indices for each
         dimension of the type.
     */
    template <std::size_t N>
    Segment<DataType<N>, typename MemSpace::execution_space> get() {
      if (num_ptcls == 0)
        return Segment<DataType<N>, typename MemSpace::execution_space>();
      MemberTypeView<DataType<N> >* view =
        static_cast<MemberTypeView<DataType<N> >*>(ptcl_data[N]);
      return Segment<DataType<N>, typename MemSpace::execution_space>(*view);
    }


    virtual void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                         MemberTypeViews<DataTypes> new_particle_info = NULL) = 0;
    virtual void migrate(kkLidView new_element, kkLidView new_process,
                         kkLidView new_particle_elements = kkLidView(),
                         MemberTypeViews<DataTypes> new_particle_info = NULL) = 0;
    virtual void printMetrics() const = 0;
  protected:
    //Element and particle Counts/capacities
    lid_t num_elems;
    lid_t num_ptcls;
    lid_t capacity_;
    lid_t num_rows;

    //Particle information
    MemberTypeViews<DataTypes> ptcl_data;

    //Number of Data types
    static constexpr std::size_t num_types = DataTypes::size;
  };


  template <class DataTypes, typename MemSpace>
  ParticleStructure<DataTypes, MemSpace>::ParticleStructure() : num_elems(0), num_ptcls(0),
                                                                capacity_(0), num_rows(0) {
  }
}
