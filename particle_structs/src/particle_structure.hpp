#pragma once

#include <PS_Types.h>
#include <Segment.h>
#include <MemberTypeLibraries.h>

namespace particle_structs {

  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class ParticleStructure {
  public:
    typedef typename MemSpace::memory_space memory_space;
    typedef typename MemSpace::execution_space execution_space;
    typedef typename MemSpace::device_type device_type;
    typedef Kokkos::View<lid_t*, device_type> kkLidView;
    typedef Kokkos::View<gid_t*, device_type> kkGidView;
    typedef typename kkLidView::HostMirror kkLidHostMirror;
    typedef typename kkGidView::HostMirror kkGidHostMirror;
    template <std::size_t N> using DataType = typename MemberTypeAtIndex<N, DataTypes>::type;
    typedef MemberTypeViews<DataTypes, device_type> MTVs;

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
    Segment<DataType<N>, execution_space> get() {
      if (num_ptcls == 0)
        return Segment<DataType<N>, execution_space>();
      MemberTypeView<DataType<N>, device_type>* view =
        static_cast<MemberTypeView<DataType<N>, device_type>*>(ptcl_data[N]);
      return Segment<DataType<N>, execution_space>(*view);
    }


    virtual void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                         MemberTypeViews<DataTypes, device_type> new_particle_info = NULL) = 0;
    virtual void migrate(kkLidView new_element, kkLidView new_process,
                         kkLidView new_particle_elements = kkLidView(),
                         MemberTypeViews<DataTypes, device_type> new_particle_info = NULL) = 0;
    virtual void printMetrics() const = 0;
  protected:
    //Element and particle Counts/capacities
    lid_t num_elems;
    lid_t num_ptcls;
    lid_t capacity_;
    lid_t num_rows;

    //Particle information
    MTVs ptcl_data;

    //Number of Data types
    static constexpr std::size_t num_types = DataTypes::size;
  };


  template <class DataTypes, typename MemSpace>
  ParticleStructure<DataTypes, MemSpace>::ParticleStructure() : num_elems(0), num_ptcls(0),
                                                                capacity_(0), num_rows(0) {
  }
}
