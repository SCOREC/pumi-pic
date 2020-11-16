#pragma once

#include <ppTypes.h>
#include <Segment.h>
#include <MemberTypeLibraries.h>
#include <psDistributor.hpp>


namespace pumipic {

  //Temporary hack
  template <typename Device, typename... Types> struct SetPtrs;

  template <class DataTypes, typename Space = DefaultMemSpace>
  class ParticleStructure {
  public:
    typedef DataTypes Types;
    typedef typename Space::memory_space memory_space;
    typedef typename Space::execution_space execution_space;
    typedef typename Space::device_type device_type;
    typedef typename Kokkos::ViewTraits<void, Space>::HostMirrorSpace HostMirrorSpace;
    typedef ParticleStructure<DataTypes, HostMirrorSpace> HostMirror;
    template <typename Space2> using Mirror = ParticleStructure<DataTypes, Space2>;

    template <class T> using View = Kokkos::View<T*, device_type>;
    typedef View<lid_t> kkLidView;
    typedef View<gid_t> kkGidView;
    typedef typename kkLidView::HostMirror kkLidHostMirror;
    typedef typename kkGidView::HostMirror kkGidHostMirror;

    template <std::size_t N> using DataType = typename MemberTypeAtIndex<N, DataTypes>::type;
    typedef MemberTypeViews MTVs;
    template <std::size_t N> using MTV = MemberTypeView<DataType<N>, device_type>;
    template <std::size_t N> using Slice = Segment<DataType<N>, device_type>;

    ParticleStructure();
    ParticleStructure(const std::string& name_);
    virtual ~ParticleStructure() {}

    const std::string& getName() const {return name;}
    lid_t nElems() const {return num_elems;}
    lid_t nPtcls() const {return num_ptcls;}
    lid_t capacity() const {return capacity_;}
    lid_t numRows() const {return num_rows;}

    /* Provides access to the particle info for Nth time of each particle

       The segment is indexed by particle index first followed by indices for each
         dimension of the type.
     */
    template <std::size_t N>
    Slice<N> get() {
      if (num_ptcls == 0)
        return Slice<N>();
      MTV<N>* view = static_cast<MTV<N>*>(ptcl_data[N]);
      return Slice<N>(*view);
    }


    virtual void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                         MTVs new_particle_info = NULL) = 0;
    virtual void migrate(kkLidView new_element, kkLidView new_process,
                         Distributor<Space> dist = Distributor<Space>(),
                         kkLidView new_particle_elements = kkLidView(),
                         MTVs new_particle_info = NULL) = 0;
    virtual void printMetrics() const = 0;
  protected:
    //String to identify the particle structure
    std::string name;
    //Element and particle Counts/capacities
    lid_t num_elems;
    lid_t num_ptcls;
    lid_t capacity_;
    lid_t num_rows;

    //Number of Data types
    static constexpr std::size_t num_types = DataTypes::size;

    //Particle information
    MTVs ptcl_data;
    //Device pointers to particle information
    void* ptcl_data_d;


    /*
      Copy a particle structure to another memory space
      Note: if the same memory space is used then the data is not duplicated
    */
    template <class Space2>
    void copy(Mirror<Space2>* old) {
      num_elems = old->num_elems;
      num_ptcls = old->num_ptcls;
      capacity_ = old->capacity_;
      num_rows = old->num_rows;
      if (std::is_same<memory_space, typename Space2::memory_space>::value) {
        ptcl_data = old->ptcl_data;
      }
      else {
        auto first_data_view = static_cast<MTV<0>*>(old->ptcl_data[0]);
        int s = first_data_view->size() / BaseType<DataType<0> >::size;
        ptcl_data = createMemberViews<DataTypes, Space>(s);
        CopyMemSpaceToMemSpace<Space, Space2, DataTypes>(ptcl_data, old->ptcl_data);
      }
    }
    template <typename DT, typename Space2> friend class ParticleStructure;

   };

  template <class DataTypes, typename Space>
  ParticleStructure<DataTypes, Space>::ParticleStructure()
    : name("ptcls"), num_elems(0), num_ptcls(0), capacity_(0), num_rows(0), ptcl_data_d(NULL) {
  }

  template <class DataTypes, typename Space>
  ParticleStructure<DataTypes, Space>::ParticleStructure(const std::string& name_)
    : name(name_), num_elems(0), num_ptcls(0), capacity_(0), num_rows(0), ptcl_data_d(NULL){
  }

}
