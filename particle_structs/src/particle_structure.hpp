#pragma once
#include <ppTypes.h>
#include <Segment.h>
#include <MemberTypeLibraries.h>
#include <psDistributor.hpp>
#ifdef PP_ENABLE_CAB
#include "psMemberTypeCabana.h"
#endif

namespace pumipic {

  //Forward Declaration of Cabana-using classes
  template <class DataTypes, typename Space>
  class CabM;
  template <class DataTypes, typename Space>
  class DPS;

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

    template <std::size_t N> using DataType =
      typename MemberTypeAtIndex<N, DataTypes>::type;
    typedef MemberTypeViews MTVs;
    template <std::size_t N> using MTV = MemberTypeView<DataType<N>, device_type>;
#ifdef PP_ENABLE_CAB
    //Cabana Values for defining generic slice
    //Some defintions are taken from cabana/Cabana_AoSoA.hpp
    static constexpr int vector_length =
      Cabana::Impl::PerformanceTraits<execution_space>::vector_length;
#endif
    template <std::size_t M>
    using member_value_type =
      typename std::remove_all_extents<DataType<M>>::type;

#ifdef PP_ENABLE_CAB
    using PS_DT=PS_DTBool<Types>;
    using soa_type = Cabana::SoA<PS_DT, vector_length>;
    template <std::size_t N> using Slice =
      Segment<DataType<N>, device_type, Cabana::DefaultAccessMemory, vector_length,
              sizeof(soa_type)/ sizeof(member_value_type<N>)>;
#else
    template <std::size_t N> using Slice = Segment<DataType<N>, device_type>;
#endif

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
      assert(N < num_types);
      if (num_ptcls == 0)
        return Slice<N>();
#ifdef PP_ENABLE_CAB
      if (dynamic_cast<CabM<DataTypes, Space>*>(this) != NULL)
        return dynamic_cast<CabM<DataTypes, Space>*>(this)->template get<N>();
      if (dynamic_cast<DPS<DataTypes, Space>*>(this) != NULL)
        return dynamic_cast<DPS<DataTypes, Space>*>(this)->template get<N>();
#endif
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

    /* checkpoint restart methods*/
    virtual void checkpoint(std::string path) { fprintf(stderr, "not supported!\n"); }

  protected:
    //String to identify the particle structure
    std::string name;
    //Element and particle Counts/capacities
    lid_t num_elems;
    lid_t num_ptcls;
    lid_t capacity_;
    lid_t num_rows;

    //Particle information
    MTVs ptcl_data;

    //Number of Data types
    static constexpr std::size_t num_types = DataTypes::size;

    /*
      Copy a particle structure to another memory space
      Note: if the same memory space is used then a the data is not duplicated
    */
    template <class Space2>
    void copy(Mirror<Space2>* old) {
      name = old->name;
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
  ParticleStructure<DataTypes, Space>::ParticleStructure() : name("ptcls"), num_elems(0), num_ptcls(0),
                                                             capacity_(0), num_rows(0) {
  }

  template <class DataTypes, typename Space>
  ParticleStructure<DataTypes, Space>::ParticleStructure(const std::string& name_) : name(name_), num_elems(0), num_ptcls(0),
                                                             capacity_(0), num_rows(0) {
  }

}
