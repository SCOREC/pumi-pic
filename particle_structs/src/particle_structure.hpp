#pragma once

#include <ppTypes.h>
#include <Segment.h>
#include <MemberTypeLibraries.h>
#include <psDistributor.hpp>
// #include <cabm_support.hpp>
namespace pumipic {

  /* The ParticleStructure class
     \template DataStructure - the underlying data structure to store particles
       * Required definitions include:
           Types - The MemberTypes (a template type of the data structure)
           memory_space, execution_space, device_type - Spaces the particle structure is defined on
           Input_T - the input type that can construct the data structure from
           Slice<N> - The return type of the get function

  */
  template <class DataStructure>
  class ParticleStructure {
  public:
    typedef typename DataStructure::Types Types;
    typedef typename DataStructure::memory_space memory_space;
    typedef typename DataStructure::execution_space execution_space;
    typedef typename DataStructure::device_type device_type;
    template <typename Space2> using Mirror = ParticleStructure<typename DataStructure::Mirror<Space2> >;

    typedef typename DataStructure::kkLidView kkLidView;
    typedef typename DataStructure::kkGidView kkGidView;
    typedef typename DataStructure::kkLidHostMirror kkLidHostMirror;
    typedef typename DataStructure::kkGidHostMirror kkGidHostMirror;
    typedef typename DataStructure::MTVs MTVs;
    // template <std::size_t N> using DataType =
    //   typename MemberTypeAtIndex<N, Types>::type;
    // template <std::size_t N> using MTV = MemberTypeView<DataType<N>, device_type>;
    //Cabana Values for defining generic slice
    //Some defintions are taken from cabana/Cabana_AoSoA.hpp
    // static constexpr int vector_length =
    //   Cabana::Impl::PerformanceTraits<execution_space>::vector_length;
    // template <std::size_t M>
    // using member_value_type =
    //   typename std::remove_all_extents<DataType<M>>::type;

    // using CM_DT=CM_DTInt<Types>;
    // using soa_type = Cabana::SoA<CM_DT, vector_length>;
    template <std::size_t N> using Slice = typename DataStructure::Slice<N>;

    //Create an empty particle structure (this should likely not be used)
    ParticleStructure();
    //Create a particle structure with a prebuilt data structure
    // This results in a deep copy of the memory
    ParticleStructure(const DataStructure& ds);
    //Constructs a particle structure using the data structure's inputs
    // The best option to construct the data structure directly
    ParticleStructure(typename DataStructure::Input_T& input);
    ~ParticleStructure() {}

    const std::string& getName() const {return structure.getName();}
    lid_t nElems() const {return structure.nElems();}
    lid_t nPtcls() const {return structure.nPtcls();}
    lid_t capacity() const {return structure.capacity();}
    lid_t numRows() const {return structure.numRows();}

    DataStructure* operator->() {
      return &structure;
    }

    /* Provides access to the particle info for Nth time of each particle

       The segment is indexed by particle index first followed by indices for each
         dimension of the type
     */
    template <std::size_t N>
    Slice<N> get() {
      return structure.template get<N>();
    }

    void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                         MTVs new_particle_info = NULL) {
      structure.rebuild(new_element, new_particle_elements, new_particle_info);
    };
    void migrate(kkLidView new_element, kkLidView new_process,
                         Distributor<memory_space> dist = Distributor<memory_space>(),
                         kkLidView new_particle_elements = kkLidView(),
                         MTVs new_particle_info = NULL) {
      structure.migrate(new_element, new_process, dist,
                        new_particle_elements, new_particle_info);
    };
    template <typename FunctionType>
    void parallel_for(FunctionType& fn, std::string s="") {
      structure.parallel_for(fn, s);
    }

    void printMetrics() const {
      structure.printMetrics();
    };
  private:
    DataStructure structure;

    /*
      Copy a particle structure to another memory space
      Note: if the same memory space is used then a the data is not duplicated
    */
    // template <class Space2>
    // void copy(Mirror<Space2>* old) {
    //   num_elems = old->num_elems;
    //   num_ptcls = old->num_ptcls;
    //   capacity_ = old->capacity_;
    //   num_rows = old->num_rows;
    //   if (std::is_same<memory_space, typename Space2::memory_space>::value) {
    //     ptcl_data = old->ptcl_data;
    //   }
    //   else {
    //     auto first_data_view = static_cast<MTV<0>*>(old->ptcl_data[0]);
    //     int s = first_data_view->size() / BaseType<DataType<0> >::size;
    //     ptcl_data = createMemberViews<DataTypes, Space>(s);
    //     CopyMemSpaceToMemSpace<Space, Space2, DataTypes>(ptcl_data, old->ptcl_data);
    //   }
    // }

    /* Friend will all structures
       Note: We only need to friend with structures that are of the same class
             but have different templates, but as far as I can find, C++ does
             not allow conditional friendship on templated classes of different
             template than the template type of the current class
     */
    template <typename Structure> friend class ParticleStructure;
  };

  template <class DataStructure>
  ParticleStructure<DataStructure>::ParticleStructure() : structure() {}

  template <class DataStructure>
  ParticleStructure<DataStructure>::ParticleStructure(const DataStructure& ds) :
    structure(ds) {}

  template <class DataStructure>
  ParticleStructure<DataStructure>::ParticleStructure(
    typename DataStructure::Input_T& input) : structure(input) {}

}
