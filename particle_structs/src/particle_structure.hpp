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
    template <typename NewSpace> using Mirror =
      ParticleStructure<typename DataStructure::Mirror<NewSpace>>;

    typedef typename DataStructure::kkLidView kkLidView;
    typedef typename DataStructure::kkGidView kkGidView;
    typedef typename DataStructure::kkLidHostMirror kkLidHostMirror;
    typedef typename DataStructure::kkGidHostMirror kkGidHostMirror;
    typedef typename DataStructure::MTVs MTVs;
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
    // This will potentially require a deep copy of the memory
    ParticleStructure(const DataStructure& ds);
    //Constructs a particle structure using the data structure's inputs
    // The best option to construct the data structure directly
    ParticleStructure(typename DataStructure::Input_T& input);

    //Copy the ParticleStructure from one memory space to another
    //Note: This constructor only works if NewDS is not the same as DataStructure
    template <typename Space>
    ParticleStructure(const Mirror<Space>& old) :
      structure(old.structure) {}
    template <typename NewDS, typename =
              typename std::enable_if<!std::is_same<DataStructure, NewDS>::value>::type>
    ParticleStructure<DataStructure>& operator=(const ParticleStructure<NewDS>& old) {
      structure = old.structure;
      return *this;
    }
    //Disallow copying/assigning when the templates are the same
    ParticleStructure(const ParticleStructure& old) = delete;
    ParticleStructure& operator=(const ParticleStructure& old) = delete;
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
