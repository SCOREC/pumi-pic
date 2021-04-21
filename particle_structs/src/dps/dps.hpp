#pragma once
#include <particle_structs.hpp>
#ifdef PP_ENABLE_CAB
#include <Cabana_Core.hpp>
#include "dps_support.hpp"
#include <sstream>

namespace pumipic {

  void enable_prebarrier();
  double prebarrier();

  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class DPS : public ParticleStructure<DataTypes, MemSpace> {
  public:
    using typename ParticleStructure<DataTypes, MemSpace>::execution_space;
    using typename ParticleStructure<DataTypes, MemSpace>::memory_space;
    using typename ParticleStructure<DataTypes, MemSpace>::device_type;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::MTVs;
    template<std::size_t N>
    using Slice = typename ParticleStructure<DataTypes, MemSpace>::Slice<N>;

    using host_space = Kokkos::HostSpace;
    typedef Kokkos::TeamPolicy<execution_space> PolicyType;
    typedef Kokkos::UnorderedMap<gid_t, lid_t, device_type> GID_Mapping;

    using DPS_DT = PS_DTBool<DataTypes>;
    using AoSoA_t = Cabana::AoSoA<DPS_DT,device_type>;

    DPS() = delete;
    DPS(const DPS&) = delete;
    DPS& operator=(const DPS&) = delete;

    DPS( PolicyType& p,
          lid_t num_elements, lid_t num_particles,
          kkLidView particles_per_element,
          kkGidView element_gids,
          kkLidView particle_elements = kkLidView(),
          MTVs particle_info = NULL);
    ~DPS();

    //Functions from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::nElems;
    using ParticleStructure<DataTypes, MemSpace>::nPtcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity;
    using ParticleStructure<DataTypes, MemSpace>::numRows;

    template <std::size_t N>
    Slice<N> get() { return Slice<N>(Cabana::slice<N, AoSoA_t>(*aosoa_, "get<>()")); }

    void migrate(kkLidView new_element, kkLidView new_process,
                 Distributor<MemSpace> dist = Distributor<MemSpace>(),
                 kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particle_info = NULL);

    void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particles = NULL);

    template <typename FunctionType>
    void parallel_for(FunctionType& fn, std::string s="");

    void printMetrics() const;
    void printFormat(const char* prefix) const;

    // Do not call these functions:
    AoSoA_t* makeAoSoA(const lid_t capacity, const lid_t num_soa);
    kkLidView buildIndices(const kkLidView particles_per_element, const lid_t capacity,
      kkLidView particleIds, kkLidView parentElms);
    void setNewActive(AoSoA_t* aosoa, const lid_t num_particles);
    void createGlobalMapping(kkGidView element_gids, kkGidView& lid_to_gid, GID_Mapping& gid_to_lid);
    void fillAoSoA(kkLidView particle_elements, MTVs particle_info);

  private:
    //The User defined Kokkos policy
    PolicyType policy;

    //Variables from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::name;
    using ParticleStructure<DataTypes, MemSpace>::num_elems;
    using ParticleStructure<DataTypes, MemSpace>::num_ptcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity_;
    using ParticleStructure<DataTypes, MemSpace>::num_rows;
    using ParticleStructure<DataTypes, MemSpace>::ptcl_data;
    using ParticleStructure<DataTypes, MemSpace>::num_types;
  
    // mappings from row to element gid and back to row
    kkGidView element_to_gid;
    GID_Mapping element_gid_to_lid;
    // number of SoA
    lid_t num_soa_;
    // SoA index for start of padding
    lid_t padding_start;
    // percentage of capacity to add as padding
    double extra_padding;
    // CSR structure for element tracking
    kkLidView particleIds_;
    kkLidView offsets;
    kkLidView parentElms_;
    // particle data
    AoSoA_t* aosoa_;
  };

  /**
   * Constructor
   * @param[in] p
   * @param[in] num_elements number of elements
   * @param[in] num_particles number of particles
   * @param[in] particle_per_element view of ints, representing number of particles
   *    in each element
   * @param[in] element_gids view of ints, representing the global ids of each element
   * @param[in] particle_elements view of ints, representing which elements
   *    particle reside in (optional)
   * @param[in] particle_info array of views filled with particle data (optional)
   * @exception num_elements != particles_per_element.size(),
   *    undefined behavior for new_particle_elements.size() != sizeof(new_particles),
   *    undefined behavior for numberoftypes(new_particles) != numberoftypes(DataTypes)
  */
  template <class DataTypes, typename MemSpace>
  DPS<DataTypes, MemSpace>::DPS( PolicyType& p,
                                   lid_t num_elements, lid_t num_particles,
                                   kkLidView particles_per_element,
                                   kkGidView element_gids,
                                   kkLidView particle_elements, // optional
                                   MTVs particle_info) :        // optional
    ParticleStructure<DataTypes, MemSpace>(),
    policy(p),
    element_gid_to_lid(num_elements),
    extra_padding(0.05) // default extra padding at 5%
  {
    assert(num_elements == particles_per_element.size());
    num_elems = num_elements;
    num_rows = num_elems;
    num_ptcls = num_particles;
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    if(!comm_rank)
      fprintf(stderr, "building DPS\n");

    // calculate num_soa_ from number of particles + extra padding
    num_soa_ = ceil(ceil(double(num_ptcls)/AoSoA_t::vector_length)*(1+extra_padding));
    // calculate capacity_ from num_soa_ and max size of an SoA
    capacity_ = num_soa_*AoSoA_t::vector_length;
    // initialize appropriately-sized AoSoA
    aosoa_ = makeAoSoA(capacity_, num_soa_);
    // build element tracking arrays
    offsets = buildIndices(particles_per_element, capacity_, particleIds_, parentElms_, padding_start);
    // set active mask
    setNewActive(aosoa_, padding_start);
    // get global ids
    if (element_gids.size() > 0)
      createGlobalMapping(element_gids, element_to_gid, element_gid_to_lid);
    // populate AoSoA with input data if given
    if (particle_elements.size() > 0 && particle_info != NULL) {
      if(!comm_rank) fprintf(stderr, "initializing DPS data\n");
      fillAoSoA(particle_elements, particle_info); // fill aosoa with data
    }

    fprintf(stderr, "[WARNING] Contructor not yet finished!\n");
  }

  template <class DataTypes, typename MemSpace>
  DPS<DataTypes, MemSpace>::~DPS() { }

  /**
   * a parallel for-loop that iterates through all particles
   * @param[in] fn function of the form fn(elm, particle_id, mask), where
   *    elm is the element the particle is in
   *    particle_id is the overall index of the particle in the structure
   *    mask is 0 if the particle is inactive and 1 if the particle is active
   * @param[in] s string for labelling purposes
  */
  template <class DataTypes, typename MemSpace>
  template <typename FunctionType>
  void DPS<DataTypes, MemSpace>::parallel_for(FunctionType& fn, std::string s) {
    fprintf(stderr, "[WARNING] parallel_for not yet implemented!\n");
  }

  template <class DataTypes, typename MemSpace>
  void DPS<DataTypes, MemSpace>::printMetrics() const {
    fprintf(stderr, "[WARNING] printMetrics not yet implemented!\n");
  }

  template <class DataTypes, typename MemSpace>
  void DPS<DataTypes, MemSpace>::printFormat(const char* prefix) const {
    fprintf(stderr, "[WARNING] printFormat not yet implemented!\n");
  }

}

// Separate files with DPS member function implementations
#include "dps_buildFns.hpp"
#include "dps_rebuild.hpp"
#include "dps_migrate.hpp"

#else
namespace pumipic {
  /*A dummy version of DPS when pumi-pic is built without Cabana so operations
    can compile without ifdef guards. The operations will report a message stating
    that the structure will not work.
  */
  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class DPS : public ParticleStructure<DataTypes, MemSpace> {
  public:
    using typename ParticleStructure<DataTypes, MemSpace>::execution_space;
    using typename ParticleStructure<DataTypes, MemSpace>::memory_space;
    using typename ParticleStructure<DataTypes, MemSpace>::device_type;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::MTVs;
    template<std::size_t N>
    using Slice = typename ParticleStructure<DataTypes, MemSpace>::Slice<N>;

    using host_space = Kokkos::HostSpace;
    typedef Kokkos::TeamPolicy<execution_space> PolicyType;
    typedef Kokkos::UnorderedMap<gid_t, lid_t, device_type> GID_Mapping;

    DPS() = delete;
    DPS(const DPS&) = delete;
    DPS& operator=(const DPS&) = delete;

    DPS( PolicyType& p,
          lid_t num_elements, lid_t num_particles,
          kkLidView particles_per_element,
          kkGidView element_gids,
          kkLidView particle_elements = kkLidView(),
          MTVs particle_info = NULL) {reportError();}
    ~DPS() {}

    //Functions from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::nElems;
    using ParticleStructure<DataTypes, MemSpace>::nPtcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity;
    using ParticleStructure<DataTypes, MemSpace>::numRows;

    template <std::size_t N>
    Slice<N> get() { reportError(); return Slice<N>();}

    void migrate(kkLidView new_element, kkLidView new_process,
                 Distributor<MemSpace> dist = Distributor<MemSpace>(),
                 kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particle_info = NULL) {reportError();}

    void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particles = NULL) {reportError();}

    template <typename FunctionType>
    void parallel_for(FunctionType& fn, std::string s="") {reportError();}

    void printMetrics() const {reportError();}
    void printFormat(const char* prefix) const {reportError();}

  private:
    void reportError() const {fprintf(stderr, "[ERROR] pumi-pic was built "
                                      "without Cabana so the DPS structure "
                                      "can not be used\n");}
  };
}
#endif // PP_ENABLE_CAB