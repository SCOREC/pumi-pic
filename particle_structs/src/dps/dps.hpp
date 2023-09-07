#pragma once
#include <particle_structs.hpp>
#ifdef PP_ENABLE_CAB
#include <Cabana_Core.hpp>
#include "psMemberTypeCabana.h"
#include "dps_input.hpp"
#include <sstream>

namespace pumipic {

  void enable_prebarrier();
  double prebarrier();

  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class DPS : public ParticleStructure<DataTypes, MemSpace> {
  public:
    template <typename MSpace> using Mirror = DPS<DataTypes, MSpace>;
    using typename ParticleStructure<DataTypes, MemSpace>::execution_space;
    using typename ParticleStructure<DataTypes, MemSpace>::memory_space;
    using typename ParticleStructure<DataTypes, MemSpace>::device_type;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::MTVs;
    template<std::size_t N>
    using Slice = typename ParticleStructure<DataTypes, MemSpace>::template Slice<N>;

    using host_space = Kokkos::HostSpace;
    typedef Kokkos::TeamPolicy<execution_space> PolicyType;
    typedef Kokkos::UnorderedMap<gid_t, lid_t, device_type> GID_Mapping;
    typedef DPS_Input<DataTypes, MemSpace> Input_T;

    using DPS_DT = PS_DTBool<DataTypes>;
    using AoSoA_t = Cabana::AoSoA<DPS_DT,device_type>;

    DPS(const DPS&) = delete;
    DPS& operator=(const DPS&) = delete;

    DPS( PolicyType& p,
          lid_t num_elements, lid_t num_particles,
          kkLidView particles_per_element,
          kkGidView element_gids,
          kkLidView particle_elements = kkLidView(),
          MTVs particle_info = NULL);
    DPS(DPS_Input<DataTypes, MemSpace>&);
    ~DPS();

    template <class MSpace>
    Mirror<MSpace>* copy();

    //Functions from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::nElems;
    using ParticleStructure<DataTypes, MemSpace>::nPtcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity;
    using ParticleStructure<DataTypes, MemSpace>::numRows;
    using ParticleStructure<DataTypes, MemSpace>::copy;

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
    void setNewActive(const lid_t num_particles);
    void createGlobalMapping(const kkGidView element_gids, kkGidView& lid_to_gid, GID_Mapping& gid_to_lid);
    void fillAoSoA(const kkLidView particle_elements, const MTVs particle_info, kkLidView& parentElms);
    void setParentElms(const kkLidView particles_per_element, kkLidView& parentElms);

    template <typename DT, typename MSpace> friend class DPS;

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
    // percentage of capacity to add as padding
    double extra_padding;
    // parent elements for all particles in AoSoA
    kkLidView parentElms_;
    // particle data
    AoSoA_t* aosoa_;

    //Private construct function
    void construct(kkLidView ptcls_per_elem,
                  kkGidView element_gids,
                  kkLidView particle_elements,
                  MTVs particle_info);

    //Private constructor for copy()
    DPS() : ParticleStructure<DataTypes, MemSpace>(), policy(100, 1) {}
  };

  template<class DataTypes, typename MemSpace>
  void DPS<DataTypes, MemSpace>::construct(kkLidView ptcls_per_elem,
                                           kkGidView element_gids,
                                           kkLidView particle_elements,
                                           MTVs particle_info) {
    Kokkos::Profiling::pushRegion("dps_construction");
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
    // set active mask
    setNewActive(num_ptcls);
    // get global ids
    if (element_gids.size() > 0)
      createGlobalMapping(element_gids, element_to_gid, element_gid_to_lid);
    // populate AoSoA with input data if given
    if (particle_elements.size() > 0 && particle_info != NULL) {
      if(!comm_rank) fprintf(stderr, "initializing DPS data\n");
      fillAoSoA(particle_elements, particle_info, parentElms_); // fill aosoa with data
    }
    else
      setParentElms(ptcls_per_elem, parentElms_);

    Kokkos::Profiling::popRegion();       
  }

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
    construct(particles_per_element, element_gids, particle_elements, particle_info);
  }

  template <class DataTypes, typename MemSpace>
  DPS<DataTypes, MemSpace>::DPS(Input_T& input) :        // optional
    ParticleStructure<DataTypes, MemSpace>(input.name),
    policy(input.policy),
    element_gid_to_lid(input.ne)
  {
    num_elems = input.ne;
    num_rows = num_elems;
    num_ptcls = input.np;
    extra_padding = input.extra_padding;
    assert(num_elems == input.ppe.size());
    construct(input.ppe, input.e_gids, input.particle_elms, input.p_info);
  }

  template <class DataTypes, typename MemSpace>
  DPS<DataTypes, MemSpace>::~DPS() { delete aosoa_; }

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
    if (nPtcls() == 0)
      return;

    // move function pointer to GPU (if needed)
    FunctionType* fn_d = gpuMemcpy(fn);
    kkLidView parentElms_cpy = parentElms_;
    const auto soa_len = AoSoA_t::vector_length;
    const auto mask = Cabana::slice<DPS_DT::size-1>(*aosoa_); // get active mask
    Cabana::SimdPolicy<soa_len,execution_space> simd_policy(0, capacity_);
    Cabana::simd_parallel_for(simd_policy,
      KOKKOS_LAMBDA( const lid_t soa, const lid_t ptcl ) {
        const lid_t particle_id = soa*soa_len + ptcl; // calculate overall index
        const lid_t elm = parentElms_cpy(particle_id); // calculate element
        (*fn_d)(elm, particle_id, mask.access(soa,ptcl));
      }, s);
#ifdef PP_USE_GPU
    gpuFree(fn_d);
#endif

  }

  template<class DataTypes, typename MemSpace>
  template <class MSpace>
  typename DPS<DataTypes, MemSpace>::template Mirror<MSpace>* DPS<DataTypes, MemSpace>::copy() {
    if (std::is_same<memory_space, typename MSpace::memory_space>::value) {
      fprintf(stderr, "[ERROR] Copy to same memory space not supported\n");
      exit(EXIT_FAILURE);
    }
    Mirror<MSpace>* mirror_copy = new DPS<DataTypes, MSpace>();
    //Call Particle structures copy
    mirror_copy->copy(this);
    //Copy constants
    mirror_copy->num_soa_ = num_soa_;
    mirror_copy->extra_padding = extra_padding;

    //Copy AoSoA
    mirror_copy->aosoa_ = new typename DPS<DataTypes, MSpace>::AoSoA_t(std::string(aosoa_->label()).append("_mirror"), aosoa_->size());
    Cabana::deep_copy(*(mirror_copy->aosoa_), *aosoa_);

    //Deep copy each view
    mirror_copy->parentElms_ = typename Mirror<MSpace>::kkLidView("mirror parentElms_",
                                                                    parentElms_.size());
    Kokkos::deep_copy(mirror_copy->parentElms_, parentElms_);
    mirror_copy->element_to_gid = typename Mirror<MSpace>::kkGidView("mirror element_to_gid",
                                                                    element_to_gid.size());
    Kokkos::deep_copy(mirror_copy->element_to_gid, element_to_gid);
    //Deep copy the gid mapping
    mirror_copy->element_gid_to_lid.create_copy_view(element_gid_to_lid);
    return mirror_copy;
  }

  template <class DataTypes, typename MemSpace>
  void DPS<DataTypes, MemSpace>::printMetrics() const {
    // Sum number of empty cells
    auto mask = Cabana::slice<DPS_DT::size-1>(*aosoa_);
    kkLidView padded_cells("num_padded_cells",1);
    Kokkos::parallel_for("count_padding", capacity_,
      KOKKOS_LAMBDA(const lid_t ptcl_id) {
        Kokkos::atomic_fetch_add(&padded_cells(0), !mask(ptcl_id));
      });
    lid_t num_padded = getLastValue<lid_t>(padded_cells);

    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    char buffer[1000];
    char* ptr = buffer;
    // Header
    ptr += sprintf(ptr, "Metrics (Rank %d)\n", comm_rank);
    // Sizes
    ptr += sprintf(ptr, "Number of Elements %d, Number of SoA %d, Number of Particles %d, Capacity %d\n",
                   num_elems, num_soa_, num_ptcls, capacity_);
    // Padded Cells
    ptr += sprintf(ptr, "Padded Cells <Tot %%> %d %.3f%%\n", num_padded,
                   num_padded * 100.0 / capacity_);
    printf("%s\n", buffer);
  }

  template <class DataTypes, typename MemSpace>
  void DPS<DataTypes, MemSpace>::printFormat(const char* prefix) const {
    kkGidHostMirror element_to_gid_host = deviceToHost(element_to_gid);
    kkLidHostMirror parents_host = deviceToHost(parentElms_);
    const auto soa_len = AoSoA_t::vector_length;

    kkLidView mask(Kokkos::ViewAllocateWithoutInitializing("mask"), capacity_);
    auto mask_slice = Cabana::slice<DPS_DT::size-1>(*aosoa_);
    Kokkos::parallel_for("copy_mask", capacity_,
      KOKKOS_LAMBDA(const lid_t ptcl_id) {
        mask(ptcl_id) = mask_slice(ptcl_id);
      });
    kkLidHostMirror mask_host = deviceToHost(mask);

    std::stringstream ss;
    char buffer[1000];
    char* ptr = buffer;
    int num_chars;

    num_chars = sprintf(ptr, "%s\n", prefix);
    num_chars += sprintf(ptr+num_chars,"Particle Structures DPS\n");
    num_chars += sprintf(ptr+num_chars,"Number of Elements: %d.\nNumber of SoA: %d.\nNumber of Particles: %d.", num_elems, num_soa_, num_ptcls);

    num_chars += sprintf(ptr+num_chars, "\n  Elements");
    for (int e = 0; e < num_elems; e++){
      num_chars += sprintf(ptr+num_chars, " %2d(%2d)", e, element_to_gid_host(e));
    }

    lid_t last_soa = -1;
    for (int i = 0; i < capacity_; i++) {
      lid_t soa = i / soa_len;
      if (last_soa == soa) {
        num_chars += sprintf(ptr+num_chars," %d", mask_host(i));
      }
      else {
        num_chars += sprintf(ptr+num_chars,"\n  SOA %d | %d", soa, mask_host(i));
      }
      last_soa = soa;
    }
    buffer[num_chars] = '\0';
    ss << buffer;
    ss << "\n";
    std::cout << ss.str();
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
    template <typename MSpace> using Mirror = DPS<DataTypes, MSpace>;
    using typename ParticleStructure<DataTypes, MemSpace>::execution_space;
    using typename ParticleStructure<DataTypes, MemSpace>::memory_space;
    using typename ParticleStructure<DataTypes, MemSpace>::device_type;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::MTVs;
    template<std::size_t N>
    using Slice = typename ParticleStructure<DataTypes, MemSpace>::template Slice<N>;

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

    template <class MSpace>
    Mirror<MSpace>* copy() {reportError(); return NULL;}

  private:
    void reportError() const {fprintf(stderr, "[ERROR] pumi-pic was built "
                                      "without Cabana so the DPS structure "
                                      "can not be used\n");}
  };
}
#endif // PP_ENABLE_CAB
