#pragma once
#include <particle_structs.hpp>
#ifdef PP_ENABLE_CAB
#include <Cabana_Core.hpp>
#include "psMemberTypeCabana.h"
#include "cabm_input.hpp"
#include <sstream>

namespace pumipic {

  void enable_prebarrier();
  double prebarrier();

  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class CabM : public ParticleStructure<DataTypes, MemSpace> {
  public:
    template <typename MSpace> using Mirror = CabM<DataTypes, MSpace>;
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
    typedef CabM_Input<DataTypes, MemSpace> Input_T;

    //from https://github.com/SCOREC/Cabana/blob/53ad18a030f19e0956fd0cab77f62a9670f31941/core/src/CabanaM.hpp#L18-L19
    using CM_DT = PS_DTBool<DataTypes>;
    using AoSoA_t = Cabana::AoSoA<CM_DT,memory_space>;

    CabM(const CabM&) = delete;
    CabM& operator=(const CabM&) = delete;

    CabM( PolicyType& p,
          lid_t num_elements, lid_t num_particles,
          kkLidView particles_per_element,
          kkGidView element_gids,
          kkLidView particle_elements = kkLidView(),
          MTVs particle_info = NULL,
          MPI_Comm mpi_comm = MPI_COMM_WORLD);
    CabM(CabM_Input<DataTypes, MemSpace>&);
    ~CabM();

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

    void printMetrics(MPI_Comm mpi_comm = MPI_COMM_WORLD) const;
    void printFormat(const char* prefix) const;

    // Do not call these functions:
    typename ParticleStructure<DataTypes, MemSpace>::kkLidView
    buildOffset(const kkLidView particles_per_element, const lid_t num_ptcls, const double padding, lid_t &padding_start);
    AoSoA_t* makeAoSoA(const lid_t capacity, const lid_t num_soa);

    typename ParticleStructure<DataTypes, MemSpace>::kkLidView
    getParentElms(const lid_t num_elements, const lid_t num_soa, const kkLidView offsets);

    void setActive(const kkLidView particles_per_element);
    void createGlobalMapping(const kkGidView element_gids, kkGidView& lid_to_gid, GID_Mapping& gid_to_lid);
    void fillAoSoA(const kkLidView particle_elements, const MTVs particle_info);

    template <typename DT, typename MSpace> friend class CabM;

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
    // offsets array for CSR structure
    kkLidView offsets; 
    // parent elements for each SoA
    kkLidView parentElms_;
    // particle data
    AoSoA_t* aosoa_;
    // extra AoSoA copy for swapping (same size as aosoa_)
    AoSoA_t* aosoa_swap;

    //Private constructor for copy()
    CabM() : ParticleStructure<DataTypes, MemSpace>(), policy(100, 1) {}
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
  CabM<DataTypes, MemSpace>::CabM( PolicyType& p,
                                   lid_t num_elements, lid_t num_particles,
                                   kkLidView particles_per_element,
                                   kkGidView element_gids,
                                   kkLidView particle_elements, // optional
                                   MTVs particle_info, // optional
                                   MPI_Comm mpi_comm) : // optional
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
    MPI_Comm_rank(mpi_comm, &comm_rank);
    if(!comm_rank)
      printInfo( "building CabM\n");

    // build view of offsets for SoA indices within particle elements
    offsets = buildOffset(particles_per_element, num_ptcls, extra_padding, padding_start);
    // set num_soa_ from the last entry of offsets
    num_soa_ = getLastValue(offsets);
    // calculate capacity_ from num_soa_ and max size of an SoA
    capacity_ = num_soa_*AoSoA_t::vector_length;
    // initialize appropriately-sized AoSoA and copy for swapping
    aosoa_ = makeAoSoA(capacity_, num_soa_);
    aosoa_swap = makeAoSoA(capacity_, num_soa_);
    // get array of parents element indices for particles
    parentElms_ = getParentElms(num_elems, num_soa_, offsets);
    // set active mask
    setActive(particles_per_element);
    // get global ids
    if (element_gids.size() > 0) {
      createGlobalMapping(element_gids, element_to_gid, element_gid_to_lid);
    }
    // populate AoSoA with input data if given
    if (particle_elements.size() > 0 && particle_info != NULL) {
      if(!comm_rank) printInfo( "initializing CabM data\n");
      fillAoSoA(particle_elements, particle_info); // initialize data
    }
  }

  template<class DataTypes, typename MemSpace>
  CabM<DataTypes, MemSpace>::CabM(Input_T& input) :
    ParticleStructure<DataTypes, MemSpace>(input.name),
    policy(input.policy),
    element_gid_to_lid(input.ne)
  {
    num_elems = input.ne;
    num_rows = num_elems;
    num_ptcls = input.np;
    extra_padding = input.extra_padding;

    assert(num_elems == input.ppe.size());

    int comm_rank;
    MPI_Comm_rank(input.mpi_comm, &comm_rank);
    if(!comm_rank)
      printInfo( "building CabM for %s\n", name.c_str());
    
    // build view of offsets for SoA indices within particle elements
    offsets = buildOffset(input.ppe, num_ptcls, extra_padding, padding_start);
    // set num_soa_ from the last entry of offsets
    num_soa_ = getLastValue(offsets);
    // calculate capacity_ from num_soa_ and max size of an SoA
    capacity_ = num_soa_*AoSoA_t::vector_length;
    // initialize appropriately-sized AoSoA and copy for swapping
    aosoa_ = makeAoSoA(capacity_, num_soa_);
    aosoa_swap = makeAoSoA(capacity_, num_soa_);
    // get array of parents element indices for particles
    parentElms_ = getParentElms(num_elems, num_soa_, offsets);
    // set active mask
    setActive(input.ppe);
    // get global ids
    if (input.e_gids.size() > 0)
      createGlobalMapping(input.e_gids, element_to_gid, element_gid_to_lid);
    // populate AoSoA with input data if given
    if (input.particle_elms.size() > 0 && input.p_info != NULL) {
      if(!comm_rank) printInfo( "initializing CabM data\n");
      fillAoSoA(input.particle_elms, input.p_info); // initialize data
    }
  }

  template <class DataTypes, typename MemSpace>
  CabM<DataTypes, MemSpace>::~CabM() {
    delete aosoa_;
    delete aosoa_swap;
  }

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
  void CabM<DataTypes, MemSpace>::parallel_for(FunctionType& fn, std::string name) {
    if (nPtcls() == 0)
      return;

    // move function pointer to GPU (if needed)
    FunctionType* fn_d = gpuMemcpy(fn);
    kkLidView parentElms_cpy = parentElms_;
    const auto soa_len = AoSoA_t::vector_length;
    const auto mask = Cabana::slice<CM_DT::size-1>(*aosoa_); // get active mask
    Cabana::SimdPolicy<soa_len,execution_space> simd_policy(0, capacity_);
    Cabana::simd_parallel_for(simd_policy,
      KOKKOS_LAMBDA( const lid_t soa, const lid_t ptcl ) {
        const lid_t elm = parentElms_cpy(soa); // calculate element
        const lid_t particle_id = soa*soa_len + ptcl; // calculate overall index
        (*fn_d)(elm, particle_id, mask.access(soa,ptcl));
      }, name);
#ifdef PP_USE_GPU
    gpuFree(fn_d);
#endif

  }

  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::printMetrics(MPI_Comm mpi_comm) const {
    // Sum number of empty cells
    auto mask = Cabana::slice<CM_DT::size-1>(*aosoa_);
    kkLidView padded_cells("num_padded_cells",1);
    Kokkos::parallel_for("count_padding", capacity_,
      KOKKOS_LAMBDA(const lid_t ptcl_id) {
        Kokkos::atomic_fetch_add(&padded_cells(0), !mask(ptcl_id));
      });
    // Sum number of empty elements
    kkLidHostMirror offsets_host = deviceToHost(offsets);
    lid_t num_empty_elements = 0;
    if (num_soa_ == 0)
      num_empty_elements = num_elems;
    else {
      for (int i = 0; i < num_elems; i++) {
        if (i != 0 && (offsets_host(i) == offsets_host(i-1)) )
          num_empty_elements++;
      }
    }

    lid_t num_padded = getLastValue(padded_cells);

    int comm_rank;
    MPI_Comm_rank(mpi_comm, &comm_rank);
    
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
    // Empty Elements
    ptr += sprintf(ptr, "Empty Elements <Tot %%> %d %.3f%%\n", num_empty_elements,
                   num_empty_elements * 100.0 / num_elems);

    printInfo("%s\n", buffer);
  }

  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::printFormat(const char* prefix) const {
    kkGidHostMirror element_to_gid_host = deviceToHost(element_to_gid);
    kkLidHostMirror offsets_host = deviceToHost(offsets);
    kkLidHostMirror parents_host = deviceToHost(parentElms_);
    const auto soa_len = AoSoA_t::vector_length;

    kkLidView mask(Kokkos::ViewAllocateWithoutInitializing("offsets_host"), capacity_);
    auto mask_slice = Cabana::slice<CM_DT::size-1>(*aosoa_);
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
    num_chars += sprintf(ptr+num_chars,"Particle Structures CabM\n");
    num_chars += sprintf(ptr+num_chars,"Number of Elements: %d.\nNumber of SoA: %d.\nNumber of Particles: %d.", num_elems, num_soa_, num_ptcls);
    buffer[num_chars] = '\0';
    ss << buffer;

    lid_t last_soa = -1;
    lid_t last_elm = -1;
    for (int i = 0; i < capacity_; i++) {
      lid_t soa = i / soa_len;
      lid_t elm = parents_host(soa);
      if (last_soa == soa) {
        num_chars = sprintf(ptr," %d", mask_host(i));
        buffer[num_chars] = '\0';
        ss << buffer;
      }
      else {
        if (element_to_gid_host.size() > 0) {
          if (last_elm != elm)
            num_chars = sprintf(ptr,"\n  Element %2d(%2ld) | %d", elm, element_to_gid_host(elm), mask_host(i));
          else
            num_chars = sprintf(ptr,"\n                 | %d", mask_host(i));
          buffer[num_chars] = '\0';
          ss << buffer;
        }
        else {
          if (last_elm != elm)
            num_chars = sprintf(ptr,"\n  Element %2d | %d", elm, mask_host(i));
          else
            num_chars = sprintf(ptr,"\n             | %d", mask_host(i));
          buffer[num_chars] = '\0';
          ss << buffer;
        }
      }

      last_soa = soa;
      last_elm = elm;
    }
    ss << "\n";
    std::cout << ss.str();
  }

  template<class DataTypes, typename MemSpace>
  template <class MSpace>
  typename CabM<DataTypes, MemSpace>::template Mirror<MSpace>* CabM<DataTypes, MemSpace>::copy() {
    if (std::is_same<memory_space, typename MSpace::memory_space>::value) {
      printError("Copy to same memory space not supported\n");
      exit(EXIT_FAILURE);
    }
    Mirror<MSpace>* mirror_copy = new CabM<DataTypes, MSpace>();
    //Call Particle structures copy
    mirror_copy->copy(this);
    //Copy constants
    mirror_copy->num_soa_ = num_soa_;
    mirror_copy->padding_start = padding_start;
    mirror_copy->extra_padding = extra_padding;

    //copy AoSoA
    mirror_copy->aosoa_ = new typename CabM<DataTypes, MSpace>::AoSoA_t(std::string(aosoa_->label()).append("_mirror"), aosoa_->size());
    Cabana::deep_copy(*(mirror_copy->aosoa_), *aosoa_);
    //Create the swap space
    mirror_copy->aosoa_swap = new typename CabM<DataTypes, MSpace>::AoSoA_t(std::string(aosoa_swap->label()).append("_mirror"), aosoa_swap->size());
    Cabana::deep_copy(*(mirror_copy->aosoa_swap), *aosoa_swap);

    //Deep copy each view
    mirror_copy->parentElms_ = typename Mirror<MSpace>::kkLidView("mirror parentElms_",
                                                                  parentElms_.size());
    Kokkos::deep_copy(mirror_copy->parentElms_, parentElms_);
    mirror_copy->offsets = typename Mirror<MSpace>::kkLidView("mirror offsets", offsets.size());
    Kokkos::deep_copy(mirror_copy->offsets, offsets);
    mirror_copy->element_to_gid = typename Mirror<MSpace>::kkGidView("mirror element_to_gid",
                                                                     element_to_gid.size());
    Kokkos::deep_copy(mirror_copy->element_to_gid, element_to_gid);
    //Deep copy the gid mapping
    mirror_copy->element_gid_to_lid.create_copy_view(element_gid_to_lid);
    return mirror_copy;
  }

}

// Separate files with CabM member function implementations
#include "cabm_buildFns.hpp"
#include "cabm_rebuild.hpp"
#include "cabm_migrate.hpp"

#else
namespace pumipic {
  /*A dummy version of CabM when pumi-pic is built without Cabana so operations
    can compile without ifdef guards. The operations will report a message stating
    that the structure will not work.
  */
  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class CabM : public ParticleStructure<DataTypes, MemSpace> {
  public:
    template <typename MSpace> using Mirror = CabM<DataTypes, MSpace>;
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

    CabM() = delete;
    CabM(const CabM&) = delete;
    CabM& operator=(const CabM&) = delete;

    CabM( PolicyType& p,
          lid_t num_elements, lid_t num_particles,
          kkLidView particles_per_element,
          kkGidView element_gids,
          kkLidView particle_elements = kkLidView(),
          MTVs particle_info = NULL) {reportError();}
    ~CabM() {}

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

    void printMetrics(MPI_Comm mpi_comm = MPI_COMM_WORLD) const {reportError();}
    void printFormat(const char* prefix) const {reportError();}

    template <class MSpace>
    Mirror<MSpace>* copy() {reportError(); return NULL;}

  private:
    void reportError() const {printError( "pumi-pic was built "
                                      "without Cabana so the CabM structure "
                                      "can not be used\n");}
  };
}
#endif // PP_ENABLE_CAB
