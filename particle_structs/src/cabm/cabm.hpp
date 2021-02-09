#pragma once

#include <Cabana_Core.hpp>
#include <particle_structs.hpp>
#include "cabm_support.hpp"

namespace {


// class to append member types
template <typename T, typename... Types>
struct MemberTypesAppend;

//Append type to the end
template <typename T, typename... Types>
struct MemberTypesAppend<T, Cabana::MemberTypes<Types...> > {
  static constexpr int size = 1 + Cabana::MemberTypes<Types...>::size;
  using type = Cabana::MemberTypes<Types..., T>; //Put T before Types... to put at beginning
};

}//end anonymous

namespace pumipic {

  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class CabM : public ParticleStructure<DataTypes, MemSpace> {
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

    //from https://github.com/SCOREC/Cabana/blob/53ad18a030f19e0956fd0cab77f62a9670f31941/core/src/CabanaM.hpp#L18-L19
    using CM_DT = CM_DTInt<DataTypes>;
    using AoSoA_t = Cabana::AoSoA<CM_DT,device_type>;

    CabM() = delete;
    CabM(const CabM&) = delete;
    CabM& operator=(const CabM&) = delete;

    CabM( PolicyType& p,
          lid_t num_elements, lid_t num_particles,
          kkLidView particles_per_element,
          kkGidView element_gids,
          kkLidView particle_elements = kkLidView(),
          MTVs particle_info = NULL);
    ~CabM();

    //Functions from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::nElems;
    using ParticleStructure<DataTypes, MemSpace>::nPtcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity;
    using ParticleStructure<DataTypes, MemSpace>::numRows;

    template <std::size_t N>
    Slice<N> get() {
      //TODO: write this function to return the pumipic::Slice for the Nth type given a Cabana::Slice
      return Slice<N>(Cabana::slice<N, AoSoA_t>(aosoa_, "get<>()"));
    }

    void migrate(kkLidView new_element, kkLidView new_process,
                 Distributor<MemSpace> dist = Distributor<MemSpace>(),
                 kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particle_info = NULL);

    void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particles = NULL);

    template <typename FunctionType>
    void parallel_for(FunctionType& fn, std::string s="");

    void printMetrics() const;


    /**
     * helper function: Builds the offset array for the CSR structure
     * @param[in] particles_per_element View representing the number of active elements in each SoA
     * @return offset array (each element is the first index of each SoA block)
    */
    kkLidView buildOffset(const kkLidView particles_per_element) {
      auto particles_per_element_h = Kokkos::create_mirror_view_and_copy(host_space(), particles_per_element);
      Kokkos::View<lid_t*,host_space> offsets_h("offsets_host", num_elems+1);
      // elem at i owns SoA offsets[i+1] - offsets[i]
      auto soa_len = AoSoA_t::vector_length;
      offsets_h(0) = 0;
      for ( int i=0; i<num_elems; i++ ) {
        const auto SoA_count = (particles_per_element_h(i)/soa_len) + 1;
        offsets_h(i+1) = SoA_count + offsets_h(i);
      }
      kkLidView offsets_d("offsets_device", offsets_h.size());
      hostToDevice(offsets_d, offsets_h.data());
      return offsets_d;
    }

    /**
     * helper function: initialize an AoSoA (including hidden active SoA)
     * @param[in] capacity maximum capacity (number of particles) of the AoSoA to be created
     * @param[in] num_soa total number of SoAs (can be greater than elem_count if
     * any element of particles_per_element is vector_length)
     * @return AoSoA of max capacity, capacity, and total number of SoAs, numSoa
    */
    AoSoA_t makeAoSoA(const lid_t capacity, const lid_t num_soa) {
      auto aosoa = AoSoA_t();
      aosoa.resize(capacity);
      assert(num_soa == aosoa.numSoA());
      return aosoa;
    }

    /**
     * helper function: builds the parent view for tracking particle position
     * @param[in] num_elements total number of element SoAs in AoSoA
     * @param[in] num_soa total number of SoAs (can be greater than elem_count if
     * any element of deg is _vector_length)
     * @param[in] offsets offset array for AoSoA, built by buildOffset
     * @return parent array, each element is an int representing the parent element each SoA resides in
    */
    kkLidView getParentElms( const lid_t num_elements, const lid_t num_soa, const kkLidView offsets ) {
      Kokkos::View<lid_t*,host_space> elms_h("parentElms_host", num_soa);
      kkLidHostMirror offsets_h = create_mirror_view(offsets);
      Kokkos::deep_copy( offsets_h, offsets );
      for ( int elm=0; elm<num_elements; elm++ )
        for ( int soa=offsets_h(elm); soa<offsets_h(elm+1); soa++)
          elms_h(soa)=elm;
      kkLidView elms_d("elements_device", elms_h.size());
      hostToDevice(elms_d, elms_h.data());
      return elms_d;
    }

    /**
     * helper function: initializes last SoAs in AoSoA as active mask
     * where 1 denotes an active particle and 0 denotes an inactive particle.
     * @param[out] aosoa the AoSoA to be edited
     * @param[in] particles_per_element pointer to an array of ints, representing the number of active elements in each SoA
     * @param[in] parentElms parent view for AoSoA, built by getParentElms
     * @param[in] offsets offset array for AoSoA, built by buildOffset
    */
    void setActive(AoSoA_t &aosoa, const kkLidView particles_per_element,
        const kkLidView parentElms, const kkLidView offsets) {

      const lid_t num_elements = particles_per_element.size();
      const auto soa_len = AoSoA_t::vector_length;

      const auto activeSliceIdx = aosoa.number_of_members-1;
      auto active = Cabana::slice<activeSliceIdx>(aosoa);
      Cabana::SimdPolicy<soa_len,execution_space> simd_policy(0, capacity_);
      Cabana::simd_parallel_for(simd_policy,
        KOKKOS_LAMBDA( const int soa, const int ptcl ) {
          const auto elm = parentElms(soa);
          const auto num_soa = offsets(elm+1)-offsets(elm);
          const auto last_soa = offsets(elm+1)-1;
          const auto elm_ppe = particles_per_element(elm);
          const auto last_soa_ppe = soa_len - ((num_soa * soa_len) - elm_ppe);
          int isActive = 0;
          if (soa < last_soa) {
            isActive = 1;
          }
          if (soa == last_soa && ptcl < last_soa_ppe) {
            isActive = 1;
          }
          active.access(soa,ptcl) = isActive;
        }, "set_active");
    }

    /**
     * helper function: fills aosoa_ with particle data
     * @param[in] particle_indices - particle_elements[i] contains the index of particle i
     *                          in its parent element
     * @param[in] particle_elements - particle_elements[i] contains the id (index)
     *                          of the parent element * of particle i
     * @param[in] particle_info - 'member type views' containing the user's data to be
     *                      associated with each particle
    */
    void fillAoSoA(kkLidView particle_indices, kkLidView particle_elements, MTVs particle_info) {
      const auto soa_len = AoSoA_t::vector_length;

      // calculate SoA and ptcl in SoA indices for next CopyMTVsToAoSoA
      kkLidView soa_indices("soa_indices", num_ptcls);
      kkLidView soa_ptcl_indices("soa_ptcl_indices", num_ptcls);
      auto offsets_cpy = offsets; // copy of offsets since GPUs don't like member variables
      Kokkos::parallel_for("soa_and_ptcl", num_ptcls,
        KOKKOS_LAMBDA(const lid_t ptcl_id) {
          soa_indices(ptcl_id) = offsets_cpy(particle_elements(ptcl_id))
            + (particle_indices(ptcl_id)/soa_len);
          soa_ptcl_indices(ptcl_id) = particle_indices(ptcl_id)%soa_len;
        });
      CopyMTVsToAoSoA<device_type, DataTypes>(aosoa_, particle_info, soa_indices,
        soa_ptcl_indices);
    }

    //---Attention User---  Do **not** call this function!
    /**
     * helper function: find indices for particle_info for fillAoSoA
     * @param[in] particle_elements - particle_elements[i] contains the id (index)
     *                          of the parent element * of particle i
     * @param[in] particle_info - 'member type views' containing the user's data to be
     *                      associated with each particle
    */
    void initCabMData(kkLidView particle_elements, MTVs particle_info) {
      assert(particle_elements.size() == num_ptcls);

      // create a pointer to the offsets array that we can access in a kokkos parallel_for
      auto offset_copy = offsets;
      kkLidView particle_indices("particle_indices", num_ptcls);
      // View for tracking particle index in elements
      kkLidView ptcl_elm_indices("ptcl_elm_indices", num_elems);
      Kokkos::parallel_for("fill_zeros", num_elems,
        KOKKOS_LAMBDA(const int& i) {
          ptcl_elm_indices(i) = 0;
        });

      // atomic_fetch_add to increment from the beginning of each element
      Kokkos::parallel_for("fill_ptcl_indices", num_ptcls,
        KOKKOS_LAMBDA(const lid_t ptcl_id) {
          particle_indices(ptcl_id) = Kokkos::atomic_fetch_add(&ptcl_elm_indices(particle_elements(ptcl_id)),1);
        });

      fillAoSoA(particle_indices, particle_elements, particle_info);
    }

  private:
    //The User defined Kokkos policy
    PolicyType policy;

    //Variables from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::num_elems;
    using ParticleStructure<DataTypes, MemSpace>::num_ptcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity_;
    using ParticleStructure<DataTypes, MemSpace>::num_rows;
    using ParticleStructure<DataTypes, MemSpace>::ptcl_data;
    using ParticleStructure<DataTypes, MemSpace>::num_types;

    lid_t num_soa_; // number of SoA
    kkLidView offsets; // Offsets array into CabM
    kkLidView parentElms_; // Parent elements for each SoA
    AoSoA_t aosoa_;
  };

  /**
   * Constructor
   * @param[in] p
   * @param[in] num_elements number of elements
   * @param[in] num_particles number of particles
   * @param[in] particle_per_element view of ints, representing number of particles
   *    in each element
   * @param[in] element_gids
   * @param[in] particle_elements view of ints, representing which elements
   *    particle reside in (optional)
   * @param[in] particle_info array of views filled with particle data (optional)
  */
  template <class DataTypes, typename MemSpace>
  CabM<DataTypes, MemSpace>::CabM( PolicyType& p,
                                   lid_t num_elements, lid_t num_particles,
                                   kkLidView particles_per_element,
                                   kkGidView element_gids,
                                   kkLidView particle_elements,
                                   MTVs particle_info) :        // optional
    ParticleStructure<DataTypes, MemSpace>(),
    policy(p)
  {
    assert(num_elements == particles_per_element.size());
    num_elems = num_elements;
    num_rows = num_elems;
    num_ptcls = num_particles;
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if(!comm_rank)
      fprintf(stderr, "building CabM\n");

    // build view of offsets for SoA indices within particle elements
    offsets = buildOffset(particles_per_element);
    // set num_soa_ from the last entry of offsets
    num_soa_ = getLastValue(offsets);
    // calculate capacity_ from num_soa_ and max size of an SoA
    capacity_ = num_soa_*AoSoA_t::vector_length;

    // initialize appropriately-sized AoSoA
    aosoa_ = makeAoSoA(capacity_, num_soa_);
    // get array of parents element indices for particles
    parentElms_ = getParentElms(num_elems, num_soa_, offsets);
    // set active mask
    setActive(aosoa_, particles_per_element, parentElms_, offsets);

    /// @todo add usage of element_gids

    // populate AoSoA with input data if given
    lid_t given_particles = particle_elements.size();
    if (given_particles > 0 && particle_info != NULL) {
      if(!comm_rank) fprintf(stderr, "initializing CabM data\n");
      initCabMData(particle_elements, particle_info); // initialize data
    }
  }

  template <class DataTypes, typename MemSpace>
  CabM<DataTypes, MemSpace>::~CabM() {
    fprintf(stderr, "[WARNING] CabM deconstructor not implemented\n");
  }

  /**
   *
   * @param[in] new_element
   * @param[in] new_process
   * @param[in] dist
   * @param[in] new_particle_elements
   * @param[in] new_particle_info
  */
  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::migrate(kkLidView new_element, kkLidView new_process,
                                         Distributor<MemSpace> dist,
                                         kkLidView new_particle_elements,
                                         MTVs new_particle_info) {
    /// @todo implement migrate
    fprintf(stderr, "[WARNING] CabM migrate(...) not implemented\n");
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
  void CabM<DataTypes, MemSpace>::parallel_for(FunctionType& fn, std::string s) {
    if (nPtcls() == 0)
      return;

    // move function pointer to GPU (if needed)
    FunctionType* fn_d;
    #ifdef PP_USE_CUDA
        cudaMalloc(&fn_d, sizeof(FunctionType));
        cudaMemcpy(fn_d,&fn, sizeof(FunctionType), cudaMemcpyHostToDevice);
    #else
        fn_d = &fn;
    #endif
    auto parentElms_cpy = parentElms_;
    const auto soa_len = AoSoA_t::vector_length;
    const auto activeSliceIdx = aosoa_.number_of_members-1;
    const auto mask = Cabana::slice<activeSliceIdx>(aosoa_); // get active mask
    Cabana::SimdPolicy<soa_len,execution_space> simd_policy( 0, capacity_);
    Cabana::simd_parallel_for(simd_policy,
      KOKKOS_LAMBDA( const int soa, const int ptcl ) {
        const lid_t elm = parentElms_cpy(soa); // calculate element
        const lid_t particle_id = soa*soa_len + ptcl; // calculate overall index
        (*fn_d)(elm, particle_id, mask.access(soa,ptcl));
      }, "parallel_for");
  }

  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::printMetrics() const {
    fprintf(stderr, "CabM capacity %d\n", capacity_);
    fprintf(stderr, "CabM num ptcls %d\n", num_ptcls);
    fprintf(stderr, "CabM num elements %d\n", num_elems);
    fprintf(stderr, "CabM num SoA %d\n", num_soa_);
  }
}

#include "cabm_rebuild.hpp" // rebuild member function in its own file