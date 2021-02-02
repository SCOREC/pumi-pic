#pragma once

#include <particle_structs.hpp>
#include <Cabana_Core.hpp>

namespace {

template <typename T, typename... Types>
struct AppendMT;

//Append type to the end
template <typename T, typename... Types>
struct AppendMT<T, particle_structs::MemberTypes<Types...> > {
  static constexpr int size = 1 + Cabana::MemberTypes<Types...>::size;
  using type = Cabana::MemberTypes<Types..., T>; //Put T before Types... to put at beginning
};


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

    using host_space = Kokkos::HostSpace;
    typedef Kokkos::TeamPolicy<execution_space> PolicyType;

    //from https://github.com/SCOREC/Cabana/blob/53ad18a030f19e0956fd0cab77f62a9670f31941/core/src/CabanaM.hpp#L18-L19
    using CM_DT = AppendMT<int,DataTypes>;
    using AoSoA_t = Cabana::AoSoA<typename CM_DT::type,MemSpace>;

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
      const lid_t num_elements = particles_per_element.size();
      Kokkos::View<lid_t,host_space> offsets_h("offsets_host", num_elements+1);
      // elem at i owns SoA offsets[i+1] - offsets[i]
      auto soa_len = AoSoA_t::vector_length;
      offsets_h(0) = 0;
      for ( int i=0; i<num_elements; i++ ) {
        const auto SoA_count = (particles_per_element(i)/soa_len) + 1;
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
      Kokkos::View<lid_t,host_space> elms_h("parentElms_host", num_soa);
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
      
      // calculate SoA and ptcl in SoA indices for next loop
      kkLidView soa_indices("soa_indices", num_ptcls);
      kkLidView soa_ptcl_indices("soa_ptcl_indices", num_ptcls);
      Kokkos::parallel_for("soa_and_ptcl", num_ptcls,
        KOKKOS_LAMBDA(const lid_t ptcl_id) {
          soa_indices(ptcl_id) = offsets(particle_elements(ptcl_id)) + (particle_indices(ptcl_id)/soa_len);
          soa_ptcl_indices(ptcl_id) = particle_indices(ptcl_id)%soa_len;
        });

      // add data from particle_info to correct position in aosoa_
      const lid_t league_size = num_types;
      const lid_t team_size = AoSoA_t::vector_length;
      const lid_t num_ptcls_cpy = num_ptcls;
      const auto aosoa_copy = aosoa_;
      const PolicyType policy(league_size, team_size);
      Kokkos::parallel_for("fill_aosoa", policy,
          KOKKOS_LAMBDA(const typename PolicyType::member_type& thread) {
          const lid_t member_type_ind = thread.league_rank();
          const auto member_slice = Cabana::slice<member_type_ind>(aosoa_copy);
          Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, num_ptcls_cpy), [=] (lid_t& ptcl_id) {
            member_slice.access( soa_indices(ptcl_id), soa_ptcl_indices(ptcl_id) ) = (*particle_info[member_type_ind])(ptcl_id);
          });
      });
      aosoa_ = aosoa_copy;
    }

    //---Attention User---  Do **not** call this function!
    /**
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

  template <class DataTypes, typename MemSpace>
  CabM<DataTypes, MemSpace>::CabM( PolicyType& p,
                                   lid_t num_elements, lid_t num_particles,
                                   kkLidView particles_per_element,
                                   kkGidView element_gids,      // optional
                                   kkLidView particle_elements, // optional
                                   MTVs particle_info) :        // optional
    ParticleStructure<DataTypes, MemSpace>(),
    policy(p)
  {
    assert(num_elements == particles_per_element.size());
    num_elems = num_elements;
    num_rows = num_elems;
    offsets = buildOffset(particles_per_element); // build offset array
    num_soa_ = getLastValue(offsets);
    capacity_ = num_soa_*AoSoA_t::vector_length;
    aosoa_ = makeAoSoA(capacity_, num_soa_); // initialize AoSoA
    // get array of parents element indices for particles
    parentElms_ = getParentElms(num_elements, num_soa_, offsets);
    // set active mask
    setActive(aosoa_, particles_per_element, parentElms_, offsets);
    initCabMData(particle_elements, particle_info); // initialize data
  }

  template <class DataTypes, typename MemSpace>
  CabM<DataTypes, MemSpace>::~CabM() {
    fprintf(stderr, "[WARNING] CabM deconstructor not implemented\n");
  }

  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::migrate(kkLidView new_element, kkLidView new_process,
                                         Distributor<MemSpace> dist,
                                         kkLidView new_particle_elements,
                                         MTVs new_particle_info) {
    fprintf(stderr, "[WARNING] CabM migrate(...) not implemented\n");
  }
  
  /// @todo edit to add new particles
  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::rebuild(kkLidView new_element,
                                         kkLidView new_particle_elements,
                                         MTVs new_particles) {
    const auto soa_len = AoSoA_t::vector_length;
    Kokkos::View<int*> elmDegree("elmDegree", num_elems);
    Kokkos::View<int*> elmOffsets("elmOffsets", num_elems);
    const auto activeSliceIdx = aosoa_.number_of_members-1;
    auto active = Cabana::slice<activeSliceIdx>(aosoa_);
    kkLidView elmDegree_d("elmDegree_device", elmDegree.size());
    hostToDevice(elmDegree_d,elmDegree.data());
    //first loop to count number of particles per new element (atomic)
    auto atomic = KOKKOS_LAMBDA(const int& soa,const int& tuple){
      if (active.access(soa,tuple) == 1){
        auto parent = new_element((soa*soa_len)+tuple);
        Kokkos::atomic_increment<int>(&elmDegree_d(parent));
      }
    };
    Cabana::SimdPolicy<soa_len,execution_space> simd_policy( 0, capacity_ );
    Cabana::simd_parallel_for( simd_policy, atomic, "atomic" );
    auto elmDegree_h = Kokkos::create_mirror_view_and_copy(host_space(), elmDegree_d);

    //prepare a new aosoa to store the shuffled particles
    auto newOffset = buildOffset(elmDegree_d);
    const auto newNumSoa = newOffset[num_elems];
    const auto newCapacity = newNumSoa*soa_len;
    auto newAosoa = makeAoSoA(newCapacity, newNumSoa);
    //assign the particles from the current aosoa to the newAosoa 
    Kokkos::View<int*,host_space> newOffset_h("newOffset_host",num_elems+1);
    for (int i=0; i<=num_elems; i++)
      newOffset_h(i) = newOffset[i];
    auto newOffset_d = Kokkos::create_mirror_view_and_copy(memory_space(), newOffset_h);
    Kokkos::View<int*, host_space> elmPtclCounter_h("elmPtclCounter_device", num_elems); 
    auto elmPtclCounter_d = Kokkos::create_mirror_view_and_copy(memory_space(), elmPtclCounter_h);
    auto newActive = Cabana::slice<activeSliceIdx>(newAosoa);
    auto aosoa_cpy = aosoa_; // copy of member variable aosoa_ (Kokkos doesn't like member variables)
    auto copyPtcls = KOKKOS_LAMBDA(const int& soa,const int& tuple){
        if (active.access(soa,tuple) == 1){
          //Compute the destSoa based on the destParent and an array of
          // counters for each destParent tracking which particle is the next
          // free position. Use atomic fetch and incriment with the
          // 'elmPtclCounter_d' array.
          auto destParent = new_element(soa*soa_len + tuple);
          auto occupiedTuples = Kokkos::atomic_fetch_add<int>(&elmPtclCounter_d(destParent), 1);
          auto oldTuple = aosoa_cpy.getTuple(soa*soa_len + tuple);
          auto firstSoa = newOffset_d(destParent);
          // use newOffset_d to figure out which soa is the first for destParent
          newAosoa.setTuple(firstSoa*soa_len + occupiedTuples, oldTuple);
        }
      };
      Cabana::simd_parallel_for(simd_policy, copyPtcls, "copyPtcls");
      //destroy the old aosoa and use the new one in the CabanaM object
      aosoa_ = newAosoa;
      setActive(aosoa_, elmDegree_d, parentElms_, offsets);
  }

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
    const auto mask = Cabana::slice<activeSliceIdx>(aosoa_); // check which particles are active
    Cabana::SimdPolicy<soa_len,execution_space> simd_policy( 0, capacity_);
    Cabana::simd_parallel_for(simd_policy,
      KOKKOS_LAMBDA( const int soa, const int ptcl ) {
        const lid_t elm = parentElms_cpy(soa);
        const lid_t particle_id = soa*soa_len + ptcl;
        (*fn_d)(elm, particle_id, mask.access(soa,ptcl));
      }, "parallel_for");
  }

  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::printMetrics() const {
    fprintf(stderr, "CabM capacity %d\n", capacity_);
    fprintf(stderr, "CabM num ptcls %d\n", num_ptcls);
    fprintf(stderr, "CabM num elements %d\n", num_elems);
  }
}
