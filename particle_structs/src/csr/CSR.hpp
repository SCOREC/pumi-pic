#pragma once

#include <particle_structs.hpp>
#include <ppTiming.hpp>
#include <sstream>

namespace ps = particle_structs;

namespace pumipic {

  void enable_prebarrier();
  double prebarrier();

  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class CSR : public ParticleStructure<DataTypes, MemSpace> {
  public:
    using typename ParticleStructure<DataTypes, MemSpace>::execution_space;
    using typename ParticleStructure<DataTypes, MemSpace>::memory_space;
    using typename ParticleStructure<DataTypes, MemSpace>::device_type;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::MTVs;

    typedef Kokkos::TeamPolicy<execution_space> PolicyType;
    typedef Kokkos::UnorderedMap<gid_t, lid_t, device_type> GID_Mapping;

    CSR() = delete;
    CSR(const CSR&) = delete;
    CSR& operator=(const CSR&) = delete;

    CSR(PolicyType& p,
        lid_t num_elements, lid_t num_particles,
        kkLidView particles_per_element,
        kkGidView element_gids,
        kkLidView particle_elements = kkLidView(),
        MTVs particle_info = NULL);
    ~CSR();

    //Functions from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::nElems;
    using ParticleStructure<DataTypes, MemSpace>::nPtcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity;
    using ParticleStructure<DataTypes, MemSpace>::numRows;
    using ParticleStructure<DataTypes, MemSpace>::copy;

    void migrate(kkLidView new_element, kkLidView new_process,
                 Distributor<MemSpace> dist = Distributor<MemSpace>(),
                 kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particle_info = NULL);

    void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particles = NULL);

    template <typename FunctionType>
    void parallel_for(FunctionType& fn, std::string name="");

    void printMetrics() const;
    void printFormat(const char* prefix) const;

    // Do not call these functions:
    void createGlobalMapping(kkGidView element_gids, kkGidView& lid_to_gid, GID_Mapping& gid_to_lid);
    void initCsrData(kkLidView particle_elements, MTVs particle_info);

  private:
    // The User defined Kokkos policy
    PolicyType policy;

    // Variables from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::num_elems;
    using ParticleStructure<DataTypes, MemSpace>::num_ptcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity_;
    using ParticleStructure<DataTypes, MemSpace>::num_rows;
    using ParticleStructure<DataTypes, MemSpace>::ptcl_data;
    using ParticleStructure<DataTypes, MemSpace>::num_types;

    // Data types for keeping track of global IDs
    kkGidView element_to_gid;
    GID_Mapping element_gid_to_lid;
    // Offsets array into CSR
    kkLidView offsets;

    //Swap memory
    MTVs ptcl_data_swap;
    lid_t swap_capacity_;
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
  CSR<DataTypes, MemSpace>::CSR(PolicyType& p,
                                lid_t num_elements, lid_t num_particles,
                                kkLidView particles_per_element,
                                kkGidView element_gids,      // optional
                                kkLidView particle_elements, // optional
                                MTVs particle_info) :        // optional
      ParticleStructure<DataTypes, MemSpace>(),
      policy(p),
      element_gid_to_lid(num_elements)
  {
    Kokkos::Profiling::pushRegion("csr_construction");
    num_elems = num_elements;
    num_rows = num_elems;
    num_ptcls = num_particles;
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if(!comm_rank)
      fprintf(stderr, "Building CSR\n");

    // SS1 allocate the offsets array and use an exclusive_scan (aka prefix sum)
    // to fill the entries of the offsets array.
    // see pumi-pic/support/SupportKK.h for the exclusive_scan helper function
    offsets = kkLidView(Kokkos::ViewAllocateWithoutInitializing("offsets"), num_elems+1);
    Kokkos::resize(particles_per_element, particles_per_element.size()+1);
    exclusive_scan(particles_per_element, offsets);

    // get global ids
    if (element_gids.size() > 0) {
      createGlobalMapping(element_gids, element_to_gid, element_gid_to_lid);
    }

    // SS2 set the 'capacity_' of the CSR storage from the last entry of offsets
    // pumi-pic/support/SupportKK.h has a helper function for this
    capacity_ = getLastValue(offsets);
    // allocate storage for user particle data
    CreateViews<device_type, DataTypes>(ptcl_data, capacity_);
    CreateViews<device_type, DataTypes>(ptcl_data_swap,capacity_ * 1.05);
    swap_capacity_ = capacity_*1.05;

    // If particle info is provided then enter the information
    lid_t given_particles = particle_elements.size();
    if (given_particles > 0 && particle_info != NULL) {
      if(!comm_rank) fprintf(stderr, "initializing CSR data\n");
      initCsrData(particle_elements, particle_info);
    }

    Kokkos::Profiling::popRegion();
  }

  template <class DataTypes, typename MemSpace>
  CSR<DataTypes, MemSpace>::~CSR() {
    destroyViews<DataTypes, memory_space>(ptcl_data);
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
  void CSR<DataTypes, MemSpace>::parallel_for(FunctionType& fn, std::string name) {
    if (nPtcls() == 0)
      return;
    FunctionType* fn_d;
#ifdef PP_USE_CUDA
    cudaMalloc(&fn_d, sizeof(FunctionType));
    cudaMemcpy(fn_d,&fn, sizeof(FunctionType), cudaMemcpyHostToDevice);
#else
    fn_d = &fn;
#endif
    const lid_t league_size = num_elems;
    const lid_t team_size = policy.team_size();
    const PolicyType policy(league_size, team_size);
    auto offsets_cpy = offsets;
    const lid_t mask = 1; // all particles are active
    Kokkos::parallel_for(name, policy,
        KOKKOS_LAMBDA(const typename PolicyType::member_type& thread) {
        const lid_t elm = thread.league_rank();
        const lid_t start = offsets_cpy(elm);
        const lid_t end = offsets_cpy(elm+1);
        const lid_t numPtcls = end-start;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, numPtcls), [=] (lid_t& j) {
          const lid_t particle_id = start+j;
          (*fn_d)(elm, particle_id, mask);
        });
    });
#ifdef PP_USE_CUDA
    cudaFree(fn_d);
#endif
  }

  template <class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::printMetrics() const {
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    char buffer[1000];
    char* ptr = buffer;

    // Header
    ptr += sprintf(ptr, "Metrics (Rank %d)\n", comm_rank);
    // Sizes
    ptr += sprintf(ptr, "Number of Elements %d, Number of Particles %d, Capacity %d\n",
                   num_elems, num_ptcls, capacity_);
    
    printf("%s\n", buffer);
  }

  template <class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::printFormat(const char* prefix) const {
    kkGidHostMirror element_to_gid_host = deviceToHost(element_to_gid);
    kkLidHostMirror offsets_host = deviceToHost(offsets);

    std::stringstream ss;
    char buffer[1000];
    char* ptr = buffer;
    int num_chars;

    num_chars = sprintf(ptr, "%s\n", prefix);
    num_chars += sprintf(ptr+num_chars,"Particle Structures CSR\n");
    num_chars += sprintf(ptr+num_chars,"Number of Elements: %d.\nNumber of Particles: %d.", num_elems, num_ptcls);
    buffer[num_chars] = '\0';
    ss << buffer;

    for (int i = 1; i < offsets_host.size(); i++) {
      if ( offsets_host[i] != offsets_host[i-1] ) {
        if (element_to_gid_host.size() > 0)
          num_chars = sprintf(ptr,"\n  Element %2d(%2d) |", i-1, element_to_gid_host(i-1));
        else
          num_chars = sprintf(ptr,"\n  Element %2d |", i-1);
        buffer[num_chars] = '\0';
        ss << buffer;

        for (int j = offsets_host[i-1]; j < offsets_host[i]; j++) {
          num_chars = sprintf(ptr," 1");
          buffer[num_chars] = '\0';
          ss << buffer;
        }
      }
    }
    ss << "\n";
    std::cout << ss.str();
  }

} // end namespace pumipic

#include "CSR_buildFns.hpp"
#include "CSR_rebuild.hpp"
#include "CSR_migrate.hpp"
