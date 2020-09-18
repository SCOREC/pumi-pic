#pragma once

#include <particle_structure.hpp>
#include <particle_structs.hpp>
namespace ps = particle_structs;

namespace {
  // print the contents of a view for debugging
  template <typename ppView>
  void printView(ppView v){
      //printf("view: %s\n", v.label().c_str());
      Kokkos::parallel_for("print_view",
          v.size(),
          KOKKOS_LAMBDA (const int& i) {
            printf("%d %d\n", i, v[i]);
          });
  }

  // count the number of elements with particles - remove this function?
  template <typename ppView>
  int countElmsWithPtcls(ppView particles_per_element){
    int count;
    Kokkos::parallel_reduce("count_elements_with_particles",
        particles_per_element.size(),
        KOKKOS_LAMBDA (const int& i, int& lsum ) {
          //SS0 use a kokkos parallel_reduce to count the number of elements
          //that have at least one particle
	  if(particles_per_element( i ) > 0) {
            lsum += 1;
	  }
        }, count);
    return count;
  }

  //helper function for rebuild to determine how much space to allocate
  template <typename ppView>
  int countParticlesOnProcess(ppView particle_elements){
    int count = 0;
    Kokkos::parallel_reduce("particle on process",
        particle_elements.size(), KOKKOS_LAMBDA (const int& i, int& lsum){
      if(particle_elements(i) > -1){
        lsum += 1;
      }
    }, count);
    return count;
  }
}


namespace pumipic {

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

    kkLidView getOffsets() { return offsets; }
    MTVs getPtcl_data() { return ptcl_data; }

    void migrate(kkLidView new_element, kkLidView new_process,
                 Distributor<MemSpace> dist = Distributor<MemSpace>(),
                 kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particle_info = NULL);

    void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particles = NULL);

    template <typename FunctionType>
    void parallel_for(FunctionType& fn, std::string name="");

    void printMetrics() const;

    //---Attention User---  Do **not** call this function! {
    /**
     * (in) particle_elements - particle_elements[i] contains the id (index) 
     *                          of the parent element * of particle i
     * (in) particle_info - 'member type views' containing the user's data to be
     *                      associated with each particle
     */
    void initCsrData(kkLidView particle_elements, MTVs particle_info) {
      //Create the 'particle_indices' array.  particle_indices[i] stores the 
      //location in the 'ptcl_data' where  particle i is stored.  Use the
      //CSR offsets array and an atomic_fetch_add to compute these entries.
      lid_t given_particles = particle_elements.size();
      assert(given_particles == num_ptcls);

      // create a pointer to the offsets array that we can access in a kokkos parallel_for
      auto offset_cpy = offsets; 
      kkLidView particle_indices("particle_indices", num_ptcls);
      //SS3 insert code to set the entries of particle_indices>
      kkLidView row_indices("row indces", num_elems+1);
      Kokkos::deep_copy(row_indices, offset_cpy);

      //atomic_fetch_add to increment from the beginning of each element
      //when filling (offset[element] is start of element)
      Kokkos::parallel_for("particle indices", given_particles, KOKKOS_LAMBDA(const int& i){
        particle_indices(i) = Kokkos::atomic_fetch_add(&row_indices(particle_elements(i)), 1);
      });

      //populate ptcl_data with input data and particle_indices mapping
      CopyViewsToViews<kkLidView, DataTypes>(ptcl_data, particle_info, particle_indices);
    }
    // } ... or else!

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
  
    //Offsets array into CSR
    kkLidView offsets;
  };


  template <class DataTypes, typename MemSpace>
  CSR<DataTypes, MemSpace>::CSR(PolicyType& p,
                                lid_t num_elements, lid_t num_particles,
                                kkLidView particles_per_element,
                                kkGidView element_gids,      //optional
                                kkLidView particle_elements, //optional
                                MTVs particle_info) :        //optional
      ParticleStructure<DataTypes, MemSpace>(), 
      policy(p)
  {
    Kokkos::Profiling::pushRegion("csr_construction");
    num_elems = num_elements;
    num_rows = num_elems;
    num_ptcls = num_particles;
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if(!comm_rank)
      fprintf(stderr, "Building CSR\n");

    //SS1 allocate the offsets array and use an exclusive_scan (aka prefix sum)
    //to fill the entries of the offsets array.
    //see pumi-pic/support/SupportKK.h for the exclusive_scan helper function
    offsets = kkLidView("offsets", num_elems+1); 
    Kokkos::resize(particles_per_element, particles_per_element.size()+1);
    exclusive_scan(particles_per_element, offsets);

    //SS2 set the 'capacity_' of the CSR storage from the last entry of offsets
    //pumi-pic/support/SupportKK.h has a helper function for this
    capacity_ = getLastValue(offsets);
    //allocate storage for user particle data
    CreateViews<device_type, DataTypes>(ptcl_data, capacity_); 

    //If particle info is provided then enter the information
    lid_t given_particles = particle_elements.size();
    if (given_particles > 0 && particle_info != NULL) {
      if(!comm_rank) fprintf(stderr, "initializing CSR data\n");
      initCsrData(particle_elements, particle_info);
    }

    if(!comm_rank)
      fprintf(stderr, "Building CSR done\n");
    Kokkos::Profiling::popRegion();
  }

  template <class DataTypes, typename MemSpace>
  CSR<DataTypes, MemSpace>::~CSR() {
    destroyViews<DataTypes, memory_space>(ptcl_data);
  }

  template <class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::migrate(kkLidView new_element, kkLidView new_process,
                                         Distributor<MemSpace> dist,
                                         kkLidView new_particle_elements,
                                         MTVs new_particle_info) {
    fprintf(stderr, "[WARNING] CSR migrate(...) not implemented\n");
  }

  template <class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::rebuild(kkLidView new_element,
                                         kkLidView new_particle_elements,
                                         MTVs new_particles) {
    //new_element - integers corresponding to which mesh element each particle
    //is now assigned to, -1 if no longer on current process
    //
    //Do new_element's indices correspond with current location in CSR representation 
    //or particle ID?
    //
    //new_particle_elements - integers corresponding to which mesh element
    //particles new to the process exist in (-1 should throw error)
    //new_particles - MTV data associated with each of the particles added
    //to the process
    //Assume these index respectively for new particles
    
    //Gameplan - count how many entries are > -1 first to determine space to allocate
    //           'merge' existing and new data for input to CSR constructor
    //           construct new CSR based on that input 

    lid_t particles_on_process = countParticlesOnProcess(new_element) + 
                                 countParticlesOnProcess(new_particle_elements);
    capacity_ = particles_on_process;

    
    //fresh filling of particles_per_element
    kkLidView particles_per_element = kkLidView("particlesPerElement", num_elems+1);
    Kokkos::parallel_for("fill particlesPerElement1", new_element.size(),
        KOKKOS_LAMBDA(const int& i){
          if(new_element[i] > -1)
            Kokkos::atomic_increment(particles_per_element[new_element[i]]);
        });
    Kokkos::parallel_for("fill particlesPerElement2", new_particle_elements.size(),
        KOKKOS_LAMBDA(const int& i){
          if(new_particle_elements[i] > -1)
            Kokkos::atomic_increment(particles_per_element[new_partilce_elements[i]]);
        });

    Kokkos::fence();

    //refill offset here 
    offests = kkLidView("offsets", num_elems+1);
    exclusive_scan(particles_per_element, offsets);
    assert(capacity_ == getLastValue(offsets)); 

    //need to combine particle->element mapping and particledata to new structures for init
    //remove all off process particles in the process
    kkLidView particle_elements = kkLidView("particle elements", particles_on_process); 
    //for now assuming all particles remain on process (no -1 elements)

    CreateViews<device_type, DataTypes>(particle_info, particles_on_process); 
    Kokkos::parallel_for("fill particlesPerElement1", new_element.size(),
        KOKKOS_LAMBDA(const int& i){
          particle_elements(i) = new_element(i);    
          particle_info(i) = ptcl_data(i);
        });
    Kokkos::parallel_for("fill particlesPerElement2", new_particle_elements.size(),
        KOKKOS_LAMBDA(const int& i){
          particle_elements(i+new_element.size()) = new_particle_elements(i);
          particle_info(i+new_element.size()) = new_particles(i);
        });
    Kokkos::fence();


    Kokkos::resize(ptcl_data, capacity_); //going to write over ptcl_data anyways
    //initCsr data could likely be used as oppose to explicitly calling the constructor
    //and moving the reference to *this 
    initCsrData(particle_elements, particle_info);
    //after all data is copied into new Views
    offsets = new_offsets;
    num_ptcls = particles_on_process;
    capacity_ = num_ptcls;
    //num_rows remains unchanged
    //num_types remains unchanged
    
    fprintf(stderr, "[WARNING] CSR rebuild(...) not implemented\n");
  }

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
    const lid_t team_size = 32;  //hack
    const PolicyType policy(league_size, team_size);
    auto offsets_cpy = offsets;
    const lid_t mask = 1; //all particles are active
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
  }

  template <class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::printMetrics() const {
    fprintf(stderr, "csr capacity %d\n", capacity_);
  }
}
