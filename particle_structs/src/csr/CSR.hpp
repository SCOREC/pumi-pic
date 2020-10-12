#pragma once

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
            printf("%d %d\n", i, v(i));
      });
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
    using ParticleStructure<DataTypes, MemSpace>::copy;

    lid_t getNumPtcls() { return num_ptcls; }
    kkLidView getOffsets() { return offsets; }
    MTVs getPtcl_data() { return ptcl_data; }
    lid_t getCapacity() { return capacity_; }

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
      auto fill_ptcl_indices = PS_LAMBDA(const lid_t elm_id, const lid_t ptcl_id, bool mask){
        particle_indices(ptcl_id) = Kokkos::atomic_fetch_add(&row_indices(particle_elements(ptcl_id)),1);
      };
      parallel_for(fill_ptcl_indices);

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
    fprintf(stderr, "num ptcls    %d\n", num_ptcls);
    fprintf(stderr, "num elements %d\n", num_elems);
  }
} //end namespace pumipic

#include "CSR_rebuild.hpp"
