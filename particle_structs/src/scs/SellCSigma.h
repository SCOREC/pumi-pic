#ifndef SELL_C_SIGMA_H_
#define SELL_C_SIGMA_H_
#include <vector>
#include <utility>
#include <functional>
#include <algorithm>
#include <mpi.h>
#include <unordered_map>
#include <climits>
#include <ppAssert.h>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Sort.hpp>
#include "SCSPair.h"
#include "scs_input.hpp"
#include <particle_structs.hpp>
#include <ppTiming.hpp>
#include <sstream>

namespace pumipic {

void enable_prebarrier();
double prebarrier();

template<class DataTypes, typename MemSpace = DefaultMemSpace>
class SellCSigma : public ParticleStructure<DataTypes, MemSpace> {
 public:

  template <typename MSpace> using Mirror = SellCSigma<DataTypes, MSpace>;
  using typename ParticleStructure<DataTypes, MemSpace>::Types;
  using typename ParticleStructure<DataTypes, MemSpace>::execution_space;
  using typename ParticleStructure<DataTypes, MemSpace>::memory_space;
  using typename ParticleStructure<DataTypes, MemSpace>::device_type;
  using typename ParticleStructure<DataTypes, MemSpace>::kkLidView;
  using typename ParticleStructure<DataTypes, MemSpace>::kkGidView;
  using typename ParticleStructure<DataTypes, MemSpace>::kkLidHostMirror;
  using typename ParticleStructure<DataTypes, MemSpace>::kkGidHostMirror;
  using typename ParticleStructure<DataTypes, MemSpace>::MTVs;

#ifdef PP_USE_GPU
  template <std::size_t N>
  using Slice = typename ParticleStructure<DataTypes, MemSpace>::Slice<N>;
#else
  template <std::size_t N> using DataType = typename MemberTypeAtIndex<N, DataTypes>::type;
template <std::size_t N> using Slice = Segment<DataType<N>, device_type>;
#endif
  typedef Kokkos::TeamPolicy<execution_space> PolicyType;
  typedef Kokkos::View<MyPair*, device_type> PairView;
  typedef Kokkos::UnorderedMap<gid_t, lid_t, device_type> GID_Mapping;
  typedef SCS_Input<DataTypes, MemSpace> Input_T;

  SellCSigma() = delete;
  SellCSigma(const SellCSigma&) = delete;
  SellCSigma& operator=(const SellCSigma&) = delete;
  /* Constructor of SellCSigma as particle structure
    p - a Kokkos::TeamPolicy that defines the value of C based on the device
    sigma - the sorting parameter 1 = no sorting, INT_MAX = full sorting
    vertical_chunk_size - tuning parameter for load balancing of irregular row lengths
    num_elements - the number of elements in the mesh
    num_particles - the number of particles needed
    particles_per_element - the number of particles in each element
    element_gids - (for MPI parallelism) global ids for each element (size 0 is ignored)
    particle_elements - parent element for each particle (optional)
    particle_info - Initial values for the particle information (optional)
  */
  SellCSigma(PolicyType& p,
             lid_t sigma, lid_t vertical_chunk_size, lid_t num_elements, lid_t num_particles,
             kkLidView particles_per_element, kkGidView element_gids,
             kkLidView particle_elements = kkLidView(),
             MTVs particle_info = NULL);
  SellCSigma(SCS_Input<DataTypes, MemSpace>&);
  ~SellCSigma();

  template <class MSpace>
  Mirror<MSpace>* copy();

  //Functions from ParticleStructure
  using ParticleStructure<DataTypes, MemSpace>::nElems;
  using ParticleStructure<DataTypes, MemSpace>::nPtcls;
  using ParticleStructure<DataTypes, MemSpace>::capacity;
  using ParticleStructure<DataTypes, MemSpace>::numRows;
  using ParticleStructure<DataTypes, MemSpace>::copy;

  //Returns the horizontal slicing(C)
  lid_t C() const {return C_;}
  //Returns the vertical slicing(V)
  lid_t V() const {return V_;}


  //Change whether or not to try shuffling
  void setShuffling(bool newS) {tryShuffling = newS;}

  /* Migrates each particle to new_process and to new_element
     Calls rebuild to recreate the SCS after migrating particles
     new_element - array sized scs->capacity with the new element for each particle
     new_process - array sized scs->capacity with the new process for each particle
  */
  void migrate(kkLidView new_element, kkLidView new_process,
               Distributor<MemSpace> dist = Distributor<MemSpace>(),
               kkLidView new_particle_elements = kkLidView(),
               MTVs new_particle_info = NULL);

  /*
    Reshuffles the scs values to the element in new_element[i]
    Calls rebuild if there is not enough space for the shuffle
    new_element - array sized scs->capacity with the new element for each particle
      Optional arguments when adding new particles to the structure
      new_particle_elements - the new element for each new particle
      new_particles - the data for the new particles
  */
  bool reshuffle(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particles = NULL);
  /*
    Rebuilds a new SCS where particles move to the element in new_element[i]
    new_element - array sized scs->capacity with the new element for each particle
      Optional arguments when adding new particles to the structure
      new_particle_elements - the new element for each new particle
      new_particles - the data for the new particles

  */
  void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
               MTVs new_particles = NULL);

  /*
    Performs a parallel for over the elements/particles in the SCS
    The passed in functor/lambda should take in 3 arguments (int elm_id, int ptcl_id, bool mask)
    Example usage with lambda:
    auto lamb = PS_LAMBDA(const int& elm_id, const int& ptcl_id, const bool& mask) {
      do stuff...
    };
    ps::parallel_for(scs, lamb, name);
  */
  template <typename FunctionType, class U = MemSpace>
  typename std::enable_if<std::is_same<typename U::execution_space, Kokkos::Serial>::value>::type parallel_for(FunctionType& fn, std::string s="");
  template <typename FunctionType, class U = MemSpace>
  typename std::enable_if<!std::is_same<typename U::execution_space, Kokkos::Serial>::value>::type parallel_for(FunctionType& fn, std::string s="");

  //Prints the format of the SCS labeled by prefix
  void printFormat(const char* prefix = "") const;

  //Prints metrics of the SCS
  void printMetrics() const;

  //Do not call these functions:
  int chooseChunkHeight(int maxC, kkLidView ptcls_per_elem);
  void sigmaSort(PairView& ptcl_pairs, lid_t num_elems,
                 kkLidView ptcls_per_elem, lid_t sigma);
  void constructChunks(PairView ptcls, lid_t& nchunks,
                       kkLidView& chunk_widths, kkLidView& row_element,
                       kkLidView& element_row);
  void createGlobalMapping(kkGidView elmGid, kkGidView& elm2Gid, GID_Mapping& elmGid2Lid);
  void constructOffsets(lid_t nChunks, lid_t& nSlices, kkLidView chunk_widths,
                        kkLidView& offs, kkLidView& s2e, lid_t& capacity);
  void setupParticleMask(Kokkos::View<bool*> mask, PairView ptcls, kkLidView chunk_widths,
                         kkLidView& chunk_starts);
  void initSCSData(kkLidView chunk_widths, kkLidView particle_elements,
                   MTVs particle_info);

  template <typename DT, typename MSpace> friend class SellCSigma;
 private:

  //Variables from ParticleStructure
  using ParticleStructure<DataTypes, MemSpace>::name;
  using ParticleStructure<DataTypes, MemSpace>::num_elems;
  using ParticleStructure<DataTypes, MemSpace>::num_ptcls;
  using ParticleStructure<DataTypes, MemSpace>::capacity_;
  using ParticleStructure<DataTypes, MemSpace>::num_rows;
  using ParticleStructure<DataTypes, MemSpace>::ptcl_data;
  using ParticleStructure<DataTypes, MemSpace>::num_types;

  //The User defined kokkos policy
  PolicyType policy;
  //Chunk size
  lid_t C_;
  //Max Chunk size from policy
  lid_t C_max;
  //Vertical slice size
  lid_t V_;
  //Sorting chunk size
  lid_t sigma;
  //Number of chunks
  lid_t num_chunks;
  //Number of slices
  lid_t num_slices;
  //chunk_element stores the id of the first row in the chunk
  //  This only matters for vertical slicing so that each slice can determine which row
  //  it is a part of.
  kkLidView slice_to_chunk;
  //particle_mask true means there is a particle at this location, false otherwise
  Kokkos::View<bool*, MemSpace> particle_mask;
  //offsets into the scs structure
  kkLidView offsets;

  //map from row to element
  // row = slice_to_chunk[slice] + row_in_chunk
  kkLidView row_to_element;
  //map from element to row
  kkLidView element_to_row;

  //mappings from row to element gid and back to row
  kkGidView element_to_gid;
  GID_Mapping element_gid_to_lid;
  //Pointers to the start of each SCS for each data type
  MTVs scs_data_swap;
  std::size_t current_size, swap_size;

  //Padding terms
  double extra_padding;
  double shuffle_padding;
  PaddingStrategy pad_strat;
  double minimize_size;
  bool always_realloc;
  //True - try shuffling every rebuild, false - only rebuild
  bool tryShuffling;
  //Metric Info
  lid_t num_empty_elements;

  //Private construct function
  void construct(kkLidView ptcls_per_elem,
                 kkGidView element_gids,
                 kkLidView particle_elements,
                 MTVs particle_info);
  void destroy();

  SellCSigma(lid_t Cmax) : ParticleStructure<DataTypes, MemSpace>(), policy(PolicyType(1000,Cmax)) {};

};

template<class DataTypes, typename MemSpace>
void SellCSigma<DataTypes, MemSpace>::construct(kkLidView ptcls_per_elem,
                                                kkGidView element_gids,
                                                kkLidView particle_elements,
                                                MTVs particle_info) {
  Kokkos::Profiling::pushRegion("scs_construction");
  tryShuffling = true;
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  C_max = policy.team_size();
  C_ = chooseChunkHeight(C_max, ptcls_per_elem);

  if(!comm_rank)
    fprintf(stderr, "Building SCS with C: %d sigma: %d V: %d\n",C_,sigma,V_);
  //Perform sorting
  PairView ptcls;
  sigmaSort(ptcls, num_elems,ptcls_per_elem, sigma);

  // Number of chunks without vertical slicing
  kkLidView chunk_widths;
  constructChunks(ptcls, num_chunks, chunk_widths, row_to_element, element_to_row);
  num_rows = num_chunks * C_;

  if (element_gids.size() > 0) {
    createGlobalMapping(element_gids, element_to_gid, element_gid_to_lid);
  }

  //Create offsets into each chunk/vertical slice
  constructOffsets(num_chunks, num_slices, chunk_widths, offsets, slice_to_chunk,capacity_);

  //Allocate the SCS and backup with extra space
  lid_t cap = capacity_;
  particle_mask = Kokkos::View<bool*>(Kokkos::ViewAllocateWithoutInitializing("particle_mask"), cap);
  if (extra_padding > 0)
    cap *= (1 + extra_padding);
  CreateViews<device_type, DataTypes>(ptcl_data, cap);
  if (!always_realloc)
    CreateViews<device_type, DataTypes>(scs_data_swap, cap);
  swap_size = current_size = cap;

  if (num_ptcls > 0) {
    kkLidView chunk_starts;
    setupParticleMask(particle_mask, ptcls, chunk_widths, chunk_starts);

    //If particle info is provided then enter the information
    lid_t given_particles = particle_elements.size();
    if (given_particles > 0 && particle_info != NULL) {
      initSCSData(chunk_starts, particle_elements, particle_info);
    }
  }
  Kokkos::Profiling::popRegion();
}


template<class DataTypes, typename MemSpace>
SellCSigma<DataTypes, MemSpace>::SellCSigma(PolicyType& p, lid_t sig, lid_t v, lid_t ne,
                                            lid_t np, kkLidView ptcls_per_elem,
                                            kkGidView element_gids,
                                            kkLidView particle_elements,
                                            MTVs particle_info) :
  ParticleStructure<DataTypes, MemSpace>(), policy(p), element_gid_to_lid(ne) {
  //Set variables
  sigma = sig;
  V_ = v;
  num_elems = ne;
  num_ptcls = np;
  shuffle_padding = 0.1;
  extra_padding = 0.05;
  minimize_size = 0.8;
  always_realloc = false;
  pad_strat = PAD_EVENLY;
  construct(ptcls_per_elem, element_gids, particle_elements, particle_info);
}

template<class DataTypes, typename MemSpace>
SellCSigma<DataTypes, MemSpace>::SellCSigma(Input_T& input) :
    ParticleStructure<DataTypes, MemSpace>(input.name), policy(input.policy),
    element_gid_to_lid(input.ne) {
  sigma = input.sig;
  V_ = input.V;
  num_elems = input.ne;
  num_ptcls = input.np;
  shuffle_padding = input.shuffle_padding;
  extra_padding = input.extra_padding;
  minimize_size = input.minimize_size;
  pad_strat = input.padding_strat;
  always_realloc = input.always_realloc;
  construct(input.ppe, input.e_gids, input.particle_elms, input.p_info);
}

template <typename Space>
typename std::enable_if<std::is_same<Kokkos::Serial, typename Space::execution_space>::value, int>::type 
  maxChunk(int C_max) {
    return 1;
}
template <typename Space>
typename std::enable_if<!std::is_same<Kokkos::Serial, typename Space::execution_space>::value, int>::type
  maxChunk(int C_max) {
    return C_max;
}

template<class DataTypes, typename MemSpace>
template <class MSpace>
SellCSigma<DataTypes, MemSpace>::Mirror<MSpace>* SellCSigma<DataTypes, MemSpace>::copy() {
  if (std::is_same<memory_space, typename MSpace::memory_space>::value) {
    fprintf(stderr, "[ERROR] Copy to same memory space not supported\n");
    exit(EXIT_FAILURE);
  }
  const auto cmax = maxChunk<MSpace>(C_max);
  Mirror<MSpace>* mirror_copy = new SellCSigma<DataTypes, MSpace>(cmax);
  //Call Particle structures copy
  mirror_copy->copy(this);
  //Copy constants
  mirror_copy->C_ = C_;
  mirror_copy->C_max = C_max;
  mirror_copy->V_ = V_;
  mirror_copy->sigma = sigma;
  mirror_copy->num_chunks = num_chunks;
  mirror_copy->num_slices = num_slices;
  mirror_copy->current_size = current_size;
  mirror_copy->swap_size = swap_size;
  mirror_copy->extra_padding = extra_padding;
  mirror_copy->shuffle_padding = shuffle_padding;
  mirror_copy->pad_strat = pad_strat;
  mirror_copy->minimize_size = minimize_size;
  mirror_copy->always_realloc = always_realloc;
  mirror_copy->tryShuffling = tryShuffling;
  mirror_copy->num_empty_elements = num_empty_elements;

  //Create the swap space
  mirror_copy->scs_data_swap = createMemberViews<DataTypes, memory_space>(swap_size);
  //Deep copy each view
  mirror_copy->slice_to_chunk = typename Mirror<MSpace>::kkLidView("mirror slice_to_chunk",
                                                                   slice_to_chunk.size());
  Kokkos::deep_copy(mirror_copy->slice_to_chunk, slice_to_chunk);
  mirror_copy->particle_mask = Kokkos::View<bool*,MSpace>("mirror particle_mask",
                                                                  particle_mask.size());
  Kokkos::deep_copy(mirror_copy->particle_mask, particle_mask);
  mirror_copy->offsets = typename Mirror<MSpace>::kkLidView("mirror offsets", offsets.size());
  Kokkos::deep_copy(mirror_copy->offsets, offsets);
  mirror_copy->row_to_element = typename Mirror<MSpace>::kkLidView("mirror row_to_element",
                                                                   row_to_element.size());
  Kokkos::deep_copy(mirror_copy->row_to_element, row_to_element);
  mirror_copy->element_to_row = typename Mirror<MSpace>::kkLidView("mirror element_to_row",
                                                                   element_to_row.size());
  Kokkos::deep_copy(mirror_copy->element_to_row, element_to_row);
  mirror_copy->element_to_gid = typename Mirror<MSpace>::kkGidView("mirror element_to_gid",
                                                                   element_to_gid.size());
  Kokkos::deep_copy(mirror_copy->element_to_gid, element_to_gid);
  //Deep copy the gid mapping
  mirror_copy->element_gid_to_lid.create_copy_view(element_gid_to_lid);
  return mirror_copy;
}


template<class DataTypes, typename MemSpace>
void SellCSigma<DataTypes, MemSpace>::destroy() {
  destroyViews<DataTypes, memory_space>(ptcl_data);
  if (!always_realloc)
    destroyViews<DataTypes, memory_space>(scs_data_swap);
}
template<class DataTypes, typename MemSpace>
SellCSigma<DataTypes, MemSpace>::~SellCSigma() {
  destroy();
}



template<class DataTypes, typename MemSpace>
void SellCSigma<DataTypes,MemSpace>::printFormat(const char* prefix) const {
  //Transfer everything to the host
  kkLidHostMirror slice_to_chunk_host = deviceToHost(slice_to_chunk);
  kkGidHostMirror element_to_gid_host = deviceToHost(element_to_gid);
  kkLidHostMirror row_to_element_host = deviceToHost(row_to_element);
  kkLidHostMirror offsets_host = deviceToHost(offsets);
  Kokkos::View<bool*>::HostMirror particle_mask_host = deviceToHost(particle_mask);

  std::stringstream ss;
  char buffer[1000];
  char* ptr = buffer;
  int num_chars;

  num_chars = sprintf(ptr, "%s\n", prefix);
  num_chars += sprintf(ptr+num_chars,"Particle Structures Sell-C-Sigma C: %d sigma: %d V: %d.\n", C_, sigma, V_);
  num_chars += sprintf(ptr+num_chars,"Number of Elements: %d.\nNumber of Particles: %d.\n", num_elems, num_ptcls);
  num_chars += sprintf(ptr+num_chars,"Number of Chunks: %d.\nNumber of Slices: %d.\n", num_chunks, num_slices);
  buffer[num_chars] = '\0';
  ss << buffer;

  lid_t last_chunk = -1;
  for (lid_t i = 0; i < num_slices; ++i) {
    lid_t chunk = slice_to_chunk_host(i);
    if (chunk != last_chunk) {
      last_chunk = chunk;
      num_chars = sprintf(ptr,"  Chunk %d. Elements", chunk);
      if (element_to_gid_host.size() > 0)
        num_chars += sprintf(ptr+num_chars,"(GID)");
      num_chars += sprintf(ptr+num_chars,":");
      buffer[num_chars] = '\0';
      ss << buffer;

      for (lid_t row = chunk*C_; row < (chunk+1)*C_; ++row) {
        lid_t elem = row_to_element_host(row);
        num_chars = sprintf(ptr," %d", elem);
        if (element_to_gid_host.size() > 0)
          num_chars += sprintf(ptr+num_chars,"(%ld)", element_to_gid_host(elem));
        buffer[num_chars] = '\0';
        ss << buffer;
      }
      num_chars = sprintf(ptr,"\n");
      buffer[num_chars] = '\0';
      ss << buffer;
    }
    num_chars = sprintf(ptr,"    Slice %d", i);
    buffer[num_chars] = '\0';
    ss << buffer;
    for (lid_t j = offsets_host(i); j < offsets_host(i+1); ++j) {
      if ((j - offsets_host(i)) % C_ == 0)
        num_chars = sprintf(ptr," |");
      num_chars += sprintf(ptr+num_chars," %d", particle_mask_host(j));
      buffer[num_chars] = '\0';
      ss << buffer;
    }
    num_chars = sprintf(ptr,"\n");
    buffer[num_chars] = '\0';
    ss << buffer;
  }
  std::cout << ss.str();
}

template <class DataTypes, typename MemSpace>
void SellCSigma<DataTypes, MemSpace>::printMetrics() const {

  //Gather metrics
  kkLidView padded_cells("padded_cells", 1);
  kkLidView padded_slices("padded_slices", 1);
  const lid_t league_size = num_slices;
  const lid_t team_size = C_;
  const PolicyType policy(league_size, team_size);
  auto offsets_cpy = offsets;
  auto slice_to_chunk_cpy = slice_to_chunk;
  auto row_to_element_cpy = row_to_element;
  auto particle_mask_cpy = particle_mask;
  Kokkos::parallel_for("GatherMetrics", policy,
                       KOKKOS_LAMBDA(const typename PolicyType::member_type& thread) {
    const lid_t slice = thread.league_rank();
    const lid_t slice_row = thread.team_rank();
    const lid_t rowLen = (offsets_cpy(slice+1)-offsets_cpy(slice))/team_size;
    const lid_t start = offsets_cpy(slice) + slice_row;
    const lid_t row = slice_to_chunk_cpy(slice) * team_size + slice_row;
    const lid_t element_id = row_to_element_cpy(row);
    lid_t np = 0;
    for (lid_t p = 0; p < rowLen; ++p) {
      const lid_t particle_id = start+(p*team_size);
      const bool mask = particle_mask_cpy(particle_id);
      np += !mask;
    }
    Kokkos::atomic_fetch_add(&padded_cells[0],np);
    thread.team_reduce(Kokkos::Sum<lid_t, MemSpace>(np));
    if (slice_row == 0)
      Kokkos::atomic_fetch_add(&padded_slices[0], np > 0);
  });

  lid_t num_padded = getLastValue<lid_t>(padded_cells);
  lid_t num_padded_slices = getLastValue<lid_t>(padded_slices);

  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  char buffer[1000];
  char* ptr = buffer;

  //Header
  ptr += sprintf(ptr, "Metrics %d, C %d, V %d, sigma %d\n", comm_rank, C_, V_, sigma);
  //Sizes
  ptr += sprintf(ptr, "Nelems %d, Nchunks %d, Nslices %d, Nptcls %d, Capacity %d, "
                 "Allocation %lu\n", nElems(), num_chunks, num_slices, nPtcls(),
                 capacity(), current_size + swap_size);
  //Padded Cells
  ptr += sprintf(ptr, "Padded Cells <Tot %%> %d %.3f\n", num_padded,
                 num_padded * 100.0 / particle_mask.size());
  //Padded Slices
  ptr += sprintf(ptr, "Padded Slices <Tot %%> %d %.3f\n", num_padded_slices,
                 num_padded_slices * 100.0 / num_slices);
  //Empty Elements
  ptr += sprintf(ptr, "Empty Rows <Tot %%> %d %.3f\n", num_empty_elements,
                 num_empty_elements * 100.0 / numRows());

  printf("%s\n",buffer);
}

template <class DataTypes, typename MemSpace>
template <typename FunctionType, typename U>
typename std::enable_if<!std::is_same<typename U::execution_space, Kokkos::Serial>::value>::type
SellCSigma<DataTypes, MemSpace>::parallel_for(FunctionType& fn, std::string name) {
  if (nPtcls() == 0)
    return;
  FunctionType* fn_d = gpuMemcpy(fn);
  const lid_t league_size = num_slices;
  const lid_t team_size = C_;
  const PolicyType policy(league_size, team_size);
  auto offsets_cpy = offsets;
  auto slice_to_chunk_cpy = slice_to_chunk;
  auto row_to_element_cpy = row_to_element;
  auto particle_mask_cpy = particle_mask;
  Kokkos::parallel_for(name, policy,
                       KOKKOS_LAMBDA(const typename PolicyType::member_type& thread) {
    const lid_t slice = thread.league_rank();
    const lid_t slice_row = thread.team_rank();
    const lid_t rowLen = (offsets_cpy(slice+1)-offsets_cpy(slice))/team_size;
    const lid_t start = offsets_cpy(slice) + slice_row;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_size), [=] (lid_t& j) {
      const lid_t row = slice_to_chunk_cpy(slice) * team_size + slice_row;
      const lid_t element_id = row_to_element_cpy(row);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, rowLen), [=] (lid_t& p) {
        const lid_t particle_id = start+(p*team_size);
        const bool mask = particle_mask_cpy(particle_id);
        (*fn_d)(element_id, particle_id, mask);
      });
    });
  });
#ifdef PP_USE_GPU
  cudaFree(fn_d);
#endif
}

template <class DataTypes, typename MemSpace>
template <typename FunctionType, typename U>
typename std::enable_if<std::is_same<typename U::execution_space, Kokkos::Serial>::value>::type
SellCSigma<DataTypes, MemSpace>::parallel_for(FunctionType& fn, std::string name) {
  if (nPtcls() == 0)
    return;
  for (int slice = 0; slice < num_slices; ++slice) {
    const lid_t rowLen = (offsets(slice+1)-offsets(slice))/C_;
    for (int slice_row = 0; slice_row < C_; ++slice_row) {
      const lid_t row = slice_to_chunk(slice) * C_ + slice_row;
      const lid_t element_id = row_to_element(row);
      const lid_t start = offsets(slice) + slice_row;
      for (int ptcl = 0; ptcl < rowLen; ++ptcl) {
        const lid_t particle_id = start + ptcl*C_;
        const lid_t mask = particle_mask[particle_id];
        fn(element_id, particle_id, mask);
      }
    }
  }
}

} // end namespace pumipic

//Seperate files with SCS member function implementations
#include "SCS_sort.h"
#include "SCS_buildFns.h"
#include "SCS_rebuild.h"
#include "SCS_migrate.h"

#endif
