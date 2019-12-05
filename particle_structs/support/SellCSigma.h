#ifndef SELL_C_SIGMA_H_
#define SELL_C_SIGMA_H_
#include <vector>
#include <utility>
#include <functional>
#include <algorithm>
#include "psAssert.h"
#include "MemberTypes.h"
#include "MemberTypeArray.h"
#include "MemberTypeLibraries.h"
#include "SCS_Macros.h"
#include "SCS_Types.h"
#include "SupportKK.h"
#include "ViewComm.h"
#include "Segment.h"
#include "SCSPair.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Sort.hpp>
#include <mpi.h>
#include <unordered_map>
#include <climits>

#ifdef SCS_USE_CUDA
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#endif

namespace particle_structs {

void enable_prebarrier();
double prebarrier();
  
template <typename ExecSpace> 
using PairView=Kokkos::View<MyPair*, typename ExecSpace::device_type>;

template<class DataTypes, typename ExecSpace = Kokkos::DefaultExecutionSpace>
class SellCSigma {
 public:
  
  typedef Kokkos::TeamPolicy<ExecSpace> PolicyType ;
  typedef Kokkos::View<lid_t*, typename ExecSpace::device_type> kkLidView;
  typedef Kokkos::View<gid_t*, typename ExecSpace::device_type> kkGidView;
  typedef typename kkLidView::HostMirror kkLidHostMirror;
  typedef typename kkGidView::HostMirror kkGidHostMirror;
  typedef Kokkos::UnorderedMap<gid_t, lid_t, typename ExecSpace::device_type> GID_Mapping;

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
             MemberTypeViews<DataTypes> particle_info = NULL);
  ~SellCSigma();

  //Returns the horizontal slicing(C)
  lid_t C() const {return C_;}
  //Returns the vertical slicing(V)
  lid_t V() const {return V_;}
  //Returns the number of rows in the scs including padded rows
  lid_t numRows() const {return num_chunks * C_;}
  //Returns the capacity of the scs including padding
  lid_t capacity() const { return capacity_;}
  //Return the number of elements in the SCS
  lid_t nElems() const {return num_elems;}
  //Returns the number of particles managed by the SCS
  lid_t nPtcls() const {return num_ptcls;}

  //Change whether or not to try shuffling
  void setShuffling(bool newS) {tryShuffling = newS;}
  
  /* Gets the Nth datatype SCS to be indexed by particle id 
     Example: auto segment = scs->get<0>()
   */ 
  template <std::size_t N> 
  Segment<typename MemberTypeAtIndex<N,DataTypes>::type, ExecSpace> get() {
    using Type=typename MemberTypeAtIndex<N, DataTypes>::type;
    if (num_ptcls == 0)
      return Segment<Type, ExecSpace>();
    MemberTypeView<Type>* view = static_cast<MemberTypeView<Type>*>(scs_data[N]);
    return Segment<Type, ExecSpace>(*view);
  }


  /* Migrates each particle to new_process and to new_element
     Calls rebuild to recreate the SCS after migrating particles
     new_element - array sized scs->capacity with the new element for each particle
     new_process - array sized scs->capacity with the new process for each particle
  */
  void migrate(kkLidView new_element, kkLidView new_process);

  /*
    Reshuffles the scs values to the element in new_element[i]
    Calls rebuild if there is not enough space for the shuffle
    new_element - array sized scs->capacity with the new element for each particle
      Optional arguments when adding new particles to the structure
      new_particle_elements - the new element for each new particle
      new_particles - the data for the new particles
  */
  bool reshuffle(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                 MemberTypeViews<DataTypes> new_particles = NULL);
  /*
    Rebuilds a new SCS where particles move to the element in new_element[i]
    new_element - array sized scs->capacity with the new element for each particle
      Optional arguments when adding new particles to the structure
      new_particle_elements - the new element for each new particle
      new_particles - the data for the new particles

  */
  void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(), 
                  MemberTypeViews<DataTypes> new_particles = NULL);

  /*
    Performs a parallel for over the elements/particles in the SCS
    The passed in functor/lambda should take in 3 arguments (int elm_id, int ptcl_id, bool mask)
    Example usage with lambda:
    auto lamb = SCS_LAMBDA(const int& elm_id, const int& ptcl_id, const bool& mask) {
      do stuff...
    };
    scs->parallel_for(lamb);
  */
  template <typename FunctionType>
  void parallel_for(FunctionType& fn, std::string s="");


  //Prints the format of the SCS labeled by prefix
  void printFormat(const char* prefix = "") const;

  //Prints metrics of the SCS
  void printMetrics() const;
  
  //Do not call these functions:
  void constructChunks(PairView<ExecSpace> ptcls, lid_t& nchunks,
                       kkLidView& chunk_widths, kkLidView& row_element,
                       kkLidView& element_row);
  void createGlobalMapping(kkGidView elmGid, kkGidView& elm2Gid, GID_Mapping& elmGid2Lid);
  void constructOffsets(lid_t nChunks, lid_t& nSlices, kkLidView chunk_widths, 
                        kkLidView& offs, kkLidView& s2e, lid_t& capacity);
  void setupParticleMask(kkLidView mask, PairView<ExecSpace> ptcls, kkLidView chunk_widths);
  void initSCSData(kkLidView chunk_widths, kkLidView particle_elements,
                   MemberTypeViews<DataTypes> particle_info);
private:
  //Number of Data types
  static constexpr std::size_t num_types = DataTypes::size;

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
  //Total entries
  lid_t num_elems;
  //Total particles
  lid_t num_ptcls;
  //num_ptcls + buffer
  lid_t capacity_;
  //chunk_element stores the id of the first row in the chunk
  //  This only matters for vertical slicing so that each slice can determine which row
  //  it is a part of.
  kkLidView slice_to_chunk;
  //particle_mask true means there is a particle at this location, false otherwise
  kkLidView particle_mask;
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
  MemberTypeViews<DataTypes> scs_data;
  MemberTypeViews<DataTypes> scs_data_swap;
  std::size_t current_size, swap_size;
  void destroy();

  //True - try shuffling every rebuild, false - only rebuild
  bool tryShuffling;
  //Metric Info
  lid_t num_empty_elements;
};

template<typename ExecSpace>
int chooseChunkHeight(int maxC,
                      Kokkos::View<lid_t*, typename ExecSpace::device_type> ptcls_per_elem) {
  lid_t num_elems_with_ptcls = 0;
  Kokkos::parallel_reduce("count_elems", ptcls_per_elem.size(), KOKKOS_LAMBDA(const lid_t& i, lid_t& sum) {
    sum += ptcls_per_elem(i) > 0;
    }, num_elems_with_ptcls);
  if (num_elems_with_ptcls == 0)
    return 1;
  if (num_elems_with_ptcls < maxC)
    return num_elems_with_ptcls;
  return maxC;
}
template <typename ExecSpace> 
void sigmaSort(PairView<ExecSpace>& ptcl_pairs, lid_t num_elems, 
               Kokkos::View<lid_t*,typename ExecSpace::device_type> ptcls_per_elem, 
               lid_t sigma){
  //Make temporary copy of the particle counts for sorting
  ptcl_pairs = PairView<ExecSpace>("ptcl_pairs", num_elems);
  //PairView<ExecSpace> ptcl_pairs("ptcl_pairs", num_elems);
  if (sigma > 1) {
    lid_t i;
#ifdef SCS_USE_CUDA
    Kokkos::View<lid_t*, typename ExecSpace::device_type> elem_ids("elem_ids", num_elems);
    Kokkos::View<lid_t*, typename ExecSpace::device_type> temp_ppe("temp_ppe", num_elems);
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
      temp_ppe(i) = ptcls_per_elem(i);
      elem_ids(i) = i;
    });
    thrust::device_ptr<lid_t> ptcls_t(temp_ppe.data());
    thrust::device_ptr<lid_t> elem_ids_t(elem_ids.data());
    for (i = 0; i < num_elems - sigma; i+=sigma) {
      thrust::sort_by_key(thrust::device, ptcls_t + i, ptcls_t + i + sigma, elem_ids_t + i);
    }
    thrust::sort_by_key(thrust::device, ptcls_t + i, ptcls_t + num_elems, elem_ids_t + i);
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
      ptcl_pairs(num_elems - 1 - i).first = temp_ppe(i);
      ptcl_pairs(num_elems - 1 - i).second = elem_ids(i);
    });
#else
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
      ptcl_pairs(num_elems).first = ptcls_per_elem(i);
      ptcl_pairs(num_elems).second = i;
    });
    typename PairView<ExecSpace>::HostMirror ptcl_pairs_host = deviceToHost(ptcl_pairs);
    MyPair* ptcl_pair_data = ptcl_pairs_host.data();
    for (i = 0; i < num_elems - sigma; i+=sigma) {
      std::sort(ptcl_pair_data + i, ptcl_pair_data + i + sigma);
    }
    std::sort(ptcl_pair_data + i, ptcl_pair_data + num_elems);
    hostToDevice(ptcl_pairs,  ptcl_pair_data);
#endif
  }
  else {
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
      ptcl_pairs(i).first = ptcls_per_elem(i);
      ptcl_pairs(i).second = i;
    });
  }
}

template <typename ExecSpace>
struct MaxChunkWidths {

  typedef lid_t value_type[];

  typedef typename PairView<ExecSpace>::size_type size_type;

  size_type value_count;

  PairView<ExecSpace> widths;

  lid_t C;

  MaxChunkWidths(const PairView<ExecSpace>& widths_, const lid_t C_, lid_t nchunks) :
    value_count(nchunks), widths(widths_), C(C_) {}

  KOKKOS_INLINE_FUNCTION void operator()(const size_type i, value_type mx) const {
    const lid_t index = i / C;
    if (widths(i).first > mx[index]) {
      mx[index] = widths(i).first;
    }
  }

  KOKKOS_INLINE_FUNCTION void join(volatile value_type dst, const volatile value_type src) const {
    for (size_type j = 0; j < value_count; ++j) {
      if (src[j] > dst[j])
        dst[j] = src[j];
    }
  }

  KOKKOS_INLINE_FUNCTION void init(value_type mx) const {
    for (size_type j = 0; j < value_count; ++j) {
      mx[j] = 0;
    }
  }
};

template<class DataTypes, typename ExecSpace> 
void SellCSigma<DataTypes, ExecSpace>::constructChunks(PairView<ExecSpace> ptcls, lid_t& nchunks, 
                                                       kkLidView& chunk_widths,
                                                       kkLidView& row_element,
                                                       kkLidView& element_row) {
  nchunks = num_elems / C_ + (num_elems % C_ != 0);
  chunk_widths = kkLidView("chunk_widths", nchunks);
  row_element = kkLidView("row_element", nchunks * C_);
  element_row = kkLidView("element_row", nchunks * C_);
  kkLidView empty("empty_elems", 1);
  Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
    const lid_t element = ptcls(i).second;
    row_element(i) = element;
    element_row(element) = i;
    Kokkos::atomic_fetch_add(&empty[0], ptcls(i).first == 0);
  });
  Kokkos::parallel_for(Kokkos::RangePolicy<>(num_elems, nchunks * C_),
                       KOKKOS_LAMBDA(const lid_t& i) {
    row_element(i) = i;
    element_row(i) = i;
    Kokkos::atomic_fetch_add(&empty[0], 1);
  });

  num_empty_elements = getLastValue<lid_t>(empty);
  typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
  const team_policy policy(nchunks, C_);
  lid_t C_local = C_;
  lid_t num_elems_local = num_elems;
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const typename team_policy::member_type& thread) {
    const lid_t chunk_id = thread.league_rank();
    const lid_t row_num = chunk_id * C_local + thread.team_rank();
    lid_t width = 0;
    if (row_num < num_elems_local) {
      width = ptcls(row_num).first;
    }
    thread.team_reduce(Kokkos::Max<lid_t,ExecSpace>(width));
    chunk_widths[chunk_id] = width;
  });
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::createGlobalMapping(kkGidView elmGid,kkGidView& elm2Gid, 
                                                           GID_Mapping& elmGid2Lid) {
  elm2Gid = kkGidView("row to element gid", numRows());
  Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
    const gid_t gid = elmGid(i);
    elm2Gid(i) = gid;
    elmGid2Lid.insert(gid, i);
  });
  Kokkos::parallel_for(Kokkos::RangePolicy<>(num_elems, numRows()), KOKKOS_LAMBDA(const lid_t& i) {
    elm2Gid(i) = -1;
  });
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::constructOffsets(lid_t nChunks, lid_t& nSlices, 
                                                        kkLidView chunk_widths, kkLidView& offs,
                                                        kkLidView& s2c, lid_t& cap) {
  kkLidView slices_per_chunk("slices_per_chunk", nChunks);
  const lid_t V_local = V_;
  Kokkos::parallel_for(nChunks, KOKKOS_LAMBDA(const lid_t& i) {
    const lid_t width = chunk_widths(i);
    const lid_t val1 = width / V_local;
    const lid_t val2 = width % V_local;
    const bool val3 = val2 != 0;
    slices_per_chunk(i) = val1 + val3;
  });
  kkLidView offset_nslices("offset_nslices",nChunks+1);
  Kokkos::parallel_scan(nChunks, KOKKOS_LAMBDA(const lid_t& i, lid_t& cur, const bool& final) {
    cur += slices_per_chunk(i);
    if (final)
      offset_nslices(i+1) += cur;
  });

  nSlices = getLastValue<lid_t>(offset_nslices);
  offs = kkLidView("SCS offset", nSlices + 1);
  s2c = kkLidView("slice to chunk", nSlices);
  kkLidView slice_size("slice_size", nSlices);
  const lid_t nat_size = V_*C_;
  const lid_t C_local = C_;
  Kokkos::parallel_for(nChunks, KOKKOS_LAMBDA(const lid_t& i) {
    const lid_t start = offset_nslices(i);
    const lid_t end = offset_nslices(i+1);
    for (lid_t j = start; j < end; ++j) {
      s2c(j) = i;
      const lid_t rem = chunk_widths(i) % V_local;
      const lid_t val = rem + (rem==0)*V_local;
      const bool is_last = (j == end-1);
      slice_size(j) = (!is_last) * nat_size;
      slice_size(j) += (is_last) * (val) * C_local;
    }
  });
  Kokkos::parallel_scan(nSlices, KOKKOS_LAMBDA(const lid_t& i, lid_t& cur, const bool final) {
    cur += slice_size(i);
    if (final) {
      const lid_t index = i+1;
      offs(index) += cur;
    }
  });
  cap = getLastValue<lid_t>(offs);
}
template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::setupParticleMask(kkLidView mask, PairView<ExecSpace> ptcls, kkLidView chunk_widths) {
  //Get start of each chunk
  auto offsets_cpy = offsets;
  auto slice_to_chunk_cpy = slice_to_chunk;
  kkLidView chunk_starts("chunk_starts", num_chunks);
  Kokkos::parallel_for(num_slices-1, KOKKOS_LAMBDA(const lid_t& i) {
    const lid_t my_chunk = slice_to_chunk_cpy(i);
    const lid_t next_chunk = slice_to_chunk_cpy(i+1);
    if (my_chunk != next_chunk) {
      chunk_starts(next_chunk) = offsets_cpy(i+1);
    }
  });
  //Fill the SCS
  const lid_t league_size = num_chunks;
  const lid_t team_size = C_;
  const lid_t ne = num_elems;
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> team_policy;
  const team_policy policy(league_size, team_size);
  auto row_to_element_cpy = row_to_element;
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const team_policy::member_type& thread) {
    const lid_t chunk = thread.league_rank();
    const lid_t chunk_row = thread.team_rank();
    const lid_t rowLen = chunk_widths(chunk);
    const lid_t start = chunk_starts(chunk) + chunk_row;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_size), [=] (lid_t& j) {
      const lid_t row = chunk * team_size + chunk_row;
      const lid_t element_id = row_to_element_cpy(row);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, rowLen), [&] (lid_t& p) {
        const lid_t particle_id = start+(p*team_size);
        if (element_id < ne)
          mask(particle_id) =  p < ptcls(row).first;
      });
    });
  });
\
}
template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::initSCSData(kkLidView chunk_widths,
                                                   kkLidView particle_elements,
                                                   MemberTypeViews<DataTypes> particle_info) {
  lid_t given_particles = particle_elements.size();
  kkLidView element_to_row_local = element_to_row;
  //Setup starting point for each row
  lid_t C_local = C_;    
  kkLidView row_index("row_index", numRows());
  Kokkos::parallel_scan(num_chunks, KOKKOS_LAMBDA(const lid_t& i, lid_t& sum, const bool& final) {
      if (final) {
        for (lid_t j = 0; j < C_local; ++j)
          row_index(i*C_local+j) = sum + j;
      }
      sum += chunk_widths(i) * C_local;
    });
  //Determine index for each particle
  kkLidView particle_indices("new_particle_scs_indices", given_particles);
  Kokkos::parallel_for(given_particles, KOKKOS_LAMBDA(const lid_t& i) {
      lid_t new_elem = particle_elements(i);
      lid_t new_row = element_to_row_local(new_elem);
      particle_indices(i) = Kokkos::atomic_fetch_add(&row_index(new_row), C_local);
    });
  
  CopyNewParticlesToSCS<SellCSigma<DataTypes, ExecSpace>, DataTypes>(this, scs_data,
                                                                     particle_info,
                                                                     given_particles,
                                                                     particle_indices);
}

template<class DataTypes, typename ExecSpace>
SellCSigma<DataTypes, ExecSpace>::SellCSigma(PolicyType& p, lid_t sig, lid_t v, lid_t ne, 
                                             lid_t np, kkLidView ptcls_per_elem, 
                                             kkGidView element_gids,
                                             kkLidView particle_elements,
                                             MemberTypeViews<DataTypes> particle_info) :
  policy(p), element_gid_to_lid(ne) {
  tryShuffling = true;
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  C_max = policy.team_size();
  C_ = chooseChunkHeight<ExecSpace>(C_max, ptcls_per_elem);
  
  sigma = sig;
  V_ = v;
  num_elems = ne;
  num_ptcls = np;

  if(!comm_rank)
    fprintf(stderr, "Building SCS with C: %d sigma: %d V: %d\n",C_,sigma,V_);
  //Perform sorting
  PairView<ExecSpace> ptcls;
  Kokkos::Timer timer;
  sigmaSort<ExecSpace>(ptcls, num_elems,ptcls_per_elem, sigma);
  if(comm_rank == 0 || comm_rank == comm_size/2)
    fprintf(stderr,"%d SCS sorting time (seconds) %f\n", comm_rank, timer.seconds());

  // Number of chunks without vertical slicing
  kkLidView chunk_widths;
  constructChunks(ptcls, num_chunks, chunk_widths, row_to_element, element_to_row);

  if (element_gids.size() > 0) {
    createGlobalMapping(element_gids, element_to_gid, element_gid_to_lid);
  }

  //Create offsets into each chunk/vertical slice
  constructOffsets(num_chunks, num_slices, chunk_widths, offsets, slice_to_chunk,capacity_);

  //Allocate the SCS and backup with 10% extra space
  lid_t cap = getLastValue<lid_t>(offsets);
  particle_mask = kkLidView("particle_mask", cap);
  CreateViews<DataTypes>(scs_data, cap*1.1);
  CreateViews<DataTypes>(scs_data_swap, cap*1.1);
  swap_size = current_size = cap*1.1;

  if (np > 0)
    setupParticleMask(particle_mask, ptcls, chunk_widths);

  //If particle info is provided then enter the information
  lid_t given_particles = particle_elements.size();
  if (given_particles > 0 && particle_info != NULL) {
    initSCSData(chunk_widths, particle_elements, particle_info);
  }
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::destroy() {
  destroyViews<DataTypes>(scs_data);
  destroyViews<DataTypes>(scs_data_swap);
}
template<class DataTypes, typename ExecSpace>
SellCSigma<DataTypes, ExecSpace>::~SellCSigma() {
  destroy();
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::migrate(kkLidView new_element, kkLidView new_process) {
  const auto btime = prebarrier();
  Kokkos::Profiling::pushRegion("scs_migrate");
  Kokkos::Timer timer;
  /********* Send # of particles being sent to each process *********/
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  if (comm_size == 1) {
    rebuild(new_element);
    if(!comm_rank || comm_rank == comm_size/2)
      fprintf(stderr, "%d ps particle migration (seconds) %f\n", comm_rank, timer.seconds());
    Kokkos::Profiling::popRegion();
    return;
  }
  kkLidView num_send_particles("num_send_particles", comm_size);
  auto count_sending_particles = SCS_LAMBDA(lid_t element_id, lid_t particle_id, bool mask) {
    const lid_t process = new_process(particle_id);
    Kokkos::atomic_fetch_add(&(num_send_particles(process)), mask * (process != comm_rank));
  };
  parallel_for(count_sending_particles);
  kkLidView num_recv_particles("num_recv_particles", comm_size);
  PS_Comm_Alltoall(num_send_particles, 1, num_recv_particles, 1, MPI_COMM_WORLD);

  lid_t num_sending_to = 0, num_receiving_from = 0;
  Kokkos::parallel_reduce("sum_senders", comm_size, KOKKOS_LAMBDA (const lid_t& i, lid_t& lsum ) {
      lsum += (num_send_particles(i) > 0);
  }, num_sending_to);
  Kokkos::parallel_reduce("sum_receivers", comm_size, KOKKOS_LAMBDA (const lid_t& i, lid_t& lsum ) {
      lsum += (num_recv_particles(i) > 0);
  }, num_receiving_from);

  if (num_sending_to == 0 && num_receiving_from == 0) {
    rebuild(new_element);
    if(!comm_rank || comm_rank == comm_size/2)
      fprintf(stderr, "%d ps particle migration (seconds) %f\n", comm_rank, timer.seconds());
    Kokkos::Profiling::popRegion();
    return;
  }
  /********** Send particle information to new processes **********/
  //Perform an ex-sum on num_send_particles & num_recv_particles
  kkLidView offset_send_particles("offset_send_particles", comm_size+1);
  kkLidView offset_send_particles_temp("offset_send_particles_temp", comm_size + 1);
  kkLidView offset_recv_particles("offset_recv_particles", comm_size+1);
  Kokkos::parallel_scan(comm_size, KOKKOS_LAMBDA(const lid_t& i, lid_t& num, const bool& final) {
    num += num_send_particles(i);
    if (final) {
      offset_send_particles(i+1) += num;
      offset_send_particles_temp(i+1) += num;
    }
  });
  Kokkos::parallel_scan(comm_size, KOKKOS_LAMBDA(const lid_t& i, lid_t& num, const bool& final) {
    num += num_recv_particles(i);
    if (final)
      offset_recv_particles(i+1) += num;
  });
  kkLidHostMirror offset_send_particles_host = deviceToHost(offset_send_particles);
  kkLidHostMirror offset_recv_particles_host = deviceToHost(offset_recv_particles);

  //Create arrays for particles being sent
  lid_t np_send = offset_send_particles_host(comm_size);
  kkLidView send_element("send_element", np_send);
  MemberTypeViews<DataTypes> send_particle;
  //Allocate views for each data type into send_particle[type]
  CreateViews<DataTypes>(send_particle, np_send);
  kkLidView send_index("send_particle_index", capacity());
  auto element_to_gid_local = element_to_gid;
  auto gatherParticlesToSend = SCS_LAMBDA(lid_t element_id, lid_t particle_id, lid_t mask) {
    const lid_t process = new_process(particle_id);
    if (mask && process != comm_rank) {
      send_index(particle_id) =
        Kokkos::atomic_fetch_add(&(offset_send_particles_temp(process)),1);
      const lid_t index = send_index(particle_id);
      send_element(index) = element_to_gid_local(new_element(particle_id));
    }
  };
  parallel_for(gatherParticlesToSend);
  //Copy the values from scs_data[type][particle_id] into send_particle[type](index) for each data type
  CopyParticlesToSend<SellCSigma<DataTypes, ExecSpace>, DataTypes>(this, send_particle, scs_data,
                                                                   new_process,
                                                                   send_index);
  
  //Create arrays for particles being received
  lid_t np_recv = offset_recv_particles_host(comm_size);
  kkLidView recv_element("recv_element", np_recv);
  MemberTypeViews<DataTypes> recv_particle;
  //Allocate views for each data type into recv_particle[type]
  CreateViews<DataTypes>(recv_particle, np_recv);

  //Get pointers to the data for MPI calls
  lid_t send_num = 0, recv_num = 0;
  lid_t num_sends = num_sending_to * (num_types + 1);
  lid_t num_recvs = num_receiving_from * (num_types + 1);
  MPI_Request* send_requests = new MPI_Request[num_sends];
  MPI_Request* recv_requests = new MPI_Request[num_recvs];
  //Send the particles to each neighbor
  for (lid_t i = 0; i < comm_size; ++i) {
    if (i == comm_rank)
      continue;
    
    //Sending
    lid_t num_send = offset_send_particles_host(i+1) - offset_send_particles_host(i);
    if (num_send > 0) {
      lid_t start_index = offset_send_particles_host(i);
      PS_Comm_Isend(send_element, start_index, num_send, i, 0, MPI_COMM_WORLD, 
                    send_requests +send_num);
      send_num++;
      SendViews<DataTypes>(send_particle, start_index, num_send, i, 1,
                           send_requests + send_num);
      send_num+=num_types;
    }
    //Receiving
    lid_t num_recv = offset_recv_particles_host(i+1) - offset_recv_particles_host(i);
    if (num_recv > 0) {
      lid_t start_index = offset_recv_particles_host(i);
      PS_Comm_Irecv(recv_element, start_index, num_recv, i, 0, MPI_COMM_WORLD, 
                    recv_requests + recv_num);
      recv_num++;
      RecvViews<DataTypes>(recv_particle,start_index, num_recv, i, 1,
                           recv_requests + recv_num);
      recv_num+=num_types;
    }
  }
  PS_Comm_Waitall<ExecSpace>(num_recvs, recv_requests, MPI_STATUSES_IGNORE);
  delete [] recv_requests;

  /********** Convert the received element from element gid to element lid *********/
  auto element_gid_to_lid_local = element_gid_to_lid;
  Kokkos::parallel_for(recv_element.size(), KOKKOS_LAMBDA(const lid_t& i) {
    const gid_t gid = recv_element(i);
    const lid_t index = element_gid_to_lid_local.find(gid);
    recv_element(i) = element_gid_to_lid_local.value_at(index);
  });
  
  /********** Set particles that were sent to non existent on this process *********/
  auto removeSentParticles = SCS_LAMBDA(lid_t element_id, lid_t particle_id, lid_t mask) {
    const bool sent = new_process(particle_id) != comm_rank;
    const lid_t elm = new_element(particle_id);
    //Subtract (its value + 1) to get to -1 if it was sent, 0 otherwise
    new_element(particle_id) -= (elm + 1) * sent;
  };
  parallel_for(removeSentParticles);

  /********** Combine and shift particles to their new destination **********/
  rebuild(new_element, recv_element, recv_particle);

  //Cleanup
  PS_Comm_Waitall<ExecSpace>(num_sends, send_requests, MPI_STATUSES_IGNORE);
  delete [] send_requests;
  destroyViews<DataTypes>(send_particle);
  destroyViews<DataTypes>(recv_particle);
  if(!comm_rank || comm_rank == comm_size/2)
    fprintf(stderr, "%d ps particle migration (seconds) %f pre-barrier (seconds) %f\n",
        comm_rank, timer.seconds(), btime);
  Kokkos::Profiling::popRegion();
}

template<class DataTypes, typename ExecSpace>
bool SellCSigma<DataTypes,ExecSpace>::reshuffle(kkLidView new_element, 
                                                kkLidView new_particle_elements, 
                                                MemberTypeViews<DataTypes> new_particles) {
  //Count current/new particles per row
  kkLidView new_particles_per_row("new_particles_per_row", numRows());
  kkLidView num_holes_per_row("num_holes_per_row", numRows());
  kkLidView element_to_row_local = element_to_row;
  auto particle_mask_local = particle_mask;  
  auto countNewParticles = SCS_LAMBDA(lid_t element_id,lid_t particle_id, bool mask){
    const lid_t new_elem = new_element(particle_id);

    const lid_t row = element_to_row_local(element_id);
    const bool is_particle = mask & new_elem != -1;
    const bool is_moving = is_particle & new_elem != element_id;
    if (is_moving) {
      const lid_t new_row = element_to_row_local(new_elem);
      Kokkos::atomic_fetch_add(&(new_particles_per_row(new_row)), mask);
    }
    particle_mask_local(particle_id) = is_particle;
    Kokkos::atomic_fetch_add(&(num_holes_per_row(row)), !is_particle);
  };
  parallel_for(countNewParticles, "countNewParticles");
  // Add new particles to counts
  Kokkos::parallel_for("reshuffle_count", new_particle_elements.size(), KOKKOS_LAMBDA(const lid_t& i) {
      const lid_t new_elem = new_particle_elements(i);
      const lid_t new_row = element_to_row_local(new_elem);
      Kokkos::atomic_fetch_add(&(new_particles_per_row(new_row)), 1);
    });

  //Check if the particles will fit in current structure
  kkLidView fail("fail",1);
  Kokkos::parallel_for(numRows(), KOKKOS_LAMBDA(const lid_t& i) {
      if( new_particles_per_row(i) > num_holes_per_row(i))
        fail(0) = 1;
  });

  if (getLastValue<lid_t>(fail)) {
    //Reshuffle fails
    return false;
  }
  
  //Offset moving particles
  kkLidView offset_new_particles("offset_new_particles", numRows() + 1);
  kkLidView counting_offset_index("counting_offset_index", numRows() + 1);
  Kokkos::parallel_scan(numRows(), KOKKOS_LAMBDA(const lid_t& i, lid_t& cur, const bool& final) {
    cur += new_particles_per_row(i);
    if (final) {
      offset_new_particles(i+1) = cur;
      counting_offset_index(i+1) = cur;
    }
  });

  int num_moving_ptcls = getLastValue<lid_t>(offset_new_particles);
  if (num_moving_ptcls == 0) {
    Kokkos::parallel_reduce(capacity(), KOKKOS_LAMBDA(const lid_t& i, lid_t& sum) {
      sum += particle_mask_local(i);
    }, num_ptcls);
    return true;
  }
  kkLidView movingPtclIndices("movingPtclIndices", num_moving_ptcls);
  kkLidView isFromSCS("isFromSCS", num_moving_ptcls);
  //Gather moving particle list
  auto gatherMovingPtcls = SCS_LAMBDA(const lid_t& element_id,const lid_t& particle_id, const bool& mask){
    const lid_t new_elem = new_element(particle_id);

    const lid_t row = element_to_row_local(element_id);
    const bool is_moving = new_elem != -1 & new_elem != element_id & mask;
    if (is_moving) {
      const lid_t new_row = element_to_row_local(new_elem);
      const lid_t index = Kokkos::atomic_fetch_add(&(counting_offset_index(new_row)), 1);
      movingPtclIndices(index) = particle_id;
      isFromSCS(index) = 1;
    }
  };
  parallel_for(gatherMovingPtcls, "gatherMovingPtcls");

  //Gather new particles in list
  Kokkos::parallel_for("reshuffle_count", new_particle_elements.size(), KOKKOS_LAMBDA(const lid_t& i) {
      const lid_t new_elem = new_particle_elements(i);
      const lid_t new_row = element_to_row_local(new_elem);
      const lid_t index = Kokkos::atomic_fetch_add(&(counting_offset_index(new_row)), 1);
      movingPtclIndices(index) = i;
      isFromSCS(index) = 0;
    });

  //Assign hole index for moving particles
  kkLidView holes("holeIndex", num_moving_ptcls);
  auto assignPtclsToHoles = SCS_LAMBDA(const lid_t& element_id,const lid_t& particle_id, const bool& mask){
    const lid_t row = element_to_row_local(element_id);
    if (!mask) {
      const lid_t moving_index = Kokkos::atomic_fetch_add(&(offset_new_particles(row)),1);
      const lid_t max_index = counting_offset_index(row);
      if (moving_index < max_index) {
        holes(moving_index) = particle_id;
      }
    }
  };
  parallel_for(assignPtclsToHoles, "assignPtclsToHoles");

  //Update particle mask
  Kokkos::parallel_for(num_moving_ptcls, KOKKOS_LAMBDA(const lid_t& i) {
      const lid_t old_index = movingPtclIndices(i);
      const lid_t new_index = holes(i);
      const lid_t fromSCS = isFromSCS(i);
      if (fromSCS == 1)
        particle_mask_local(old_index) = 0;
      particle_mask_local(new_index) = 1;
  });
  
  //Shift SCS values
  ShuffleParticles<kkLidView, DataTypes>(scs_data, new_particles, movingPtclIndices, holes,
                                         isFromSCS);

  //Count number of active particles
  Kokkos::parallel_reduce(capacity(), KOKKOS_LAMBDA(const lid_t& i, lid_t& sum) {
      sum += particle_mask_local(i);
  }, num_ptcls);
  return true;
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes,ExecSpace>::rebuild(kkLidView new_element, 
                                              kkLidView new_particle_elements, 
                                              MemberTypeViews<DataTypes> new_particles) {
  const auto btime = prebarrier();
  Kokkos::Profiling::pushRegion("scs_rebuild");
  Kokkos::Timer timer;
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  //If tryShuffling is on and shuffling works then rebuild is complete
  if (tryShuffling && reshuffle(new_element, new_particle_elements, new_particles))
    return;
  kkLidView new_particles_per_elem("new_particles_per_elem", numRows());
  auto countNewParticles = SCS_LAMBDA(lid_t element_id,lid_t particle_id, bool mask){
    const lid_t new_elem = new_element(particle_id);
    if (new_elem != -1)
      Kokkos::atomic_fetch_add(&(new_particles_per_elem(new_elem)), mask);
  };
  parallel_for(countNewParticles, "countNewParticles");
  // Add new particles to counts
  Kokkos::parallel_for("rebuild_count", new_particle_elements.size(), KOKKOS_LAMBDA(const lid_t& i) {
    const lid_t new_elem = new_particle_elements(i);
    Kokkos::atomic_fetch_add(&(new_particles_per_elem(new_elem)), 1);
  });
  lid_t activePtcls;
  Kokkos::parallel_reduce(numRows(), KOKKOS_LAMBDA(const lid_t& i, lid_t& sum) {
    sum+= new_particles_per_elem(i);
  }, activePtcls);
  //If there are no particles left, then destroy the structure
  if(activePtcls == 0) {
    num_ptcls = 0;
    num_slices = 0;
    capacity_ = 0;
    return;
  }
  lid_t new_num_ptcls = activePtcls;

  int new_C = chooseChunkHeight<ExecSpace>(C_max, new_particles_per_elem);
  int old_C = C_;
  C_ = new_C;
  //Perform sorting
  Kokkos::Profiling::pushRegion("Sorting");
  PairView<ExecSpace> ptcls;
  sigmaSort<ExecSpace>(ptcls,num_elems,new_particles_per_elem, sigma);
  Kokkos::Profiling::popRegion();

  // Number of chunks without vertical slicing
  kkLidView chunk_widths;
  lid_t new_nchunks;
  kkLidView new_row_to_element;
  kkLidView new_element_to_row;
  constructChunks(ptcls, new_nchunks, chunk_widths, new_row_to_element, new_element_to_row);

  lid_t new_num_slices;
  lid_t new_capacity;
  kkLidView new_offsets;
  kkLidView new_slice_to_chunk;
  //Create offsets into each chunk/vertical slice
  constructOffsets(new_nchunks, new_num_slices, chunk_widths, new_offsets, new_slice_to_chunk,
                   new_capacity);

  //Allocate the SCS
  lid_t new_cap = getLastValue<lid_t>(new_offsets);
  kkLidView new_particle_mask("new_particle_mask", new_cap);
  if (swap_size < new_cap) {
    destroyViews<DataTypes>(scs_data_swap);
    CreateViews<DataTypes>(scs_data_swap, new_cap*1.1);
    swap_size = new_cap * 1.1;
  }

  
  /* //Fill the SCS */
  kkLidView interior_slice_of_chunk("interior_slice_of_chunk", new_num_slices);
  Kokkos::parallel_for("set_interior_slice_of_chunk", Kokkos::RangePolicy<>(1,new_num_slices),
    KOKKOS_LAMBDA(const lid_t& i) {
      const lid_t my_chunk = new_slice_to_chunk(i);
      const lid_t prev_chunk = new_slice_to_chunk(i-1);
      interior_slice_of_chunk(i) = my_chunk == prev_chunk;
  });
  lid_t C_local = C_;
  kkLidView element_index("element_index", new_nchunks * C_local);
  Kokkos::parallel_for("set_element_index", new_num_slices, KOKKOS_LAMBDA(const lid_t& i) {
      const lid_t chunk = new_slice_to_chunk(i);
      for (lid_t e = 0; e < C_local; ++e) {
        Kokkos::atomic_fetch_add(&element_index(chunk*C_local + e),
                                 (new_offsets(i) + e) * !interior_slice_of_chunk(i));
      }
  });
  C_ = old_C;
  kkLidView new_indices("new_scs_index", capacity());
  auto copySCS = SCS_LAMBDA(lid_t elm_id, lid_t ptcl_id, bool mask) {
    const lid_t new_elem = new_element(ptcl_id);
    //TODO remove conditional
    if (mask && new_elem != -1) {
      const lid_t new_row = new_element_to_row(new_elem);
      new_indices(ptcl_id) = Kokkos::atomic_fetch_add(&element_index(new_row), new_C);
      const lid_t new_index = new_indices(ptcl_id);
      new_particle_mask(new_index) = 1;
    }
  };
  parallel_for(copySCS);

  CopySCSToSCS<SellCSigma<DataTypes, ExecSpace>, DataTypes>(this, scs_data_swap, scs_data,
                                                            new_element, new_indices);
  //Add new particles
  lid_t num_new_ptcls = new_particle_elements.size(); 
  kkLidView new_particle_indices("new_particle_scs_indices", num_new_ptcls);

  Kokkos::parallel_for("set_new_particle", num_new_ptcls, KOKKOS_LAMBDA(const lid_t& i) {
    lid_t new_elem = new_particle_elements(i);
    lid_t new_row = new_element_to_row(new_elem);
    new_particle_indices(i) = Kokkos::atomic_fetch_add(&element_index(new_row), new_C);
    lid_t new_index = new_particle_indices(i);
    new_particle_mask(new_index) = 1;
  });
  
  if (new_particle_elements.size() > 0)
    CopyNewParticlesToSCS<SellCSigma<DataTypes, ExecSpace>, DataTypes>(this, scs_data_swap,
                                                                       new_particles,
                                                                       num_new_ptcls,
                                                                       new_particle_indices);

  //set scs to point to new values
  C_ = new_C;
  num_ptcls = new_num_ptcls;
  num_chunks = new_nchunks;
  num_slices = new_num_slices;
  capacity_ = new_capacity;
  row_to_element = new_row_to_element;
  element_to_row = new_element_to_row;
  offsets = new_offsets;
  slice_to_chunk = new_slice_to_chunk;
  particle_mask = new_particle_mask;
  MemberTypeViews<DataTypes> tmp = scs_data;
  scs_data = scs_data_swap;
  scs_data_swap = tmp;
  std::size_t tmp_size = current_size;
  current_size = swap_size;
  swap_size = tmp_size;
  if(!comm_rank || comm_rank == comm_size/2)
    fprintf(stderr, "%d ps rebuild (seconds) %f pre-barrier (seconds) %f\n",
        comm_rank, timer.seconds(), btime);
  Kokkos::Profiling::popRegion();
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes,ExecSpace>::printFormat(const char* prefix) const {
  //Transfer everything to the host
  kkLidHostMirror slice_to_chunk_host = deviceToHost(slice_to_chunk);
  kkGidHostMirror element_to_gid_host = deviceToHost(element_to_gid);
  kkLidHostMirror row_to_element_host = deviceToHost(row_to_element);
  kkLidHostMirror offsets_host = deviceToHost(offsets);
  kkLidHostMirror particle_mask_host = deviceToHost(particle_mask);
  char message[10000];
  char* cur = message;
  cur += sprintf(cur, "%s\n", prefix);
  cur += sprintf(cur,"Particle Structures Sell-C-Sigma C: %d sigma: %d V: %d.\n", C_, sigma, V_);
  cur += sprintf(cur,"Number of Elements: %d.\nNumber of Particles: %d.\n", num_elems, num_ptcls);
  cur += sprintf(cur,"Number of Chunks: %d.\nNumber of Slices: %d.\n", num_chunks, num_slices);
  lid_t last_chunk = -1;
  for (lid_t i = 0; i < num_slices; ++i) {
    lid_t chunk = slice_to_chunk_host(i);
    if (chunk != last_chunk) {
      last_chunk = chunk;
      cur += sprintf(cur,"  Chunk %d. Elements", chunk);
      if (element_to_gid_host.size() > 0)
        cur += sprintf(cur,"(GID)");
      cur += sprintf(cur,":");
      for (lid_t row = chunk*C_; row < (chunk+1)*C_; ++row) {
        lid_t elem = row_to_element_host(row);
        cur += sprintf(cur," %d", elem);
        if (element_to_gid_host.size() > 0) {
          cur += sprintf(cur,"(%ld)", element_to_gid_host(elem));
        }
      }
      cur += sprintf(cur,"\n");
    }
    cur += sprintf(cur,"    Slice %d", i);
    for (lid_t j = offsets_host(i); j < offsets_host(i+1); ++j) {
      if ((j - offsets_host(i)) % C_ == 0)
        cur += sprintf(cur," |");
      cur += sprintf(cur," %d", particle_mask_host(j));
    }
    cur += sprintf(cur,"\n");
  }
  printf("%s", message);
}

template <class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::printMetrics() const {

  //Gather metrics
  kkLidView padded_cells("padded_cells", 1);
  kkLidView padded_slices("padded_slices", 1);
  const lid_t league_size = num_slices;
  const lid_t team_size = C_;
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> team_policy;
  const team_policy policy(league_size, team_size);
  auto offsets_cpy = offsets;
  auto slice_to_chunk_cpy = slice_to_chunk;
  auto row_to_element_cpy = row_to_element;
  auto particle_mask_cpy = particle_mask;
  Kokkos::parallel_for("GatherMetrics", policy, KOKKOS_LAMBDA(const team_policy::member_type& thread) {
    const lid_t slice = thread.league_rank();
    const lid_t slice_row = thread.team_rank();
    const lid_t rowLen = (offsets_cpy(slice+1)-offsets_cpy(slice))/team_size;
    const lid_t start = offsets_cpy(slice) + slice_row;
    const lid_t row = slice_to_chunk_cpy(slice) * team_size + slice_row;
    const lid_t element_id = row_to_element_cpy(row);
    lid_t np = 0;
    for (lid_t p = 0; p < rowLen; ++p) {
      const lid_t particle_id = start+(p*team_size);
      const lid_t mask = particle_mask_cpy[particle_id];
      np += !mask;
    }
    Kokkos::atomic_fetch_add(&padded_cells[0],np);
    thread.team_reduce(Kokkos::Sum<lid_t, ExecSpace>(np));
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
  ptr += sprintf(ptr, "Nelems %d, Nchunks %d, Nslices %d, Nptcls %d, Capacity %d, Allocation %d\n",
                 nElems(), num_chunks, num_slices, nPtcls(), capacity(), current_size + swap_size);
  //Padded Cells
  ptr += sprintf(ptr, "Padded Cells <Tot %> %d %.3f\n", num_padded,
                 num_padded * 100.0 / particle_mask.size());
  //Padded Slices
  ptr += sprintf(ptr, "Padded Slices <Tot %> %d %.3f\n", num_padded_slices,
                 num_padded_slices * 100.0 / num_slices);
  //Empty Elements
  ptr += sprintf(ptr, "Empty Rows <Tot %> %d %.3f\n", num_empty_elements,
                 num_empty_elements * 100.0 / numRows());

  printf("%s\n",buffer);
}
  
template <class DataTypes, typename ExecSpace>
template <typename FunctionType>
void SellCSigma<DataTypes, ExecSpace>::parallel_for(FunctionType& fn, std::string name) {
  FunctionType* fn_d;
#ifdef SCS_USE_CUDA
  cudaMalloc(&fn_d, sizeof(FunctionType));
  cudaMemcpy(fn_d,&fn, sizeof(FunctionType), cudaMemcpyHostToDevice);
#else
  fn_d = &fn;
#endif
  const lid_t league_size = num_slices;
  const lid_t team_size = C_;
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> team_policy;
  const team_policy policy(league_size, team_size);
  auto offsets_cpy = offsets;
  auto slice_to_chunk_cpy = slice_to_chunk;
  auto row_to_element_cpy = row_to_element;
  auto particle_mask_cpy = particle_mask;
  Kokkos::parallel_for(name, policy, KOKKOS_LAMBDA(const team_policy::member_type& thread) {
    const lid_t slice = thread.league_rank();
    const lid_t slice_row = thread.team_rank();
    const lid_t rowLen = (offsets_cpy(slice+1)-offsets_cpy(slice))/team_size;
    const lid_t start = offsets_cpy(slice) + slice_row;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_size), [=] (lid_t& j) {
      const lid_t row = slice_to_chunk_cpy(slice) * team_size + slice_row;
      const lid_t element_id = row_to_element_cpy(row);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, rowLen), [&] (lid_t& p) {
        const lid_t particle_id = start+(p*team_size);
        const lid_t mask = particle_mask_cpy[particle_id];
        (*fn_d)(element_id, particle_id, mask);
      });
    });
  });
}

} // end namespace particle_structs

#endif
