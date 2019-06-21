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
namespace particle_structs {

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
    element_gids - (for MPI parallelism) global ids for each element
  */
  SellCSigma(PolicyType& p,
	     lid_t sigma, lid_t vertical_chunk_size, lid_t num_elements, lid_t num_particles,
             kkLidView particles_per_element, kkGidView element_gids);
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
  
  /* Gets the Nth datatype SCS to be indexed by particle id 
     Example: auto segment = scs->get<0>()
   */ 
  template <std::size_t N> 
  Segment<DataTypes, N, ExecSpace> get() {
    if (num_ptcls == 0)
      return Segment<DataTypes, N, ExecSpace>();
    using Type=typename MemberTypeAtIndex<N, DataTypes>::type;
    MemberTypeView<Type>* view = static_cast<MemberTypeView<Type>*>(scs_data[N]);
    return Segment<DataTypes, N, ExecSpace>(*view);
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
  void reshuffle(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
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
  void parallel_for(FunctionType& fn);


  //Prints the format of the SCS labeled by prefix
  void printFormat(const char* prefix = "") const;

  //Do not call these functions:
  void constructChunks(PairView<ExecSpace> ptcls, int& nchunks,
                       kkLidView& chunk_widths, kkLidView& row_element);
  void createGlobalMapping(kkGidView elmGid, kkGidView& elm2Gid, GID_Mapping& elmGid2Lid);
  void constructOffsets(lid_t nChunks, lid_t& nSlices, kkLidView chunk_widths, 
                        kkLidView& offs, kkLidView& s2e, lid_t& capacity);
  void setupParticleMask(kkLidView mask, PairView<ExecSpace> ptcls, kkLidView chunk_widths);

private:
  //Number of Data types
  static constexpr std::size_t num_types = DataTypes::size;

  //The User defined kokkos policy
  PolicyType policy;
  //Chunk size
  lid_t C_;
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

  //mappings from row to element gid and back to row
  kkGidView element_to_gid;
  GID_Mapping element_gid_to_lid;
  //Pointers to the start of each SCS for each data type
  MemberTypeViews<DataTypes> scs_data;

  void destroy(bool destroyGid2Row=true);

};

template <typename ExecSpace> 
void sigmaSort(PairView<ExecSpace>& ptcl_pairs, lid_t num_elems, 
               Kokkos::View<lid_t*,typename ExecSpace::device_type> ptcls_per_elem, 
               int sigma){
  //Make temporary copy of the particle counts for sorting
  Kokkos::resize(ptcl_pairs, num_elems);
  //PairView<ExecSpace> ptcl_pairs("ptcl_pairs", num_elems);
  Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
    ptcl_pairs(i).first = ptcls_per_elem(i);
    ptcl_pairs(i).second = i;
  });
  if (sigma > 1) {
    int i;
#ifdef SORT_ON_DEVICE
    for (i = 0; i < num_elems - sigma; i+=sigma) {
      Kokkos::sort(ptcl_pairs, i, i + sigma);
    }
    Kokkos::sort(ptcl_pairs, i, num_elems);
#else
    typename PairView<ExecSpace>::HostMirror ptcl_pairs_host = deviceToHost(ptcl_pairs);
    MyPair* ptcl_pair_data = ptcl_pairs_host.data();
    for (i = 0; i < num_elems - sigma; i+=sigma) {
      std::sort(ptcl_pair_data + i, ptcl_pair_data + i + sigma);
    }
    std::sort(ptcl_pair_data + i, ptcl_pair_data + num_elems);
    hostToDevice(ptcl_pairs,  ptcl_pair_data);
#endif
  }
}

template <typename ExecSpace>
struct MaxChunkWidths {

  typedef lid_t value_type[];

  typedef typename PairView<ExecSpace>::size_type size_type;

  size_type value_count;

  PairView<ExecSpace> widths;

  lid_t C;

  MaxChunkWidths(const PairView<ExecSpace>& widths_, const int C_, int nchunks) :
    value_count(nchunks), widths(widths_), C(C_) {}

  KOKKOS_INLINE_FUNCTION void operator()(const size_type i, value_type mx) const {
    const int index = i / C;
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
void SellCSigma<DataTypes, ExecSpace>::constructChunks(PairView<ExecSpace> ptcls, int& nchunks, 
                                                       kkLidView& chunk_widths,
                                                       kkLidView& row_element) {
  nchunks = num_elems / C_ + (num_elems % C_ != 0);
  Kokkos::resize(chunk_widths, nchunks);
  Kokkos::resize(row_element, nchunks * C_);

  Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const int& i) {
    row_element(i) = ptcls(i).second;
  });
  Kokkos::parallel_for(Kokkos::RangePolicy<>(num_elems, nchunks * C_),
                       KOKKOS_LAMBDA(const int& i) {
    row_element(i) = i;
  });

  MaxChunkWidths<ExecSpace> maxer(ptcls, C_, nchunks);
  lid_t* widths = new lid_t[nchunks];
  Kokkos::parallel_reduce(num_elems, maxer, widths);
  hostToDevice<lid_t>(chunk_widths, widths);
  delete [] widths;
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::createGlobalMapping(kkGidView elmGid,kkGidView& elm2Gid, 
                                                           GID_Mapping& elmGid2Lid) {
  Kokkos::resize(elm2Gid, numRows());
  Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const int& i) {
    const gid_t gid = elmGid(i);
    elm2Gid(i) = gid;
    elmGid2Lid.insert(gid, i);
  });
  Kokkos::parallel_for(Kokkos::RangePolicy<>(num_elems, numRows()), KOKKOS_LAMBDA(const int& i) {
    elm2Gid(i) = -1;
  });
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::constructOffsets(lid_t nChunks, lid_t& nSlices, 
                                                        kkLidView chunk_widths, kkLidView& offs,
                                                        kkLidView& s2c, lid_t& cap) {
  kkLidView slices_per_chunk("slices_per_chunk", nChunks);
  const int V_local = V_;
  Kokkos::parallel_for(nChunks, KOKKOS_LAMBDA(const int& i) {
    const lid_t width = chunk_widths(i);
    const lid_t val1 = width / V_local;
    const lid_t val2 = width % V_local;
    const bool val3 = val2 != 0;
    slices_per_chunk(i) = val1 + val3;
  });
  kkLidView offset_nslices("offset_nslices",nChunks+1);
  Kokkos::parallel_scan(nChunks, KOKKOS_LAMBDA(const int& i, int& cur, const bool& final) {
    cur += slices_per_chunk(i);
    if (final)
      offset_nslices(i+1) += cur;
  });

  nSlices = getLastValue<lid_t>(offset_nslices);
  Kokkos::resize(offs,nSlices + 1);
  Kokkos::resize(s2c, nSlices);
  kkLidView slice_size("slice_size", nSlices);
  const int nat_size = V_*C_;
  const int C_local = C_;
  Kokkos::parallel_for(nChunks, KOKKOS_LAMBDA(const int& i) {
    const int start = offset_nslices(i);
    const int end = offset_nslices(i+1);
    for (int j = start; j < end; ++j) {
      s2c(j) = i;
      const lid_t rem = chunk_widths(i) % V_local;
      const lid_t val = rem + (rem==0)*V_local;
      const bool is_last = (j == end-1);
      slice_size(j) = (!is_last) * nat_size;
      slice_size(j) += (is_last) * (val) * C_local;
    }
  });
  Kokkos::parallel_scan(nSlices, KOKKOS_LAMBDA(const int& i, lid_t& cur, const bool final) {
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
  Kokkos::parallel_for(num_slices-1, KOKKOS_LAMBDA(const int& i) {
    const int my_chunk = slice_to_chunk_cpy(i);
    const int next_chunk = slice_to_chunk_cpy(i+1);
    if (my_chunk != next_chunk) {
      chunk_starts(next_chunk) = offsets_cpy(i+1);
    }
  });
  //Fill the SCS
  const int league_size = num_chunks;
  const int team_size = C_;
  const int ne = num_elems;
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> team_policy;
  const team_policy policy(league_size, team_size);
  auto row_to_element_cpy = row_to_element;
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const team_policy::member_type& thread) {
    const int chunk = thread.league_rank();
    const int chunk_row = thread.team_rank();
    const int rowLen = chunk_widths(chunk);
    const int start = chunk_starts(chunk) + chunk_row;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_size), [=] (int& j) {
      const int row = chunk * team_size + chunk_row;
      const int element_id = row_to_element_cpy(row);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, rowLen), [&] (int& p) {
        const int particle_id = start+(p*team_size);
        if (element_id < ne)
          mask(particle_id) =  p < ptcls(row).first;
      });
    });
  });

}

template<class DataTypes, typename ExecSpace>
SellCSigma<DataTypes, ExecSpace>::SellCSigma(PolicyType& p, lid_t sig, lid_t v, lid_t ne, 
                                             lid_t np, kkLidView ptcls_per_elem, 
                                             kkGidView element_gids)  : policy(p),
                                                                        element_gid_to_lid(ne) {
  C_ = policy.team_size();
  sigma = sig;
  V_ = v;
  num_elems = ne;
  num_ptcls = np;
  
  printf("Building SCS with C: %d sigma: %d V: %d\n",C_,sigma,V_);
  //Perform sorting
  PairView<ExecSpace> ptcls;
  sigmaSort<ExecSpace>(ptcls, num_elems,ptcls_per_elem, sigma);

  // Number of chunks without vertical slicing
  kkLidView chunk_widths;
  constructChunks(ptcls, num_chunks, chunk_widths, row_to_element);

  if (element_gids.size() > 0) {
    createGlobalMapping(element_gids, element_to_gid, element_gid_to_lid);
  }

  //Create offsets into each chunk/vertical slice
  constructOffsets(num_chunks, num_slices, chunk_widths, offsets, slice_to_chunk,capacity_);

  //Allocate the SCS  
  lid_t cap = getLastValue<lid_t>(offsets);
  Kokkos::resize(particle_mask, cap);
  CreateViews<DataTypes>(scs_data, cap);

  if (np > 0)
    setupParticleMask(particle_mask, ptcls, chunk_widths);
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::destroy(bool destroyGid2Row) {
  DestroyViews<DataTypes>(scs_data+0);
  if (destroyGid2Row)
    element_gid_to_lid.clear();
}
template<class DataTypes, typename ExecSpace>
SellCSigma<DataTypes, ExecSpace>::~SellCSigma() {
  destroy();
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::migrate(kkLidView new_element, kkLidView new_process) {
  /********* Send # of particles being sent to each process *********/
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  kkLidView num_send_particles("num_send_particles", comm_size);
  auto count_sending_particles = SCS_LAMBDA(int element_id, int particle_id, bool mask) {
    const int process = new_process(particle_id);
    Kokkos::atomic_fetch_add(&(num_send_particles(process)), mask * (process != comm_rank));
  };
  parallel_for(count_sending_particles);
  kkLidView num_recv_particles("num_recv_particles", comm_size);
  PS_Comm_Alltoall(num_send_particles, 1, num_recv_particles, 1, MPI_COMM_WORLD);

  int num_sending_to = 0, num_receiving_from = 0;
  Kokkos::parallel_reduce("sum_senders", comm_size, KOKKOS_LAMBDA (const int& i, int& lsum ) {
      lsum += (num_send_particles(i) > 0);
  }, num_sending_to);
  Kokkos::parallel_reduce("sum_receivers", comm_size, KOKKOS_LAMBDA (const int& i, int& lsum ) {
      lsum += (num_recv_particles(i) > 0);
  }, num_receiving_from);
  
  /********** Send particle information to new processes **********/
  //Perform an ex-sum on num_send_particles & num_recv_particles
  kkLidView offset_send_particles("offset_send_particles", comm_size+1);
  kkLidView offset_send_particles_temp("offset_send_particles_temp", comm_size + 1);
  kkLidView offset_recv_particles("offset_recv_particles", comm_size+1);
  Kokkos::parallel_scan(comm_size, KOKKOS_LAMBDA(const int& i, int& num, const bool& final) {
    num += num_send_particles(i);
    if (final) {
      offset_send_particles(i+1) += num;
      offset_send_particles_temp(i+1) += num;
    }
  });
  Kokkos::parallel_scan(comm_size, KOKKOS_LAMBDA(const int& i, int& num, const bool& final) {
    num += num_recv_particles(i);
    if (final)
      offset_recv_particles(i+1) += num;
  });
  kkLidHostMirror offset_send_particles_host = deviceToHost(offset_send_particles);
  kkLidHostMirror offset_recv_particles_host = deviceToHost(offset_recv_particles);

  //Create arrays for particles being sent
  int np_send = offset_send_particles_host(comm_size);
  kkLidView send_element("send_element", np_send);
  MemberTypeViews<DataTypes> send_particle;
  //Allocate views for each data type into send_particle[type]
  CreateViews<DataTypes>(send_particle, np_send);
  kkLidView send_index("send_particle_index", capacity());
  auto element_to_gid_local = element_to_gid;
  auto gatherParticlesToSend = SCS_LAMBDA(int element_id, int particle_id, int mask) {
    const int process = new_process(particle_id);
    if (mask && process != comm_rank) {
      send_index(particle_id) =
        Kokkos::atomic_fetch_add(&(offset_send_particles_temp(process)),1);
      const int index = send_index(particle_id);
      send_element(index) = element_to_gid_local(new_element(particle_id));
    }
  };
  parallel_for(gatherParticlesToSend);
  //Copy the values from scs_data[type][particle_id] into send_particle[type](index) for each data type
  CopyParticlesToSend<SellCSigma<DataTypes, ExecSpace>, DataTypes>(this, send_particle, scs_data,
                                                                   new_process,
                                                                   send_index);
  
  //Create arrays for particles being received
  int np_recv = offset_recv_particles_host(comm_size);
  kkLidView recv_element("recv_element", np_recv);
  MemberTypeViews<DataTypes> recv_particle;
  //Allocate views for each data type into recv_particle[type]
  CreateViews<DataTypes>(recv_particle, np_recv);

  //Get pointers to the data for MPI calls
  int send_num = 0, recv_num = 0;
  int num_sends = num_sending_to * (num_types + 1);
  int num_recvs = num_receiving_from * (num_types + 1);
  MPI_Request* send_requests = new MPI_Request[num_sends];
  MPI_Request* recv_requests = new MPI_Request[num_recvs];
  //Send the particles to each neighbor
  for (int i = 0; i < comm_size; ++i) {
    if (i == comm_rank)
      continue;
    
    //Sending
    int num_send = offset_send_particles_host(i+1) - offset_send_particles_host(i);
    if (num_send > 0) {
      int start_index = offset_send_particles_host(i);
      PS_Comm_Isend(send_element, start_index, num_send, i, 0, MPI_COMM_WORLD, 
                    send_requests +send_num);
      send_num++;
      SendViews<DataTypes>(send_particle, start_index, num_send, i, 1,
                           send_requests + send_num);
      send_num+=num_types;
    }
    //Receiving
    int num_recv = offset_recv_particles_host(i+1) - offset_recv_particles_host(i);
    if (num_recv > 0) {
      int start_index = offset_recv_particles_host(i);
      PS_Comm_Irecv(recv_element, start_index, num_recv, i, 0, MPI_COMM_WORLD, 
                    recv_requests + recv_num);
      recv_num++;
      RecvViews<DataTypes>(recv_particle,start_index, num_recv, i, 1,
                           recv_requests + recv_num);
      recv_num+=num_types;
    }
  }
  PS_Comm_Waitall<ExecSpace>(num_sends, send_requests, MPI_STATUSES_IGNORE);
  PS_Comm_Waitall<ExecSpace>(num_recvs, recv_requests, MPI_STATUSES_IGNORE);
  delete [] send_requests;
  delete [] recv_requests;
  DestroyViews<DataTypes>(send_particle+0);

  /********** Convert the received element from element gid to element lid *********/
  auto element_gid_to_lid_local = element_gid_to_lid;
  Kokkos::parallel_for(recv_element.size(), KOKKOS_LAMBDA(const int& i) {
    const gid_t gid = recv_element(i);
    const int index = element_gid_to_lid_local.find(gid);
    recv_element(i) = element_gid_to_lid_local.value_at(index);
  });
  
  /********** Set particles that went sent to non existent on this process *********/
  auto removeSentParticles = SCS_LAMBDA(int element_id, int particle_id, int mask) {
    const bool notSent = new_process(particle_id) != comm_rank;
    const lid_t elm = new_element(particle_id);
    Kokkos::atomic_fetch_add(&new_element(particle_id), -1 * (elm + 1) * notSent);
  };
  parallel_for(removeSentParticles);

  /********** Combine and shift particles to their new destination **********/
  rebuild(new_element, recv_element, recv_particle);

  DestroyViews<DataTypes>(recv_particle+0);
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes,ExecSpace>::reshuffle(kkLidView new_element, 
                                                kkLidView new_particle_elements, 
                                                MemberTypeViews<DataTypes> new_particles) {
  //For now call rebuild
  rebuild(new_element, new_particle_elements, new_particles);
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes,ExecSpace>::rebuild(kkLidView new_element, 
                                              kkLidView new_particle_elements, 
                                              MemberTypeViews<DataTypes> new_particles) {
  kkLidView new_particles_per_elem("new_particles_per_elem", numRows());
  auto countNewParticles = SCS_LAMBDA(int element_id,int particle_id, bool mask){
    const lid_t new_elem = new_element(particle_id);
    if (new_elem != -1)
      Kokkos::atomic_fetch_add(&(new_particles_per_elem(new_elem)), mask);
  };
  parallel_for(countNewParticles);
  // Add new particles to counts
  Kokkos::parallel_for(new_particle_elements.size(), KOKKOS_LAMBDA(const int& i) {
    const lid_t new_elem = new_particle_elements(i);
    Kokkos::atomic_fetch_add(&(new_particles_per_elem(new_elem)), 1);
  });
  lid_t activePtcls;
  Kokkos::parallel_reduce(numRows(), KOKKOS_LAMBDA(const int& i, lid_t& sum) {
    sum+= new_particles_per_elem(i);
  }, activePtcls);
  //If there are no particles left, then destroy the structure
  if(activePtcls == 0) {
    num_ptcls = 0;
    num_slices = 0;
    capacity_ = 0;
    return;
  }
  int new_num_ptcls = activePtcls;
  
  //Perform sorting
  PairView<ExecSpace> ptcls;
  sigmaSort<ExecSpace>(ptcls,num_elems,new_particles_per_elem, sigma);

  // Number of chunks without vertical slicing
  kkLidView chunk_widths;
  int new_nchunks;
  kkLidView new_row_to_element;
  constructChunks(ptcls, new_nchunks, chunk_widths, new_row_to_element);

  //Create mapping from element to its new row
  kkLidView element_to_new_row("element_to_new_row", num_chunks*C_);
  Kokkos::parallel_for(num_chunks*C_, KOKKOS_LAMBDA(const int& i) {
    const lid_t elem = new_row_to_element(i);
    element_to_new_row(elem) = i;
  });

  int new_num_slices;
  int new_capacity;
  kkLidView new_offsets;
  kkLidView new_slice_to_chunk;
  //Create offsets into each chunk/vertical slice
  constructOffsets(new_nchunks, new_num_slices, chunk_widths, new_offsets, new_slice_to_chunk,
                   new_capacity);

  //Allocate the SCS
  int new_cap = getLastValue<lid_t>(new_offsets);
  kkLidView new_particle_mask("new_particle_mask", new_cap);
  MemberTypeViews<DataTypes> new_scs_data;
  CreateViews<DataTypes>(new_scs_data, new_cap);

  
  /* //Fill the SCS */
  kkLidView interior_slice_of_chunk("interior_slice_of_chunk", new_num_slices);
  Kokkos::parallel_for(Kokkos::RangePolicy<>(1,new_num_slices), KOKKOS_LAMBDA(const int& i) {
      const int my_chunk = new_slice_to_chunk(i);
      const int prev_chunk = new_slice_to_chunk(i-1);
      interior_slice_of_chunk(i) = my_chunk == prev_chunk;
  });
  lid_t C_local = C_;
  kkLidView element_index("element_index", new_nchunks * C_);
  Kokkos::parallel_for(new_num_slices, KOKKOS_LAMBDA(const int& i) {
      const int chunk = new_slice_to_chunk(i);
      for (int e = 0; e < C_local; ++e) {
        Kokkos::atomic_fetch_add(&element_index(chunk*C_local + e),
                                 (new_offsets(i) + e) * !interior_slice_of_chunk(i));
      }
  });
  kkLidView new_indices("new_scs_index", capacity());
  auto copySCS = SCS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
    const lid_t new_elem = new_element(ptcl_id);
    //TODO remove conditional
    if (mask && new_elem != -1) {
      const lid_t new_row = element_to_new_row(new_elem);
      new_indices(ptcl_id) = Kokkos::atomic_fetch_add(&element_index(new_row), C_local);
      const lid_t new_index = new_indices(ptcl_id);
      new_particle_mask(new_index) = 1;
    }
  };
  parallel_for(copySCS);

  CopySCSToSCS<SellCSigma<DataTypes, ExecSpace>, DataTypes>(this, new_scs_data, scs_data,
                                                            new_element, new_indices);
  //Add new particles
  int num_new_ptcls = new_particle_elements.size(); 
  kkLidView new_particle_indices("new_particle_scs_indices", num_new_ptcls);

  Kokkos::parallel_for(num_new_ptcls, KOKKOS_LAMBDA(const int& i) {
    int new_elem = new_particle_elements(i);
    int new_row = element_to_new_row(new_elem);
    new_particle_indices(i) = Kokkos::atomic_fetch_add(&element_index(new_row), C_local);
    int new_index = new_particle_indices(i);
    new_particle_mask(new_index) = 1;
  });
  
  if (new_particle_elements.size() > 0)
    CopyNewParticlesToSCS<SellCSigma<DataTypes, ExecSpace>, DataTypes>(this, new_scs_data,
                                                                       new_particles,
                                                                       num_new_ptcls,
                                                                       new_particle_indices);
  //Destroy old scs
  destroy(false);

  //set scs to point to new values
  num_ptcls = new_num_ptcls;
  num_chunks = new_nchunks;
  num_slices = new_num_slices;
  capacity_ = new_capacity;
  row_to_element = new_row_to_element;
  offsets = new_offsets;
  slice_to_chunk = new_slice_to_chunk;
  particle_mask = new_particle_mask;
  for (size_t i = 0; i < num_types; ++i)
    scs_data[i] = new_scs_data[i];
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
  int last_chunk = -1;
  for (int i = 0; i < num_slices; ++i) {
    int chunk = slice_to_chunk_host(i);
    if (chunk != last_chunk) {
      last_chunk = chunk;
      cur += sprintf(cur,"  Chunk %d. Elements", chunk);
      if (element_to_gid_host.size() > 0)
        cur += sprintf(cur,"(GID)");
      cur += sprintf(cur,":");
      for (int row = chunk*C_; row < (chunk+1)*C_; ++row) {
        lid_t elem = row_to_element_host(row);
        cur += sprintf(cur," %d", elem);
        if (element_to_gid_host.size() > 0) {
          cur += sprintf(cur,"(%ld)", element_to_gid_host(elem));
        }
      }
      cur += sprintf(cur,"\n");
    }
    cur += sprintf(cur,"    Slice %d", i);
    for (int j = offsets_host(i); j < offsets_host(i+1); ++j) {
      if ((j - offsets_host(i)) % C_ == 0)
        cur += sprintf(cur," |");
      cur += sprintf(cur," %d", particle_mask_host(j));
    }
    cur += sprintf(cur,"\n");
  }
  printf("%s", message);
}

template <class DataTypes, typename ExecSpace>
template <typename FunctionType>
void SellCSigma<DataTypes, ExecSpace>::parallel_for(FunctionType& fn) {
  FunctionType* fn_d;
#ifdef SCS_USE_CUDA
  cudaMalloc(&fn_d, sizeof(FunctionType));
  cudaMemcpy(fn_d,&fn, sizeof(FunctionType), cudaMemcpyHostToDevice);
#else
  fn_d = &fn;
#endif
  const int league_size = num_slices;
  const int team_size = C_;
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> team_policy;
  const team_policy policy(league_size, team_size);
  auto offsets_cpy = offsets;
  auto slice_to_chunk_cpy = slice_to_chunk;
  auto row_to_element_cpy = row_to_element;
  auto particle_mask_cpy = particle_mask;
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const team_policy::member_type& thread) {
    const int slice = thread.league_rank();
    const int slice_row = thread.team_rank();
    const int rowLen = (offsets_cpy(slice+1)-offsets_cpy(slice))/team_size;
    const int start = offsets_cpy(slice) + slice_row;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_size), [=] (int& j) {
      const int row = slice_to_chunk_cpy(slice) * team_size + slice_row;
      const int element_id = row_to_element_cpy(row);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, rowLen), [&] (int& p) {
        const int particle_id = start+(p*team_size);
        const int mask = particle_mask_cpy[particle_id];
        (*fn_d)(element_id, particle_id, mask);
      });
    });
  });
}

} // end namespace particle_structs

#endif
