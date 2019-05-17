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
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Sort.hpp>
#include <mpi.h>
#include <unordered_map>
namespace particle_structs {

template <typename ExecSpace> 
using PairView=Kokkos::View<Kokkos::pair<lid_t,lid_t>*, typename ExecSpace::device_type>;


template<class DataTypes, typename ExecSpace = Kokkos::DefaultExecutionSpace>
class SellCSigma {
 public:
  typedef Kokkos::TeamPolicy<ExecSpace> PolicyType ;
  typedef Kokkos::View<lid_t*, typename ExecSpace::device_type> kkLidView;
  typedef Kokkos::View<gid_t*, typename ExecSpace::device_type> kkGidView;
  typedef typename kkLidView::HostMirror kkLidHostMirror;


  SellCSigma(PolicyType& p,
	     int sigma, int vertical_chunk_size, int num_elements, int num_particles,
             int* particles_per_element, std::vector<int>* particle_id_bins, int* element_gids);
  SellCSigma(PolicyType& p,
	     lid_t sigma, lid_t vertical_chunk_size, lid_t num_elements, lid_t num_particles,
             kkLidView particles_per_element, kkGidView element_gids);
  ~SellCSigma();

  //Returns the capacity of the scs including padding
  lid_t capacity() const { return offsets[num_slices];}
  //Returns the number of rows in the scs including padded rows
  lid_t numRows() const {return num_chunks * C;}

  //Gets the Nth member type SCS
  template <std::size_t N>
  typename MemberTypeAtIndex<N,DataTypes>::type* getSCS();

 /*
   tempalte <std::size_t N>
   typename SCS<DataTypes, N> get();
  */

  void printFormat(const char* prefix = "") const;
  void printFormatDevice(const char* prefix = "") const;

  /* Migrates each particle to new_process and to new_element
     Calls rebuildSCS to recreate the SCS after migrating particles
  */
  void migrate(kkLidView new_element, kkLidView new_process);

  /*
    Reshuffles the scs values to the element in new_element[i]
    Calls rebuildSCS if there is not enough space for the shuffle
  */
  void reshuffleSCS(int* new_element);
  /*
    Rebuilds a new SCS where particles move to the element in new_element[i]
  */
  void rebuildSCS(int* new_element, kkLidView new_particle_elements = kkLidView(), 
                  MemberTypeViews<DataTypes> new_particles = NULL);

  //Number of Data types
  static constexpr std::size_t num_types = DataTypes::size;

  //The User defined kokkos policy
  PolicyType policy;
  //Chunk size
  int C;
  //Vertical slice size
  int V;
  //Sorting chunk size
  int sigma;
  //Number of chunks
  int num_chunks;
  //Number of slices
  int num_slices;
  //Total entries
  int num_elems;
  //Total particles
  int num_ptcls;
  //chunk_element stores the id of the first row in the chunk
  //  This only matters for vertical slicing so that each slice can determine which row
  //  it is a part of.
  int* slice_to_chunk;
  kkLidView slice_to_chunk_v;
  //particle_mask true means there is a particle at this location, false otherwise
  int* particle_mask;
  kkLidView particle_mask_v;
  //offsets into the scs structure
  int* offsets;
  kkLidView offsets_v;

  //map from row to element
  // row = slice_to_chunk[slice] + row_in_chunk
  int* row_to_element;
  kkLidView row_to_element_v;

  //mappings from row to element gid and back to row
  int* row_to_element_gid;
  kkLidView row_to_element_gid_v;
  typedef std::unordered_map<int, int> GID_Mapping;
  GID_Mapping element_gid_to_row;
  typedef Kokkos::UnorderedMap<gid_t, lid_t, typename ExecSpace::device_type> GID_Mapping_KK;
  GID_Mapping_KK element_gid_to_row_v;
 //Kokkos Views
#ifdef KOKKOS_ENABLED
  void transferToDevice();
  template <typename FunctionType>
  void parallel_for(FunctionType& fn);

 kkLidView offsets_d;
 kkLidView slice_to_chunk_d;
 kkLidView num_particles_d;
 kkLidView chunksz_d;
 kkLidView slicesz_d;
 kkLidView num_elems_d;
 kkLidView row_to_element_d;
 kkLidView particle_mask_d;

#endif
 private: 
  //Pointers to the start of each SCS for each data type
  MemberTypeArray<DataTypes> scs_data;
  MemberTypeViews<DataTypes> scs_data_v;

  SellCSigma() {throw 1;}
  SellCSigma(const SellCSigma&) {throw 1;}
  SellCSigma& operator=(const SellCSigma&) {throw 1;}
  void destroySCS(bool destroyGid2Row=true);

  void constructChunks(std::pair<int,int>* ptcls, int& nchunks, int& nslices, int*& chunk_widths, 
                       int*& row_element);
  void constructChunks(PairView<ExecSpace> ptcls, int& nchunks,
                       kkLidView& chunk_widths, kkLidView& row_element);

  void createGlobalMapping(int* row2Elm, int* elmGid, int*& row2ElmGid, GID_Mapping& elmGid2Row);
  void createGlobalMapping(kkLidView row2Elm, kkGidView elmGid, kkLidView& row2ElmGid, 
                           GID_Mapping_KK& elmGid2Row);
  void constructOffsets(int nChunks, int nSlices, int* chunk_widths, int*& offs, int*& s2e);
  void constructOffsets(lid_t nChunks, lid_t& nSlices, kkLidView chunk_widths, 
                        kkLidView& offs, kkLidView& s2e);
};


void sigmaSort(int num_elems, int* ptcls_per_elem, int sigma, 
	       std::pair<int, int>*& ptcl_pairs);


template <typename ExecSpace> 
PairView<ExecSpace> sigmaSort(lid_t num_elems, 
                              Kokkos::View<lid_t*,typename ExecSpace::device_type> ptcls_per_elem, 
                              int sigma){
  //Make temporary copy of the particle counts for sorting
  PairView<ExecSpace> ptcl_pairs("ptcl_pairs", num_elems);
  Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
    ptcl_pairs(i).first = ptcls_per_elem(i);
    ptcl_pairs(i).second = i;
  });
  /*
    Sorting Disabled due to kokkos problems
  int i;
  if (sigma > 1) {
    for (i = 0; i < num_elems - sigma; i+=sigma) {
      Kokkos::sort(ptcl_pairs, i, i + sigma);
    }
    Kokkos::sort(ptcl_pairs, i, num_elems);
  }
  */
  return ptcl_pairs;
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::constructChunks(std::pair<int,int>* ptcls, int& nchunks, 
							 int& nslices, int*& chunk_widths, 
							 int*& row_element) {
  nchunks = num_elems / C + (num_elems % C != 0);
  nslices = 0;
  chunk_widths = new int[nchunks];
  row_element = new int[nchunks * C];

  //Add chunks for vertical slicing
  int i;
  for (i = 0; i < nchunks; ++i) {
    chunk_widths[i] = 0;
    for (int j = i * C; j < (i + 1) * C && j < num_elems; ++j)  {
      row_element[j] = ptcls[j].second;
      chunk_widths[i] = std::max(chunk_widths[i],ptcls[j].first);
    }
    int num_vertical_slices = chunk_widths[i] / V + (chunk_widths[i] % V != 0);
    nslices += num_vertical_slices;
  }

  //Set the padded row_to_element values
  for (i = num_elems; i < nchunks * C; ++i) {
    row_element[i] = i;
  }
}
template<class DataTypes, typename ExecSpace> 
void SellCSigma<DataTypes, ExecSpace>::constructChunks(PairView<ExecSpace> ptcls, int& nchunks, 
                                                       kkLidView& chunk_widths,
                                                       kkLidView& row_element) {
  nchunks = num_elems / C + (num_elems % C != 0);
  Kokkos::resize(chunk_widths, nchunks);
  Kokkos::resize(row_element, nchunks * C);

  //Add chunks for vertical slicing
  typedef Kokkos::TeamPolicy<> TeamPolicy;
  const TeamPolicy pol(nchunks, C);
  Kokkos::parallel_for(pol, KOKKOS_LAMBDA(const TeamPolicy::member_type& thread) {
    const int chunk = thread.league_rank();
    const int chunk_row = thread.team_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, C), KOKKOS_LAMBDA(const int& j) {
      row_element(chunk_row) = ptcls(chunk_row).second;
      //NOTE this is happening twice per chunk_row
    });
    int maxL;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, C), KOKKOS_LAMBDA(const int& j, lid_t& mx) {
      //TODO remove conditional
      if (ptcls(chunk_row).first > mx)
        mx = ptcls(chunk_row).first;
      }, Kokkos::Max<lid_t>(maxL));
    chunk_widths(chunk) = maxL;
  });

  Kokkos::parallel_for(Kokkos::RangePolicy<>(num_elems, nchunks * C), KOKKOS_LAMBDA(const int& i) {
    row_element(i) = i;
  });
}


template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::createGlobalMapping(int* row2Elm, int* elmGid,
                                                           int*& row2ElmGid, 
                                                           GID_Mapping& elmGid2Row) {
  row2ElmGid = new int[num_chunks*C];
  for (int i = 0; i < num_elems; ++i) {
    int gid = elmGid[row2Elm[i]];
    row2ElmGid[i] = gid;
    elmGid2Row[gid] = i;
  }
  for (int j = num_elems; j < num_chunks*C; ++j) {
    row2ElmGid[j] = -1;
  }
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::createGlobalMapping(kkLidView row2Elm, kkGidView elmGid,
                                                           kkLidView& row2ElmGid, 
                                                           GID_Mapping_KK& elmGid2Row) {
  Kokkos::resize(row2ElmGid, num_chunks*C);
  Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const int& i) {
    const int elm = row2Elm(i);
    const int gid = elmGid(elm);
    row2ElmGid(i) = gid;
    elmGid2Row.insert(gid, i);
  });
  Kokkos::parallel_for(Kokkos::RangePolicy<>(num_elems, num_chunks*C), KOKKOS_LAMBDA(const int& i) {
    row2ElmGid(i) = -1;
  });
}


template<class DataTypes, typename ExecSpace>
 void SellCSigma<DataTypes, ExecSpace>::constructOffsets(int nChunks, int nSlices, 
							 int* chunk_widths, int*& offs, int*& s2e) {
  offs = new int[nSlices + 1];
  s2e = new int[nSlices];
  offs[0] = 0;
  int index = 1;
  for (int i = 0; i < nChunks; ++i) {
    for (int j = V; j <= chunk_widths[i]; j+=V) {
      s2e[index-1] = i;
      offs[index] = offs[index-1] + V * C;
      ++index;
    }
    int rem = chunk_widths[i] % V;
    if (rem > 0) {
      s2e[index-1] = i;
      offs[index] = offs[index-1] + rem * C;
      ++index;
    }
  }
  PS_ALWAYS_ASSERT(index == nSlices + 1);
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::constructOffsets(lid_t nChunks, lid_t& nSlices, 
                                                        kkLidView chunk_widths, kkLidView& offs,
                                                        kkLidView& s2c) {
  kkLidView slices_per_chunk("slices_per_chunk", nChunks);
  Kokkos::parallel_for(nChunks, KOKKOS_LAMBDA(const int& i) {
    slices_per_chunk(i) = chunk_widths(i) / V + (chunk_widths(i) % V != 0);
  });
  Kokkos::parallel_reduce(nChunks, KOKKOS_LAMBDA(const int& i, lid_t& sum) {
      sum+= slices_per_chunk(i);
  }, nSlices);
  kkLidView offset_nslices("offset_nslices",nChunks+1);
  Kokkos::parallel_scan(nChunks, KOKKOS_LAMBDA(const int& i, int& cur, const bool final) {
    cur += slices_per_chunk(i);
    offset_nslices(i+1) += cur * final;
  });

  Kokkos::resize(offs,nSlices + 1);
  Kokkos::resize(s2c, nSlices);
  kkLidView slice_size("slice_size", nSlices);
  const int nat_size = V*C;
  Kokkos::parallel_for(nChunks, KOKKOS_LAMBDA(const int& i) {
    const int start = offset_nslices(i);
    const int end = offset_nslices(i+1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(start,end), KOKKOS_LAMBDA(const int& j) {
      s2c(j) = i;
      const lid_t rem = chunk_widths(i) % V;
      slice_size(j) = (j!=end-1)*nat_size  + (j==end-1)*(rem + (rem==0) * V) * C;
    });
  });

  Kokkos::parallel_scan(nSlices, KOKKOS_LAMBDA(const int& i, lid_t& cur, const bool final) {
    cur += slice_size(i);
    offs(i+1) += cur * final;
  });
}

template<class DataTypes, typename ExecSpace>
 SellCSigma<DataTypes, ExecSpace>::SellCSigma(Kokkos::TeamPolicy<ExecSpace>& p,
					       int sig, int v, int ne, int np,
					       int* ptcls_per_elem, std::vector<int>* ids,
                                               int* element_gids)  : policy(p) {
  C = policy.team_size();
  sigma = sig;
  V = v;
  num_elems = ne;
  num_ptcls = np;
  
  printf("Building SCS with C: %d sigma: %d V: %d\n",C,sigma,V);

  //Perform sorting
  std::pair<int, int>* ptcls;
  sigmaSort(num_elems,ptcls_per_elem, sigma, ptcls);

  //Number of chunks without vertical slicing
  int* chunk_widths;
  constructChunks(ptcls, num_chunks, num_slices, chunk_widths, row_to_element);

  row_to_element_gid = NULL;
  if (element_gids) {
    createGlobalMapping(row_to_element, element_gids, row_to_element_gid, element_gid_to_row);
  }

  //Create offsets into each chunk/vertical slice
  constructOffsets(num_chunks, num_slices, chunk_widths, offsets, slice_to_chunk);
  delete [] chunk_widths;
  
  //Allocate the SCS
  particle_mask = new int[offsets[num_slices]];
  CreateArrays<DataTypes>(scs_data, offsets[num_slices]);

  //Fill the SCS
  int index = 0;
  int start = 0;
  int old_elem = 0;
  int i;
  for (i = 0; i < num_slices; ++i) { //for each slice
    int elem = slice_to_chunk[i];
    if (old_elem!=elem) {
      old_elem = elem;
      start = 0;
    }
    int width = (offsets[i + 1] - offsets[i]) / C;
    for (int j = 0; j < width; ++j) { //for the width of that slice
      for (int k = elem * C; k < (elem + 1) * C; ++k) { //for each row in the slice
        if (k < num_elems && ptcls[k].first > start + j) {
          int ent_id = ptcls[k].second;
          int ptcl = ids[ent_id][start + j];
          PS_ALWAYS_ASSERT(ptcl<np);
          particle_mask[index++] = true;
        }
        else
          particle_mask[index++] = false;
      }
    }
    start+=width;
  }

  delete [] ptcls;
}

template<class DataTypes, typename ExecSpace>
SellCSigma<DataTypes, ExecSpace>::SellCSigma(PolicyType& p, lid_t sig, lid_t v, lid_t ne, 
                                             lid_t np, kkLidView ptcls_per_elem, 
                                             kkGidView element_gids)  : policy(p) {
  C = policy.team_size();
  sigma = sig;
  V = v;
  num_elems = ne;
  num_ptcls = np;
  
  printf("Building SCS with C: %d sigma: %d V: %d\n",C,sigma,V);

  //Perform sorting
  PairView<ExecSpace> ptcls = sigmaSort<ExecSpace>(num_elems,ptcls_per_elem, sigma);

  
  // Number of chunks without vertical slicing
  kkLidView chunk_widths;
  constructChunks(ptcls, num_chunks, chunk_widths, row_to_element_v);

  if (element_gids.size() > 0) {
    createGlobalMapping(row_to_element_v, element_gids, row_to_element_gid_v, element_gid_to_row_v);
  }

  //Create offsets into each chunk/vertical slice
  constructOffsets(num_chunks, num_slices, chunk_widths, offsets_v, slice_to_chunk_v);
  
  /* //Allocate the SCS */
  //TODO get the last value of a view from the device
  Kokkos::resize(particle_mask_v, offsets_v(num_slices));
  CreateViews<DataTypes>(scs_data_v, offsets_v(num_slices));

  //Fill the SCS
  const int league_size = num_slices;
  const int team_size = C;
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> team_policy;
  const team_policy policy(league_size, team_size);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const team_policy::member_type& thread) {
    const int slice = thread.league_rank();
    const int slice_row = thread.team_rank();
    const int rowLen = (offsets_v(slice+1)-offsets_v(slice))/team_size;
    const int start = offsets_v(slice) + slice_row;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_size), [=] (int& j) {
      const int row = slice_to_chunk_v(slice) * team_size + slice_row;
      const int element_id = row_to_element_v(row);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, rowLen), [&] (int& p) {
        const int particle_id = start+(p*team_size);
        //TODO? examine ways to avoid the &&
        particle_mask_v(particle_id) =  element_id < num_elems && p < ptcls(element_id).first;
      });
    });
  });
}


template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::destroySCS(bool destroyGid2Row) {
  DestroyArrays<DataTypes>({scs_data});
  
  delete [] slice_to_chunk;
  delete [] row_to_element;
  delete [] offsets;
  delete [] particle_mask;
  if (row_to_element_gid)
    delete [] row_to_element_gid;
  if (destroyGid2Row)
    element_gid_to_row.clear();
}
template<class DataTypes, typename ExecSpace>
SellCSigma<DataTypes, ExecSpace>::~SellCSigma() {
  destroySCS();
}


template<class DataTypes, typename ExecSpace>
template <std::size_t N>
typename MemberTypeAtIndex<N,DataTypes>::type* SellCSigma<DataTypes,ExecSpace>::getSCS() {
  return static_cast<typename MemberTypeAtIndex<N,DataTypes>::type*>(scs_data[N]);
}


template <class T>
typename Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type>::HostMirror deviceToHost(Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type> view) {
  typename Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type>::HostMirror hv = 
    Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(hv, view);
  return hv;
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
  MPI_Alltoall(num_send_particles.data(), 1, MPI_INT, 
               num_recv_particles.data(), 1, MPI_INT, MPI_COMM_WORLD);

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
    offset_send_particles(i+1) += num*final;
    offset_send_particles_temp(i+1) += num*final;
  });
  Kokkos::parallel_scan(comm_size, KOKKOS_LAMBDA(const int& i, int& num, const bool& final) {
    num += num_recv_particles(i);
    offset_recv_particles(i+1) += num*final;
  });
  kkLidHostMirror offset_send_particles_host = deviceToHost(offset_send_particles);
  kkLidHostMirror offset_recv_particles_host = deviceToHost(offset_recv_particles);

  //Create arrays for particles being sent
  int np_send = offset_send_particles_host(comm_size);
  kkLidView send_element("send_element", np_send);
  MemberTypeViews<DataTypes> send_particle;
  //Allocate views for each data type into send_particle[type]
  CreateViews<DataTypes>(send_particle, np_send);
  auto gatherParticlesToSend = SCS_LAMBDA(int element_id, int particle_id, int mask) {
    const int process = new_process[particle_id];
    if (mask && process != comm_rank) {
      const int index = Kokkos::atomic_fetch_add(&(offset_send_particles_temp[process]),1);
      send_element(index) = new_element(particle_id);
      //Copy the values from scs_data[type][particle_id] into send_particle[type](index) for each data type
      //TODO Replace with view to view once scs_data is on the device
      CopyArrayToView<DataTypes>(send_particle, index, scs_data, particle_id);
    }
  };
  parallel_for(gatherParticlesToSend);

  //Create arrays for particles being received
  int np_recv = offset_recv_particles_host(comm_size);
  kkLidView recv_element("recv_element", np_recv);
  MemberTypeViews<DataTypes> recv_particle;
  //Allocate views for each data type into recv_particle[type]
  CreateViews<DataTypes>(recv_particle, np_recv);

  //Get pointers to the data for MPI calls
  int* recv_element_data = recv_element.data();
  int* send_element_data = send_element.data();
  int send_num = 0, recv_num = 0;
  int num_sends = num_sending_to * (num_types + 1);
  int num_recvs = num_receiving_from * (num_types + 1);
  MPI_Request* send_requests = new MPI_Request[num_sends];
  MPI_Request* recv_requests = new MPI_Request[num_recvs];
  //Send the particles to each neighbor
  //TODO? Combine new_element & scs_data arrays into one array to reduce number of sends/recvs
  for (int i = 0; i < comm_size; ++i) {
    if (i == comm_rank)
      continue;
    
    //Sending
    int num_send = offset_send_particles_host(i+1) - offset_send_particles_host(i);
    if (num_send > 0) {
      int start_index = offset_send_particles_host(i);
      MPI_Isend(send_element_data + start_index, num_send, MPI_INT, i, 0,
                MPI_COMM_WORLD, send_requests + send_num);
      send_num++;
      SendViews<DataTypes>(send_particle, start_index, num_send, i, 1,
                           send_requests + send_num);
      send_num+=num_types;
    }
    //Receiving
    int num_recv = offset_recv_particles_host(i+1) - offset_recv_particles_host(i);
    if (num_recv > 0) {
      MPI_Irecv(recv_element_data + offset_recv_particles_host(i), num_recv, MPI_INT, i, 0,
                MPI_COMM_WORLD, recv_requests + recv_num);
      recv_num++;
      RecvViews<DataTypes>(recv_particle,offset_recv_particles_host(i), num_recv, i, 1,
                           recv_requests + recv_num);
      recv_num+=num_types;
    }
  }
  MPI_Waitall(num_sends, send_requests, MPI_STATUSES_IGNORE);
  MPI_Waitall(num_recvs, recv_requests, MPI_STATUSES_IGNORE);
  delete [] send_requests;
  delete [] recv_requests;

  kkLidHostMirror recv_element_host = deviceToHost(recv_element);
  for (int i = 0; i < recv_element_host.size(); ++i)
    printf("Rank %d received particle in element %d\n", comm_rank, recv_element_host(i));

  
  /********** Set particles that went sent to non existent on this process *********/
  auto removeSentParticles = SCS_LAMBDA(int element_id, int particle_id, int mask) {
    new_element(particle_id) -= (new_element(particle_id) + 1) *
                                (new_process(particle_id) != comm_rank);
  };
  parallel_for(removeSentParticles);

  kkLidHostMirror new_element_host = deviceToHost(new_element);
  int* new_element_data = new_element_host.data();
  /********** Combine and shift particles to their new destination **********/
  rebuildSCS(new_element_data, recv_element, recv_particle);

}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes,ExecSpace>::reshuffleSCS(int* new_element) {
  //For now call rebuild
  rebuildSCS(new_element);
}


template<class DataTypes, typename ExecSpace>
  void SellCSigma<DataTypes,ExecSpace>::rebuildSCS(int* new_element, kkLidView new_particle_elements, MemberTypeViews<DataTypes> new_particles) {

  int* new_particles_per_elem = new int[num_chunks*C];
  for (int i =0; i < num_elems; ++i)
    new_particles_per_elem[i] = 0;
  int activePtcls = 0;
  for (int slice = 0; slice < num_slices; ++slice) {
    for (int j = offsets[slice]; j < offsets[slice+1]; j+= C) {
      for (int k = 0; k < C; ++k) {
	if (particle_mask[j+k] && new_element[j+k] != -1) {
	  new_particles_per_elem[new_element[j+k]] += particle_mask[j+k];
          ++activePtcls;
	}
      }
    }
  }
  //Add new particles to counts
  for (int i = 0; i < new_particle_elements.size(); ++i) {
    new_particles_per_elem[new_particle_elements[i]]++;
    ++activePtcls;
  }

  //If there are no particles left, then destroy the structure
  if(!activePtcls) {
    delete [] new_particles_per_elem;
    destroySCS();
    num_ptcls = 0;
    num_chunks = 0;
    num_slices = 0;
    row_to_element = 0;
    row_to_element_gid = 0;
    offsets = 0;
    slice_to_chunk = 0;
    particle_mask = 0;
    for (size_t i = 0; i < num_types; ++i)
      scs_data[i] = 0;
    return;
  }
  int new_num_ptcls = activePtcls;
  //Perform sorting
  std::pair<int, int>* ptcls;
  sigmaSort(num_elems, new_particles_per_elem, sigma, ptcls);


  //Create chunking and count slices
  int* chunk_widths;
  int new_nchunks, new_nslices;
  int* new_row_to_element;
  constructChunks(ptcls, new_nchunks,new_nslices,chunk_widths,new_row_to_element);

  //Create a mapping from element to new scs row index
  //Also recreate the original gid mapping
  int* element_to_new_row = new int[num_chunks * C];
  int* gid_mapping = new int[num_chunks * C];
  for (int i = 0; i < num_chunks * C; ++i) {
    element_to_new_row[new_row_to_element[i]] = i;
    if (row_to_element_gid)
      gid_mapping[row_to_element[i]] = row_to_element_gid[i];
  }

  int* new_row_to_element_gid;
  element_gid_to_row.clear();
  if (row_to_element_gid)
    createGlobalMapping(new_row_to_element, gid_mapping, 
                        new_row_to_element_gid, element_gid_to_row);
  delete [] gid_mapping;

  //Create offsets for each slice
  int* new_offsets;
  int* new_slice_to_chunk;
  constructOffsets(new_nchunks, new_nslices, chunk_widths,new_offsets, new_slice_to_chunk);
  delete [] chunk_widths;

  //Allocate the Chunks
  int* new_particle_mask = new int[new_offsets[new_nslices]];
  std::memset(new_particle_mask,0,new_offsets[new_nslices]*sizeof(int));
  MemberTypeArray<DataTypes> new_scs_data;
  CreateArrays<DataTypes>(new_scs_data, new_offsets[new_nslices]);
  
  //Fill the SCS
  int* element_index = new int[new_nchunks * C];
  int chunk = -1;
  for (int i =0; i < new_nslices; ++i) {
    if ( new_slice_to_chunk[i] == chunk)
      continue;
    chunk = new_slice_to_chunk[i];
    for (int e = 0; e < C; ++e) {
      element_index[chunk*C + e] = new_offsets[i] + e;
    }
  }

  for (int slice = 0; slice < num_slices; ++slice) {
    for (int j = offsets[slice]; j < offsets[slice+1]; j+= C) {
      for (int k = 0; k < C; ++k) {
	int particle = j + k;
	if (particle_mask[particle] && new_element[particle] != -1) {
	  int new_elem = new_element[particle];
          int new_row = element_to_new_row[new_elem];
	  int new_index = element_index[new_row];
	  CopyEntries<DataTypes>(new_scs_data,new_index, scs_data, particle);
	  element_index[new_row] += C;
	  new_particle_mask[new_index] = 1;
	}
      }
    }
  }
  for (int i = 0; i < new_particle_elements.size(); ++i) {
    int new_elem = new_particle_elements[i];
    int new_row = element_to_new_row[new_elem];
    int new_index = element_index[new_row];
    new_particle_mask[new_index] = 1;
    element_index[new_row] += C;
    //TODO copy the data once the scs is on the device
    //CopyViewToView<DataTypes>(new_scs_data, new_index, new_particles, i);
  }
  delete [] element_index;
  delete [] element_to_new_row;
  delete [] ptcls;
  delete [] new_particles_per_elem;

  //Destroy old scs 
  destroySCS(false);

  //set scs to point to new values
  num_ptcls = new_num_ptcls;
  num_chunks = new_nchunks;
  num_slices = new_nslices;
  row_to_element = new_row_to_element;
  if (row_to_element_gid)
    row_to_element_gid = new_row_to_element_gid;
  offsets = new_offsets;
  slice_to_chunk = new_slice_to_chunk;
  particle_mask = new_particle_mask;
  for (size_t i = 0; i < num_types; ++i)
    scs_data[i] = new_scs_data[i];
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes,ExecSpace>::printFormat(const char* prefix) const {
  char message[10000];
  char* cur = message;
  cur += sprintf(cur, "%s\n", prefix);
  cur += sprintf(cur,"Particle Structures Sell-C-Sigma C: %d sigma: %d V: %d.\n", C, sigma, V);
  cur += sprintf(cur,"Number of Elements: %d.\nNumber of Particles: %d.\n", num_elems, num_ptcls);
  cur += sprintf(cur,"Number of Chunks: %d.\nNumber of Slices: %d.\n", num_chunks, num_slices);
  int last_chunk = -1;
  for (int i = 0; i < num_slices; ++i) {
    int chunk = slice_to_chunk[i];
    if (chunk != last_chunk) {
      last_chunk = chunk;
      cur += sprintf(cur,"  Chunk %d. Elements", chunk);
      if (row_to_element_gid)
        cur += sprintf(cur,"(GID)");
      cur += sprintf(cur,":");
      for (int row = chunk*C; row < (chunk+1)*C; ++row) {
        cur += sprintf(cur," %d", row_to_element[row]);
        if (row_to_element_gid) {
          cur += sprintf(cur,"(%d)", row_to_element_gid[row]);
        }
      }
      cur += sprintf(cur,"\n");
    }
    cur += sprintf(cur,"    Slice %d", i);
    for (int j = offsets[i]; j < offsets[i+1]; ++j) {
      if ((j - offsets[i]) % C == 0)
        cur += sprintf(cur," |");
      cur += sprintf(cur," %d", particle_mask[j]);
    }
    cur += sprintf(cur,"\n");
  }
  printf("%s", message);
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes,ExecSpace>::printFormatDevice(const char* prefix) const {
  //Transfer everything to the host
  kkLidHostMirror slice_to_chunk_host = deviceToHost(slice_to_chunk_v);
  kkLidHostMirror row_to_element_gid_host = deviceToHost(row_to_element_gid_v);
  kkLidHostMirror row_to_element_host = deviceToHost(row_to_element_v);
  kkLidHostMirror offsets_host = deviceToHost(offsets_v);
  kkLidHostMirror particle_mask_host = deviceToHost(particle_mask_v);
  char message[10000];
  char* cur = message;
  cur += sprintf(cur, "%s\n", prefix);
  cur += sprintf(cur,"Particle Structures Sell-C-Sigma C: %d sigma: %d V: %d.\n", C, sigma, V);
  cur += sprintf(cur,"Number of Elements: %d.\nNumber of Particles: %d.\n", num_elems, num_ptcls);
  cur += sprintf(cur,"Number of Chunks: %d.\nNumber of Slices: %d.\n", num_chunks, num_slices);
  int last_chunk = -1;
  for (int i = 0; i < num_slices; ++i) {
    int chunk = slice_to_chunk_host(i);
    if (chunk != last_chunk) {
      last_chunk = chunk;
      cur += sprintf(cur,"  Chunk %d. Elements", chunk);
      if (row_to_element_gid_host.size() > 0)
        cur += sprintf(cur,"(GID)");
      cur += sprintf(cur,":");
      for (int row = chunk*C; row < (chunk+1)*C; ++row) {
        cur += sprintf(cur," %d", row_to_element_host(row));
        if (row_to_element_gid_host.size() > 0) {
          cur += sprintf(cur,"(%d)", row_to_element_gid_host(row));
        }
      }
      cur += sprintf(cur,"\n");
    }
    cur += sprintf(cur,"    Slice %d", i);
    for (int j = offsets_host(i); j < offsets_host(i+1); ++j) {
      if ((j - offsets_host(i)) % C == 0)
        cur += sprintf(cur," |");
      cur += sprintf(cur," %d", particle_mask_host(j));
    }
    cur += sprintf(cur,"\n");
  }
  printf("%s", message);
}

template <class T>
void hostToDevice(Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type>& view, T* data) {
  typename Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type>::HostMirror hv = 
    Kokkos::create_mirror_view(view);
  for (size_t i = 0; i < hv.size(); ++i)
    hv(i) = data[i];
  Kokkos::deep_copy(view, hv);
}

template <class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::transferToDevice() {
  offsets_d = kkLidView("offsets_d",num_slices+1);
  hostToDevice(offsets_d,offsets);

  slice_to_chunk_d = kkLidView("slice_to_chunk_d",num_slices);
  hostToDevice(slice_to_chunk_d,slice_to_chunk);

  num_particles_d = kkLidView("num_particles_d",1);
  hostToDevice(num_particles_d, &num_ptcls);

  chunksz_d = kkLidView("chunksz_d",1);
  hostToDevice(chunksz_d,&C);

  row_to_element_d = kkLidView("row_to_element_d", C * num_chunks);
  hostToDevice(row_to_element_d,row_to_element);

  particle_mask_d = kkLidView("particle_mask_d", offsets[num_slices]);
  hostToDevice(particle_mask_d,particle_mask);
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
  const int team_size = C;
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> team_policy;
  const team_policy policy(league_size, team_size);
  auto offsets_cpy = offsets_d;
  auto slice_to_chunk_cpy = slice_to_chunk_d;
  auto row_to_element_cpy = row_to_element_d;
  auto particle_mask_cpy = particle_mask_d;
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
